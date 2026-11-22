/*
 * Resources with CPU-side backing: surfaces, 2D and cube textures, vertex
 * and index buffers, vertex declarations, shaders and queries.
 *
 * Every object has the descriptions, levels and memory the engine asks
 * for, so Lock and Unlock behave and the bridge's bounce copies land in
 * memory that is kept. Texture memory is allocated zeroed on first use,
 * which also matches DXVK's texture zero-init, and is uploaded to the GPU
 * by gpu.m when a draw binds it. Buffer memory is a shared MTLBuffer's
 * contents from the start. Declarations and shader bytecode are copied
 * and kept; shaders are analysed here and translated at their first draw.
 */
#include "mtl_d3d9.h"
#include "shader.h"

#include <stdlib.h>
#include <string.h>

#define MAX_BYTES 0x7FFF0000ull

void mtl_obj_init(mtl_obj *o, void *const *vtbl, mtl_kind kind, mtl_device *dev)
{
    o->vtbl = vtbl;
    o->kind = kind;
    o->refs = 0; /* the creator's mtl_addref hands out the first reference */
    o->priv = 0;
    o->container = NULL;
    o->dev = dev;
}

mtl_obj *mtl_root(mtl_obj *o)
{
    while (o->container) o = o->container;
    return o;
}

static void destroy(mtl_obj *o);

/* A resource the guest holds holds its device, as DXVK's device children
   do: one public device reference while the resource's own public count
   is above zero. The bridge relies on it when the device is released: it
   compares the count Release returns with the three readback surfaces it
   keeps for the luma probe, and releases those itself. */
static int holds_device(const mtl_obj *o) { return o->kind != MTL_K_D3D && o->kind != MTL_K_DEVICE && o->dev; }

uint32_t mtl_addref(mtl_obj *o)
{
    o = mtl_root(o);
    if (++o->refs == 1 && holds_device(o)) mtl_addref(&o->dev->o);
    return o->refs;
}

uint32_t mtl_release(mtl_obj *o)
{
    o = mtl_root(o);
    if (!o->refs) {
        static int once;
        if (!once++) mtl_log("Release of an object with no references left (kind %d)", o->kind);
        return 0;
    }
    uint32_t r = --o->refs;
    if (!r) {
        mtl_device *dev = holds_device(o) ? o->dev : NULL;
        if (!o->priv) destroy(o);
        if (dev) mtl_release(&dev->o);
    }
    return r;
}

void mtl_hold(mtl_obj *o)
{
    if (o) mtl_root(o)->priv++;
}

void mtl_unhold(mtl_obj *o)
{
    if (!o) return;
    o = mtl_root(o);
    if (o->priv && !--o->priv && !o->refs) destroy(o);
}

static void free_surface_backing(mtl_surface *s)
{
    /* A readback surface's memory is its MTLBuffer's contents and goes
       with the buffer; only plain CPU memory is freed here. */
    int in_buffer = s->buf != NULL;
    mtl_surface_gpu_free(s);
    if (!in_buffer) free(s->mem);
    s->mem = NULL;
}

static void destroy(mtl_obj *o)
{
    switch (o->kind) {
    case MTL_K_D3D:
        free(o);
        return;
    case MTL_K_DEVICE:
        mtl_device_destroy((mtl_device *)o);
        return;
    case MTL_K_SURFACE:
        free_surface_backing((mtl_surface *)o);
        free(o);
        return;
    case MTL_K_TEXTURE:
    case MTL_K_CUBE: {
        mtl_texture *t = (mtl_texture *)o;
        mtl_texture_gpu_free(t);
        for (uint32_t i = 0; i < t->levels * t->faces; i++) free_surface_backing(&t->sub[i]);
        if (t->pool == D3DPOOL_DEFAULT && o->dev) o->dev->vidmem -= t->bytes;
        free(t->sub);
        free(t);
        return;
    }
    case MTL_K_VB:
    case MTL_K_IB: {
        mtl_buffer *b = (mtl_buffer *)o;
        if (b->gbuf) mtl_buffer_gpu_free(b);
        else free(b->mem);
        free(o);
        return;
    }
    case MTL_K_DECL:
    case MTL_K_VS:
    case MTL_K_PS:
        mtl_blob_gpu_free((mtl_blob *)o);
        free(((mtl_blob *)o)->data);
        free(o);
        return;
    case MTL_K_QUERY:
        free(o);
        return;
    default:
        mtl_log("destroy of unknown kind %d", o->kind);
        return;
    }
}

/* Shared by every interface: the object pointer comes first and the ABI is
   the same whatever the declared type. */
static uint32_t obj_addref(mtl_obj *o) { return mtl_addref(o); }
static uint32_t obj_release(mtl_obj *o) { return mtl_release(o); }
static void obj_preload(mtl_obj *o) { (void)o; }

static HRESULT obj_device(mtl_obj *o, mtl_device **pp)
{
    if (!pp) return D3DERR_INVALIDCALL;
    *pp = o->dev;
    if (!o->dev) return D3DERR_INVALIDCALL;
    mtl_addref(&o->dev->o);
    return D3D_OK;
}

/* Surfaces */

static void surface_fields(mtl_surface *s, uint32_t w, uint32_t h, uint32_t fmt, uint32_t pool, uint32_t usage)
{
    s->format = fmt;
    s->pool = pool;
    s->usage = usage;
    s->width = w;
    s->height = h;
    s->pitch = (uint32_t)mtl_fmt_pitch(fmt, w);
    s->rows = (uint32_t)mtl_fmt_rows(fmt, h);
}

mtl_surface *mtl_surface_new(mtl_device *d, uint32_t w, uint32_t h, uint32_t fmt, uint32_t pool, uint32_t usage)
{
    if (!w || !h || mtl_fmt_pitch(fmt, w) * mtl_fmt_rows(fmt, h) > MAX_BYTES) return NULL;
    mtl_surface *s = calloc(1, sizeof *s);
    if (!s) return NULL;
    mtl_obj_init(&s->o, mtl_vt_surface, MTL_K_SURFACE, d);
    surface_fields(s, w, h, fmt, pool, usage);
    return s;
}

uint8_t *mtl_surface_mem(mtl_surface *s)
{
    if (!s->mem && !s->no_cpu) s->mem = calloc(1, (size_t)s->pitch * s->rows + 16);
    return s->mem;
}

static HRESULT surface_desc(mtl_surface *s, mtl_surface_desc *d)
{
    if (!d) return D3DERR_INVALIDCALL;
    *d = (mtl_surface_desc){s->format, D3DRTYPE_SURFACE, s->usage, s->pool, 0, 0, s->width, s->height};
    return D3D_OK;
}

static HRESULT surface_lock(mtl_surface *s, mtl_locked_rect *lr, const mtl_rect *r, uint32_t flags)
{
    if (!lr || s->no_cpu || s->locked) return D3DERR_INVALIDCALL;
    /* UpdateTexture copies a source texture only while it is dirty, and a
       lock dirties it unless the engine says otherwise. */
    if (s->o.container && !(flags & 0x8000u)) ((mtl_texture *)s->o.container)->update_dirty = 1;
    size_t off = 0;
    if (r) {
        if (r->left < 0 || r->top < 0 || r->right <= r->left || r->bottom <= r->top || (uint32_t)r->right > s->width ||
            (uint32_t)r->bottom > s->height)
            return D3DERR_INVALIDCALL;
        uint32_t x = (uint32_t)r->left, y = (uint32_t)r->top;
        if (mtl_fmt_dxt(s->format)) off = (size_t)(y / 4) * s->pitch + (size_t)(x / 4) * (s->format == D3DFMT_DXT1 ? 8 : 16);
        else off = (size_t)y * s->pitch + (size_t)x * (mtl_fmt_bits(s->format) / 8);
    }
    mtl_surface_sync(s);
    uint8_t *m = mtl_surface_mem(s);
    if (!m) return E_OUTOFMEMORY;
    lr->Pitch = (int32_t)s->pitch;
    lr->pBits = m + off;
    s->locked = 1;
    return D3D_OK;
}

static HRESULT surface_unlock(mtl_surface *s)
{
    s->locked = 0;
    s->dirty = 1;
    if (s->o.container) ((mtl_texture *)s->o.container)->dirty = 1;
    return D3D_OK;
}

static uint32_t surface_set_priority(mtl_surface *s, uint32_t p) { (void)s; (void)p; return 0; }
static uint32_t surface_get_priority(mtl_surface *s) { (void)s; return 0; }
static uint32_t surface_type(mtl_surface *s) { (void)s; return D3DRTYPE_SURFACE; }

/* Textures, 2D and cube. A cube is six faces of the same chain. */

static uint32_t full_chain(uint32_t w, uint32_t h)
{
    uint32_t n = 1, m = w > h ? w : h;
    while (m > 1) {
        m >>= 1;
        n++;
    }
    return n;
}

static uint32_t level_dim(uint32_t v, uint32_t l)
{
    v >>= l;
    return v ? v : 1;
}

HRESULT mtl_texture_new(mtl_device *d, int cube, uint32_t w, uint32_t h, uint32_t levels, uint32_t usage, uint32_t fmt,
                        uint32_t pool, mtl_texture **out)
{
    *out = NULL;
    if (!w || !h) return D3DERR_INVALIDCALL;
    uint32_t n = full_chain(w, h);
    if (usage & D3DUSAGE_AUTOGENMIPMAP) levels = 1;
    if (!levels || levels > n) levels = n;
    uint32_t faces = cube ? 6 : 1;
    uint64_t bytes = 0;
    for (uint32_t l = 0; l < levels; l++) {
        uint64_t b = mtl_fmt_pitch(fmt, level_dim(w, l)) * mtl_fmt_rows(fmt, level_dim(h, l));
        if (b > MAX_BYTES) return E_OUTOFMEMORY;
        bytes += b * faces;
    }
    if (pool == D3DPOOL_DEFAULT && d->vidmem + bytes > MTL_VIDMEM_CAP) return D3DERR_OUTOFVIDEOMEMORY;
    mtl_texture *t = calloc(1, sizeof *t);
    mtl_surface *sub = calloc((size_t)levels * faces, sizeof *sub);
    if (!t || !sub) {
        free(t);
        free(sub);
        return E_OUTOFMEMORY;
    }
    mtl_obj_init(&t->o, cube ? mtl_vt_cube : mtl_vt_texture, cube ? MTL_K_CUBE : MTL_K_TEXTURE, d);
    t->format = fmt;
    t->usage = usage;
    t->pool = pool;
    t->width = w;
    t->height = h;
    t->levels = levels;
    t->faces = faces;
    t->filter = 2; /* D3DTEXF_LINEAR, the D3D9 default */
    t->bytes = bytes;
    t->sub = sub;
    t->update_dirty = 1; /* a new texture is dirty as a whole, as in D3D9 */
    for (uint32_t f = 0; f < faces; f++)
        for (uint32_t l = 0; l < levels; l++) {
            mtl_surface *s = &sub[f * levels + l];
            mtl_obj_init(&s->o, mtl_vt_surface, MTL_K_SURFACE, d);
            s->o.container = &t->o;
            surface_fields(s, level_dim(w, l), level_dim(h, l), fmt, pool, usage);
        }
    if (pool == D3DPOOL_DEFAULT) d->vidmem += bytes;
    mtl_addref(&t->o);
    *out = t;
    return D3D_OK;
}

static mtl_surface *level_of(mtl_texture *t, uint32_t face, uint32_t level)
{
    return face < t->faces && level < t->levels ? &t->sub[face * t->levels + level] : NULL;
}

static uint32_t tex_set_priority(mtl_texture *t, uint32_t p)
{
    uint32_t old = t->priority;
    t->priority = p;
    return old;
}

static uint32_t tex_get_priority(mtl_texture *t) { return t->priority; }
static uint32_t tex_type(mtl_texture *t) { return t->faces == 6 ? D3DRTYPE_CUBETEXTURE : D3DRTYPE_TEXTURE; }

static uint32_t tex_set_lod(mtl_texture *t, uint32_t lod)
{
    uint32_t old = t->lod;
    t->lod = lod < t->levels ? lod : t->levels - 1;
    return old;
}

static uint32_t tex_get_lod(mtl_texture *t) { return t->lod; }
static uint32_t tex_levels(mtl_texture *t) { return t->levels; }

static HRESULT tex_set_filter(mtl_texture *t, uint32_t f)
{
    t->filter = f;
    return D3D_OK;
}

static uint32_t tex_get_filter(mtl_texture *t) { return t->filter; }
static void tex_genmips(mtl_texture *t) { (void)t; }

static HRESULT tex_level_desc(mtl_texture *t, uint32_t level, mtl_surface_desc *d)
{
    mtl_surface *s = level_of(t, 0, level);
    return s ? surface_desc(s, d) : D3DERR_INVALIDCALL;
}

static HRESULT tex_surface(mtl_texture *t, uint32_t level, mtl_surface **pp)
{
    if (!pp) return D3DERR_INVALIDCALL;
    mtl_surface *s = level_of(t, 0, level);
    *pp = s;
    if (!s) return D3DERR_INVALIDCALL;
    mtl_addref(&s->o);
    return D3D_OK;
}

static HRESULT tex_lock(mtl_texture *t, uint32_t level, mtl_locked_rect *lr, const mtl_rect *r, uint32_t flags)
{
    mtl_surface *s = level_of(t, 0, level);
    return s ? surface_lock(s, lr, r, flags) : D3DERR_INVALIDCALL;
}

static HRESULT tex_unlock(mtl_texture *t, uint32_t level)
{
    mtl_surface *s = level_of(t, 0, level);
    return s ? surface_unlock(s) : D3DERR_INVALIDCALL;
}

static HRESULT tex_dirty(mtl_texture *t, const mtl_rect *r)
{
    (void)r;
    t->update_dirty = 1;
    return D3D_OK;
}

static HRESULT cube_surface(mtl_texture *t, uint32_t face, uint32_t level, mtl_surface **pp)
{
    if (!pp) return D3DERR_INVALIDCALL;
    mtl_surface *s = level_of(t, face, level);
    *pp = s;
    if (!s) return D3DERR_INVALIDCALL;
    mtl_addref(&s->o);
    return D3D_OK;
}

static HRESULT cube_lock(mtl_texture *t, uint32_t face, uint32_t level, mtl_locked_rect *lr, const mtl_rect *r,
                         uint32_t flags)
{
    mtl_surface *s = level_of(t, face, level);
    return s ? surface_lock(s, lr, r, flags) : D3DERR_INVALIDCALL;
}

static HRESULT cube_unlock(mtl_texture *t, uint32_t face, uint32_t level)
{
    mtl_surface *s = level_of(t, face, level);
    return s ? surface_unlock(s) : D3DERR_INVALIDCALL;
}

static HRESULT cube_dirty(mtl_texture *t, uint32_t face, const mtl_rect *r)
{
    (void)face;
    (void)r;
    t->update_dirty = 1;
    return D3D_OK;
}

/* Vertex and index buffers */

HRESULT mtl_buffer_new(mtl_device *d, mtl_kind kind, uint32_t len, uint32_t usage, uint32_t fmt_or_fvf, uint32_t pool,
                       mtl_buffer **out)
{
    *out = NULL;
    if (!len || (uint64_t)len + 16 > MAX_BYTES) return D3DERR_INVALIDCALL;
    mtl_buffer *b = calloc(1, sizeof *b);
    if (!b) return E_OUTOFMEMORY;
    mtl_obj_init(&b->o, kind == MTL_K_VB ? mtl_vt_vb : mtl_vt_ib, kind, d);
    b->length = len;
    b->usage = usage;
    b->format_or_fvf = fmt_or_fvf;
    b->pool = pool;
    if (!mtl_buffer_gpu_init(b)) {
        free(b);
        return E_OUTOFMEMORY;
    }
    mtl_addref(&b->o);
    *out = b;
    return D3D_OK;
}

static uint32_t buf_set_priority(mtl_buffer *b, uint32_t p)
{
    uint32_t old = b->priority;
    b->priority = p;
    return old;
}

static uint32_t buf_get_priority(mtl_buffer *b) { return b->priority; }
static uint32_t buf_type(mtl_buffer *b) { return b->o.kind == MTL_K_VB ? D3DRTYPE_VERTEXBUFFER : D3DRTYPE_INDEXBUFFER; }

static HRESULT buf_lock(mtl_buffer *b, uint32_t off, uint32_t size, void **pp, uint32_t flags)
{
    (void)size;
    if (!pp) return D3DERR_INVALIDCALL;
    *pp = NULL;
    if (off > b->length) return D3DERR_INVALIDCALL;
    mtl_buffer_lock_prepare(b, flags);
    *pp = b->mem + off;
    return D3D_OK;
}

static HRESULT buf_unlock(mtl_buffer *b) { (void)b; return D3D_OK; }

static HRESULT vb_desc(mtl_buffer *b, uint32_t *d)
{
    if (!d) return D3DERR_INVALIDCALL;
    uint32_t v[6] = {D3DFMT_VERTEXDATA, D3DRTYPE_VERTEXBUFFER, b->usage, b->pool, b->length, b->format_or_fvf};
    memcpy(d, v, sizeof v);
    return D3D_OK;
}

static HRESULT ib_desc(mtl_buffer *b, uint32_t *d)
{
    if (!d) return D3DERR_INVALIDCALL;
    uint32_t v[5] = {b->format_or_fvf, D3DRTYPE_INDEXBUFFER, b->usage, b->pool, b->length};
    memcpy(d, v, sizeof v);
    return D3D_OK;
}

/* Declarations and shaders: kept as copies of what the engine handed in. */

static uint32_t decl_bytes(const uint8_t *e)
{
    /* D3DVERTEXELEMENT9 is 8 bytes; the list ends with Stream 0xFF and
       Type D3DDECLTYPE_UNUSED (17), and holds at most 64 elements before it. */
    for (uint32_t i = 0; i <= 64; i++)
        if (e[8 * i] == 0xFF && e[8 * i + 1] == 0x00 && e[8 * i + 4] == 17) return 8 * (i + 1);
    return 0;
}

static uint32_t shader_bytes(const uint32_t *t)
{
    /* Walk the token stream to the end token. Comments carry their length
       in bits 16 to 30 and are skipped whole, since their payload may hold
       any value. From shader model 2 on, instructions carry their length in
       bits 24 to 27. Before that, parameter tokens have bit 31 set and
       instruction tokens do not, so only an instruction token can be a
       comment or the end; the one exception is def (0x51), whose four
       float literals are raw values and are skipped with it, six tokens
       in all. */
    uint32_t major = (t[0] >> 8) & 0xFF;
    for (uint32_t i = 1; i < (1u << 20);) {
        uint32_t k = t[i];
        int instr = !(k & 0x80000000u);
        if (k == 0x0000FFFFu) return 4 * (i + 1);
        if (instr && (k & 0xFFFF) == 0xFFFE) i += 1 + ((k >> 16) & 0x7FFF);
        else if (instr && major >= 2) i += 1 + ((k >> 24) & 0xF);
        else if (instr && (k & 0xFFFF) == 0x51) i += 6;
        else i++;
    }
    return 0;
}

HRESULT mtl_blob_new(mtl_device *d, mtl_kind kind, const void *src, mtl_blob **out)
{
    *out = NULL;
    if (!src) return D3DERR_INVALIDCALL;
    uint32_t n = kind == MTL_K_DECL ? decl_bytes(src) : shader_bytes(src);
    if (!n) return D3DERR_INVALIDCALL;
    mtl_blob *b = calloc(1, sizeof *b);
    uint8_t *m = malloc(n);
    if (!b || !m) {
        free(b);
        free(m);
        return E_OUTOFMEMORY;
    }
    memcpy(m, src, n);
    struct mtl_sc *sc = NULL;
    if (kind != MTL_K_DECL) {
        char err[512];
        sc = mtl_sc_new(m, n, err, sizeof err);
        if (!sc) {
            mtl_log("shader analysis failed (version 0x%08X, %u bytes): %s", *(const uint32_t *)src, n, err);
            free(b);
            free(m);
            return D3DERR_INVALIDCALL;
        }
    }
    void *const *vt = kind == MTL_K_DECL ? mtl_vt_decl : kind == MTL_K_VS ? mtl_vt_vs : mtl_vt_ps;
    mtl_obj_init(&b->o, vt, kind, d);
    b->bytes = n;
    b->version = kind == MTL_K_DECL ? 0 : *(const uint32_t *)src;
    b->data = m;
    b->uid = mtl_next_uid();
    b->sc = sc;
    /* D3DDECLUSAGE_POSITIONT (9) means pretransformed vertices, which DXVK
       draws with its fixed-function vertex shader whatever is bound. The
       other facts are the ones that shader reads (DXVK's Classify). */
    for (uint32_t i = 0; kind == MTL_K_DECL && i + 1 < n / 8; i++) {
        static const uint8_t k_count[17] = {1, 2, 3, 4, 4, 4, 2, 4, 4, 2, 4, 2, 4, 3, 3, 2, 4};
        uint8_t type = m[8 * i + 4], usage = m[8 * i + 6], index = m[8 * i + 7];
        if (usage == 10 && index == 0) b->decl_flags |= MTL_DECL_COLOR0;
        else if (usage == 10 && index == 1) b->decl_flags |= MTL_DECL_COLOR1;
        else if (usage == 9) b->decl_flags |= MTL_DECL_POSITIONT;
        else if (usage == 4) b->decl_flags |= MTL_DECL_PSIZE;
        else if (usage == 11) b->decl_flags |= MTL_DECL_FOG;
        else if (usage == 1) b->decl_flags |= MTL_DECL_BLENDWEIGHT;
        else if (usage == 2) b->decl_flags |= MTL_DECL_BLENDINDICES;
        if (usage == 5 && index < 8 && type < 17) b->tc_mask |= (uint32_t)k_count[type] << (index * 3);
    }
    b->has_post = (b->decl_flags & MTL_DECL_POSITIONT) != 0;
    mtl_addref(&b->o);
    *out = b;
    return D3D_OK;
}

static HRESULT decl_get(mtl_blob *b, void *elems, uint32_t *num)
{
    if (!num) return D3DERR_INVALIDCALL;
    *num = b->bytes / 8;
    if (elems) memcpy(elems, b->data, b->bytes);
    return D3D_OK;
}

static HRESULT shader_function(mtl_blob *b, void *data, uint32_t *size)
{
    if (!size) return D3DERR_INVALIDCALL;
    if (data) {
        if (*size < b->bytes) return D3DERR_INVALIDCALL;
        memcpy(data, b->data, b->bytes);
    }
    *size = b->bytes;
    return D3D_OK;
}

/* Queries. Nothing is drawn, so every query has its answer at once: GetData
   returns S_OK with a one in the first DWORD, which reads as "event done"
   and "one pixel passed". An answer of S_FALSE would leave the engine
   spinning on it. */

HRESULT mtl_query_new(mtl_device *d, uint32_t type, mtl_query **out)
{
    mtl_query *q = calloc(1, sizeof *q);
    *out = q;
    if (!q) return E_OUTOFMEMORY;
    mtl_obj_init(&q->o, mtl_vt_query, MTL_K_QUERY, d);
    q->type = type;
    mtl_addref(&q->o);
    return D3D_OK;
}

static uint32_t query_type(mtl_query *q) { return q->type; }

static uint32_t query_size(mtl_query *q)
{
    switch (q->type) {
    case 4: return 16;           /* VCACHE */
    case 8: case 9: case 11: return 4; /* EVENT, OCCLUSION, TIMESTAMPDISJOINT */
    case 10: case 12: return 8;  /* TIMESTAMP, TIMESTAMPFREQ */
    default: return 4;
    }
}

static HRESULT query_issue(mtl_query *q, uint32_t flags) { (void)q; (void)flags; return D3D_OK; }

static HRESULT query_data(mtl_query *q, void *data, uint32_t size, uint32_t flags)
{
    (void)q;
    (void)flags;
    if (data && size) {
        memset(data, 0, size);
        if (size >= 4) *(uint32_t *)data = 1;
    }
    return D3D_OK;
}

void mtl_fill_resources(void)
{
    void **v = mtl_vt_surface;
    v[SLOT_IDirect3DSurface9_AddRef] = (void *)obj_addref;
    v[SLOT_IDirect3DSurface9_Release] = (void *)obj_release;
    v[SLOT_IDirect3DSurface9_GetDevice] = (void *)obj_device;
    v[SLOT_IDirect3DSurface9_SetPriority] = (void *)surface_set_priority;
    v[SLOT_IDirect3DSurface9_GetPriority] = (void *)surface_get_priority;
    v[SLOT_IDirect3DSurface9_PreLoad] = (void *)obj_preload;
    v[SLOT_IDirect3DSurface9_GetType] = (void *)surface_type;
    v[SLOT_IDirect3DSurface9_GetDesc] = (void *)surface_desc;
    v[SLOT_IDirect3DSurface9_LockRect] = (void *)surface_lock;
    v[SLOT_IDirect3DSurface9_UnlockRect] = (void *)surface_unlock;

#define TEXTURE_COMMON(V, I) \
    V[SLOT_##I##_AddRef] = (void *)obj_addref; \
    V[SLOT_##I##_Release] = (void *)obj_release; \
    V[SLOT_##I##_GetDevice] = (void *)obj_device; \
    V[SLOT_##I##_SetPriority] = (void *)tex_set_priority; \
    V[SLOT_##I##_GetPriority] = (void *)tex_get_priority; \
    V[SLOT_##I##_PreLoad] = (void *)obj_preload; \
    V[SLOT_##I##_GetType] = (void *)tex_type; \
    V[SLOT_##I##_SetLOD] = (void *)tex_set_lod; \
    V[SLOT_##I##_GetLOD] = (void *)tex_get_lod; \
    V[SLOT_##I##_GetLevelCount] = (void *)tex_levels; \
    V[SLOT_##I##_SetAutoGenFilterType] = (void *)tex_set_filter; \
    V[SLOT_##I##_GetAutoGenFilterType] = (void *)tex_get_filter; \
    V[SLOT_##I##_GenerateMipSubLevels] = (void *)tex_genmips; \
    V[SLOT_##I##_GetLevelDesc] = (void *)tex_level_desc;
    TEXTURE_COMMON(mtl_vt_texture, IDirect3DTexture9)
    TEXTURE_COMMON(mtl_vt_cube, IDirect3DCubeTexture9)
#undef TEXTURE_COMMON
    mtl_vt_texture[SLOT_IDirect3DTexture9_GetSurfaceLevel] = (void *)tex_surface;
    mtl_vt_texture[SLOT_IDirect3DTexture9_LockRect] = (void *)tex_lock;
    mtl_vt_texture[SLOT_IDirect3DTexture9_UnlockRect] = (void *)tex_unlock;
    mtl_vt_texture[SLOT_IDirect3DTexture9_AddDirtyRect] = (void *)tex_dirty;
    mtl_vt_cube[SLOT_IDirect3DCubeTexture9_GetCubeMapSurface] = (void *)cube_surface;
    mtl_vt_cube[SLOT_IDirect3DCubeTexture9_LockRect] = (void *)cube_lock;
    mtl_vt_cube[SLOT_IDirect3DCubeTexture9_UnlockRect] = (void *)cube_unlock;
    mtl_vt_cube[SLOT_IDirect3DCubeTexture9_AddDirtyRect] = (void *)cube_dirty;

#define BUFFER_COMMON(V, I) \
    V[SLOT_##I##_AddRef] = (void *)obj_addref; \
    V[SLOT_##I##_Release] = (void *)obj_release; \
    V[SLOT_##I##_GetDevice] = (void *)obj_device; \
    V[SLOT_##I##_SetPriority] = (void *)buf_set_priority; \
    V[SLOT_##I##_GetPriority] = (void *)buf_get_priority; \
    V[SLOT_##I##_PreLoad] = (void *)obj_preload; \
    V[SLOT_##I##_GetType] = (void *)buf_type; \
    V[SLOT_##I##_Lock] = (void *)buf_lock; \
    V[SLOT_##I##_Unlock] = (void *)buf_unlock;
    BUFFER_COMMON(mtl_vt_vb, IDirect3DVertexBuffer9)
    BUFFER_COMMON(mtl_vt_ib, IDirect3DIndexBuffer9)
#undef BUFFER_COMMON
    mtl_vt_vb[SLOT_IDirect3DVertexBuffer9_GetDesc] = (void *)vb_desc;
    mtl_vt_ib[SLOT_IDirect3DIndexBuffer9_GetDesc] = (void *)ib_desc;

    mtl_vt_decl[SLOT_IDirect3DVertexDeclaration9_AddRef] = (void *)obj_addref;
    mtl_vt_decl[SLOT_IDirect3DVertexDeclaration9_Release] = (void *)obj_release;
    mtl_vt_decl[SLOT_IDirect3DVertexDeclaration9_GetDevice] = (void *)obj_device;
    mtl_vt_decl[SLOT_IDirect3DVertexDeclaration9_GetDeclaration] = (void *)decl_get;
    mtl_vt_vs[SLOT_IDirect3DVertexShader9_AddRef] = (void *)obj_addref;
    mtl_vt_vs[SLOT_IDirect3DVertexShader9_Release] = (void *)obj_release;
    mtl_vt_vs[SLOT_IDirect3DVertexShader9_GetDevice] = (void *)obj_device;
    mtl_vt_vs[SLOT_IDirect3DVertexShader9_GetFunction] = (void *)shader_function;
    mtl_vt_ps[SLOT_IDirect3DPixelShader9_AddRef] = (void *)obj_addref;
    mtl_vt_ps[SLOT_IDirect3DPixelShader9_Release] = (void *)obj_release;
    mtl_vt_ps[SLOT_IDirect3DPixelShader9_GetDevice] = (void *)obj_device;
    mtl_vt_ps[SLOT_IDirect3DPixelShader9_GetFunction] = (void *)shader_function;

    v = mtl_vt_query;
    v[SLOT_IDirect3DQuery9_AddRef] = (void *)obj_addref;
    v[SLOT_IDirect3DQuery9_Release] = (void *)obj_release;
    v[SLOT_IDirect3DQuery9_GetDevice] = (void *)obj_device;
    v[SLOT_IDirect3DQuery9_GetType] = (void *)query_type;
    v[SLOT_IDirect3DQuery9_GetDataSize] = (void *)query_size;
    v[SLOT_IDirect3DQuery9_Issue] = (void *)query_issue;
    v[SLOT_IDirect3DQuery9_GetData] = (void *)query_data;
}
