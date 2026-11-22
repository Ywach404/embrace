/*
 * The draw path: render passes and clears, the device state setters, and
 * DrawPrimitive / DrawIndexedPrimitive through translated game shaders.
 *
 * Each call is split in two. Recording (game thread) resolves device
 * state into a command that holds everything by value: pipeline and
 * depth-stencil keys, constants, push and spec data already in the
 * upload ring, and the storage of every buffer and texture as it is now.
 * Replay (the mtl_emit_* functions, on the render thread, or at once on
 * the game thread with RT_METAL_INLINE=1) reads only that command and
 * the replay-side caches: shader compilation, pipelines, depth-stencil
 * states and the encoder all live there.
 *
 * Stages are linked at pipeline creation. shader.cpp translates each
 * shader alone, and every sm1/sm2 vertex shader writes the same twelve
 * output locations, a superset of what any pixel shader reads, with the
 * same types; Metal matches [[user(locnN)]] by name, so any pair links
 * with no per-pair recompile.
 *
 * State DXVK keeps in specialisation constants (alpha test, fog, sampler
 * types, clip plane count, boolean constants) reaches the shaders in the
 * 80-byte spec data buffer DXVK itself lowers them to when it compiles
 * without a linkage. A pipeline therefore depends only on the shader
 * pair, the vertex layout, blend state and the attachment formats.
 *
 * One recorded render pass stays open across draws until the targets
 * change or a blit (readback, Present, StretchRect) needs the command
 * buffer. A Clear
 * of a whole target becomes the next pass's load action; a partial Clear
 * is a draw under the scissor.
 *
 * Fixed-function draws (no vertex or pixel shader bound, or POSITIONT
 * vertices) go through DXVK's two fixed-function ubershaders, fed what
 * DXVK's device feeds them: the ShaderData block its
 * UpdateFixedFunctionVS builds (transforms, lights, material, texcoord
 * and vertex facts), the texture stage ops and arguments in spec data
 * words 8 to 19 as its UpdateFixedFunctionPS packs them, the stage
 * constants and bump matrices in the shared pixel CBV, and the texture
 * factor and point scale in push data. Either stage may be fixed while
 * the other is the game's: all of them write and read the same twelve
 * locations.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "mtl_objc.h"
#include "pipelines.h"
#include "shader.h"
#include "stereo.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MTLD(d) ((__bridge id<MTLDevice>)(d)->mtl)
#define ENC(e) ((__bridge id<MTLRenderCommandEncoder>)(e)->enc)
#define CB(e) ((__bridge id<MTLCommandBuffer>)(e)->cb)
#define OBJ(p) ((__bridge id)(p))

/* D3DRENDERSTATETYPE values used here */
enum {
    RS_ZENABLE = 7, RS_FILLMODE = 8, RS_SHADEMODE = 9, RS_ZWRITEENABLE = 14, RS_ALPHATESTENABLE = 15,
    RS_LASTPIXEL = 16, RS_SRCBLEND = 19, RS_DESTBLEND = 20, RS_CULLMODE = 22, RS_ZFUNC = 23, RS_ALPHAREF = 24,
    RS_ALPHAFUNC = 25, RS_ALPHABLENDENABLE = 27, RS_FOGENABLE = 28, RS_FOGCOLOR = 34, RS_FOGTABLEMODE = 35,
    RS_FOGSTART = 36, RS_FOGEND = 37, RS_FOGDENSITY = 38, RS_STENCILENABLE = 52, RS_STENCILFAIL = 53,
    RS_STENCILZFAIL = 54, RS_STENCILPASS = 55, RS_STENCILFUNC = 56, RS_STENCILREF = 57, RS_STENCILMASK = 58,
    RS_STENCILWRITEMASK = 59, RS_TEXTUREFACTOR = 60, RS_CLIPPING = 136, RS_LIGHTING = 137,
    RS_FOGVERTEXMODE = 140, RS_COLORVERTEX = 141, RS_LOCALVIEWER = 142, RS_DIFFUSEMATERIALSOURCE = 145,
    RS_SPECULARMATERIALSOURCE = 146, RS_CLIPPLANEENABLE = 152, RS_POINTSIZE = 154, RS_POINTSIZE_MIN = 155,
    RS_POINTSPRITEENABLE = 156, RS_POINTSCALE_A = 158, RS_MULTISAMPLEANTIALIAS = 161,
    RS_MULTISAMPLEMASK = 162, RS_POINTSIZE_MAX = 166, RS_COLORWRITEENABLE = 168, RS_BLENDOP = 171,
    RS_POSITIONDEGREE = 172, RS_NORMALDEGREE = 173, RS_SCISSORTESTENABLE = 174, RS_SLOPESCALEDEPTHBIAS = 175,
    RS_MINTESSELLATIONLEVEL = 178, RS_MAXTESSELLATIONLEVEL = 179, RS_TWOSIDEDSTENCILMODE = 185,
    RS_CCW_STENCILFAIL = 186, RS_CCW_STENCILZFAIL = 187, RS_CCW_STENCILPASS = 188, RS_CCW_STENCILFUNC = 189,
    RS_COLORWRITEENABLE1 = 190, RS_COLORWRITEENABLE2 = 191, RS_COLORWRITEENABLE3 = 192, RS_BLENDFACTOR = 193,
    RS_SRGBWRITEENABLE = 194, RS_DEPTHBIAS = 195, RS_SEPARATEALPHABLENDENABLE = 206, RS_SRCBLENDALPHA = 207,
    RS_DESTBLENDALPHA = 208, RS_BLENDOPALPHA = 209, RS_SPECULARENABLE = 29, RS_POINTSCALEENABLE = 157,
    RS_POINTSCALE_B = 159, RS_POINTSCALE_C = 160,
};

/* D3DSAMPLERSTATETYPE */
enum {
    SS_ADDRESSU = 1, SS_ADDRESSV = 2, SS_ADDRESSW = 3, SS_BORDERCOLOR = 4, SS_MAGFILTER = 5, SS_MINFILTER = 6,
    SS_MIPFILTER = 7, SS_MIPMAPLODBIAS = 8, SS_MAXMIPLEVEL = 9, SS_MAXANISOTROPY = 10, SS_SRGBTEXTURE = 11,
};

#define TSS_COLOROP 1
#define TSS_COLORARG1 2
#define TSS_COLORARG2 3
#define TSS_ALPHAOP 4
#define TSS_ALPHAARG1 5
#define TSS_ALPHAARG2 6
#define TSS_BUMPENVMAT00 7
#define TSS_TEXCOORDINDEX 11
#define TSS_BUMPENVLSCALE 22
#define TSS_BUMPENVLOFFSET 23
#define TSS_TEXTURETRANSFORMFLAGS 24
#define TSS_COLORARG0 26
#define TSS_ALPHAARG0 27
#define TSS_RESULTARG 28
#define TSS_CONSTANT 32

/* D3DTEXTUREOP and D3DTA values the stage packing tests */
#define TOP_DISABLE 1
#define TOP_SELECTARG1 2
#define TA_DIFFUSE 0
#define TA_CURRENT 1
#define TA_TEXTURE 2
#define TA_TEMP 5

/* The fixed-function pixel shader's push data after the shared block:
   the texture factor, then the sampler heap indices of stages 0 to 7
   (DXVK's GetPushSamplerOffset). */
#define FF_PS_PUSH_TFACTOR 32
#define FF_PS_PUSH_SAMPLERS 36
/* and the vertex shader's point scale, after its point size words */
#define FF_VS_PUSH_SCALE 40

_Static_assert(MTL_NXF == MTL_XF_N, "transform slots agree with shader.h");

#define HEAP_MAX 4096
#define FF_KEYS 256
#define RING_CHUNK (4u << 20)
#define RING_MAX 64

static float as_float(uint32_t v)
{
    float f;
    memcpy(&f, &v, 4);
    return f;
}

static uint32_t as_bits(float f)
{
    uint32_t v;
    memcpy(&v, &f, 4);
    return v;
}

static int clip_rect(mtl_rect *r, const mtl_rect *by)
{
    if (r->left < by->left) r->left = by->left;
    if (r->top < by->top) r->top = by->top;
    if (r->right > by->right) r->right = by->right;
    if (r->bottom > by->bottom) r->bottom = by->bottom;
    return r->right > r->left && r->bottom > r->top;
}

uint64_t mtl_next_uid(void)
{
    static _Atomic uint64_t n;
    return ++n;
}

/* The per-device caches. */

typedef struct {
    void *buf; /* id<MTLBuffer> */
    uint8_t *base; /* its contents */
    uint64_t serial;
} ring_chunk;

/* Recorded commands (frame.m holds the lists). Objective-C objects are
   plain pointers here: the list retains each one it records (or it is one
   of the cache's own, which live as long as the device), so a replay
   finds them as they were when recorded. */
enum { CMD_PASS = 1, CMD_END_PASS, CMD_DRAW, CMD_CLEAR, CMD_READBACK, CMD_UPLOAD, CMD_STRETCH, CMD_FILL, CMD_PRESENT };

typedef struct {
    uint32_t type, bytes;
} cmd_head;

typedef struct {
    cmd_head h;
    void *ct, *dt; /* colour and depth-stencil textures; dt may be NULL */
    uint32_t level, slice, width, height;
    uint32_t clear, color, stencil; /* D3DCLEAR flags turned into load actions */
    float z;
} cmd_pass;

typedef struct {
    cmd_head h;
    uint32_t pf, has_ds, flags, color, stencil, width, height, n;
    float z;
    uint32_t pad;
    mtl_rect r[];
} cmd_clear;

typedef struct {
    cmd_head h;
    void *src, *dst; /* texture, shared buffer */
    uint32_t level, slice, width, height, pitch, rows;
} cmd_readback;

typedef struct {
    cmd_head h;
    void *src, *dst; /* staging buffer, texture */
    uint32_t row_bytes, rows, width, height, level, slice, x, y;
} cmd_upload;

typedef struct {
    cmd_head h;
    void *src, *dst;
    uint32_t src_level, src_slice, dst_level, dst_slice, filter;
    mtl_rect s, t;
} cmd_stretch;

typedef struct {
    cmd_head h;
    void *tex;
    uint32_t level, slice, whole, color, lw, lh;
    mtl_rect r;
} cmd_fill;

typedef struct {
    cmd_head h;
    void *back;
    uint64_t presented_ns; /* the game's Present, for the immersive handoff's ages */
    float pose[3];         /* the head pose the frame was drawn for, latched at Present */
} cmd_present;

@interface MtlDrawCache : NSObject {
@public
    NSMutableDictionary<NSData *, id> *psos;       /* pipeline key -> state or NSNull */
    NSMutableDictionary<NSNumber *, id> *dss;      /* depth stencil key -> state */
    NSMutableDictionary<NSNumber *, NSNumber *> *samplers; /* sampler key -> heap index */
    NSMutableDictionary<NSNumber *, id> *clear_psos;
    NSMutableSet<NSString *> *ff_seen;
    NSMutableDictionary<NSNumber *, NSValue *> *fvf_decls; /* FVF -> mtl_blob *, held privately */
    mtl_blob *ff_vs, *ff_ps;                       /* the ubershaders, as stage objects for the pipeline cache */
    mtl_sc_info ff_vs_info, ff_ps_info;
    NSMutableSet<NSString *> *notes;
    NSMutableArray *sampler_objs;                  /* keeps every heap sampler alive */
    id<MTLBuffer> heap;
    uint32_t heap_n;
    id<MTLBuffer> zero;                            /* the null stream, and spare zero CBVs */
    id<MTLTexture> dummy2d, dummy3d, dummycube;
    id<MTLLibrary> util_lib;
    NSString *cache_dir;
    ring_chunk ring[RING_MAX];
    uint32_t nring, cur;
    uint32_t cur_off;
    double compile_ms;                             /* replay side, like every cache above but samplers and fvf_decls */
    uint64_t ff_keys[FF_KEYS][2];                  /* fixed-function draw kinds already logged: a set without
                                                      formatting a string at every draw */
    id<MTLSamplerState> stretch_samp[2];           /* StretchRect's linear and point samplers, replay side */
    id<MTLTexture> aura_ct, aura_dt;               /* full view: a camera-held effect drawn once, small (emit_aura) */
    id<MTLSamplerState> aura_samp;
    id<MTLTexture> arm_ct, arm_dt;                 /* full view: the gauges' cards, both sides side by side (arm_enter) */
    uint32_t arm_col[2][2];                        /* each side's back buffer columns, [x0, x1) */
    uint32_t arm_x[2];                             /* and where each starts in arm_ct */
    int arm_used, arm_drawn[2], arm_broken;        /* this frame cleared it, and drew each side; no target to be had */
    mtl_arm_frame arm_last;
    mtl_pipes *pipes;                              /* functions and pipelines across runs (pipelines.h) */
}
@end

@implementation MtlDrawCache
@end

#define CACHE(d) ((__bridge MtlDrawCache *)(d)->draw)

static mtl_pipes *open_pipelines(mtl_device *d, const char *cache_dir);

static void note_once(mtl_device *d, NSString *key, const char *fmt, ...) __attribute__((format(printf, 3, 4)));
static void note_once(mtl_device *d, NSString *key, const char *fmt, ...)
{
    __unsafe_unretained MtlDrawCache *c = CACHE(d);
    if ([c->notes containsObject:key]) return;
    [c->notes addObject:key];
    char b[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    mtl_log("%s", b);
}

/* Whether this fixed-function combination is new. A full table stops
   remembering, which only means a later kind is logged again. */
static int ff_first_seen(MtlDrawCache *c, uint64_t vs, uint64_t ps, uint32_t fvf, int post)
{
    uint64_t a = vs << 32 ^ ps, b = (uint64_t)fvf << 1 | (uint64_t)(post != 0) | 1u << 31;
    uint32_t h = (uint32_t)((a * 0x9E3779B97F4A7C15ull ^ b) >> 40) % FF_KEYS;
    for (uint32_t i = 0; i < FF_KEYS; i++) {
        uint64_t *e = c->ff_keys[(h + i) % FF_KEYS];
        if (e[0] == a && e[1] == b) return 0;
        if (!e[1]) {
            e[0] = a;
            e[1] = b;
            return 1;
        }
    }
    return 1;
}

/* Upload ring: per-draw constants, converted indices, UP vertex data. A
   chunk is reused once every command buffer that used it has completed. */
static void *ring_alloc(mtl_device *d, uint32_t size, void **buf, uint32_t *off)
{
    __unsafe_unretained MtlDrawCache *c = CACHE(d);
    size = (size + 255u) & ~255u;
    if (size > RING_CHUNK) return NULL;
    if (c->nring == 0 || c->cur_off + size > RING_CHUNK) {
        uint32_t pick = UINT32_MAX;
        for (uint32_t i = 0; i < c->nring; i++)
            if (i != c->cur && !mtl_busy(d, c->ring[i].serial) && c->ring[i].serial != d->serial) {
                pick = i;
                break;
            }
        if (pick == UINT32_MAX) {
            if (c->nring == RING_MAX) {
                /* Every chunk is in flight. This is called from inside a
                   draw, with the pass open, so it must not end the pass or
                   commit to wait: it hands out a buffer of its own, which
                   the command buffer keeps alive while it runs. */
                note_once(d, @"ring-full", "upload ring full (%u MiB in flight): allocating past it",
                          RING_MAX * (RING_CHUNK >> 20));
                id<MTLBuffer> b = [MTLD(d) newBufferWithLength:size options:MTLResourceStorageModeShared];
                if (!b) return NULL;
                mtl_rec_keep(d, (__bridge void *)b); /* the list keeps it until it completes */
                *buf = (__bridge void *)b;
                *off = 0;
                return b.contents;
            } else {
                id<MTLBuffer> b = [MTLD(d) newBufferWithLength:RING_CHUNK options:MTLResourceStorageModeShared];
                if (!b) return NULL;
                c->ring[c->nring].buf = (void *)CFBridgingRetain(b);
                c->ring[c->nring].base = b.contents;
                pick = c->nring++;
            }
        }
        c->cur = pick;
        c->cur_off = 0;
    }
    ring_chunk *r = &c->ring[c->cur];
    r->serial = d->serial;
    *buf = r->buf;
    *off = c->cur_off;
    c->cur_off += size;
    return r->base + *off;
}

uint32_t mtl_draw_ring_chunks(mtl_device *d) { return d->draw ? CACHE(d)->nring : 0; }

/* Surfaces as attachments */

id<MTLTexture> mtl_surface_texture(mtl_surface *s, uint32_t *level, uint32_t *slice)
{
    *level = 0;
    *slice = 0;
    if (!s) return nil;
    if (s->tex) return (__bridge id<MTLTexture>)s->tex;
    if (s->o.container) {
        mtl_texture *t = (mtl_texture *)s->o.container;
        uint32_t idx = (uint32_t)(s - t->sub);
        *slice = idx / t->levels;
        *level = idx % t->levels;
        id<MTLTexture> g = mtl_texture_gpu(t);
        /* Whatever asked is about to record a read or write of it (a
           pass, a StretchRect, a readback), so a later upload must not
           change it in place under that list. */
        if (g) t->last_use = t->o.dev->serial;
        return g;
    }
    return nil;
}

int mtl_surface_make_target(mtl_device *d, mtl_surface *s)
{
    if (s->tex) return 1;
    @autoreleasepool {
        int depth = mtl_format_is_depth(s->format);
        MTLPixelFormat pf = depth ? (MTLPixelFormat)mtl_depth_pixel_format() : (MTLPixelFormat)mtl_color_pixel_format(d, s->format);
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pf
                                                                                      width:s->width
                                                                                     height:s->height
                                                                                  mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModePrivate;
        id<MTLTexture> t = [MTLD(d) newTextureWithDescriptor:td];
        if (!t) return 0;
        s->tex = (void *)CFBridgingRetain(t);
    }
    s->no_cpu = 1;
    return 1;
}

/* Passes. The game thread tracks which attachments the recorded pass has
   open, as the encoder did, and records a new pass only when they change;
   the replay makes the encoder. */

static int ds_usable(mtl_device *d)
{
    mtl_surface *rt = d->rt[0], *ds = d->ds;
    return ds && ds->tex && rt && ds->width >= rt->width && ds->height >= rt->height;
}

static int begin_pass(mtl_device *d)
{
    mtl_surface *rt = d->rt[0];
    uint32_t level, slice;
    __unsafe_unretained id<MTLTexture> ct = mtl_surface_texture(rt, &level, &slice);
    if (!ct) return 0;
    mtl_surface *ds = ds_usable(d) ? d->ds : NULL;
    void *ctp = (__bridge void *)ct, *dtp = ds ? ds->tex : NULL;
    if (d->pass_open && d->pass_rt == ctp && d->pass_ds == dtp && d->pass_level == level && d->pass_slice == slice)
        return 1;
    if (d->pass_open) mtl_rec_cmd(d, CMD_END_PASS, sizeof(cmd_head));
    cmd_pass *p = mtl_rec_cmd(d, CMD_PASS, sizeof *p);
    p->ct = ctp;
    p->dt = dtp;
    p->level = level;
    p->slice = slice;
    p->width = rt->width;
    p->height = rt->height;
    p->clear = d->pend_clear & (ds ? 7u : D3DCLEAR_TARGET);
    p->color = d->pend_color;
    p->z = d->pend_z;
    p->stencil = d->pend_stencil;
    mtl_rec_keep(d, ctp);
    mtl_rec_keep(d, dtp);
    d->pend_clear = 0;
    d->pass_open = 1;
    d->pass_rt = ctp;
    d->pass_ds = dtp;
    d->pass_level = level;
    d->pass_slice = slice;
    d->pass_pf = (uint32_t)ct.pixelFormat;
    return 1;
}

void mtl_end_pass(mtl_device *d)
{
    if (!d->fq) return;
    if (d->pend_clear && !d->pass_open) begin_pass(d);
    d->pend_clear = 0;
    if (d->pass_open) {
        mtl_rec_cmd(d, CMD_END_PASS, sizeof(cmd_head));
        d->pass_open = 0;
        d->pass_rt = d->pass_ds = NULL;
    }
    mtl_rec_end(d);
}


/* Clear */

static id<MTLLibrary> util_library(mtl_device *d)
{
    __unsafe_unretained MtlDrawCache *c = CACHE(d);
    if (c->util_lib) return c->util_lib;
    /* A full-screen triangle at a given depth in one colour, cut to the
       rectangle by the scissor (partial clears), and a textured quad over
       the viewport (StretchRect). */
    static NSString *const src = @"#include <metal_stdlib>\n"
                                  "using namespace metal;\n"
                                  "struct v_out { float4 p [[position]]; float2 uv; };\n"
                                  "vertex v_out clear_vs(uint v [[vertex_id]], constant float &z [[buffer(0)]]) {\n"
                                  "    float2 t = float2((v << 1) & 2, v & 2);\n"
                                  "    v_out o; o.p = float4(t * 2.0 - 1.0, z, 1.0); o.uv = t; return o;\n"
                                  "}\n"
                                  "fragment float4 clear_fs(constant float4 &c [[buffer(0)]]) { return c; }\n"
                                  "vertex v_out blit_vs(uint v [[vertex_id]], constant float4 &r [[buffer(0)]]) {\n"
                                  "    float2 t = float2((v << 1) & 2, v & 2);\n"
                                  "    v_out o; o.p = float4(t.x * 2.0 - 1.0, 1.0 - t.y * 2.0, 0.0, 1.0);\n"
                                  "    o.uv = r.xy + t * r.zw; return o;\n"
                                  "}\n"
                                  "fragment float4 blit_fs(v_out i [[stage_in]], texture2d<float> t [[texture(0)]],\n"
                                  "                        sampler s [[sampler(0)]]) { return t.sample(s, i.uv); }\n";
    NSError *err = nil;
    c->util_lib = [MTLD(d) newLibraryWithSource:src options:nil error:&err];
    if (!c->util_lib) mtl_log("utility library failed: %s", err.localizedDescription.UTF8String);
    return c->util_lib;
}

static id<MTLRenderPipelineState> clear_pipeline(mtl_device *d, MTLPixelFormat cf, int has_ds, uint32_t flags)
{
    __unsafe_unretained MtlDrawCache *c = CACHE(d);
    NSNumber *key = @(((uint64_t)cf << 8) | ((uint64_t)has_ds << 4) | (flags & 7));
    id p = c->clear_psos[key];
    if (p) return p == [NSNull null] ? nil : p;
    id<MTLLibrary> lib = util_library(d);
    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction = [lib newFunctionWithName:@"clear_vs"];
    pd.fragmentFunction = [lib newFunctionWithName:@"clear_fs"];
    pd.colorAttachments[0].pixelFormat = cf;
    pd.colorAttachments[0].writeMask = (flags & D3DCLEAR_TARGET) ? MTLColorWriteMaskAll : MTLColorWriteMaskNone;
    if (has_ds) {
        pd.depthAttachmentPixelFormat = (MTLPixelFormat)mtl_depth_pixel_format();
        pd.stencilAttachmentPixelFormat = (MTLPixelFormat)mtl_depth_pixel_format();
    }
    NSError *err = nil;
    id<MTLRenderPipelineState> pso = lib ? [MTLD(d) newRenderPipelineStateWithDescriptor:pd error:&err] : nil;
    if (!pso) mtl_log("clear pipeline failed: %s", err.localizedDescription.UTF8String);
    c->clear_psos[key] = pso ? (id)pso : (id)[NSNull null];
    return pso;
}

static id<MTLDepthStencilState> clear_dss(mtl_device *d, uint32_t flags)
{
    __unsafe_unretained MtlDrawCache *c = CACHE(d);
    NSNumber *key = @(0x8000000000000000ull | flags); /* above every draw key's bits */
    id s = c->dss[key];
    if (s) return s;
    MTLDepthStencilDescriptor *dd = [MTLDepthStencilDescriptor new];
    dd.depthCompareFunction = MTLCompareFunctionAlways;
    dd.depthWriteEnabled = (flags & D3DCLEAR_ZBUFFER) != 0;
    if (flags & D3DCLEAR_STENCIL) {
        MTLStencilDescriptor *sd = [MTLStencilDescriptor new];
        sd.stencilCompareFunction = MTLCompareFunctionAlways;
        sd.depthStencilPassOperation = MTLStencilOperationReplace;
        sd.writeMask = 0xFF;
        dd.frontFaceStencil = sd;
        dd.backFaceStencil = sd;
    }
    id<MTLDepthStencilState> st = [MTLD(d) newDepthStencilStateWithDescriptor:dd];
    c->dss[key] = st;
    return st;
}

static void cpu_clear(mtl_surface *t, const mtl_rect *r, uint32_t n, uint32_t color)
{
    uint32_t bits = mtl_fmt_bits(t->format);
    if (mtl_fmt_dxt(t->format) || (bits != 32 && t->format != D3DFMT_R5G6B5)) return;
    mtl_surface_sync(t);
    uint8_t *m = mtl_surface_mem(t);
    if (!m) return;
    uint16_t c16 = (uint16_t)(((color >> 8) & 0xF800) | ((color >> 5) & 0x07E0) | ((color >> 3) & 0x001F));
    for (uint32_t i = 0; i < n; i++)
        for (int32_t y = r[i].top; y < r[i].bottom; y++) {
            uint8_t *row = m + (size_t)y * t->pitch;
            for (int32_t x = r[i].left; x < r[i].right; x++) {
                if (bits == 32) ((uint32_t *)row)[x] = color;
                else ((uint16_t *)row)[x] = c16;
            }
        }
    t->dirty = 1;
    if (t->o.container) ((mtl_texture *)t->o.container)->dirty = 1;
}

/* D3D9 Clear: with no rectangles it clears the viewport, with rectangles
   each one cut to the viewport; render target, depth and stencil as the
   flags say. RT_METAL_CLEAR overrides the colour (device.m). */
HRESULT mtl_clear(mtl_device *d, uint32_t count, const mtl_rect *rects, uint32_t flags, uint32_t color, float z,
                  uint32_t stencil)
{
    d->clears++;
    uint32_t game = color;
    if (d->force_clear) color = d->force_color;
    if (d->clears <= 4)
        mtl_log("Clear %llu: %u rects flags 0x%X colour 0x%08X%s z %.3f stencil %u", (unsigned long long)d->clears, count,
                flags, color, d->force_clear && game != color ? " (forced by RT_METAL_CLEAR)" : "", z, stencil);
    mtl_surface *t = d->rt[0];
    if (!t) return D3D_OK;
    if (!d->ds || !d->ds->tex) flags &= ~(uint32_t)(D3DCLEAR_ZBUFFER | D3DCLEAR_STENCIL);
    flags &= D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER | D3DCLEAR_STENCIL;
    if (!flags) return D3D_OK;
    /* A depth clear of the back buffer starts a new group of world draws
       for mtl_fv_scan: the 3D sky is the group before the world's. */
    if ((flags & D3DCLEAR_ZBUFFER) && t == d->back) d->fv_group++;
    mtl_rect full = {0, 0, (int32_t)t->width, (int32_t)t->height};
    mtl_rect vp = {(int32_t)d->vp.X, (int32_t)d->vp.Y, (int32_t)(d->vp.X + d->vp.Width), (int32_t)(d->vp.Y + d->vp.Height)};
    if (!clip_rect(&vp, &full)) return D3D_OK;
    mtl_rect one[1] = {vp};
    mtl_rect *r = one;
    uint32_t n = 1;
    if (count && rects) {
        r = malloc(count * sizeof *r);
        if (!r) return E_OUTOFMEMORY;
        n = 0;
        for (uint32_t i = 0; i < count; i++) {
            r[n] = rects[i];
            n += clip_rect(&r[n], &vp);
        }
    }
    uint32_t lvl, sl;
    if (!n) {
    } else if (!mtl_surface_texture(t, &lvl, &sl)) {
        if (flags & D3DCLEAR_TARGET) cpu_clear(t, r, n, color);
    } else if (n == 1 && !memcmp(&r[0], &full, sizeof full) &&
               (!(flags & (D3DCLEAR_ZBUFFER | D3DCLEAR_STENCIL)) || ds_usable(d))) {
        /* The whole target: the next pass loads with these values. A pass
           already open on it ends here, so its draws keep their order. */
        mtl_end_pass(d);
        d->pend_clear = flags;
        d->pend_color = color;
        d->pend_z = z < 0.0f ? 0.0f : z > 1.0f ? 1.0f : z;
        d->pend_stencil = stencil;
        /* A depth-only clear still needs the colour kept. */
    } else if (begin_pass(d)) {
        /* A partial clear is a draw under the scissor, recorded into the
           pass with its rectangles. */
        int has_ds = ds_usable(d);
        if (!has_ds) flags &= D3DCLEAR_TARGET;
        if (flags) {
            cmd_clear *c = mtl_rec_cmd(d, CMD_CLEAR, (uint32_t)(sizeof *c + n * sizeof *r));
            c->pf = d->pass_pf;
            c->has_ds = (uint32_t)has_ds;
            c->flags = flags;
            c->color = color;
            c->z = z;
            c->stencil = stencil;
            c->width = t->width;
            c->height = t->height;
            c->n = n;
            memcpy(c->r, r, n * sizeof *r);
        }
    }
    mtl_rec_end(d);
    if (r != one) free(r);
    return D3D_OK;
}

/* Default state, as the D3D9 runtime sets it at device creation. */
static void default_state(mtl_device *d)
{
    uint32_t *rs = d->rs;
    memset(d->rs, 0, sizeof d->rs);
    rs[RS_ZENABLE] = d->pp.EnableAutoDepthStencil ? 1 : 0;
    rs[RS_FILLMODE] = 3;
    rs[RS_SHADEMODE] = 2;
    rs[RS_ZWRITEENABLE] = 1;
    rs[RS_LASTPIXEL] = 1;
    rs[RS_SRCBLEND] = 2;
    rs[RS_DESTBLEND] = 1;
    rs[RS_CULLMODE] = 3;
    rs[RS_ZFUNC] = 4;
    rs[RS_ALPHAFUNC] = 8;
    rs[RS_FOGEND] = as_bits(1.0f);
    rs[RS_FOGDENSITY] = as_bits(1.0f);
    rs[RS_STENCILFAIL] = rs[RS_STENCILZFAIL] = rs[RS_STENCILPASS] = 1;
    rs[RS_STENCILFUNC] = 8;
    rs[RS_STENCILMASK] = rs[RS_STENCILWRITEMASK] = 0xFFFFFFFFu;
    rs[RS_TEXTUREFACTOR] = 0xFFFFFFFFu;
    rs[RS_CLIPPING] = rs[RS_LIGHTING] = rs[RS_COLORVERTEX] = rs[RS_LOCALVIEWER] = 1;
    rs[RS_DIFFUSEMATERIALSOURCE] = 1;
    rs[RS_SPECULARMATERIALSOURCE] = 2;
    rs[RS_POINTSIZE] = rs[RS_POINTSIZE_MIN] = rs[RS_POINTSCALE_A] = as_bits(1.0f);
    rs[RS_POINTSIZE_MAX] = as_bits(64.0f);
    rs[RS_MULTISAMPLEANTIALIAS] = 1;
    rs[RS_MULTISAMPLEMASK] = 0xFFFFFFFFu;
    rs[RS_COLORWRITEENABLE] = rs[RS_COLORWRITEENABLE1] = rs[RS_COLORWRITEENABLE2] = rs[RS_COLORWRITEENABLE3] = 0xF;
    rs[RS_BLENDOP] = rs[RS_BLENDOPALPHA] = 1;
    rs[RS_POSITIONDEGREE] = 3;
    rs[RS_NORMALDEGREE] = 1;
    rs[RS_MINTESSELLATIONLEVEL] = rs[RS_MAXTESSELLATIONLEVEL] = as_bits(1.0f);
    rs[RS_CCW_STENCILFAIL] = rs[RS_CCW_STENCILZFAIL] = rs[RS_CCW_STENCILPASS] = 1;
    rs[RS_CCW_STENCILFUNC] = 8;
    rs[RS_BLENDFACTOR] = 0xFFFFFFFFu;
    rs[RS_SRCBLENDALPHA] = 2;
    rs[RS_DESTBLENDALPHA] = 1;
    mtl_state *st = &d->st;
    for (uint32_t s = 0; s < MTL_NSAMP; s++) {
        memset(st->samp[s], 0, sizeof st->samp[s]);
        st->samp[s][SS_ADDRESSU] = st->samp[s][SS_ADDRESSV] = st->samp[s][SS_ADDRESSW] = 1;
        st->samp[s][SS_MAGFILTER] = st->samp[s][SS_MINFILTER] = 1;
        st->samp[s][SS_MAXANISOTROPY] = 1;
        st->samp_heap[s] = -1;
    }
    /* Stage 0 modulates texture by diffuse, the rest are off; the second
       and zeroth arguments and the result default to CURRENT. */
    for (uint32_t s = 0; s < 8; s++) {
        memset(st->tss[s], 0, sizeof st->tss[s]);
        st->tss[s][TSS_COLOROP] = s ? 1 : 4;
        st->tss[s][TSS_ALPHAOP] = s ? 1 : 2;
        st->tss[s][TSS_COLORARG1] = st->tss[s][TSS_ALPHAARG1] = TA_TEXTURE;
        st->tss[s][TSS_COLORARG2] = st->tss[s][TSS_ALPHAARG2] = TA_CURRENT;
        st->tss[s][TSS_COLORARG0] = st->tss[s][TSS_ALPHAARG0] = TA_CURRENT;
        st->tss[s][TSS_RESULTARG] = TA_CURRENT;
        st->tss[s][TSS_TEXCOORDINDEX] = s;
    }
    for (uint32_t s = 0; s < 16; s++) st->freq[s] = 1;
    st->scissor = (mtl_rect){0, 0, (int32_t)d->pp.BackBufferWidth, (int32_t)d->pp.BackBufferHeight};
    /* Identity transforms, a zero material, no lights. */
    memset(st->xf, 0, sizeof st->xf);
    for (uint32_t i = 0; i < MTL_NXF; i++)
        for (uint32_t k = 0; k < 4; k++) st->xf[i][5 * k] = 1.0f;
    memset(st->material, 0, sizeof st->material);
    free(st->lights);
    st->lights = NULL;
    st->nlights = 0;
}

/* Reset: what CreateDevice leaves, minus the targets and viewport the
   caller has already set to the new back buffer. Bound objects are let go
   first, so nothing the engine released stays alive through state. */
void mtl_state_reset(mtl_device *d)
{
    mtl_end_pass(d);
    mtl_state_release(d);
    mtl_state *st = &d->st;
    memset(st->vsf, 0, sizeof st->vsf);
    memset(st->vsi, 0, sizeof st->vsi);
    memset(st->vsb, 0, sizeof st->vsb);
    memset(st->psf, 0, sizeof st->psf);
    memset(st->psi, 0, sizeof st->psi);
    memset(st->psb, 0, sizeof st->psb);
    memset(st->vb_off, 0, sizeof st->vb_off);
    memset(st->vb_stride, 0, sizeof st->vb_stride);
    memset(st->clip, 0, sizeof st->clip);
    st->vs_changed_floats = 0;
    st->fvf = 0;
    default_state(d);
}

int mtl_draw_init(mtl_device *d)
{
    d->fv_vm_state = -1;
    @autoreleasepool {
        MtlDrawCache *c = [MtlDrawCache new];
        c->psos = [NSMutableDictionary new];
        c->dss = [NSMutableDictionary new];
        c->samplers = [NSMutableDictionary new];
        c->clear_psos = [NSMutableDictionary new];
        c->ff_seen = [NSMutableSet new];
        c->fvf_decls = [NSMutableDictionary new];
        c->ff_vs = calloc(1, sizeof *c->ff_vs);
        c->ff_ps = calloc(1, sizeof *c->ff_ps);
        if (!c->ff_vs || !c->ff_ps) return 0;
        c->ff_vs->o.kind = MTL_K_VS;
        c->ff_ps->o.kind = MTL_K_PS;
        c->ff_vs->uid = mtl_next_uid();
        c->ff_ps->uid = mtl_next_uid();
        mtl_ff_input_signature(&c->ff_vs_info);
        /* The pixel ubershader reads stages 0 to 7, each through whichever
           view its texture has, with heap indices after the factor. */
        memset(&c->ff_ps_info, 0, sizeof c->ff_ps_info);
        c->ff_ps_info.is_ps = 1;
        c->ff_ps_info.sampler_mask = 0xFF;
        c->ff_ps_info.sampler_push_base = FF_PS_PUSH_SAMPLERS;
        mtl_tex_layout(&c->ff_ps_info);
        c->notes = [NSMutableSet new];
        c->sampler_objs = [NSMutableArray new];
        c->heap = [MTLD(d) newBufferWithLength:HEAP_MAX * 8 options:MTLResourceStorageModeShared];
        c->zero = [MTLD(d) newBufferWithLength:MTL_SC_MAX_CONST options:MTLResourceStorageModeShared];
        if (!c->heap || !c->zero) return 0;
        memset(c->zero.contents, 0, c->zero.length);
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                      width:1
                                                                                     height:1
                                                                                  mipmapped:NO];
        td.storageMode = MTLStorageModeShared;
        c->dummy2d = [MTLD(d) newTextureWithDescriptor:td];
        td.textureType = MTLTextureType3D;
        c->dummy3d = [MTLD(d) newTextureWithDescriptor:td];
        td.textureType = MTLTextureTypeCube;
        c->dummycube = [MTLD(d) newTextureWithDescriptor:td];
        const char *dir = getenv("RT_METAL_CACHE_DIR");
        c->cache_dir = dir && *dir ? @(dir) : nil;
        /* Opened here, on the game thread at device creation, so the warm
           queue starts on the recorded pipelines while the menu loads. */
        c->pipes = open_pipelines(d, c->cache_dir.UTF8String);
        d->draw = (void *)CFBridgingRetain(c);
    }
    default_state(d);
    return 1;
}

void mtl_state_release(mtl_device *d)
{
    mtl_state *st = &d->st;
    for (uint32_t i = 0; i < MTL_NSAMP; i++) {
        mtl_unhold(st->tex[i] ? &st->tex[i]->o : NULL);
        st->tex[i] = NULL;
    }
    for (uint32_t i = 0; i < 16; i++) {
        mtl_unhold(st->vb[i] ? &st->vb[i]->o : NULL);
        st->vb[i] = NULL;
    }
    mtl_obj *objs[4] = {st->ib ? &st->ib->o : NULL, st->decl ? &st->decl->o : NULL, st->vs ? &st->vs->o : NULL,
                        st->ps ? &st->ps->o : NULL};
    st->ib = NULL;
    st->decl = st->vs = st->ps = NULL;
    for (int i = 0; i < 4; i++) mtl_unhold(objs[i]);
}

void mtl_draw_finish(mtl_device *d)
{
    if (d->draw) mtl_pipes_finish(CACHE(d)->pipes);
}

void mtl_draw_destroy(mtl_device *d)
{
    free(d->fv_draws);
    d->fv_draws = NULL;
    d->fv_n = d->fv_cap = 0;
    if (!d->draw) return;
    MtlDrawCache *c = CFBridgingRelease(d->draw);
    d->draw = NULL;
    /* First: the warm queue may still be compiling the ubershaders below. */
    mtl_pipes_close(c->pipes);
    c->pipes = NULL;
    for (uint32_t i = 0; i < c->nring; i++) CFRelease(c->ring[i].buf);
    for (NSValue *v in c->fvf_decls.allValues) {
        mtl_blob *b = v.pointerValue;
        mtl_unhold(&b->o);
    }
    mtl_blob *ff[2] = {c->ff_vs, c->ff_ps};
    for (int i = 0; i < 2; i++)
        if (ff[i]) {
            if (ff[i]->fn) CFRelease(ff[i]->fn);
            free(ff[i]);
        }
    free(d->st.lights);
    d->st.lights = NULL;
    d->st.nlights = 0;
    mtl_log("stereo classes: world %llu, offscreen %llu, pretransformed %llu, flat %llu, unknown %llu, before a view "
            "projection %llu; last a %.4f",
            (unsigned long long)d->stereo_class[MTL_STEREO_WORLD], (unsigned long long)d->stereo_class[MTL_STEREO_OFFSCREEN],
            (unsigned long long)d->stereo_class[MTL_STEREO_PRETRANSFORMED],
            (unsigned long long)d->stereo_class[MTL_STEREO_FLAT], (unsigned long long)d->stereo_class[MTL_STEREO_UNKNOWN],
            (unsigned long long)d->stereo_class[MTL_STEREO_NO_A], d->stereo_a);
    mtl_log("stereo: screen passes %llu (flat over a render target, kept with the eye's image)",
            (unsigned long long)d->stereo_class[MTL_STEREO_SCREEN_PASS]);
    mtl_log("draw path: %llu draws (%llu with a fixed-function stage), %llu dropped; shaders %llu translated, %llu from "
            "cache, %.0f ms compiling; %lu pipelines; buffer renames %llu (%llu old storages dropped, free list full), "
            "texture uploads %llu (%llu rebuilt)",
            (unsigned long long)d->draws, (unsigned long long)d->ff_draws, (unsigned long long)d->dropped_draws,
            (unsigned long long)d->shaders_built, (unsigned long long)d->shaders_cached, c->compile_ms,
            (unsigned long)c->psos.count, (unsigned long long)d->renames, (unsigned long long)d->free_drops,
            (unsigned long long)d->tex_uploads,
            (unsigned long long)d->tex_renames);
}

void mtl_blob_gpu_free(mtl_blob *b)
{
    if (b->fn) CFRelease(b->fn);
    b->fn = NULL;
    if (b->sc) mtl_sc_free(b->sc);
    b->sc = NULL;
}

/* A stage's function, compiled at its first draw (the engine creates
   about a thousand shaders at the menu and draws with few), or taken
   from the warm queue, which may have it already. b is NULL for the
   fixed-function stage on that side. nil if it failed, which drops the
   draw (logged once per shader), or with *pending set in skip mode if it
   is not ready yet. */
static id<MTLFunction> stage_function(mtl_device *d, mtl_blob *b, int fragment, mtl_pipes_wait *w, int *pending)
{
    __unsafe_unretained MtlDrawCache *c = CACHE(d);
    int ff = !b;
    if (ff) b = fragment ? c->ff_ps : c->ff_vs;
    if (b->fn) return (__bridge id<MTLFunction>)b->fn;
    if (b->failed || (!ff && !b->sc)) return nil;
    const char *key = ff ? (fragment ? MTL_PIPES_FF_PS : MTL_PIPES_FF_VS) : mtl_sc_hash(b->sc);
    NSString * (^msl)(void) = ^NSString *(void) {
        char err[1024];
        if (ff) {
            const char *src = mtl_sc_ff_msl(fragment, err, sizeof err);
            if (!src) mtl_log("fixed-function %s did not build: %s", fragment ? "pixel shader" : "vertex shader", err);
            return src ? @(src) : nil;
        }
        int cached = 0;
        const char *src = mtl_sc_msl(b->sc, c->cache_dir.UTF8String, &cached, err, sizeof err);
        if (!src) {
            mtl_log("shader %s (0x%08X) did not translate: %s", mtl_sc_hash(b->sc), b->version, err);
            return nil;
        }
        if (cached) d->shaders_cached++;
        else d->shaders_built++;
        return @(src);
    };
    id<MTLFunction> fn = nil;
    NSString *why = nil;
    double t0 = w->ms;
    mtl_pipes_status r = mtl_pipes_function(c->pipes, key, msl, &fn, w, &why);
    c->compile_ms += w->ms - t0;
    if (r == MTL_PIPES_PENDING) {
        *pending = 1;
        return nil;
    }
    if (r == MTL_PIPES_FAILED) {
        if (why) {
            if (ff) mtl_log("fixed-function %s did not build: %s", fragment ? "pixel shader" : "vertex shader", why.UTF8String);
            else mtl_log("shader %s (0x%08X) did not compile: %s", mtl_sc_hash(b->sc), b->version, why.UTF8String);
        }
        b->failed = 1;
        return nil;
    }
    b->fn = (void *)CFBridgingRetain(fn);
    return fn;
}

/* Vertex layout */

static MTLVertexFormat vertex_format(uint32_t t)
{
    switch (t) {
    case 0: return MTLVertexFormatFloat;
    case 1: return MTLVertexFormatFloat2;
    case 2: return MTLVertexFormatFloat3;
    case 3: return MTLVertexFormatFloat4;
    case 4: return MTLVertexFormatUChar4Normalized_BGRA; /* D3DCOLOR */
    case 5: return MTLVertexFormatUChar4;                /* UBYTE4: integers read as floats */
    case 6: return MTLVertexFormatShort2;
    case 7: return MTLVertexFormatShort4;
    case 8: return MTLVertexFormatUChar4Normalized;
    case 9: return MTLVertexFormatShort2Normalized;
    case 10: return MTLVertexFormatShort4Normalized;
    case 11: return MTLVertexFormatUShort2Normalized;
    case 12: return MTLVertexFormatUShort4Normalized;
    case 13: return MTLVertexFormatUInt1010102Normalized; /* UDEC3, approximated */
    case 14: return MTLVertexFormatInt1010102Normalized;
    case 15: return MTLVertexFormatHalf2;
    case 16: return MTLVertexFormatHalf4;
    default: return MTLVertexFormatInvalid;
    }
}

typedef struct {
    uint16_t stream, offset;
    uint8_t type, method, usage, index;
} decl_elem;

/* For each vertex shader input, the declaration element that feeds it:
   matched by usage and index, POSITIONT as POSITION, the last match
   winning (DXVK BindInputLayout). -1 for none. */
static void match_inputs(const mtl_blob *decl, const mtl_sc_info *vi, int match[MTL_MAX_INPUTS])
{
    const decl_elem *e = (const decl_elem *)decl->data;
    uint32_t n = decl->bytes / 8 - 1;
    for (uint32_t i = 0; i < vi->n_inputs; i++) {
        match[i] = -1;
        for (uint32_t k = 0; k < n; k++) {
            uint8_t u = e[k].usage == 9 ? 0 : e[k].usage;
            if (u == vi->in_usage[i] && e[k].index == vi->in_index[i] && e[k].type < 17) match[i] = (int)k;
        }
    }
}

/* Pipelines */

typedef struct {
    uint64_t vs, ps, decl;
    uint16_t stride[MTL_MAX_STREAMS];
    uint32_t inst_mask, color_pf, has_ds, x8;
    uint32_t blend, srcb, dstb, op, srca, dsta, opa, wmask;
} pso_key;

static MTLBlendFactor blend_factor(uint32_t f, int x8, int alpha_slot)
{
    switch (f) {
    case 1: return MTLBlendFactorZero;
    case 2: return MTLBlendFactorOne;
    case 3: return alpha_slot ? MTLBlendFactorSourceAlpha : MTLBlendFactorSourceColor;
    case 4: return alpha_slot ? MTLBlendFactorOneMinusSourceAlpha : MTLBlendFactorOneMinusSourceColor;
    case 5: return MTLBlendFactorSourceAlpha;
    case 6: return MTLBlendFactorOneMinusSourceAlpha;
    /* An X8 target has no alpha: D3D9 reads destination alpha as one. */
    case 7: return x8 ? MTLBlendFactorOne : MTLBlendFactorDestinationAlpha;
    case 8: return x8 ? MTLBlendFactorZero : MTLBlendFactorOneMinusDestinationAlpha;
    case 9: return alpha_slot ? (x8 ? MTLBlendFactorOne : MTLBlendFactorDestinationAlpha) : MTLBlendFactorDestinationColor;
    case 10: return alpha_slot ? (x8 ? MTLBlendFactorZero : MTLBlendFactorOneMinusDestinationAlpha)
                               : MTLBlendFactorOneMinusDestinationColor;
    case 11: return MTLBlendFactorSourceAlphaSaturated;
    case 14: return alpha_slot ? MTLBlendFactorBlendAlpha : MTLBlendFactorBlendColor;
    case 15: return alpha_slot ? MTLBlendFactorOneMinusBlendAlpha : MTLBlendFactorOneMinusBlendColor;
    default: return MTLBlendFactorOne;
    }
}

static MTLBlendOperation blend_op(uint32_t op)
{
    switch (op) {
    case 2: return MTLBlendOperationSubtract;
    case 3: return MTLBlendOperationReverseSubtract;
    case 4: return MTLBlendOperationMin;
    case 5: return MTLBlendOperationMax;
    default: return MTLBlendOperationAdd;
    }
}

/* BOTHSRCALPHA and BOTHINVSRCALPHA set both factors from the source one. */
static void resolve_both(uint32_t *src, uint32_t *dst)
{
    if (*src == 12) {
        *src = 5;
        *dst = 6;
    } else if (*src == 13) {
        *src = 6;
        *dst = 5;
    }
}

/* What a pipeline is made from, resolved from device state when a draw
   is recorded. The key is the cache key, as before; the attribute layout
   is derived from parts of the key (the declaration and the vertex
   shader), so it is carried beside it rather than in it. */
typedef struct {
    pso_key k;
    uint32_t n_inputs;
    uint8_t a_type[MTL_MAX_INPUTS]; /* D3DDECLTYPE, or 0xFF: the input reads the null stream */
    uint8_t a_stream[MTL_MAX_INPUTS];
    uint16_t a_off[MTL_MAX_INPUTS];
    uint32_t rate[MTL_MAX_STREAMS]; /* instanced streams' step rates */
} pso_desc;

static void pipeline_desc(mtl_device *d, const mtl_blob *vs, const mtl_blob *ps, const mtl_sc_info *vi,
                          const int match[MTL_MAX_INPUTS], MTLPixelFormat cpf, int has_ds, int x8,
                          const uint16_t *stride_override, pso_desc *out)
{
    mtl_state *st = &d->st;
    const uint32_t *rs = d->rs;
    const decl_elem *e = (const decl_elem *)st->decl->data;
    pso_key *k = &out->k;
    memset(out, 0, sizeof *out);
    k->vs = vs->uid;
    k->ps = ps->uid;
    k->decl = st->decl->uid;
    for (uint32_t i = 0; i < vi->n_inputs; i++)
        if (match[i] >= 0) {
            uint32_t s = e[match[i]].stream;
            if (s < MTL_MAX_STREAMS) {
                k->stride[s] = stride_override ? stride_override[s] : (uint16_t)st->vb_stride[s];
                if (st->freq[s] & 0x80000000u) k->inst_mask |= 1u << s;
            }
        }
    k->color_pf = (uint32_t)cpf;
    k->has_ds = (uint32_t)has_ds;
    k->x8 = (uint32_t)x8;
    k->wmask = rs[RS_COLORWRITEENABLE] & 0xF;
    k->blend = rs[RS_ALPHABLENDENABLE] != 0;
    if (k->blend) {
        k->srcb = rs[RS_SRCBLEND];
        k->dstb = rs[RS_DESTBLEND];
        k->op = rs[RS_BLENDOP];
        resolve_both(&k->srcb, &k->dstb);
        if (rs[RS_SEPARATEALPHABLENDENABLE]) {
            k->srca = rs[RS_SRCBLENDALPHA];
            k->dsta = rs[RS_DESTBLENDALPHA];
            k->opa = rs[RS_BLENDOPALPHA];
            resolve_both(&k->srca, &k->dsta);
        } else {
            k->srca = k->srcb;
            k->dsta = k->dstb;
            k->opa = k->op;
        }
    }
    out->n_inputs = vi->n_inputs;
    for (uint32_t i = 0; i < vi->n_inputs; i++) {
        if (match[i] < 0 || e[match[i]].stream >= MTL_MAX_STREAMS ||
            vertex_format(e[match[i]].type) == MTLVertexFormatInvalid) {
            out->a_type[i] = 0xFF;
            continue;
        }
        const decl_elem *el = &e[match[i]];
        out->a_type[i] = el->type;
        out->a_stream[i] = (uint8_t)el->stream;
        out->a_off[i] = el->offset;
        if (k->inst_mask & (1u << el->stream)) out->rate[el->stream] = st->freq[el->stream] & 0x7FFFFFu;
    }
}

/* Bump when pso_desc or describe_pipeline changes: recorded pipelines
   and the archive (pipelines.h) are kept only while this matches. */
static const char *const k_pipeline_version = "draw-pso 1";

/* A pipeline descriptor from a recorded description, with everything
   but the functions. Any thread: it reads the description alone. */
static MTLRenderPipelineDescriptor *describe_pipeline(const void *desc)
{
    const pso_desc *pd = desc;
    const pso_key *k = &pd->k;
    MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
    int null_used = 0;
    for (uint32_t i = 0; i < pd->n_inputs; i++) {
        if (pd->a_type[i] == 0xFF) {
            vd.attributes[i].format = MTLVertexFormatFloat4;
            vd.attributes[i].offset = 0;
            vd.attributes[i].bufferIndex = MTL_BUF_NULL;
            null_used = 1;
            continue;
        }
        uint32_t stream = pd->a_stream[i];
        vd.attributes[i].format = vertex_format(pd->a_type[i]);
        vd.attributes[i].offset = pd->a_off[i];
        vd.attributes[i].bufferIndex = MTL_BUF_STREAM + stream;
        MTLVertexBufferLayoutDescriptor *l = vd.layouts[MTL_BUF_STREAM + stream];
        uint32_t s = k->stride[stream];
        if (!s) {
            l.stepFunction = MTLVertexStepFunctionConstant;
            l.stepRate = 0;
            l.stride = 4;
        } else if (k->inst_mask & (1u << stream)) {
            l.stepFunction = MTLVertexStepFunctionPerInstance;
            l.stepRate = pd->rate[stream];
            l.stride = s;
        } else {
            l.stepFunction = MTLVertexStepFunctionPerVertex;
            l.stepRate = 1;
            l.stride = s;
        }
    }
    if (null_used) {
        vd.layouts[MTL_BUF_NULL].stepFunction = MTLVertexStepFunctionConstant;
        vd.layouts[MTL_BUF_NULL].stepRate = 0;
        vd.layouts[MTL_BUF_NULL].stride = 16;
    }

    int x8 = (int)k->x8;
    MTLRenderPipelineDescriptor *pdesc = [MTLRenderPipelineDescriptor new];
    pdesc.vertexDescriptor = vd;
    MTLRenderPipelineColorAttachmentDescriptor *ca = pdesc.colorAttachments[0];
    ca.pixelFormat = (MTLPixelFormat)k->color_pf;
    ca.writeMask = ((k->wmask & 1) ? MTLColorWriteMaskRed : 0) | ((k->wmask & 2) ? MTLColorWriteMaskGreen : 0) |
                   ((k->wmask & 4) ? MTLColorWriteMaskBlue : 0) | ((k->wmask & 8) ? MTLColorWriteMaskAlpha : 0);
    if (k->blend) {
        ca.blendingEnabled = YES;
        ca.sourceRGBBlendFactor = blend_factor(k->srcb, x8, 0);
        ca.destinationRGBBlendFactor = blend_factor(k->dstb, x8, 0);
        ca.rgbBlendOperation = blend_op(k->op);
        ca.sourceAlphaBlendFactor = blend_factor(k->srca, x8, 1);
        ca.destinationAlphaBlendFactor = blend_factor(k->dsta, x8, 1);
        ca.alphaBlendOperation = blend_op(k->opa);
    }
    if (k->has_ds) {
        pdesc.depthAttachmentPixelFormat = (MTLPixelFormat)mtl_depth_pixel_format();
        pdesc.stencilAttachmentPixelFormat = (MTLPixelFormat)mtl_depth_pixel_format();
    }
    return pdesc;
}

static mtl_pipes *open_pipelines(mtl_device *d, const char *cache_dir)
{
    return mtl_pipes_open(MTLD(d), d->caps.archive, cache_dir, k_pipeline_version, sizeof(pso_desc), describe_pipeline);
}

/* Replay side: the pipeline for a recorded description, made on first
   use or taken from the warm queue (pipelines.h). The dictionary here is
   keyed by uids, cheap to probe at every draw; pipelines.m keys the same
   pipeline by what it is made of, which is what survives a run. In the
   persistent key the uids are zero: the description already carries the
   whole vertex layout they select. */
static id<MTLRenderPipelineState> get_pipeline(mtl_device *d, mtl_blob *vs, mtl_blob *ps, id<MTLFunction> vf,
                                               id<MTLFunction> pf, const pso_desc *pd, mtl_pipes_wait *w, int *pending)
{
    __unsafe_unretained MtlDrawCache *c = CACHE(d);
    const pso_key *k = &pd->k;
    NSData *probe = [NSData dataWithBytesNoCopy:(void *)k length:sizeof *k freeWhenDone:NO];
    id p = c->psos[probe];
    if (p) return p == [NSNull null] ? nil : p;

    pso_desc stable = *pd;
    stable.k.vs = stable.k.ps = stable.k.decl = 0;
    const char *vk = vs->sc ? mtl_sc_hash(vs->sc) : MTL_PIPES_FF_VS, *pk = ps->sc ? mtl_sc_hash(ps->sc) : MTL_PIPES_FF_PS;
    id<MTLRenderPipelineState> pso = nil;
    NSString *why = nil;
    mtl_pipes_status r = mtl_pipes_pipeline(c->pipes, vk, pk, &stable, vf, pf, &pso, w, &why);
    if (r == MTL_PIPES_PENDING) {
        *pending = 1;
        return nil;
    }
    if (r == MTL_PIPES_FAILED && why) mtl_log("pipeline for vs %s ps %s failed: %s", vs->sc ? mtl_sc_hash(vs->sc) : "fixed-function",
                ps->sc ? mtl_sc_hash(ps->sc) : "fixed-function", why.UTF8String);
    /* The key is copied: the probe points into a list that will be reused. */
    c->psos[[NSData dataWithBytes:k length:sizeof *k]] = pso ? (id)pso : (id)[NSNull null];
    return pso;
}

static MTLCompareFunction compare_fn(uint32_t f)
{
    return f >= 1 && f <= 8 ? (MTLCompareFunction)(f - 1) : MTLCompareFunctionAlways;
}

static MTLStencilOperation stencil_op(uint32_t o)
{
    return o >= 1 && o <= 8 ? (MTLStencilOperation)(o - 1) : MTLStencilOperationKeep;
}

/* Depth and stencil state as a key that holds every field, resolved when
   the draw is recorded; the replay makes the state from the key alone. */
static uint64_t dss_key(mtl_device *d, int has_ds)
{
    const uint32_t *rs = d->rs;
    uint64_t ze = has_ds && rs[RS_ZENABLE], zw = ze && rs[RS_ZWRITEENABLE], se = has_ds && rs[RS_STENCILENABLE];
    uint64_t key = ze | zw << 1 | se << 2 | (uint64_t)(rs[RS_ZFUNC] & 15) << 3;
    if (se) {
        int two = rs[RS_TWOSIDEDSTENCILMODE] != 0;
        key |= (uint64_t)(rs[RS_STENCILFAIL] & 15) << 7 | (uint64_t)(rs[RS_STENCILZFAIL] & 15) << 11 |
               (uint64_t)(rs[RS_STENCILPASS] & 15) << 15 | (uint64_t)(rs[RS_STENCILFUNC] & 15) << 19 |
               (uint64_t)(rs[RS_STENCILMASK] & 0xFF) << 23 | (uint64_t)(rs[RS_STENCILWRITEMASK] & 0xFF) << 31 |
               (uint64_t)two << 39;
        if (two)
            key |= (uint64_t)(rs[RS_CCW_STENCILFAIL] & 15) << 40 | (uint64_t)(rs[RS_CCW_STENCILZFAIL] & 15) << 44 |
                   (uint64_t)(rs[RS_CCW_STENCILPASS] & 15) << 48 | (uint64_t)(rs[RS_CCW_STENCILFUNC] & 15) << 52;
    }
    return key;
}

static id<MTLDepthStencilState> get_dss(mtl_device *d, uint64_t key)
{
    __unsafe_unretained MtlDrawCache *c = CACHE(d);
    NSNumber *nk = @(key);
    id s = c->dss[nk];
    if (s) return s;
#define F(shift) ((uint32_t)(key >> (shift)) & 15u)
    int ze = key & 1, zw = (key >> 1) & 1, se = (key >> 2) & 1;
    MTLDepthStencilDescriptor *dd = [MTLDepthStencilDescriptor new];
    dd.depthCompareFunction = ze ? compare_fn(F(3)) : MTLCompareFunctionAlways;
    dd.depthWriteEnabled = zw != 0;
    if (se) {
        MTLStencilDescriptor *f = [MTLStencilDescriptor new];
        f.stencilFailureOperation = stencil_op(F(7));
        f.depthFailureOperation = stencil_op(F(11));
        f.depthStencilPassOperation = stencil_op(F(15));
        f.stencilCompareFunction = compare_fn(F(19));
        f.readMask = (uint32_t)(key >> 23) & 0xFF;
        f.writeMask = (uint32_t)(key >> 31) & 0xFF;
        MTLStencilDescriptor *b = f;
        if ((key >> 39) & 1) {
            b = [MTLStencilDescriptor new];
            b.stencilFailureOperation = stencil_op(F(40));
            b.depthFailureOperation = stencil_op(F(44));
            b.depthStencilPassOperation = stencil_op(F(48));
            b.stencilCompareFunction = compare_fn(F(52));
            b.readMask = f.readMask;
            b.writeMask = f.writeMask;
        }
        dd.frontFaceStencil = f;
        dd.backFaceStencil = b;
    }
#undef F
    id<MTLDepthStencilState> st = [MTLD(d) newDepthStencilStateWithDescriptor:dd];
    c->dss[nk] = st;
    return st;
}

/* Samplers: one MTLSamplerState per distinct D3D9 sampler state, its
   resource ID appended to the heap the shaders index through push data. */

static MTLSamplerAddressMode address_mode(uint32_t a)
{
    switch (a) {
    case 2: return MTLSamplerAddressModeMirrorRepeat;
    case 3: return MTLSamplerAddressModeClampToEdge;
    case 4: return MTLSamplerAddressModeClampToBorderColor;
    case 5: return MTLSamplerAddressModeMirrorClampToEdge;
    default: return MTLSamplerAddressModeRepeat;
    }
}

static uint32_t aniso_floor(void)
{
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("RT_METAL_ANISO");
#if TARGET_OS_VISION
        v = 8;
#else
        v = 1;
#endif
        if (e && *e) v = atoi(e);
        if (v < 1) v = 1;
        if (v > 16) v = 16;
    }
    return (uint32_t)v;
}

static uint16_t sampler_index(mtl_device *d, const uint32_t *ss)
{
    __unsafe_unretained MtlDrawCache *c = CACHE(d);
    uint32_t mag = ss[SS_MAGFILTER] & 3, min = ss[SS_MINFILTER] & 3, mip = ss[SS_MIPFILTER] & 3;
    uint32_t aniso = ss[SS_MAXANISOTROPY] < 1 ? 1 : ss[SS_MAXANISOTROPY] > 16 ? 16 : ss[SS_MAXANISOTROPY];
    if (mag != 3 && min != 3) aniso = 1;
    /* A floor under the game's anisotropy for linear mipmapped sampling.
       The game asks for none, and on the headset the street and floor seen
       at a glancing angle shimmer and read as grain (2026-11-20 run):
       anisotropic filtering is what keeps those sharp, at little cost on
       this GPU. RT_METAL_ANISO=N sets the floor (1 is the game's own). */
    uint32_t floor_aniso = aniso_floor();
    if (floor_aniso > aniso && min == 2 && mip != 0) aniso = floor_aniso;
    uint32_t border = ss[SS_BORDERCOLOR];
    uint32_t bkind = (border >> 24) < 128 ? 0 : (border & 0xFFFFFF) > 0x7F7F7F ? 2 : 1;
    uint64_t key = (uint64_t)(ss[SS_ADDRESSU] & 7) | (uint64_t)(ss[SS_ADDRESSV] & 7) << 3 |
                   (uint64_t)(ss[SS_ADDRESSW] & 7) << 6 | (uint64_t)mag << 9 | (uint64_t)min << 11 |
                   (uint64_t)mip << 13 | (uint64_t)aniso << 15 | (uint64_t)bkind << 20 |
                   (uint64_t)(ss[SS_MAXMIPLEVEL] & 0xFF) << 22;
    NSNumber *nk = @(key);
    NSNumber *idx = c->samplers[nk];
    if (idx) return idx.unsignedShortValue;
    if (c->heap_n >= HEAP_MAX) {
        note_once(d, @"heap-full", "sampler heap full at %u entries; reusing entry 0", HEAP_MAX);
        return 0;
    }
    MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
    sd.sAddressMode = address_mode(ss[SS_ADDRESSU]);
    sd.tAddressMode = address_mode(ss[SS_ADDRESSV]);
    sd.rAddressMode = address_mode(ss[SS_ADDRESSW]);
    sd.magFilter = mag >= 2 ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    sd.minFilter = min >= 2 ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    sd.mipFilter = mip == 0 ? MTLSamplerMipFilterNotMipmapped : mip == 1 ? MTLSamplerMipFilterNearest : MTLSamplerMipFilterLinear;
    sd.maxAnisotropy = aniso;
    sd.lodMinClamp = (float)(ss[SS_MAXMIPLEVEL] & 0xFF);
    sd.borderColor = bkind == 0 ? MTLSamplerBorderColorTransparentBlack
                     : bkind == 1 ? MTLSamplerBorderColorOpaqueBlack
                                  : MTLSamplerBorderColorOpaqueWhite;
    /* Without this the resource ID is not valid in an argument buffer, and
       every sample reads garbage with no error. */
    sd.supportArgumentBuffers = YES;
    id<MTLSamplerState> s = [MTLD(d) newSamplerStateWithDescriptor:sd];
    if (!s) return 0;
    [c->sampler_objs addObject:s];
    uint32_t i = c->heap_n++;
    MTLResourceID rid = s.gpuResourceID;
    memcpy((uint8_t *)c->heap.contents + 8 * i, &rid, 8);
    c->samplers[nk] = @(i);
    return (uint16_t)i;
}

/* A texture binding as recorded: what each of the binding's views gets
   (the texture in its own view, dummies in the others) and the texture
   index of each view the stage has (shader.h, mtl_tex_layout). */
typedef struct {
    uint32_t fragment, binding;
    void *view[MTL_TEX_VIEWS];
    uint8_t slot[MTL_TEX_VIEWS];
} tex_ref;

/* Per-stage resources, resolved when the draw is recorded: textures into
   their view slots (uploaded or renamed now, so the list reads the
   version current at this draw), heap indices into the push data,
   sampler types into the spec data. Returns the bindings written. */
static uint32_t bind_textures(mtl_device *d, tex_ref *out, const mtl_sc_info *in, uint8_t *push, uint32_t *spec,
                              int fragment)
{
    __unsafe_unretained MtlDrawCache *c = CACHE(d);
    mtl_state *st = &d->st;
    uint32_t k = 0;
    for (uint32_t b = 0; b < MTL_NSAMP; b++) {
        if (!(in->sampler_mask & (1u << b))) continue;
        mtl_texture *t = st->tex[b];
        /* Raw pointers: every texture here is owned by its object, the
           cache or (from here on) the list, so counting references on
           each draw would only cost time. */
        void *g = !t ? NULL : t->gtex && !t->dirty ? t->gtex : (__bridge void *)mtl_texture_gpu(t);
        uint32_t type = 3; /* null */
        if (g) {
            /* make_texture makes cubes and 2D textures only. */
            type = t->faces == 6 ? 2 : 0;
            t->last_use = d->serial;
            mtl_rec_keep(d, g);
        }
        uint32_t view = in->view[b];
        void *v2 = (__bridge void *)c->dummy2d, *v3 = (__bridge void *)c->dummy3d, *vc = (__bridge void *)c->dummycube;
        if (view == MTL_VIEW_ANY || view == 0) {
            if (type == 0) v2 = g;
            else if (type == 1) v3 = g;
            else if (type == 2) vc = g;
        } else {
            uint32_t want = view == MTL_VIEW_CUBE ? 2 : view == MTL_VIEW_3D ? 1 : 0;
            if (type == want) {
                if (want == 0) v2 = g;
                else if (want == 1) v3 = g;
                else vc = g;
            } else if (type != 3) {
                note_once(d, [NSString stringWithFormat:@"view-%u-%u", b, type],
                          "texture binding %u: shader declares view %u, texture is type %u; it reads as null", b, view, type);
                type = 3;
            }
        }
        tex_ref *r = &out[k];
        r->fragment = (uint32_t)fragment;
        r->binding = b;
        r->view[0] = v2;
        r->view[1] = v3;
        r->view[2] = vc;
        memcpy(r->slot, in->tex_slot[b], sizeof r->slot);
        if (fragment) {
            spec[4] = (spec[4] & ~(3u << (2 * b))) | type << (2 * b);
        } else {
            uint32_t vb = b >= 17 ? b - 17 : 0;
            spec[3] = (spec[3] & ~(3u << (2 * vb))) | type << (2 * vb);
        }
        /* The heap index only changes with the binding's sampler state,
           so the dictionary is asked once per change, not per draw. */
        if (st->samp_heap[b] < 0) st->samp_heap[b] = sampler_index(d, st->samp[b]);
        uint16_t si = (uint16_t)st->samp_heap[b];
        if (st->samp[b][SS_SRGBTEXTURE])
            note_once(d, @"srgb", "SRGBTEXTURE sampler state is set; textures are sampled linear");
        memcpy(push + in->sampler_push_base + 2 * k, &si, 2);
        k++;
    }
    return k;
}

/* Which stages of a draw are DXVK's fixed-function ones (its
   UseProgrammableVS and UseProgrammablePS), and the facts both the spec
   data and the constants need. */
typedef struct {
    int ff_vs, ff_ps;
    uint32_t prim;
    uint32_t decl_flags;
} draw_kind;

/* D3D9DeviceEx::GetTextureStageArgMask: which of arguments 0, 1, 2 an op
   reads. */
static uint32_t stage_arg_mask(uint32_t op)
{
    switch (op) {
    case 1: case 22: case 23: return 0; /* DISABLE, BUMPENVMAP, BUMPENVMAPLUMINANCE */
    case 2: case 17: return 2;          /* SELECTARG1, PREMODULATE */
    case 3: return 4;                   /* SELECTARG2 */
    case 25: case 26: return 7;         /* MULTIPLYADD, LERP */
    default: return 6;
    }
}

/* D3D9SpecData::getTextureArg: the selector in three bits and the
   COMPLEMENT and ALPHAREPLICATE flags shifted down into the next two. */
static uint32_t stage_arg(uint32_t a) { return ((a & 0xFu) | ((a & ~0xFu) >> 1)) & 0x1Fu; }

/* D3D9DeviceEx::UpdateFixedFunctionPS: the active stages run up to the
   first disabled one, or the first whose colour reads a texture that is
   not bound; ops and arguments go into spec words 8 to 19. There is
   always a colour target here, so DXVK's colour discard never applies. */
static void ff_stage_spec(mtl_device *d, uint32_t spec[20])
{
    mtl_state *st = &d->st;
    uint32_t active = 0;
    for (uint32_t i = 0; i < 8; i++) {
        const uint32_t *t = st->tss[i];
        uint32_t cop = t[TSS_COLOROP];
        if (cop == TOP_DISABLE) break;
        if (!st->tex[i]) {
            uint32_t m = stage_arg_mask(cop);
            uint32_t a0 = ((m & 1) ? t[TSS_COLORARG0] : TA_CURRENT) & 0xF;
            uint32_t a1 = ((m & 2) ? t[TSS_COLORARG1] : TA_CURRENT) & 0xF;
            uint32_t a2 = ((m & 4) ? t[TSS_COLORARG2] : TA_CURRENT) & 0xF;
            if (a0 == TA_TEXTURE || a1 == TA_TEXTURE || a2 == TA_TEXTURE) break;
        }
        active++;
    }
    for (uint32_t i = 0; i < 8; i++) {
        uint32_t ops, args = 0;
        if (i < active) {
            const uint32_t *t = st->tss[i];
            uint32_t result = t[TSS_RESULTARG];
            int result_temp = result == TA_TEMP;
            uint32_t cop = t[TSS_COLOROP], aop = t[TSS_ALPHAOP];
            uint32_t cm = stage_arg_mask(cop), am = stage_arg_mask(aop);
            uint32_t c0 = (cm & 1) ? t[TSS_COLORARG0] : 0, c1 = (cm & 2) ? t[TSS_COLORARG1] : 0,
                     c2 = (cm & 4) ? t[TSS_COLORARG2] : 0;
            uint32_t a0 = (am & 1) ? t[TSS_ALPHAARG0] : 0, a1 = (am & 2) ? t[TSS_ALPHAARG1] : 0,
                     a2 = (am & 4) ? t[TSS_ALPHAARG2] : 0;
            /* A first stage writing temp with alpha left undefined takes
               diffuse alpha. */
            if (!i && result_temp && cop != TOP_DISABLE && aop == TOP_DISABLE) {
                aop = TOP_SELECTARG1;
                a0 = 0;
                a1 = TA_DIFFUSE;
                a2 = 0;
            }
            /* The last stage always writes current. */
            if (i + 1 == active) result = TA_CURRENT;
            ops = (cop & 0x1F) | (aop & 0x1F) << 8 | ((result & 0xF) == TA_TEMP ? 1u << 15 : 0);
            if (cop != TOP_DISABLE) args |= stage_arg(c0) | stage_arg(c1) << 5 | stage_arg(c2) << 10;
            if (aop != TOP_DISABLE) args |= stage_arg(a0) << 16 | stage_arg(a1) << 21 | stage_arg(a2) << 26;
        } else {
            ops = TOP_DISABLE | TOP_DISABLE << 5; /* disableTextureStage, bit for bit */
        }
        spec[8 + i / 2] |= ops << (16 * (i & 1));
        spec[12 + i] = args;
    }
}

/* Spec data words (D3D9SpecData, DXVK d3d9_state.h). Words not set here
   stay zero. */
static void build_spec(mtl_device *d, uint32_t spec[20], uint32_t clip_count, const draw_kind *k)
{
    const uint32_t *rs = d->rs;
    mtl_state *st = &d->st;
    memset(spec, 0, 80);
    int points = k->prim == 1;
    /* Alpha test off is compare op ALWAYS (7); zero would be NEVER and
       discard every pixel. Precision 0 is the 8-bit unorm targets. Point
       sprites count for point lists only, as DXVK's UpdatePointMode has
       it: the fixed-function pixel shader replaces every texcoord with
       the point coordinate while the bit is set. */
    uint32_t aop = rs[RS_ALPHATESTENABLE] ? (uint32_t)compare_fn(rs[RS_ALPHAFUNC]) : 7u;
    spec[0] = aop | (points && rs[RS_POINTSPRITEENABLE] ? 1u << 8 : 0) | (clip_count & 7) << 16;
    if (rs[RS_FOGENABLE]) {
        uint32_t table = rs[RS_FOGTABLEMODE] & 3, vertex = rs[RS_FOGVERTEXMODE] & 3;
        /* Z fog only for a fixed-function vertex stage with an orthographic
           projection (DXVK's UpdateFog); POSITIONT means W fog. */
        int use_z = k->ff_vs && !(k->decl_flags & MTL_DECL_POSITIONT) && mtl_ff_fog_use_z(st->xf[MTL_XF_PROJ]);
        spec[1] = 1u | (table ? 0 : vertex) << 8 | table << 16 | (uint32_t)use_z << 24;
    }
    for (uint32_t s = 0; s < 8; s++)
        if (st->tss[s][TSS_TEXTURETRANSFORMFLAGS] & 256) spec[2] |= 1u << s;
    if (rs[RS_SPECULARENABLE]) spec[2] |= 1u << 8;
    if (points && rs[RS_POINTSCALEENABLE] && k->ff_vs) spec[2] |= 1u << 16;
    if (!k->ff_ps && ((st->ps->version >> 8) & 0xFF) >= 3) spec[2] |= 1u << 24;
    /* Every sampler starts null (3) and bind_textures fills the bound. */
    spec[3] = 0xFF;
    spec[4] = 0xFFFFFFFFu;
    uint32_t vb = 0, pb = 0;
    for (uint32_t i = 0; i < 16; i++) {
        vb |= (st->vsb[i] ? 1u : 0u) << i;
        pb |= (st->psb[i] ? 1u : 0u) << i;
    }
    vb = !k->ff_vs && st->vs->sc ? vb & mtl_sc_get_info(st->vs->sc)->bool_mask : 0;
    pb = !k->ff_ps && st->ps->sc ? pb & mtl_sc_get_info(st->ps->sc)->bool_mask : 0;
    spec[3] |= vb << 16;
    spec[6] = pb;
    if (k->ff_ps) ff_stage_spec(d, spec);
}

static void build_push(mtl_device *d, uint8_t push[64], int vertex, uint32_t cf_size, const draw_kind *k)
{
    const uint32_t *rs = d->rs;
    memset(push, 0, 64);
    uint32_t fog = rs[RS_FOGCOLOR] & 0xFFFFFFu;
    uint32_t aref = rs[RS_ALPHAREF] & 0xFF;
    uint32_t w0 = fog | aref << 24;
    float start = as_float(rs[RS_FOGSTART]), end = as_float(rs[RS_FOGEND]);
    float scale = end != start ? 1.0f / (end - start) : FLT_MAX;
    float dens = as_float(rs[RS_FOGDENSITY]);
    memcpy(push, &w0, 4);
    memcpy(push + 4, &scale, 4);
    memcpy(push + 8, &end, 4);
    memcpy(push + 12, &dens, 4);
    if (vertex) {
        float ps = as_float(rs[RS_POINTSIZE]), pmin = as_float(rs[RS_POINTSIZE_MIN]), pmax = as_float(rs[RS_POINTSIZE_MAX]);
#define FX13(f) ((uint32_t)fminf(fmaxf((f) * 8.0f, 0.0f), 65535.0f))
        uint32_t w8 = (cf_size & 0xFFFF) | FX13(ps) << 16, w9 = FX13(pmin) | FX13(pmax) << 16;
#undef FX13
        memcpy(push + 32, &w8, 4);
        memcpy(push + 36, &w9, 4);
        if (k->ff_vs && k->prim == 1 && rs[RS_POINTSCALEENABLE]) {
            /* UpdatePointMode: the D3D9 factors over the viewport height
               squared. */
            float h = (float)d->vp.Height, f = h > 0 ? 1.0f / (h * h) : 0.0f;
            float abc[3] = {f * as_float(rs[RS_POINTSCALE_A]), f * as_float(rs[RS_POINTSCALE_B]),
                            f * as_float(rs[RS_POINTSCALE_C])};
            memcpy(push + FF_VS_PUSH_SCALE, abc, sizeof abc);
        }
    } else if (k->ff_ps) {
        memcpy(push + FF_PS_PUSH_TFACTOR, &rs[RS_TEXTUREFACTOR], 4);
    }
}

/* A buffer and offset as a recorded draw binds them. */
typedef struct {
    void *buf; /* id<MTLBuffer>: a ring chunk, the zero buffer, or storage the list keeps */
    uint32_t off, pad;
} buf_ref;

/* Constant data goes into the upload ring when the draw is recorded, so
   the list carries it by value and the replay only binds it. NULL data
   or zero bytes bind the zero buffer, as the shaders declare every CBV. */
static buf_ref ring_cbv(mtl_device *d, const void *data, uint32_t bytes)
{
    buf_ref r = {(__bridge void *)CACHE(d)->zero, 0, 0};
    if (!bytes) return r;
    void *b;
    uint32_t off;
    void *p = ring_alloc(d, bytes, &b, &off);
    if (!p) return r;
    memcpy(p, data, bytes);
    r.buf = b;
    r.off = off;
    return r;
}

/* Packs a programmable stage's constants straight into the ring, sized
   first so each buffer takes only what it uses. */
static void ring_constants(mtl_device *d, const mtl_sc *sc, const float *f, const int32_t *i, const uint32_t *b,
                           uint32_t changed, buf_ref *stat, buf_ref *dyn, uint32_t *cf_size)
{
    buf_ref zero = {(__bridge void *)CACHE(d)->zero, 0, 0};
    static uint8_t scratch[MTL_SC_MAX_CONST];
    *stat = zero;
    if (dyn) *dyn = zero;
    uint32_t sn = 0, dn = 0, cf = 0;
    mtl_sc_constant_sizes(sc, changed, &sn, &dn);
    if (!dyn) dn = 0;
    void *sb = NULL, *db = NULL;
    uint32_t so = 0, dof = 0;
    uint8_t *sp = sn ? ring_alloc(d, sn, &sb, &so) : NULL;
    uint8_t *dp = dn ? ring_alloc(d, dn, &db, &dof) : NULL;
    uint32_t got_s = 0, got_d = 0;
    mtl_sc_constants(sc, f, i, b, changed, sp ? sp : scratch, &got_s, dp ? dp : scratch, &got_d, &cf);
    if (sp && got_s) *stat = (buf_ref){sb, so, 0};
    if (dp && got_d) *dyn = (buf_ref){db, dof, 0};
    if (cf_size) *cf_size = cf;
}

static const char *prim_name(uint32_t t)
{
    static const char *const n[] = {"?", "points", "lines", "line strip", "triangles", "triangle strip", "triangle fan"};
    return t <= 6 ? n[t] : "?";
}

static uint32_t prim_verts(uint32_t type, uint32_t count)
{
    switch (type) {
    case 1: return count;
    case 2: return count * 2;
    case 3: return count + 1;
    case 4: return count * 3;
    case 5: case 6: return count + 2;
    default: return 0;
    }
}

static MTLPrimitiveType prim_type(uint32_t t)
{
    switch (t) {
    case 1: return MTLPrimitiveTypePoint;
    case 2: return MTLPrimitiveTypeLine;
    case 3: return MTLPrimitiveTypeLineStrip;
    case 5: return MTLPrimitiveTypeTriangleStrip;
    default: return MTLPrimitiveTypeTriangle; /* lists, and fans once converted */
    }
}

static NSString *stage_name(mtl_blob *b, const char *ff)
{
    return b && b->sc ? [NSString stringWithFormat:@"%s %u.%u", mtl_sc_hash(b->sc), (b->version >> 8) & 0xFF, b->version & 0xFF]
                      : @(ff);
}

/* The fixed-function vertex stage's CBVs: ShaderData, and the blend
   matrices only when vertex blending is on (a zero buffer otherwise,
   since the shader declares the binding either way). */
static void ff_vs_constants(mtl_device *d, const mtl_blob *decl, buf_ref *data_ref, buf_ref *blend_ref)
{
    mtl_state *st = &d->st;
    uint8_t blend[MTL_FF_BLEND_BYTES];
    const void *lights[8];
    uint32_t n = 0;
    for (uint32_t i = 0; i < st->nlights && n < 8; i++)
        if (st->lights[i].enabled) {
            uint32_t type;
            memcpy(&type, st->lights[i].light, 4);
            /* An enabled light of no valid type is skipped here, as in
               mtl_ff_vs_data, so it does not take one of the eight places. */
            if (type >= 1 && type <= 3) lights[n++] = st->lights[i].light;
        }
    mtl_ff_vs_in in = {
        .xf = &st->xf[0][0],
        .material = st->material,
        .lights = lights,
        .nlights = n,
        .rs = d->rs,
        .tss = &st->tss[0][0],
        .decl_flags = decl->decl_flags,
        .tc_mask = decl->tc_mask,
        .vp_x = (float)d->vp.X,
        .vp_y = (float)d->vp.Y,
        .vp_w = (float)d->vp.Width,
        .vp_h = (float)d->vp.Height,
        .ztest = d->rs[RS_ZENABLE] && d->ds,
    };
    void *b;
    uint32_t off;
    static uint8_t scratch[MTL_FF_VS_BYTES];
    uint8_t *p = ring_alloc(d, MTL_FF_VS_BYTES, &b, &off);
    uint32_t mode = mtl_ff_vs_data(&in, p ? p : scratch, blend);
    *data_ref = p ? (buf_ref){b, off, 0} : (buf_ref){(__bridge void *)CACHE(d)->zero, 0, 0};
    *blend_ref = ring_cbv(d, blend, mode == 1 ? MTL_FF_BLEND_BYTES : 0);
}

/* The fixed-function pixel stage's shared CBV: each stage's constant and
   bump environment, the matrix stored transposed as DXVK stores it. */
static buf_ref ff_ps_constants(mtl_device *d)
{
    struct {
        uint32_t constant, pad;
        float bump[2][2];
        float lscale, loffset;
    } stages[8];
    _Static_assert(sizeof stages == 8 * 32, "D3D9SharedPS is 32 bytes a stage");
    for (uint32_t i = 0; i < 8; i++) {
        const uint32_t *t = d->st.tss[i];
        stages[i].constant = t[TSS_CONSTANT];
        stages[i].pad = 0;
        stages[i].bump[0][0] = as_float(t[TSS_BUMPENVMAT00]);
        stages[i].bump[1][0] = as_float(t[TSS_BUMPENVMAT00 + 1]);
        stages[i].bump[0][1] = as_float(t[TSS_BUMPENVMAT00 + 2]);
        stages[i].bump[1][1] = as_float(t[TSS_BUMPENVMAT00 + 3]);
        stages[i].lscale = as_float(t[TSS_BUMPENVLSCALE]);
        stages[i].loffset = as_float(t[TSS_BUMPENVLOFFSET]);
    }
    return ring_cbv(d, stages, sizeof stages);
}

/* A draw as recorded: everything the replay binds, by value. */
enum { CALL_NONE, CALL_ARRAYS, CALL_INDEXED };

/* Offsets of the small per-draw data in one ring block. Each is 256
   aligned, the strictest constant buffer offset rule Metal has. */
#define BLOCK_SPEC 0
#define BLOCK_VPUSH 256
#define BLOCK_PPUSH 512
#define BLOCK_CLIP 768
#define BLOCK_BYTES 1024

typedef struct {
    cmd_head h;
    mtl_blob *vs, *ps; /* NULL for a fixed-function stage; the list holds the others */
    pso_desc pd;
    uint64_t dss;
    uint32_t stencil_on, stencil_ref, cull, fill;
    float blend[4], bias, slope;
    float vp[6];
    uint32_t sc[4];
    uint32_t streams; /* bit s: stream s bound */
    buf_ref stream[MTL_MAX_STREAMS];
    buf_ref vcbv[7], fcbv[7];
    uint32_t vcbv_mask, fcbv_mask;
    buf_ref block; /* spec data, vertex push, pixel push, clip planes */
    uint32_t call, prim, index_type, count, start, instances;
    int32_t base;
    buf_ref ib;
    uint32_t ntex;
    float stereo_a; /* the game's clip x per view unit for a world draw, else 0: left as it is in every eye */
    int stereo_hud; /* a HUD or 2D draw: put on the HUD plane in each eye */
    mtl_stereo_proj_full stereo_p; /* a world draw's whole projection, for full view */
    int stereo_sky; /* a world draw of the 3D sky (mtl_fv_scan): full view keeps it at infinity */
    int stereo_screen; /* a camera-held effect (mtl_fv_scan): full view puts the game's screen of it on the aura panel */
    int arm_mesh;      /* one of the view model's two arm meshes, 1 or 2 (mtl_fv_scan), for RT_HIDE_ARMS */
    /* A HUD draw's position element and the rows that take it to clip
       x, y and w, so full view can tell one covering the whole view
       (fv_cover) from one on the panel. fv_full: -1 not yet known. */
    int8_t fv_full;
    /* Its arm HUD side from the same box (mtl_stereo_arm_side), 0 the
       panel, -1 not yet known: read in the first eye, kept for the second. */
    int8_t arm_side;
    uint8_t cov_stream, cov_type, cov_ok;
    uint16_t cov_off, cov_stride;
    float cov_rows[3][4];
    tex_ref tex[];
} cmd_draw;

struct mtl_fv_draw {
    uint32_t off, group;
    float a;
    float cam[3];
    int has_cam;
};

/* The common part of every draw: everything but the call, resolved from
   device state now. Returns the recorded command for the caller to fill
   in the call, or NULL when the draw is dropped (logged). */
typedef struct {
    void *buf; /* id<MTLBuffer>: a ring chunk */
    uint32_t off, stride;
} stream_override;

/* Full view reads each list's world draws into the back buffer once it
   is complete (mtl_fv_scan), so it only notes where they are: offsets,
   since the list's memory moves as it grows. Threaded mode only: inline
   mode has already encoded a draw by the time the list ends. */
static void fv_note_draw(mtl_device *d, cmd_draw *cd)
{
    if (d->fv_n == d->fv_cap) {
        uint32_t n = d->fv_cap ? d->fv_cap * 2 : 1024;
        struct mtl_fv_draw *p = realloc(d->fv_draws, n * sizeof *p);
        if (!p) return;
        d->fv_draws = p;
        d->fv_cap = n;
    }
    d->fv_draws[d->fv_n++] = (struct mtl_fv_draw){(uint32_t)((uint8_t *)cd - mtl_rec_base(d)),
                                                  d->fv_group,
                                                  cd->stereo_p.a,
                                                  {d->fv_cam[0], d->fv_cam[1], d->fv_cam[2]},
                                                  d->fv_cam_ok};
}

/* The source engine draws the 3D sky (the distant city, a model at a
   small scale around a sky camera) into the back buffer first, clears
   depth, then draws the world. So the world draws before the group with
   the most of them, drawn with the same field of view, are the sky. The
   view model is drawn with its own field of view, so a frame whose world
   draws all share one a drew none, which is how third person looks from
   here (the game switches it per weapon, not only through its camera
   commands). Both are logged on change. */
void mtl_fv_scan(mtl_device *d, uint8_t *cmds, int present)
{
    uint32_t n = d->fv_n;
    d->fv_n = 0;
    d->fv_group = 0;
    d->fv_cam_ok = 0;
    if (!n) return;
    /* The game clears no depth between its sky and its world (la_hub_1's
       captures, 2026-11-20), so the views are told apart by their cameras:
       the 3D sky's is a point in the sky's own small copy of the city, far
       from the player's. Draws are grouped by camera; one that shows none
       goes with the main view. */
    enum { MAX_CAMS = 8 };
    float cams[MAX_CAMS][3];
    uint32_t cam_n[MAX_CAMS] = {0}, ncams = 0;
    for (uint32_t i = 0; i < n; i++) {
        struct mtl_fv_draw *w = &d->fv_draws[i];
        w->group = UINT32_MAX;
        if (!w->has_cam) continue;
        uint32_t g = 0;
        for (; g < ncams; g++) {
            float dx = w->cam[0] - cams[g][0], dy = w->cam[1] - cams[g][1], dz = w->cam[2] - cams[g][2];
            float tol = 1.0f + 1e-4f * (fabsf(cams[g][0]) + fabsf(cams[g][1]) + fabsf(cams[g][2]));
            if (dx * dx + dy * dy + dz * dz <= tol * tol) break;
        }
        if (g == ncams) {
            if (ncams == MAX_CAMS) continue;
            memcpy(cams[ncams++], w->cam, sizeof cams[0]);
        }
        w->group = g;
        cam_n[g]++;
    }
    /* The camera with the most world draws is the main view's. */
    uint32_t best = UINT32_MAX, best_n = 0;
    for (uint32_t g = 0; g < ncams; g++)
        if (cam_n[g] > best_n) best_n = cam_n[g], best = g;
    if (best == UINT32_MAX) return;
    uint32_t ngroups = ncams;
    for (uint32_t i = 0; i < n; i++)
        if (d->fv_draws[i].group == UINT32_MAX) d->fv_draws[i].group = best;
    /* Its a: the first draw's in it (the world comes before the model). */
    float main_a = 0.0f;
    for (uint32_t i = 0; i < n && main_a <= 0.0f; i++)
        if (d->fv_draws[i].group == best) main_a = d->fv_draws[i].a;
    if (!(main_a > 0.0f)) return;
    /* A discipline's aura (Presence's smoke) is a cloud held in front of
       the camera, drawn with the view model's field of view and meant to
       fill the screen. At its true angle it shrank to a clump in the
       middle of the wider eye (la_hub_1's captures, 2026-11-21), so a
       view-model draw from the fixed-function pipe that writes no depth
       is drawn as the game's screen, on a head-locked panel of its own.
       Filling the whole eye instead put the ring at the lens's edge and
       cost four to five times the GPU time. The model itself is shaded
       and writes depth. */
    uint32_t sky = 0, vm = 0, screen = 0;
    void *arm_tex = NULL;
    float vm_a = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        const struct mtl_fv_draw *w = &d->fv_draws[i];
        cmd_draw *cd = (cmd_draw *)(cmds + w->off);
        int same = fabsf(w->a - main_a) <= 0.02f * main_a;
        if (w->group != best && same) {
            cd->stereo_sky = 1;
            sky++;
        } else if (w->group == best && !same) {
            vm++;
            vm_a = w->a;
        }
        /* The first person arms: the view model's skinned meshes drawn with
           one shader whatever the weapon (a capture of the shotgun and the
           Desert Eagle, 2026-11-22: three draws on each of two textures),
           the first texture seen in the frame one arm and the other the
           second. Marked so RT_HIDE_ARMS can leave either out. */
        if (w->group == best && !same && cd->vs && cd->vs->sc && cd->ntex &&
            !strcmp(mtl_sc_hash(cd->vs->sc), "0578c91b49f192fe099969e6b265821a")) {
            void *t = cd->tex[0].view[0];
            if (!arm_tex) arm_tex = t;
            cd->arm_mesh = t == arm_tex ? 1 : 2;
        }
        /* Whatever camera it reports: the aura's is not the model's. */
        if (!same && !cd->vs && !(cd->dss & 2)) {
            cd->stereo_screen = 1;
            screen++;
        }
    }
    if (screen && d->fv_screen_logs < 4) {
        d->fv_screen_logs++;
        mtl_log("full view: %u camera-held effect draws on the aura panel (the world's a %.4f)", screen, main_a);
    }
    if (sky && d->fv_sky_logs < 4) {
        d->fv_sky_logs++;
        mtl_log("full view: %u world draws from another camera taken as the 3D sky (%u cameras, the world's with %u "
                "draws at %.1f %.1f %.1f)",
                sky, ngroups, best_n, cams[best][0], cams[best][1], cams[best][2]);
        for (uint32_t g = 0; g < ncams; g++)
            if (g != best) mtl_log("full view:   camera at %.1f %.1f %.1f, %u draws", cams[g][0], cams[g][1], cams[g][2], cam_n[g]);
    }
    if (!present) return;
    /* A view model in the frame, or none: a few frames in a row before it
       counts, and nothing at all until one has been seen this session, so
       a game whose model classes differently leaves this off. */
    int has = vm != 0;
    if (has && !d->fv_vm_ever) {
        d->fv_vm_ever = 1;
        mtl_log("full view: a view model seen (a %.4f against the world's %.4f): no view model now reads as third "
                "person",
                vm_a, main_a);
    }
    if (!d->fv_vm_ever) return;
    d->fv_vm_run = has == (d->fv_vm_state > 0) ? 0 : d->fv_vm_run + 1;
    if (d->fv_vm_state < 0 || d->fv_vm_run >= 8) {
        d->fv_vm_state = has;
        d->fv_vm_run = 0;
        mtl_immersive_view_model(has);
        if (d->fv_vm_logs < 64) {
            d->fv_vm_logs++;
            mtl_log("full view: view model %s", has ? "drawn: first person" : "gone: third person");
        }
    }
}

/* Whether each eye corrects this draw, decided from what the draw itself
   says (stereo.h): its target, its vertices, and the projection its
   vertex stage reads. Counted always, so a mono run's log already shows
   how the game's draws class. */
static float stereo_class(mtl_device *d, const mtl_surface *rt, const mtl_blob *decl, const draw_kind *k,
                          const mtl_sc_info *vi, const mtl_sc_info *pi, int *hud, mtl_stereo_proj_full *proj)
{
    int samples_rt = 0;
    for (uint32_t b = 0; b < MTL_NSAMP; b++)
        if ((pi->sampler_mask & (1u << b)) && d->st.tex[b] && (d->st.tex[b]->usage & D3DUSAGE_RENDERTARGET)) samples_rt = 1;
    mtl_stereo_draw_in in = {
        .to_back_buffer = rt == d->back,
        .positiont = decl->has_post,
        .ff_vs = k->ff_vs,
        .ff_proj = d->st.xf[MTL_XF_PROJ],
        .pos_regs = k->ff_vs ? 0 : vi->pos_regs,
        .vsf = d->st.vsf,
        .last_a = d->stereo_a,
        .samples_rt = samples_rt,
        .last_proj = &d->stereo_p,
    };
    mtl_stereo_proj_full seen_p;
    float a, seen;
    int c = mtl_stereo_classify_proj(&in, &a, &seen, proj, &seen_p);
    if (seen > 0.0f) {
        d->stereo_a = seen;
        d->stereo_p = seen_p;
    }
    /* The camera the draw is seen from (mtl_fv_scan tells the 3D sky by
       it). A shader draw has it in c8 whichever registers place its
       vertices; a fixed-function one only when it shows a projection. One
       that shows none keeps the last. */
    if (seen > 0.0f || !k->ff_vs) {
        float cam[3];
        int ok = mtl_stereo_draw_camera(k->ff_vs, in.vsf, d->st.xf[MTL_XF_VIEW], cam);
        if (ok) memcpy(d->fv_cam, cam, sizeof cam);
        if (ok || seen > 0.0f) d->fv_cam_ok = ok;
    }
    if (a > 0.0f) mtl_stereo_tally_add(&d->stereo_tally, a);
    *hud = mtl_stereo_class_is_hud(c);
    d->stereo_class[c]++;
    if (!(d->stereo_seen & (1u << c))) {
        d->stereo_seen |= 1u << c;
        mtl_log("stereo: first %s draw (%s vertex stage%s, a %.4f)", mtl_stereo_class_name(c),
                k->ff_vs ? "fixed-function" : in.pos_regs == MTL_POS_VIEWPROJ ? "view projection c8"
                                            : in.pos_regs == MTL_POS_MVP     ? "model view projection c4"
                                                                             : "unrecognised",
                decl->has_post ? ", POSITIONT" : "", a);
        /* The plane is sized by a and by b from the back buffer's aspect;
           the y row says whether the game's b agrees (square pixels). */
        if (c == MTL_STEREO_WORLD && in.pos_regs == MTL_POS_VIEWPROJ && rt->height)
            mtl_log("stereo: that draw's b %.4f from its y row, a x %u/%u = %.4f",
                    mtl_stereo_a_from_rows(in.vsf + 4 * (MTL_POS_VIEWPROJ + 1), in.vsf + 4 * (MTL_POS_VIEWPROJ + 3)),
                    rt->width, rt->height, a * (float)rt->width / (float)rt->height);
        /* Full view assumes the game's projection is centred (cx and cy
           0) and re-projects its depth by A; this says whether it is. */
        if (c == MTL_STEREO_WORLD)
            mtl_log("stereo: that draw's projection a %.4f cx %.4f b %.4f cy %.4f A %.4f", proj->a, proj->cx, proj->b,
                    proj->cy, proj->A);
    }
    return a;
}

/* Notes what fv_cover needs to read a HUD draw's positions back: the
   position element, its stream's stride, and the rows taking it to clip
   x, y and w. POSITIONT vertices are already in pixels, so their rows map
   the viewport onto normalized device coordinates. */
static void cov_note(mtl_device *d, cmd_draw *cd, const mtl_blob *decl, const draw_kind *k, const mtl_sc_info *vi,
                     const stream_override *up)
{
    const decl_elem *e = (const decl_elem *)decl->data;
    uint32_t n = decl->bytes / 8 - 1;
    int pos = -1;
    for (uint32_t i = 0; i < n; i++)
        if ((e[i].usage == 0 || e[i].usage == 9) && e[i].index == 0 && (e[i].type == 1 || e[i].type == 2 || e[i].type == 3))
            pos = (int)i;
    if (pos < 0 || e[pos].stream >= MTL_MAX_STREAMS) return;
    uint32_t s = e[pos].stream;
    uint32_t stride = up && s == 0 ? up->stride : d->st.vb_stride[s];
    if (!stride) return;
    float (*r)[4] = cd->cov_rows;
    memset(r, 0, sizeof cd->cov_rows);
    if (decl->has_post) {
        float vw = cd->vp[2], vh = cd->vp[3];
        if (!(vw > 0.0f) || !(vh > 0.0f)) return;
        r[0][0] = 2.0f / vw, r[0][3] = -1.0f - 2.0f * cd->vp[0] / vw;
        r[1][1] = -2.0f / vh, r[1][3] = 1.0f + 2.0f * cd->vp[1] / vh;
        r[2][3] = 1.0f;
    } else if (k->ff_vs) {
        /* D3D row vectors: clip = p W V P. */
        float wv[16], m[16];
        const float *w = d->st.xf[MTL_XF_WORLD0], *v = d->st.xf[MTL_XF_VIEW], *p = d->st.xf[MTL_XF_PROJ];
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) {
                wv[i * 4 + j] = 0.0f;
                for (int q = 0; q < 4; q++) wv[i * 4 + j] += w[i * 4 + q] * v[q * 4 + j];
            }
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) {
                m[i * 4 + j] = 0.0f;
                for (int q = 0; q < 4; q++) m[i * 4 + j] += wv[i * 4 + q] * p[q * 4 + j];
            }
        static const int col[3] = {0, 1, 3};
        for (int o = 0; o < 3; o++)
            for (int i = 0; i < 4; i++) r[o][i] = m[i * 4 + col[o]];
    } else {
        int pr = vi->pos_regs;
        if (pr != MTL_POS_VIEWPROJ && pr != MTL_POS_MVP) return;
        memcpy(r[0], d->st.vsf + 4 * pr, 16);
        memcpy(r[1], d->st.vsf + 4 * (pr + 1), 16);
        memcpy(r[2], d->st.vsf + 4 * (pr + 3), 16);
    }
    cd->cov_stream = (uint8_t)s;
    cd->cov_type = (uint8_t)(e[pos].type | (decl->has_post ? 0x80 : 0)); /* 0x80: POSITIONT */
    cd->cov_off = e[pos].offset;
    cd->cov_stride = (uint16_t)stride;
    cd->cov_ok = 1;
}

static cmd_draw *record_draw(mtl_device *d, uint32_t type, const stream_override *up)
{
    mtl_state *st = &d->st;
    __unsafe_unretained MtlDrawCache *c = CACHE(d);
    mtl_blob *decl = st->decl;
    if (type < 1 || type > 6) {
        d->dropped_draws++;
        return NULL;
    }
    if (!decl) {
        d->dropped_draws++;
        note_once(d, @"no-decl", "draw with neither a vertex declaration nor an FVF: dropped");
        return NULL;
    }
    /* DXVK's UseProgrammableVS and UseProgrammablePS. */
    draw_kind k = {!st->vs || decl->has_post, !st->ps, type, decl->decl_flags};
    mtl_blob *vs = k.ff_vs ? c->ff_vs : st->vs, *ps = k.ff_ps ? c->ff_ps : st->ps;
    if ((k.ff_vs || k.ff_ps) && ff_first_seen(c, vs->uid, ps->uid, st->fvf, decl->has_post)) {
        /* Logged by the pair of stages and the vertex layout, first time
           each combination is seen. */
        NSString *key = [NSString stringWithFormat:@"VS %@, PS %@%@%@", stage_name(k.ff_vs ? NULL : vs, "fixed-function"),
                                                   stage_name(k.ff_ps ? NULL : ps, "fixed-function"),
                                                   st->fvf ? [NSString stringWithFormat:@", FVF 0x%X", st->fvf] : @"",
                                                   decl->has_post ? @", POSITIONT vertices" : @""];
        if (![c->ff_seen containsObject:key]) {
            [c->ff_seen addObject:key];
            mtl_log("fixed-function draw (%s, first of its kind): %s", prim_name(type), key.UTF8String);
        }
    }
    if (k.ff_vs || k.ff_ps) d->ff_draws++;
    mtl_surface *rt = d->rt[0];
    uint32_t lvl, sl;
    __unsafe_unretained id<MTLTexture> ct = mtl_surface_texture(rt, &lvl, &sl);
    if (!ct) {
        d->dropped_draws++;
        note_once(d, @"no-rt", "draw with a render target that has no GPU texture: dropped");
        return NULL;
    }
    const mtl_sc_info *vi = k.ff_vs ? &c->ff_vs_info : mtl_sc_get_info(vs->sc);
    const mtl_sc_info *pi = k.ff_ps ? &c->ff_ps_info : mtl_sc_get_info(ps->sc);
    int match[MTL_MAX_INPUTS];
    match_inputs(decl, vi, match);
    int has_ds = ds_usable(d);
    int x8 = rt->format == D3DFMT_X8R8G8B8;
    uint16_t strides[MTL_MAX_STREAMS];
    if (up) {
        memset(strides, 0, sizeof strides);
        strides[0] = (uint16_t)up->stride;
    }
    if (!begin_pass(d)) {
        d->dropped_draws++;
        return NULL;
    }
    const uint32_t *rs = d->rs;
    if (rs[RS_SHADEMODE] == 1) note_once(d, @"flat", "SHADEMODE FLAT is set; it needs a linked compile and is drawn smooth");
    if (rs[RS_SRGBWRITEENABLE]) note_once(d, @"srgbw", "SRGBWRITEENABLE is set; targets are written linear");

    /* D3D9 samples pixels at integer coordinates, Metal at half-integer
       ones: the viewport moves by half a pixel, as DXVK's does. Metal's
       y points up in clip space like D3D9's, so there is no flip. */
    mtl_viewport vp = d->vp;
    mtl_rect sc = {(int32_t)vp.X, (int32_t)vp.Y, (int32_t)(vp.X + vp.Width), (int32_t)(vp.Y + vp.Height)};
    mtl_rect full = {0, 0, (int32_t)rt->width, (int32_t)rt->height};
    if (rs[RS_SCISSORTESTENABLE]) clip_rect(&sc, &st->scissor);
    if (!clip_rect(&sc, &full)) {
        d->dropped_draws++;
        return NULL;
    }

    /* Constants into the ring, before the command is laid out. */
    buf_ref vcbv[7], fcbv[7];
    uint32_t vmask = 0, fmask = 0, cf_size = 0;
    if (!k.ff_vs) {
        ring_constants(d, vs->sc, st->vsf, st->vsi, st->vsb, st->vs_changed_floats, &vcbv[MTL_CBV_VS_STATIC],
                       &vcbv[MTL_CBV_VS_DYNAMIC], &cf_size);
        vmask |= 1u << MTL_CBV_VS_STATIC | 1u << MTL_CBV_VS_DYNAMIC;
    } else {
        ff_vs_constants(d, decl, &vcbv[MTL_CBV_VS_FF], &vcbv[MTL_CBV_VS_BLEND]);
        vmask |= 1u << MTL_CBV_VS_FF | 1u << MTL_CBV_VS_BLEND;
    }
    if (!k.ff_ps) {
        ring_constants(d, ps->sc, st->psf, st->psi, st->psb, 0, &fcbv[MTL_CBV_PS_STATIC], NULL, NULL);
        fmask |= 1u << MTL_CBV_PS_STATIC;
    } else {
        fcbv[MTL_CBV_PS_SHARED] = ff_ps_constants(d);
        fmask |= 1u << MTL_CBV_PS_SHARED;
    }
    void *block_buf;
    uint32_t block_off;
    uint8_t *block = ring_alloc(d, BLOCK_BYTES, &block_buf, &block_off);
    if (!block) {
        d->dropped_draws++;
        return NULL;
    }
    float *clip = (float *)(block + BLOCK_CLIP);
    memset(clip, 0, 96);
    uint32_t nclip = 0;
    for (uint32_t i = 0; i < 6; i++)
        if ((rs[RS_CLIPPLANEENABLE] & (1u << i)) &&
            (st->clip[i][0] != 0 || st->clip[i][1] != 0 || st->clip[i][2] != 0 || st->clip[i][3] != 0))
            memcpy(clip + 4 * nclip++, st->clip[i], 16);
    vcbv[MTL_CBV_CLIP] = (buf_ref){block_buf, block_off + BLOCK_CLIP, 0};
    vmask |= 1u << MTL_CBV_CLIP;

    uint32_t ntex = (uint32_t)__builtin_popcount(vi->sampler_mask) + (uint32_t)__builtin_popcount(pi->sampler_mask);
    cmd_draw *cd = mtl_rec_cmd(d, CMD_DRAW, (uint32_t)(sizeof *cd + ntex * sizeof(tex_ref)));
    if (!k.ff_vs) {
        cd->vs = vs;
        mtl_rec_hold(d, vs);
    }
    if (!k.ff_ps) {
        cd->ps = ps;
        mtl_rec_hold(d, ps);
    }
    pipeline_desc(d, vs, ps, vi, match, (MTLPixelFormat)d->pass_pf, has_ds, x8, up ? strides : NULL, &cd->pd);
    cd->dss = dss_key(d, has_ds);
    cd->stencil_on = has_ds && rs[RS_STENCILENABLE];
    cd->stencil_ref = rs[RS_STENCILREF] & 0xFF;
    cd->cull = rs[RS_CULLMODE];
    cd->fill = rs[RS_FILLMODE];
    uint32_t bf = rs[RS_BLENDFACTOR];
    cd->blend[0] = ((bf >> 16) & 255) / 255.0f;
    cd->blend[1] = ((bf >> 8) & 255) / 255.0f;
    cd->blend[2] = (bf & 255) / 255.0f;
    cd->blend[3] = (bf >> 24) / 255.0f;
    /* D3D9's constant bias is in depth units; DXVK scales it for a 24-bit
       format, which is what the engine asked for. */
    cd->bias = as_float(rs[RS_DEPTHBIAS]) * 16777216.0f;
    cd->slope = as_float(rs[RS_SLOPESCALEDEPTHBIAS]);
    float zmin = fminf(fmaxf(vp.MinZ, 0.0f), 1.0f);
    float zmax = fminf(fmaxf(fmaxf(vp.MaxZ, vp.MinZ + (vp.MinZ < 0.5f ? 0.001f : 0.0f)), 0.0f), 1.0f);
    cd->vp[0] = vp.X + 0.5f;
    cd->vp[1] = vp.Y + 0.5f;
    cd->vp[2] = (float)vp.Width;
    cd->vp[3] = (float)vp.Height;
    cd->vp[4] = zmin;
    cd->vp[5] = zmax;
    cd->sc[0] = (uint32_t)sc.left;
    cd->sc[1] = (uint32_t)sc.top;
    cd->sc[2] = (uint32_t)(sc.right - sc.left);
    cd->sc[3] = (uint32_t)(sc.bottom - sc.top);

    /* Streams: the storage each buffer has now, which a later lock that
       could change it moves away from (gpu.m). */
    const decl_elem *e = (const decl_elem *)decl->data;
    for (uint32_t i = 0; i < vi->n_inputs; i++) {
        if (match[i] < 0) continue;
        uint32_t s = e[match[i]].stream;
        if (s >= MTL_MAX_STREAMS || (cd->streams & (1u << s))) continue;
        cd->streams |= 1u << s;
        if (up && s == 0) {
            cd->stream[0] = (buf_ref){up->buf, up->off, 0};
            continue;
        }
        mtl_buffer *vb = st->vb[s];
        if (!vb) {
            cd->stream[s] = (buf_ref){(__bridge void *)c->zero, 0, 0};
            continue;
        }
        vb->last_use = d->serial;
        mtl_rec_keep(d, vb->gbuf);
        cd->stream[s] = (buf_ref){vb->gbuf, st->vb_off[s], 0};
    }
    memcpy(cd->vcbv, vcbv, sizeof vcbv);
    memcpy(cd->fcbv, fcbv, sizeof fcbv);
    cd->vcbv_mask = vmask;
    cd->fcbv_mask = fmask;

    uint32_t spec[20];
    build_spec(d, spec, nclip, &k);
    uint8_t vpush[64], ppush[64];
    build_push(d, vpush, 1, cf_size, &k);
    build_push(d, ppush, 0, 0, &k);
    uint32_t nv = bind_textures(d, cd->tex, vi, vpush, spec, 0);
    uint32_t np = bind_textures(d, cd->tex + nv, pi, ppush, spec, 1);
    cd->ntex = nv + np;
    memcpy(block + BLOCK_SPEC, spec, sizeof spec);
    memcpy(block + BLOCK_VPUSH, vpush, sizeof vpush);
    memcpy(block + BLOCK_PPUSH, ppush, sizeof ppush);
    cd->block = (buf_ref){block_buf, block_off, 0};
    cd->instances = 1;
    cd->stereo_a = stereo_class(d, rt, decl, &k, vi, pi, &cd->stereo_hud, &cd->stereo_p);
    if (cd->stereo_a > 0.0f && rt == d->back && mtl_threaded(d)) fv_note_draw(d, cd);
    cd->fv_full = -1;
    cd->arm_side = -1;
    cd->cov_ok = 0;
    if (cd->stereo_hud) cov_note(d, cd, decl, &k, vi, up);
    return cd;
}

static uint32_t instance_count(mtl_device *d)
{
    uint32_t f = d->st.freq[0];
    return (f & 0x40000000u) ? (f & 0x7FFFFFu ? f & 0x7FFFFFu : 1) : 1;
}

/* Metal has no triangle fan: a fan becomes a list of 32-bit indices in
   the upload ring. src is NULL for a non-indexed fan from first. */
static int fan_indices(mtl_device *d, const void *src, uint32_t isize, uint32_t first, uint32_t prims, void **buf,
                       uint32_t *off)
{
    uint32_t *o = ring_alloc(d, prims * 3 * 4, buf, off);
    if (!o) return 0;
    note_once(d, @"fan", "triangle fans are converted to lists");
#define FAN_V(i) (!src ? first + (i) : isize == 2 ? ((const uint16_t *)src)[i] : ((const uint32_t *)src)[i])
    uint32_t hub = FAN_V(0);
    for (uint32_t t = 0; t < prims; t++) {
        o[3 * t] = hub;
        o[3 * t + 1] = FAN_V(t + 1);
        o[3 * t + 2] = FAN_V(t + 2);
    }
#undef FAN_V
    return 1;
}

static void call_arrays(cmd_draw *cd, uint32_t type, uint32_t start, uint32_t n, uint32_t instances)
{
    cd->call = CALL_ARRAYS;
    cd->prim = (uint32_t)prim_type(type);
    cd->start = start;
    cd->count = n;
    cd->instances = instances;
}

static void call_indexed(cmd_draw *cd, MTLPrimitiveType prim, uint32_t n, MTLIndexType it, void *ib, uint32_t off,
                         uint32_t instances, int32_t base)
{
    cd->call = CALL_INDEXED;
    cd->prim = (uint32_t)prim;
    cd->count = n;
    cd->index_type = (uint32_t)it;
    cd->ib = (buf_ref){ib, off, 0};
    cd->instances = instances;
    cd->base = base;
}

static HRESULT dev_dp(mtl_device *d, uint32_t type, uint32_t start, uint32_t count)
{
    if (!count) return D3D_OK;
    @autoreleasepool {
        cmd_draw *cd = record_draw(d, type, NULL);
        if (cd) {
            if (type == 6) {
                void *ib;
                uint32_t off;
                if (fan_indices(d, NULL, 0, start, count, &ib, &off))
                    call_indexed(cd, MTLPrimitiveTypeTriangle, count * 3, MTLIndexTypeUInt32, ib, off, instance_count(d), 0);
            } else {
                call_arrays(cd, type, start, prim_verts(type, count), instance_count(d));
            }
            d->draws++;
        }
    }
    mtl_rec_end(d);
    return D3D_OK;
}

/* The index part of an indexed draw. src is the indices in CPU memory,
   src_buf the GPU storage they are also in (nil for the UP draws). */
static void draw_indexed(mtl_device *d, cmd_draw *cd, uint32_t type, int32_t base, uint32_t count, const uint8_t *src,
                         void *src_buf, uint32_t src_off, uint32_t isize, uint32_t instances)
{
    uint32_t n = prim_verts(type, count);
    MTLIndexType it = isize == 2 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32;
    void *ib = src_buf;
    uint32_t off = src_off;
    if (type == 6) {
        if (!fan_indices(d, src, isize, 0, count, &ib, &off)) return;
        n = count * 3;
        it = MTLIndexTypeUInt32;
    } else if (!ib || (off & 3)) {
        /* Metal wants the index offset 4-aligned; D3D9 does not. */
        void *p = ring_alloc(d, n * isize, &ib, &off);
        if (!p) return;
        memcpy(p, src, (size_t)n * isize);
    }
    call_indexed(cd, prim_type(type), n, it, ib, off, instances, base);
    d->draws++;
}

static HRESULT dev_dip(mtl_device *d, uint32_t type, int32_t base, uint32_t min, uint32_t num, uint32_t start,
                       uint32_t count)
{
    (void)min;
    (void)num;
    if (!count) return D3D_OK;
    mtl_buffer *ib = d->st.ib;
    if (!ib) return D3DERR_INVALIDCALL;
    uint32_t isize = ib->format_or_fvf == D3DFMT_INDEX32 ? 4 : 2;
    uint64_t end = ((uint64_t)start + prim_verts(type, count)) * isize;
    if (end > ib->length) {
        note_once(d, @"ib-range", "DrawIndexedPrimitive reads past its index buffer: dropped");
        d->dropped_draws++;
        return D3D_OK;
    }
    @autoreleasepool {
        cmd_draw *cd = record_draw(d, type, NULL);
        if (cd) {
            ib->last_use = d->serial;
            mtl_rec_keep(d, ib->gbuf);
            draw_indexed(d, cd, type, base, count, ib->mem + (size_t)start * isize, ib->gbuf, start * isize, isize,
                         instance_count(d));
        }
    }
    mtl_rec_end(d);
    return D3D_OK;
}

/* The UP draws read vertices (and indices) from the caller's memory, which
   is copied into the ring as stream 0. D3D9 leaves stream 0 and the
   indices unset afterwards. */
static void unset_after_up(mtl_device *d)
{
    mtl_state *st = &d->st;
    mtl_unhold(st->vb[0] ? &st->vb[0]->o : NULL);
    st->vb[0] = NULL;
    st->vb_off[0] = st->vb_stride[0] = 0;
    mtl_unhold(st->ib ? &st->ib->o : NULL);
    st->ib = NULL;
}

static HRESULT dev_dp_up(mtl_device *d, uint32_t type, uint32_t count, const void *data, uint32_t stride)
{
    if (!data || !stride) return D3DERR_INVALIDCALL;
    uint32_t n = prim_verts(type, count);
    @autoreleasepool {
        stream_override up;
        void *ub = NULL;
        void *p = n ? ring_alloc(d, n * stride, &ub, &up.off) : NULL;
        up.buf = ub;
        if (p) {
            memcpy(p, data, (size_t)n * stride);
            up.stride = stride;
            cmd_draw *cd = record_draw(d, type, &up);
            if (cd) {
                if (type == 6) {
                    void *ib;
                    uint32_t off;
                    if (fan_indices(d, NULL, 0, 0, count, &ib, &off))
                        call_indexed(cd, MTLPrimitiveTypeTriangle, count * 3, MTLIndexTypeUInt32, ib, off, 1, 0);
                } else {
                    call_arrays(cd, type, 0, n, 1);
                }
                d->draws++;
            }
        }
    }
    unset_after_up(d);
    mtl_rec_end(d);
    return D3D_OK;
}

static HRESULT dev_dip_up(mtl_device *d, uint32_t type, uint32_t min, uint32_t num, uint32_t count, const void *idx,
                          uint32_t fmt, const void *data, uint32_t stride)
{
    if (!idx || !data || !stride) return D3DERR_INVALIDCALL;
    uint32_t isize = fmt == D3DFMT_INDEX32 ? 4 : 2;
    @autoreleasepool {
        stream_override up;
        void *ub = NULL;
        void *p = ring_alloc(d, (min + num) * stride, &ub, &up.off);
        up.buf = ub;
        if (p) {
            memcpy(p, data, (size_t)(min + num) * stride);
            up.stride = stride;
            cmd_draw *cd = record_draw(d, type, &up);
            if (cd) draw_indexed(d, cd, type, 0, count, idx, NULL, 0, isize, instance_count(d));
        }
    }
    unset_after_up(d);
    mtl_rec_end(d);
    return D3D_OK;
}

/* State setters. Bound objects are held privately, as DXVK's state
   holds them, so a resource the engine releases while bound stays. */

static void swap_hold(mtl_obj **slot, mtl_obj *o)
{
    mtl_hold(o);
    mtl_unhold(*slot);
    *slot = o;
}

static int sampler_slot(uint32_t s)
{
    if (s < 16) return (int)s;
    if (s >= 257 && s <= 260) return (int)(17 + s - 257);
    return -1;
}

static HRESULT dev_set_texture(mtl_device *d, uint32_t stage, mtl_obj *t)
{
    int i = sampler_slot(stage);
    if (i < 0) return D3D_OK; /* the displacement map sampler */
    if (t && t->kind != MTL_K_TEXTURE && t->kind != MTL_K_CUBE) {
        note_once(d, @"tex-kind", "SetTexture with an object of kind %d: unbound", t->kind);
        t = NULL;
    }
    swap_hold((mtl_obj **)&d->st.tex[i], t);
    return D3D_OK;
}

static HRESULT dev_get_texture(mtl_device *d, uint32_t stage, mtl_obj **pp)
{
    if (!pp) return D3DERR_INVALIDCALL;
    int i = sampler_slot(stage);
    *pp = i >= 0 && d->st.tex[i] ? &d->st.tex[i]->o : NULL;
    if (*pp) mtl_addref(*pp);
    return D3D_OK;
}

static HRESULT dev_set_ss(mtl_device *d, uint32_t sampler, uint32_t type, uint32_t value)
{
    int i = sampler_slot(sampler);
    if (i >= 0 && type < 14 && d->st.samp[i][type] != value) {
        d->st.samp[i][type] = value;
        d->st.samp_heap[i] = -1;
    }
    return D3D_OK;
}

static HRESULT dev_get_ss(mtl_device *d, uint32_t sampler, uint32_t type, uint32_t *value)
{
    int i = sampler_slot(sampler);
    if (!value || i < 0 || type >= 14) return D3DERR_INVALIDCALL;
    *value = d->st.samp[i][type];
    return D3D_OK;
}

static HRESULT dev_set_tss(mtl_device *d, uint32_t stage, uint32_t type, uint32_t value)
{
    if (stage < 8 && type < 33) d->st.tss[stage][type] = value;
    return D3D_OK;
}

static HRESULT dev_get_tss(mtl_device *d, uint32_t stage, uint32_t type, uint32_t *value)
{
    if (!value || stage >= 8 || type >= 33) return D3DERR_INVALIDCALL;
    *value = d->st.tss[stage][type];
    return D3D_OK;
}

static HRESULT dev_stream(mtl_device *d, uint32_t stream, mtl_buffer *vb, uint32_t offset, uint32_t stride)
{
    if (stream >= 16) return D3DERR_INVALIDCALL;
    swap_hold((mtl_obj **)&d->st.vb[stream], vb ? &vb->o : NULL);
    d->st.vb_off[stream] = offset;
    d->st.vb_stride[stream] = stride;
    if (vb && stream < MTL_MAX_STREAMS && ((stride & 3) || (offset & 3)))
        note_once(d, @"stride", "stream %u stride %u offset %u not 4-aligned; Metal fetch may misread", stream, stride, offset);
    return D3D_OK;
}

static HRESULT dev_get_stream(mtl_device *d, uint32_t stream, mtl_buffer **pp, uint32_t *offset, uint32_t *stride)
{
    if (stream >= 16 || !pp || !offset || !stride) return D3DERR_INVALIDCALL;
    *pp = d->st.vb[stream];
    if (*pp) mtl_addref(&(*pp)->o);
    *offset = d->st.vb_off[stream];
    *stride = d->st.vb_stride[stream];
    return D3D_OK;
}

static HRESULT dev_stream_freq(mtl_device *d, uint32_t stream, uint32_t freq)
{
    if (stream >= 16) return D3DERR_INVALIDCALL;
    d->st.freq[stream] = freq;
    return D3D_OK;
}

static HRESULT dev_set_indices(mtl_device *d, mtl_buffer *ib)
{
    swap_hold((mtl_obj **)&d->st.ib, ib ? &ib->o : NULL);
    return D3D_OK;
}

static HRESULT dev_get_indices(mtl_device *d, mtl_buffer **pp)
{
    if (!pp) return D3DERR_INVALIDCALL;
    *pp = d->st.ib;
    if (*pp) mtl_addref(&(*pp)->o);
    return D3D_OK;
}

static HRESULT dev_set_decl(mtl_device *d, mtl_blob *b)
{
    swap_hold((mtl_obj **)&d->st.decl, b ? &b->o : NULL);
    d->st.fvf = 0;
    return D3D_OK;
}

static HRESULT dev_get_decl(mtl_device *d, mtl_blob **pp)
{
    if (!pp) return D3DERR_INVALIDCALL;
    *pp = d->st.decl;
    if (*pp) mtl_addref(&(*pp)->o);
    return D3D_OK;
}

/* D3D9VertexDecl::SetFVF: the declaration an FVF code stands for, all in
   stream 0, elements packed in FVF order. */
static uint32_t fvf_elements(uint32_t fvf, uint8_t out[17][8])
{
    static const uint8_t k_size[17] = {4, 8, 12, 16, 4, 4, 4, 8, 4, 4, 8, 4, 8, 4, 4, 4, 8};
    uint8_t type[16], usage[16], index[16];
    uint32_t n = 0;
#define ADD(t, u, i) (type[n] = (t), usage[n] = (u), index[n] = (i), n++)
    uint32_t pos = fvf & 0x400Eu;
    if (pos == 0x2 || (pos >= 0x6 && pos <= 0xE)) { /* XYZ, XYZB1 to XYZB5 */
        ADD(2, 0, 0);
        if (pos != 0x2) {
            uint32_t betas = ((pos - 0x6) >> 1) + 1, beta_type = 0xFF;
            if (fvf & 0x8000u) beta_type = 4;           /* LASTBETA_D3DCOLOR */
            else if (fvf & 0x1000u) beta_type = 5;      /* LASTBETA_UBYTE4 */
            else if (pos == 0xE) beta_type = 0;         /* XYZB5 */
            if (beta_type != 0xFF) betas--;
            if (betas >= 1 && betas <= 4) ADD(betas - 1, 1, 0);
            if (beta_type != 0xFF) ADD(beta_type, 2, 0);
        }
    } else if (pos == 0x4002 || pos == 0x4) { /* XYZW, XYZRHW */
        ADD(3, pos == 0x4002 ? 0 : 9, 0);
    }
    if (fvf & 0x10u) ADD(2, 3, 0);  /* NORMAL */
    if (fvf & 0x20u) ADD(0, 4, 0);  /* PSIZE */
    if (fvf & 0x40u) ADD(4, 10, 0); /* DIFFUSE */
    if (fvf & 0x80u) ADD(4, 10, 1); /* SPECULAR */
    uint32_t tex = (fvf >> 8) & 0xF;
    if (tex > 8) tex = 8;
    for (uint32_t i = 0; i < tex; i++) {
        /* TEXTUREFORMAT2 is 0, 3 is 1, 4 is 2, 1 is 3 */
        static const uint8_t k_tc[4] = {1, 2, 3, 0};
        ADD(k_tc[(fvf >> (16 + 2 * i)) & 3], 5, (uint8_t)i);
    }
#undef ADD
    uint16_t off = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint8_t *e = out[i];
        e[0] = e[1] = 0;
        memcpy(e + 2, &off, 2);
        e[4] = type[i];
        e[5] = 0;
        e[6] = usage[i];
        e[7] = index[i];
        off += k_size[type[i]];
    }
    static const uint8_t k_end[8] = {0xFF, 0, 0, 0, 17, 0, 0, 0};
    memcpy(out[n], k_end, 8);
    return n;
}

/* SetFVF binds the declaration the code stands for, made once per code
   and kept by the device (DXVK's m_fvfTable). Zero changes nothing. */
static HRESULT dev_set_fvf(mtl_device *d, uint32_t fvf)
{
    if (!fvf) return D3D_OK;
    __unsafe_unretained MtlDrawCache *c = CACHE(d);
    NSValue *v = c->fvf_decls[@(fvf)];
    mtl_blob *b = v.pointerValue;
    if (!b) {
        uint8_t el[17][8];
        fvf_elements(fvf, el);
        if (mtl_blob_new(d, MTL_K_DECL, el, &b)) return D3DERR_INVALIDCALL;
        /* Owned by the device, not the engine: a private hold, and the
           public reference creation handed out given back, so it neither
           shows in nor holds up the device's own count. */
        mtl_hold(&b->o);
        mtl_release(&b->o);
        c->fvf_decls[@(fvf)] = [NSValue valueWithPointer:b];
    }
    swap_hold((mtl_obj **)&d->st.decl, &b->o);
    d->st.fvf = fvf;
    return D3D_OK;
}

static HRESULT dev_get_fvf(mtl_device *d, uint32_t *fvf)
{
    if (!fvf) return D3DERR_INVALIDCALL;
    *fvf = d->st.fvf;
    return D3D_OK;
}

static HRESULT dev_set_vs(mtl_device *d, mtl_blob *b)
{
    swap_hold((mtl_obj **)&d->st.vs, b ? &b->o : NULL);
    return D3D_OK;
}

static HRESULT dev_get_vs(mtl_device *d, mtl_blob **pp)
{
    if (!pp) return D3DERR_INVALIDCALL;
    *pp = d->st.vs;
    if (*pp) mtl_addref(&(*pp)->o);
    return D3D_OK;
}

static HRESULT dev_set_ps(mtl_device *d, mtl_blob *b)
{
    swap_hold((mtl_obj **)&d->st.ps, b ? &b->o : NULL);
    return D3D_OK;
}

static HRESULT dev_get_ps(mtl_device *d, mtl_blob **pp)
{
    if (!pp) return D3DERR_INVALIDCALL;
    *pp = d->st.ps;
    if (*pp) mtl_addref(&(*pp)->o);
    return D3D_OK;
}

static HRESULT set_consts(void *dst, uint32_t cap, uint32_t elem, uint32_t start, const void *v, uint32_t count)
{
    if (!v || start >= cap || count > cap - start) return D3DERR_INVALIDCALL;
    memcpy((uint8_t *)dst + (size_t)start * elem, v, (size_t)count * elem);
    return D3D_OK;
}

static HRESULT get_consts(const void *src, uint32_t cap, uint32_t elem, uint32_t start, void *v, uint32_t count)
{
    if (!v || start >= cap || count > cap - start) return D3DERR_INVALIDCALL;
    memcpy(v, (const uint8_t *)src + (size_t)start * elem, (size_t)count * elem);
    return D3D_OK;
}

static HRESULT dev_vs_f(mtl_device *d, uint32_t start, const float *v, uint32_t count)
{
    HRESULT hr = set_consts(d->st.vsf, 256, 16, start, v, count);
    if (!hr && start + count > d->st.vs_changed_floats) d->st.vs_changed_floats = start + count;
    return hr;
}
static HRESULT dev_vs_f_get(mtl_device *d, uint32_t s, float *v, uint32_t n) { return get_consts(d->st.vsf, 256, 16, s, v, n); }
static HRESULT dev_vs_i(mtl_device *d, uint32_t s, const int32_t *v, uint32_t n) { return set_consts(d->st.vsi, 16, 16, s, v, n); }
static HRESULT dev_vs_i_get(mtl_device *d, uint32_t s, int32_t *v, uint32_t n) { return get_consts(d->st.vsi, 16, 16, s, v, n); }
static HRESULT dev_vs_b(mtl_device *d, uint32_t s, const uint32_t *v, uint32_t n) { return set_consts(d->st.vsb, 16, 4, s, v, n); }
static HRESULT dev_vs_b_get(mtl_device *d, uint32_t s, uint32_t *v, uint32_t n) { return get_consts(d->st.vsb, 16, 4, s, v, n); }
static HRESULT dev_ps_f(mtl_device *d, uint32_t s, const float *v, uint32_t n) { return set_consts(d->st.psf, 224, 16, s, v, n); }
static HRESULT dev_ps_f_get(mtl_device *d, uint32_t s, float *v, uint32_t n) { return get_consts(d->st.psf, 224, 16, s, v, n); }
static HRESULT dev_ps_i(mtl_device *d, uint32_t s, const int32_t *v, uint32_t n) { return set_consts(d->st.psi, 16, 16, s, v, n); }
static HRESULT dev_ps_i_get(mtl_device *d, uint32_t s, int32_t *v, uint32_t n) { return get_consts(d->st.psi, 16, 16, s, v, n); }
static HRESULT dev_ps_b(mtl_device *d, uint32_t s, const uint32_t *v, uint32_t n) { return set_consts(d->st.psb, 16, 4, s, v, n); }
static HRESULT dev_ps_b_get(mtl_device *d, uint32_t s, uint32_t *v, uint32_t n) { return get_consts(d->st.psb, 16, 4, s, v, n); }

/* Fixed-function state: transforms, material, lights. */

static int xf_slot(uint32_t t)
{
    if (t == 2) return MTL_XF_VIEW;
    if (t == 3) return MTL_XF_PROJ;
    if (t >= 16 && t <= 23) return (int)(MTL_XF_TEX0 + t - 16);
    if (t >= 256 && t < 256 + MTL_NXF - MTL_XF_WORLD0) return (int)(MTL_XF_WORLD0 + t - 256);
    return -1;
}

static HRESULT dev_set_transform(mtl_device *d, uint32_t t, const float *m)
{
    if (!m) return D3DERR_INVALIDCALL;
    int i = xf_slot(t);
    if (i < 0) {
        /* World matrices 8 to 255 index software vertex blending only. */
        note_once(d, @"xf-range", "SetTransform %u: not a transform the hardware path reads; kept nowhere", t);
        return D3D_OK;
    }
    memcpy(d->st.xf[i], m, 64);
    return D3D_OK;
}

static HRESULT dev_get_transform(mtl_device *d, uint32_t t, float *m)
{
    if (!m) return D3DERR_INVALIDCALL;
    int i = xf_slot(t);
    if (i < 0) {
        memset(m, 0, 64);
        for (int k = 0; k < 4; k++) m[5 * k] = 1.0f;
        return D3D_OK;
    }
    memcpy(m, d->st.xf[i], 64);
    return D3D_OK;
}

/* The given matrix times the current one, in D3D's row-vector terms. */
static HRESULT dev_mul_transform(mtl_device *d, uint32_t t, const float *m)
{
    if (!m) return D3DERR_INVALIDCALL;
    int i = xf_slot(t);
    if (i < 0) return D3D_OK;
    float r[16], *cur = d->st.xf[i];
    for (int row = 0; row < 4; row++)
        for (int col = 0; col < 4; col++) {
            float v = 0;
            for (int k = 0; k < 4; k++) v += m[4 * row + k] * cur[4 * k + col];
            r[4 * row + col] = v;
        }
    memcpy(cur, r, 64);
    return D3D_OK;
}

static HRESULT dev_set_material(mtl_device *d, const void *m)
{
    if (!m) return D3DERR_INVALIDCALL;
    memcpy(d->st.material, m, sizeof d->st.material);
    return D3D_OK;
}

static HRESULT dev_get_material(mtl_device *d, void *m)
{
    if (!m) return D3DERR_INVALIDCALL;
    memcpy(m, d->st.material, sizeof d->st.material);
    return D3D_OK;
}

/* DXVK's DefaultLight: a white directional light along +z, which a light
   enabled before it was ever set takes. */
static void default_light(mtl_light *l)
{
    memset(l->light, 0, sizeof l->light);
    uint32_t type = 3;
    float diffuse[4] = {1, 1, 1, 0}, dir[3] = {0, 0, 1};
    memcpy(l->light, &type, 4);
    memcpy(l->light + 4, diffuse, 16);
    memcpy(l->light + 68, dir, 12);
    l->enabled = 0;
}

static mtl_light *light_slot(mtl_device *d, uint32_t i)
{
    mtl_state *st = &d->st;
    if (i >= 4096) return NULL; /* D3D9 allows any index; no engine needs this many */
    if (i >= st->nlights) {
        mtl_light *l = realloc(st->lights, (size_t)(i + 1) * sizeof *l);
        if (!l) return NULL;
        for (uint32_t k = st->nlights; k <= i; k++) default_light(&l[k]);
        st->lights = l;
        st->nlights = i + 1;
    }
    return &st->lights[i];
}

static HRESULT dev_set_light(mtl_device *d, uint32_t i, const void *light)
{
    if (!light) return D3DERR_INVALIDCALL;
    mtl_light *l = light_slot(d, i);
    if (!l) return D3DERR_INVALIDCALL;
    memcpy(l->light, light, sizeof l->light);
    return D3D_OK;
}

static HRESULT dev_get_light(mtl_device *d, uint32_t i, void *light)
{
    if (!light || i >= d->st.nlights) return D3DERR_INVALIDCALL;
    memcpy(light, d->st.lights[i].light, sizeof d->st.lights[i].light);
    return D3D_OK;
}

static HRESULT dev_light_enable(mtl_device *d, uint32_t i, int32_t enable)
{
    mtl_light *l = light_slot(d, i);
    if (!l) return D3DERR_INVALIDCALL;
    l->enabled = enable != 0;
    return D3D_OK;
}

static HRESULT dev_get_light_enable(mtl_device *d, uint32_t i, int32_t *enable)
{
    if (!enable || i >= d->st.nlights) return D3DERR_INVALIDCALL;
    /* D3D9 reports an enabled light as 128, not 1. */
    *enable = d->st.lights[i].enabled ? 128 : 0;
    return D3D_OK;
}

static HRESULT dev_set_clip(mtl_device *d, uint32_t i, const float *p)
{
    if (i >= 6 || !p) return D3DERR_INVALIDCALL;
    memcpy(d->st.clip[i], p, 16);
    return D3D_OK;
}

static HRESULT dev_get_clip(mtl_device *d, uint32_t i, float *p)
{
    if (i >= 6 || !p) return D3DERR_INVALIDCALL;
    memcpy(p, d->st.clip[i], 16);
    return D3D_OK;
}

static HRESULT dev_set_scissor(mtl_device *d, const mtl_rect *r)
{
    if (!r) return D3DERR_INVALIDCALL;
    d->st.scissor = *r;
    return D3D_OK;
}

static HRESULT dev_get_scissor(mtl_device *d, mtl_rect *r)
{
    if (!r) return D3DERR_INVALIDCALL;
    *r = d->st.scissor;
    return D3D_OK;
}

/* StretchRect: a textured draw from the source rectangle into the
   destination one, through the utility library. A source with CPU memory
   only (an offscreen plain surface, as the bridge's movie path uses) is
   uploaded into a temporary texture first. */
static HRESULT dev_stretch(mtl_device *d, mtl_surface *src, const mtl_rect *sr, mtl_surface *dst, const mtl_rect *dr,
                           uint32_t filter)
{
    if (!src || !dst) return D3DERR_INVALIDCALL;
    @autoreleasepool {
        uint32_t sl, ss, dl, ds;
        id<MTLTexture> st = mtl_surface_texture(src, &sl, &ss);
        id<MTLTexture> dt = mtl_surface_texture(dst, &dl, &ds);
        if (!dt) {
            note_once(d, @"stretch-dst", "StretchRect into a surface with no GPU texture (format %u): not done", dst->format);
            return D3D_OK;
        }
        if (!st) {
            /* The source's CPU memory as it is now, in a texture of its
               own that the list keeps. */
            uint8_t *m = mtl_surface_mem(src);
            if (!m || mtl_format_is_depth(src->format)) return D3D_OK;
            /* Decoded or widened like any texture upload where the GPU
               cannot sample the format. */
            st = mtl_texture_from_mem(d, src->format, src->width, src->height, m, src->pitch, src->rows);
            if (!st) return D3D_OK;
            sl = ss = 0;
        }
        mtl_rect s = sr ? *sr : (mtl_rect){0, 0, (int32_t)src->width, (int32_t)src->height};
        mtl_rect t = dr ? *dr : (mtl_rect){0, 0, (int32_t)dst->width, (int32_t)dst->height};
        if (s.right <= s.left || s.bottom <= s.top || t.right <= t.left || t.bottom <= t.top) return D3DERR_INVALIDCALL;
        mtl_end_pass(d);
        cmd_stretch *c = mtl_rec_cmd(d, CMD_STRETCH, sizeof *c);
        c->src = (__bridge void *)st;
        c->dst = (__bridge void *)dt;
        c->src_level = sl;
        c->src_slice = ss;
        c->dst_level = dl;
        c->dst_slice = ds;
        c->filter = filter;
        c->s = s;
        c->t = t;
        mtl_rec_keep(d, c->src);
        mtl_rec_keep(d, c->dst);
        if (src == d->back && dst != d->back) {
            uint32_t n = d->eye_copy_n, k = 0;
            while (k < n && d->eye_copy[k] != c->dst) k++;
            if (k == n && n < MTL_EYE_COPIES) {
                d->eye_copy[n] = (void *)CFBridgingRetain(dt);
                __atomic_store_n(&d->eye_copy_n, n + 1, __ATOMIC_RELEASE);
                mtl_log("stereo: the back buffer copied into a %lux%lu texture: each eye keeps its own",
                        (unsigned long)dt.width, (unsigned long)dt.height);
            }
        }
        note_once(d, @"stretch", "StretchRect %ux%u format %u -> %ux%u format %u", src->width, src->height, src->format,
                  dst->width, dst->height, dst->format);
    }
    mtl_rec_end(d);
    return D3D_OK;
}

/* Surfaces whose contents live on the GPU only: render targets, the back
   buffer, depth surfaces and the levels of render target textures. Every
   other surface keeps its CPU memory as the master copy, uploaded when a
   draw binds it. */
static id<MTLTexture> gpu_only_texture(mtl_surface *s, uint32_t *level, uint32_t *slice)
{
    *level = *slice = 0;
    if (s->tex) return (__bridge id<MTLTexture>)s->tex;
    if (s->o.container && (((mtl_texture *)s->o.container)->usage & D3DUSAGE_RENDERTARGET))
        return mtl_surface_texture(s, level, slice);
    return nil;
}

static void mark_written(mtl_surface *s)
{
    s->dirty = 1;
    if (s->o.container) ((mtl_texture *)s->o.container)->dirty = 1;
}

/* Byte geometry of a rectangle in a surface's format: whole 4x4 blocks
   for DXT, pixels otherwise. */
typedef struct {
    size_t x_off, row_bytes;
    uint32_t y0, rows;
} rect_bytes;

static rect_bytes rect_geometry(uint32_t fmt, uint32_t x, uint32_t y, uint32_t w, uint32_t h)
{
    rect_bytes g;
    if (mtl_fmt_dxt(fmt)) {
        uint32_t blk = fmt == D3DFMT_DXT1 ? 8 : 16;
        g.x_off = (size_t)(x / 4) * blk;
        g.row_bytes = (size_t)((x + w + 3) / 4 - x / 4) * blk;
        g.y0 = y / 4;
        g.rows = (y + h + 3) / 4 - y / 4;
    } else {
        uint32_t bpp = mtl_fmt_bits(fmt) / 8;
        g.x_off = (size_t)x * bpp;
        g.row_bytes = (size_t)w * bpp;
        g.y0 = y;
        g.rows = h;
    }
    return g;
}

/* Writes w x h texels from src (rows src_pitch apart, starting at the
   rectangle's first texel) into dst at (x, y): a copy into its memory, or
   a staged blit into its texture, ordered after every draw encoded so
   far. */
static HRESULT write_surface(mtl_device *d, mtl_surface *dst, uint32_t x, uint32_t y, uint32_t w, uint32_t h,
                             const uint8_t *src, size_t src_pitch)
{
    rect_bytes g = rect_geometry(dst->format, x, y, w, h);
    uint32_t level, slice;
    id<MTLTexture> t = gpu_only_texture(dst, &level, &slice);
    if (!t) {
        if (dst->no_cpu) return D3DERR_INVALIDCALL;
        mtl_surface_sync(dst);
        uint8_t *m = mtl_surface_mem(dst);
        if (!m) return E_OUTOFMEMORY;
        for (uint32_t r = 0; r < g.rows; r++)
            memcpy(m + (size_t)(g.y0 + r) * dst->pitch + g.x_off, src + (size_t)r * src_pitch, g.row_bytes);
        mark_written(dst);
        return D3D_OK;
    }
    if (mtl_format_is_depth(dst->format) || t.pixelFormat == MTLPixelFormatDepth32Float_Stencil8) return D3DERR_INVALIDCALL;
    @autoreleasepool {
        /* The texels are staged now, so the list uploads what the caller
           passed at this call. */
        id<MTLBuffer> b = [MTLD(d) newBufferWithLength:g.row_bytes * g.rows options:MTLResourceStorageModeShared];
        if (!b) return E_OUTOFMEMORY;
        for (uint32_t r = 0; r < g.rows; r++)
            memcpy((uint8_t *)b.contents + (size_t)r * g.row_bytes, src + (size_t)r * src_pitch, g.row_bytes);
        mtl_end_pass(d);
        cmd_upload *c = mtl_rec_cmd(d, CMD_UPLOAD, sizeof *c);
        c->src = (__bridge void *)b;
        c->dst = (__bridge void *)t;
        c->row_bytes = (uint32_t)g.row_bytes;
        c->rows = g.rows;
        c->width = w;
        c->height = h;
        c->level = level;
        c->slice = slice;
        c->x = x;
        c->y = y;
        mtl_rec_keep(d, c->src);
        mtl_rec_keep(d, c->dst);
    }
    mtl_rec_end(d);
    return D3D_OK;
}

/* UpdateSurface: a system memory surface's rectangle into a default pool
   surface at a point. The bridge's movie path uploads each Bink frame
   this way into an offscreen plain surface, then StretchRects it. */
static HRESULT dev_update_surface(mtl_device *d, mtl_surface *src, const mtl_rect *sr, mtl_surface *dst,
                                  const int32_t *point)
{
    if (!src || !dst || src->o.kind != MTL_K_SURFACE || dst->o.kind != MTL_K_SURFACE) return D3DERR_INVALIDCALL;
    if (src->pool != D3DPOOL_SYSTEMMEM || dst->pool != D3DPOOL_DEFAULT || src->format != dst->format)
        return D3DERR_INVALIDCALL;
    mtl_rect r = sr ? *sr : (mtl_rect){0, 0, (int32_t)src->width, (int32_t)src->height};
    int32_t x = point ? point[0] : 0, y = point ? point[1] : 0; /* no point means the origin, as in DXVK */
    if (r.left < 0 || r.top < 0 || r.right <= r.left || r.bottom <= r.top || (uint32_t)r.right > src->width ||
        (uint32_t)r.bottom > src->height || x < 0 || y < 0 || (uint32_t)(x + r.right - r.left) > dst->width ||
        (uint32_t)(y + r.bottom - r.top) > dst->height)
        return D3DERR_INVALIDCALL;
    if (mtl_fmt_dxt(src->format) && ((r.left | r.top | x | y) & 3)) return D3DERR_INVALIDCALL;
    mtl_surface_sync(src);
    const uint8_t *m = mtl_surface_mem(src);
    if (!m) return E_OUTOFMEMORY;
    rect_bytes g = rect_geometry(src->format, (uint32_t)r.left, (uint32_t)r.top, 1, 1);
    return write_surface(d, dst, (uint32_t)x, (uint32_t)y, (uint32_t)(r.right - r.left), (uint32_t)(r.bottom - r.top),
                         m + (size_t)g.y0 * src->pitch + g.x_off, src->pitch);
}

/* UpdateTexture: a system memory texture into a default pool one, level
   by level, after DXVK's rules: the same format, and when the chains
   differ in length the source's smallest levels match the destination's.
   DXVK copies the dirty box of level 0 (scaled down the chain); this
   tracks dirtiness per texture, so a dirty source is copied whole and a
   clean one not at all. */
static HRESULT dev_update_texture(mtl_device *d, mtl_texture *src, mtl_texture *dst)
{
    if (!src || !dst || (src->o.kind != MTL_K_TEXTURE && src->o.kind != MTL_K_CUBE) || src->o.kind != dst->o.kind)
        return D3DERR_INVALIDCALL;
    if (src->pool != D3DPOOL_SYSTEMMEM || dst->pool != D3DPOOL_DEFAULT || src->format != dst->format ||
        src->levels < dst->levels)
        return D3DERR_INVALIDCALL;
    uint32_t off = 0;
    if ((src->levels > 1 || dst->levels > 1) && (src->width != dst->width || src->height != dst->height)) {
        off = src->levels - dst->levels;
        uint32_t w = src->width >> off, h = src->height >> off;
        if ((w ? w : 1) != dst->width || (h ? h : 1) != dst->height) return D3DERR_INVALIDCALL;
    }
    if (!src->update_dirty) return D3D_OK;
    for (uint32_t f = 0; f < dst->faces; f++)
        for (uint32_t l = 0; l < dst->levels; l++) {
            mtl_surface *s = &src->sub[f * src->levels + l + off], *t = &dst->sub[f * dst->levels + l];
            const uint8_t *m = mtl_surface_mem(s);
            if (!m) return E_OUTOFMEMORY;
            uint32_t w = s->width < t->width ? s->width : t->width, h = s->height < t->height ? s->height : t->height;
            HRESULT hr = write_surface(d, t, 0, 0, w, h, m, s->pitch);
            if (hr) return hr;
        }
    src->update_dirty = 0;
    return D3D_OK;
}

/* ColorFill: a default pool surface's rectangle (all of it without one)
   set to a colour. Unlike Clear it ignores the viewport, scissor and
   bound targets. A GPU surface gets a pass of its own, a load action
   clear when the whole surface is filled; a CPU one is written in place. */
static HRESULT dev_color_fill(mtl_device *d, mtl_surface *s, const mtl_rect *rect, uint32_t color)
{
    if (!s || s->o.kind != MTL_K_SURFACE || s->pool != D3DPOOL_DEFAULT || mtl_format_is_depth(s->format))
        return D3DERR_INVALIDCALL;
    mtl_rect full = {0, 0, (int32_t)s->width, (int32_t)s->height};
    mtl_rect r = rect ? *rect : full;
    if (!clip_rect(&r, &full)) return D3D_OK;
    uint32_t level, slice;
    id<MTLTexture> t = gpu_only_texture(s, &level, &slice);
    if (!t) {
        if (mtl_fmt_dxt(s->format) || (mtl_fmt_bits(s->format) != 32 && s->format != D3DFMT_R5G6B5)) {
            note_once(d, [NSString stringWithFormat:@"fill-%u", s->format], "ColorFill of a format %u surface: not done",
                      s->format);
            return D3D_OK;
        }
        cpu_clear(s, &r, 1, color);
        return D3D_OK;
    }
    mtl_end_pass(d);
    uint32_t lw = (uint32_t)t.width >> level, lh = (uint32_t)t.height >> level;
    lw = lw ? lw : 1;
    lh = lh ? lh : 1;
    cmd_fill *c = mtl_rec_cmd(d, CMD_FILL, sizeof *c);
    c->tex = (__bridge void *)t;
    c->level = level;
    c->slice = slice;
    c->whole = r.left == 0 && r.top == 0 && (uint32_t)r.right >= lw && (uint32_t)r.bottom >= lh;
    c->color = color;
    c->lw = lw;
    c->lh = lh;
    c->r = r;
    mtl_rec_keep(d, c->tex);
    mtl_rec_end(d);
    return D3D_OK;
}

/* Commands device.m records. */

void mtl_rec_readback(mtl_device *d, void *src, uint32_t level, uint32_t slice, uint32_t w, uint32_t h, void *dst,
                      uint32_t pitch, uint32_t rows)
{
    mtl_end_pass(d);
    cmd_readback *c = mtl_rec_cmd(d, CMD_READBACK, sizeof *c);
    c->src = src;
    c->dst = dst;
    c->level = level;
    c->slice = slice;
    c->width = w;
    c->height = h;
    c->pitch = pitch;
    c->rows = rows;
    mtl_rec_keep(d, src);
    mtl_rec_keep(d, dst);
    mtl_rec_end(d);
}

void mtl_rec_present(mtl_device *d, void *back)
{
    mtl_end_pass(d);
    cmd_present *c = mtl_rec_cmd(d, CMD_PRESENT, sizeof *c);
    c->back = back;
    c->presented_ns = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    mtl_immersive_pose_now(c->pose);
    mtl_rec_keep(d, back);
}

/* Replay. Everything below runs on the thread that replays (the render
   thread, or the game thread in inline mode) and reads only the command
   and the replay-side caches, never device state. */

static void *swap_tex(const mtl_emit *e, void *t)
{
    for (int i = 0; i < 2 + MTL_EYE_COPIES; i++)
        if (t && t == e->swap_from[i] && e->swap_to[i]) return e->swap_to[i];
    return t;
}

void mtl_emit_close(mtl_emit *e)
{
    if (!e->enc) return;
    id<MTLRenderCommandEncoder> enc = CFBridgingRelease(e->enc);
    e->enc = NULL;
    [enc endEncoding];
}

static void emit_pass(mtl_emit *e, const cmd_pass *p)
{
    mtl_emit_close(e);
    e->pass = p;
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = OBJ(swap_tex(e, p->ct));
    rp.colorAttachments[0].level = p->level;
    rp.colorAttachments[0].slice = p->slice;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    if (p->clear & D3DCLEAR_TARGET) {
        uint32_t c = p->color;
        rp.colorAttachments[0].loadAction = MTLLoadActionClear;
        rp.colorAttachments[0].clearColor = MTLClearColorMake(((c >> 16) & 255) / 255.0, ((c >> 8) & 255) / 255.0,
                                                              (c & 255) / 255.0, (c >> 24) / 255.0);
    } else {
        rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
    }
    if (p->dt) {
        id<MTLTexture> dt = OBJ(swap_tex(e, p->dt));
        rp.depthAttachment.texture = dt;
        rp.depthAttachment.storeAction = MTLStoreActionStore;
        rp.depthAttachment.loadAction = (p->clear & D3DCLEAR_ZBUFFER) ? MTLLoadActionClear : MTLLoadActionLoad;
        rp.depthAttachment.clearDepth = p->z;
        rp.stencilAttachment.texture = dt;
        rp.stencilAttachment.storeAction = MTLStoreActionStore;
        rp.stencilAttachment.loadAction = (p->clear & D3DCLEAR_STENCIL) ? MTLLoadActionClear : MTLLoadActionLoad;
        rp.stencilAttachment.clearStencil = p->stencil & 0xFF;
    }
    rp.renderTargetWidth = p->width;
    rp.renderTargetHeight = p->height;
    id<MTLRenderCommandEncoder> enc = [CB(e) renderCommandEncoderWithDescriptor:rp];
    if (!enc) return;
    [enc setFrontFacingWinding:MTLWindingClockwise];
    e->enc = (void *)CFBridgingRetain(enc);
}

static void emit_clear(mtl_device *d, mtl_emit *e, const cmd_clear *c)
{
    id<MTLRenderCommandEncoder> enc = ENC(e);
    id<MTLRenderPipelineState> pso = enc ? clear_pipeline(d, (MTLPixelFormat)c->pf, (int)c->has_ds, c->flags) : nil;
    if (!pso) return;
    uint32_t color = c->color;
    float col[4] = {((color >> 16) & 255) / 255.0f, ((color >> 8) & 255) / 255.0f, (color & 255) / 255.0f,
                    (color >> 24) / 255.0f};
    float zz = c->z;
    [enc setRenderPipelineState:pso];
    if (c->has_ds) {
        [enc setDepthStencilState:clear_dss(d, c->flags)];
        [enc setStencilReferenceValue:c->stencil & 0xFF];
    }
    [enc setCullMode:MTLCullModeNone];
    [enc setTriangleFillMode:MTLTriangleFillModeFill];
    [enc setViewport:(MTLViewport){0, 0, c->width, c->height, 0, 1}];
    [enc setVertexBytes:&zz length:sizeof zz atIndex:0];
    [enc setFragmentBytes:col length:sizeof col atIndex:0];
    for (uint32_t i = 0; i < c->n; i++) {
        const mtl_rect *r = &c->r[i];
        [enc setScissorRect:(MTLScissorRect){(NSUInteger)r->left, (NSUInteger)r->top, (NSUInteger)(r->right - r->left),
                                             (NSUInteger)(r->bottom - r->top)}];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }
}

/* Where a HUD draw's recorded vertices are read from: the storage the
   list keeps, so a replay reads what the game wrote. */
typedef struct {
    const uint8_t *base, *ix;
    size_t len, isz;
} cov_src;

static int cov_open(const cmd_draw *c, cov_src *s)
{
    if (!c->cov_ok || c->instances != 1) return 0;
    if (!(c->streams & (1u << c->cov_stream))) return 0;
    id<MTLBuffer> vb = OBJ(c->stream[c->cov_stream].buf);
    if (!vb || vb.storageMode == MTLStorageModePrivate) return 0;
    s->base = (const uint8_t *)vb.contents + c->stream[c->cov_stream].off;
    s->len = vb.length - c->stream[c->cov_stream].off;
    s->ix = NULL;
    s->isz = 0;
    if (c->call == CALL_INDEXED) {
        id<MTLBuffer> ib = OBJ(c->ib.buf);
        if (!ib || ib.storageMode == MTLStorageModePrivate) return 0;
        s->isz = c->index_type == MTLIndexTypeUInt32 ? 4 : 2;
        if (c->ib.off + (size_t)c->count * s->isz > ib.length) return 0;
        s->ix = (const uint8_t *)ib.contents + c->ib.off;
    } else if (c->call != CALL_ARRAYS) return 0;
    return 1;
}

/* Clip x, y and w of vertices from..from+n of the draw. */
static int cov_xyw(const cmd_draw *c, const cov_src *s, uint32_t from, uint32_t n, float (*xyw)[3])
{
    uint32_t ty = c->cov_type & 0x7F, post = c->cov_type & 0x80;
    uint32_t ncomp = ty == 1 ? 2 : ty == 2 ? 3 : 4;
    for (uint32_t j = 0; j < n; j++) {
        uint32_t i = from + j;
        int64_t v;
        if (s->ix) {
            uint32_t k = s->isz == 4 ? ((const uint32_t *)s->ix)[i] : ((const uint16_t *)s->ix)[i];
            v = (int64_t)k + c->base;
        } else v = (int64_t)c->start + i;
        size_t at = (size_t)v * c->cov_stride + c->cov_off;
        if (v < 0 || at + ncomp * 4 > s->len) return 0;
        float q[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        memcpy(q, s->base + at, ncomp * 4);
        if (post) q[3] = 1.0f; /* POSITIONT: the fourth is rhw, not w */
        for (int o = 0; o < 3; o++)
            xyw[j][o] = c->cov_rows[o][0] * q[0] + c->cov_rows[o][1] * q[1] + c->cov_rows[o][2] * q[2] +
                        c->cov_rows[o][3] * q[3];
    }
    return 1;
}

/* Whether a HUD draw covers the whole view, read once from its recorded
   vertices and kept on the command. A few vertices only: overlays are
   quads, the HUD's own parts small batches. */
static int fv_cover(const cmd_draw *cc)
{
    cmd_draw *c = (cmd_draw *)cc;
    if (c->fv_full >= 0) return c->fv_full;
    c->fv_full = 0;
    cov_src s;
    if (c->count < 3 || c->count > 64 || !cov_open(c, &s)) return 0;
    float xyw[64][3];
    if (!cov_xyw(c, &s, 0, c->count, xyw)) return 0;
    c->fv_full = (int8_t)mtl_stereo_covers_view((const float(*)[3])xyw, c->count);
    return c->fv_full;
}

/* A HUD draw's box in the game's NDC (x0 x1 y0 y1), for the shot dump
   only, to tell the gauges' draws from the center text's. Read in
   chunks, since a line of text is a batch past fv_cover's few vertices. */
static int cov_box(const cmd_draw *c, float bb[4])
{
    cov_src s;
    if (!c->count || !cov_open(c, &s)) return 0;
    bb[0] = bb[2] = INFINITY, bb[1] = bb[3] = -INFINITY;
    float xyw[64][3];
    for (uint32_t from = 0; from < c->count; from += 64) {
        uint32_t n = c->count - from < 64 ? c->count - from : 64;
        if (!cov_xyw(c, &s, from, n, xyw)) return 0;
        for (uint32_t i = 0; i < n; i++) {
            float w = xyw[i][2];
            if (!(w > 1e-6f)) return 0; /* behind the eye: no box in NDC */
            float x = xyw[i][0] / w, y = xyw[i][1] / w;
            bb[0] = fminf(bb[0], x), bb[1] = fmaxf(bb[1], x);
            bb[2] = fminf(bb[2], y), bb[3] = fmaxf(bb[3], y);
        }
    }
    return 1;
}

/* RT_FULLVIEW_AURA=off drops a camera-held effect in full view, to tell
   its cost from the rest of a discipline's frame. */
static int aura_off(void)
{
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("RT_FULLVIEW_AURA");
        v = e && !strcmp(e, "off");
        if (v) mtl_log("full view: camera-held effects dropped (RT_FULLVIEW_AURA=off)");
    }
    return v;
}

static void emit_draw(mtl_device *d, mtl_emit *e, const cmd_draw *c);

/* The small target a camera-held effect is drawn into: a quarter of the
   pass's width and height, so a sixteenth of its pixels, kept across
   frames. RT_FULLVIEW_AURA_SCALE (2 to 8) is the divisor. */
/* How far past the game's screen the small target reaches, in screens:
   the smoke runs on past the screen's edge, and a target that stopped
   there showed its edge as a rectangle held in front of the head. At 2
   the edge is outside the eye. RT_FULLVIEW_AURA_REACH, 1 to 3. */
static float aura_reach(void)
{
    static float v = 0.0f;
    if (!(v > 0.0f)) {
        const char *s = getenv("RT_FULLVIEW_AURA_REACH");
        v = s && *s ? (float)atof(s) : 2.0f;
        if (!(v >= 1.0f && v <= 3.0f)) v = 2.0f;
    }
    return v;
}

static int aura_targets(mtl_device *d, const cmd_pass *p, MTLPixelFormat pf)
{
    __unsafe_unretained MtlDrawCache *dc = CACHE(d);
    static int div = 0;
    if (!div) {
        const char *s = getenv("RT_FULLVIEW_AURA_SCALE");
        div = s && *s ? atoi(s) : 4;
        if (div < 2 || div > 8) div = 4;
    }
    NSUInteger w = MAX(1u, p->width / (uint32_t)div), h = MAX(1u, p->height / (uint32_t)div);
    if (dc->aura_ct && dc->aura_ct.width == w && dc->aura_ct.height == h && dc->aura_ct.pixelFormat == pf) return 1;
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pf width:w height:h mipmapped:NO];
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    dc->aura_ct = [MTLD(d) newTextureWithDescriptor:td];
    td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:(MTLPixelFormat)mtl_depth_pixel_format()
                                                            width:w
                                                           height:h
                                                        mipmapped:NO];
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageRenderTarget;
    dc->aura_dt = [MTLD(d) newTextureWithDescriptor:td];
    if (!dc->aura_samp) {
        MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
        sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterLinear;
        sd.sAddressMode = sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
        dc->aura_samp = [MTLD(d) newSamplerStateWithDescriptor:sd];
    }
    if (dc->aura_ct && dc->aura_dt) mtl_log("full view: camera-held effects drawn at %lux%lu", (unsigned long)w, (unsigned long)h);
    return dc->aura_ct && dc->aura_dt;
}

/* A camera-held effect (Presence's smoke) is many layers over most of
   the screen. Drawn into each eye at its size it took the GPU from 8 to
   70 ms a frame (la_hub_1, 2026-11-21), so it is drawn once into a small
   target with the game's own projection, as the flat game draws it, and
   that is laid on the eye's aura panel with one blend. The eye's pass is
   then reopened where it was. */
static void emit_aura(mtl_device *d, mtl_emit *e, const cmd_draw *c)
{
    __unsafe_unretained MtlDrawCache *dc = CACHE(d);
    const cmd_pass *p = e->pass;
    id<MTLTexture> eye_ct = OBJ(swap_tex(e, p->ct));
    if (!aura_targets(d, p, (MTLPixelFormat)c->pd.k.color_pf)) return;
    mtl_emit_close(e);

    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = dc->aura_ct;
    rp.colorAttachments[0].loadAction = MTLLoadActionClear;
    rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    rp.depthAttachment.texture = dc->aura_dt;
    rp.depthAttachment.loadAction = MTLLoadActionClear;
    rp.depthAttachment.clearDepth = 1.0;
    rp.depthAttachment.storeAction = MTLStoreActionDontCare;
    rp.stencilAttachment.texture = dc->aura_dt;
    rp.stencilAttachment.loadAction = MTLLoadActionClear;
    rp.stencilAttachment.storeAction = MTLStoreActionDontCare;
    id<MTLRenderCommandEncoder> aenc = [CB(e) renderCommandEncoderWithDescriptor:rp];
    if (aenc) {
        [aenc setFrontFacingWinding:MTLWindingClockwise];
        e->enc = (void *)CFBridgingRetain(aenc);
        e->in_aura = 1;
        emit_draw(d, e, c);
        e->in_aura = 0;
        mtl_emit_close(e);
    }

    /* Back into the eye's pass, loading what it has drawn so far. */
    cmd_pass again = *p;
    again.clear = 0;
    emit_pass(e, &again);
    e->pass = p;
    id<MTLRenderCommandEncoder> enc = ENC(e);
    if (!enc || !aenc) return;
    int has_ds = p->dt != NULL;
    NSNumber *key = @(0x400000000ull | ((uint64_t)eye_ct.pixelFormat << 1) | (uint64_t)has_ds);
    id<MTLRenderPipelineState> pso = dc->clear_psos[key];
    if (!pso) {
        id<MTLLibrary> lib = util_library(d);
        MTLRenderPipelineDescriptor *pdesc = [MTLRenderPipelineDescriptor new];
        pdesc.vertexFunction = [lib newFunctionWithName:@"blit_vs"];
        pdesc.fragmentFunction = [lib newFunctionWithName:@"blit_fs"];
        MTLRenderPipelineColorAttachmentDescriptor *ca = pdesc.colorAttachments[0];
        ca.pixelFormat = eye_ct.pixelFormat;
        ca.blendingEnabled = YES;
        ca.sourceRGBBlendFactor = ca.sourceAlphaBlendFactor = MTLBlendFactorOne;
        ca.destinationRGBBlendFactor = ca.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        if (has_ds) {
            pdesc.depthAttachmentPixelFormat = (MTLPixelFormat)mtl_depth_pixel_format();
            pdesc.stencilAttachmentPixelFormat = (MTLPixelFormat)mtl_depth_pixel_format();
        }
        NSError *err = nil;
        pso = [MTLD(d) newRenderPipelineStateWithDescriptor:pdesc error:&err];
        if (!pso) {
            mtl_log("aura composite pipeline failed: %s", err.localizedDescription.UTF8String);
            return;
        }
        dc->clear_psos[key] = pso;
    }
    /* The panel's corners in this eye, as mtl_stereo_panel_correction
       puts game NDC there, then in the target's pixels. */
    int eye = e->eye ? 1 : 0;
    mtl_stereo_fix fix;
    float reach = aura_reach();
    mtl_stereo_panel_correction(&e->fv.eye_p[eye], e->fv.eye_t[eye], e->hud_z, e->fv.aura[0] * reach,
                                e->fv.aura[1] * reach, &fix);
    double x0 = -fix.x[0] + fix.x[1], x1 = fix.x[0] + fix.x[1];
    double y0 = fix.y[0] + fix.y[1], y1 = -fix.y[0] + fix.y[1]; /* top, bottom */
    double W = p->width, H = p->height;
    double vx = (x0 + 1.0) * 0.5 * W, vw = (x1 - x0) * 0.5 * W;
    double vy = (1.0 - y0) * 0.5 * H, vh = (y0 - y1) * 0.5 * H;
    if (!(vw > 0.0 && vh > 0.0)) return;
    /* Kept inside the target, the texture trimmed to match: the panel
       can reach past an eye's narrow side. */
    double cx0 = fmax(vx, 0.0), cy0 = fmax(vy, 0.0), cx1 = fmin(vx + vw, W), cy1 = fmin(vy + vh, H);
    if (!(cx1 > cx0 && cy1 > cy0)) return;
    const float r[4] = {(float)((cx0 - vx) / vw), (float)((cy0 - vy) / vh), (float)((cx1 - cx0) / vw),
                        (float)((cy1 - cy0) / vh)};
    vx = cx0, vy = cy0, vw = cx1 - cx0, vh = cy1 - cy0;
    [enc setRenderPipelineState:pso];
    [enc setDepthStencilState:get_dss(d, 0)];
    [enc setCullMode:MTLCullModeNone];
    [enc setViewport:(MTLViewport){vx, vy, vw, vh, 0, 1}];
    [enc setScissorRect:(MTLScissorRect){0, 0, p->width, p->height}];
    [enc setVertexBytes:r length:sizeof r atIndex:0];
    [enc setFragmentTexture:dc->aura_ct atIndex:0];
    [enc setFragmentSamplerState:dc->aura_samp atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
}

/* The arm HUD: the game's left gauge on a card at the left controller and
   its right one at the right, drawn by the compositor (immersive.h); the
   rest of the HUD stays on the panel. RT_ARM_HUD=0 is read where the
   frame is latched (immersive.m), as is which sides the compositor can
   show: a side whose controller is not tracked keeps its gauge on the
   panel. A draw's side is read from its box once, in the first eye, and
   kept on the command, so the second eye leaves out exactly what the
   first drew on the cards. */
static int arm_side(mtl_device *d, const mtl_emit *e, const cmd_draw *cc)
{
    if (!e->stereo || !e->fv.on || e->fv.game_eye || !(e->fv.arm[0] || e->fv.arm[1])) return 0;
    if (!cc->stereo_hud || cc->stereo_a > 0.0f || CACHE(d)->arm_broken) return 0;
    cmd_draw *c = (cmd_draw *)cc;
    if (c->arm_side < 0) {
        float bb[4];
        c->arm_side = fv_cover(c) || !cov_box(c, bb) ? 0 : (int8_t)mtl_stereo_arm_side(bb);
    }
    return c->arm_side > 0 && e->fv.arm[c->arm_side - 1] ? c->arm_side : 0;
}

/* The cards' target: each side's columns of the back buffer at its own
   pixels, side by side, the full height. A few hundred columns, so the
   gauges are as sharp on the card as in the flat game. */
static int arm_targets(mtl_device *d, const cmd_pass *p, MTLPixelFormat pf)
{
    __unsafe_unretained MtlDrawCache *dc = CACHE(d);
    uint32_t col[2][2];
    for (int s = 0; s < 2; s++) mtl_stereo_arm_columns(s + 1, p->width, &col[s][0], &col[s][1]);
    NSUInteger w = (col[0][1] - col[0][0]) + (col[1][1] - col[1][0]), h = p->height;
    if (!w || !h) return 0;
    if (dc->arm_ct && dc->arm_ct.width == w && dc->arm_ct.height == h && dc->arm_ct.pixelFormat == pf) return 1;
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pf width:w height:h mipmapped:NO];
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    dc->arm_ct = [MTLD(d) newTextureWithDescriptor:td];
    dc->arm_ct.label = @"arm HUD cards";
    td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:(MTLPixelFormat)mtl_depth_pixel_format()
                                                            width:w
                                                           height:h
                                                        mipmapped:NO];
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageRenderTarget;
    dc->arm_dt = [MTLD(d) newTextureWithDescriptor:td];
    memcpy(dc->arm_col, col, sizeof col);
    dc->arm_x[0] = 0;
    dc->arm_x[1] = col[0][1] - col[0][0];
    dc->arm_used = 0;
    if (dc->arm_ct && dc->arm_dt)
        mtl_log("full view: arm HUD cards %lux%lu (left columns %u to %u, right %u to %u of %u)", (unsigned long)w,
                (unsigned long)h, col[0][0], col[0][1], col[1][0], col[1][1], p->width);
    return dc->arm_ct && dc->arm_dt;
}

/* Leaves the cards' pass; reopen puts the eye's pass back, loading what
   it has drawn so far, for a command that draws into it. */
static void arm_leave(mtl_emit *e, int reopen)
{
    mtl_emit_close(e);
    e->in_arm = 0;
    if (!reopen || !e->pass) return;
    const cmd_pass *p = e->pass;
    cmd_pass again = *p;
    again.clear = 0;
    emit_pass(e, &again);
    e->pass = p;
}

/* Opens the cards' pass in place of the eye's, cleared at the frame's
   first card draw and loaded after. It stays open across consecutive card
   draws (mtl_emit_run), since the gauges' draws come together and each
   break of the eye's pass stores and loads the whole eye. A target that
   cannot be made, or a pass that cannot be opened, turns the arm HUD off
   for the session, and the gauges stay on the panel in both eyes. */
static int arm_enter(mtl_device *d, mtl_emit *e, const cmd_draw *c)
{
    __unsafe_unretained MtlDrawCache *dc = CACHE(d);
    const cmd_pass *p = e->pass;
    if (!p || !e->enc) return 0;
    if (!arm_targets(d, p, (MTLPixelFormat)c->pd.k.color_pf)) {
        dc->arm_broken = 1;
        mtl_log("full view: no arm HUD target; the gauges stay on the panel");
        return 0;
    }
    mtl_emit_close(e);
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = dc->arm_ct;
    rp.colorAttachments[0].loadAction = dc->arm_used ? MTLLoadActionLoad : MTLLoadActionClear;
    rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    /* A depth target only as the eye's pass has one: the pipeline's
       formats are the eye's. */
    if (p->dt) {
        rp.depthAttachment.texture = dc->arm_dt;
        rp.depthAttachment.loadAction = MTLLoadActionClear;
        rp.depthAttachment.clearDepth = 1.0;
        rp.depthAttachment.storeAction = MTLStoreActionDontCare;
        rp.stencilAttachment.texture = dc->arm_dt;
        rp.stencilAttachment.loadAction = MTLLoadActionClear;
        rp.stencilAttachment.storeAction = MTLStoreActionDontCare;
    }
    id<MTLRenderCommandEncoder> aenc = [CB(e) renderCommandEncoderWithDescriptor:rp];
    if (!aenc) {
        dc->arm_broken = 1;
        mtl_log("full view: the arm HUD's pass did not open; the gauges stay on the panel");
        e->in_arm = 1;
        arm_leave(e, 1);
        return 0;
    }
    [aenc setFrontFacingWinding:MTLWindingClockwise];
    e->enc = (void *)CFBridgingRetain(aenc);
    e->in_arm = 1;
    dc->arm_used = 1;
    return 1;
}

static void arm_fill(MtlDrawCache *dc, mtl_arm_frame *out)
{
    memset(out, 0, sizeof *out);
    if (!dc->arm_ct) return;
    out->tex = (__bridge void *)dc->arm_ct;
    for (int s = 0; s < 2; s++) {
        out->x[s] = dc->arm_x[s];
        out->w[s] = dc->arm_col[s][1] - dc->arm_col[s][0];
    }
    out->h = (uint32_t)dc->arm_ct.height;
}

void mtl_arm_take(mtl_device *d, mtl_arm_frame *out)
{
    __unsafe_unretained MtlDrawCache *dc = CACHE(d);
    arm_fill(dc, out);
    for (int s = 0; s < 2; s++) out->drawn[s] = dc->arm_used && dc->arm_drawn[s];
    dc->arm_last = *out;
    dc->arm_used = 0;
    dc->arm_drawn[0] = dc->arm_drawn[1] = 0;
}

void mtl_arm_last(mtl_device *d, mtl_arm_frame *out) { *out = CACHE(d)->arm_last; }

/* RT_HIDE_ARMS: which of the view model's arm meshes full view leaves
   out, a mask (1 the first, 2 the second, 3 both), to find which arm is
   which before a two handed weapon drops its off hand. */
static int hide_arms(void)
{
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("RT_HIDE_ARMS");
        v = e && *e ? atoi(e) & 3 : 0;
        if (v) mtl_log("full view: arm meshes left out (RT_HIDE_ARMS=%d)", v);
    }
    return v;
}

static void emit_draw(mtl_device *d, mtl_emit *e, const cmd_draw *c)
{
    if (c->arm_mesh && e->stereo && e->fv.on && (hide_arms() & c->arm_mesh)) return;
    if (c->stereo_screen && e->stereo && e->fv.on && !e->in_aura) {
        if (aura_off()) return;
        if (!e->fv.game_eye && e->pass && e->enc) {
            emit_aura(d, e, c);
            return;
        }
    }
    __unsafe_unretained MtlDrawCache *dc = CACHE(d);
    /* A gauge's draw goes on its card once, in the first eye's replay, and
       neither eye's panel shows it. */
    int arm = e->in_aura ? 0 : arm_side(d, e, c);
    if (e->in_arm && !arm) arm_leave(e, 1);
    if (arm) {
        if (e->eye) return;
        if (!e->in_arm && !arm_enter(d, e, c)) arm = 0;
    }
    id<MTLRenderCommandEncoder> enc = ENC(e);
    /* Functions and the pipeline, at first use: a failure drops the
       draw (logged once per shader) as it did at record time before.
       Whatever was not ready costs this thread, and is logged per draw
       with what it cost; in skip mode the draw is skipped instead. */
    mtl_pipes_wait w = {0};
    int pending = 0;
    mtl_blob *vs = c->vs ? c->vs : dc->ff_vs, *ps = c->ps ? c->ps : dc->ff_ps;
    id<MTLFunction> vfn = enc ? stage_function(d, c->vs, 0, &w, &pending) : nil;
    /* After a pending vertex function too, so skip mode compiles both at once. */
    id<MTLFunction> pfn = vfn || pending ? stage_function(d, c->ps, 1, &w, &pending) : nil;
    /* In the effect's own pass the colour is kept premultiplied over a
       clear target, so the eye can take it with one blend: an additive
       draw leaves alpha alone, any other covers by its alpha. */
    pso_desc apd;
    const pso_desc *pd = &c->pd;
    if (e->in_aura || e->in_arm) {
        apd = c->pd;
        if (!apd.k.blend) apd.k.blend = 1, apd.k.srcb = 2, apd.k.dstb = 1, apd.k.op = 1;
        int additive = apd.k.dstb == 2;
        apd.k.srca = additive ? 1 : 2;
        apd.k.dsta = additive ? 2 : 6;
        apd.k.opa = 1;
        apd.k.wmask |= 8;
        pd = &apd;
    }
    id<MTLRenderPipelineState> pso = vfn && pfn ? get_pipeline(d, vs, ps, vfn, pfn, pd, &w, &pending) : nil;
    if (w.built || w.waited) {
        mtl_pipes_note_wait(dc->pipes, &w);
        mtl_log("pipeline wait %.1f ms: vs %s ps %s (%u built here, %u waited for; pipeline %s)", w.ms,
                vs->sc ? mtl_sc_hash(vs->sc) : "fixed-function", ps->sc ? mtl_sc_hash(ps->sc) : "fixed-function",
                w.built, w.waited, w.pso_how ? w.pso_how : pending ? "not ready" : "ready");
    }
    if (!pso) {
        if (!e->second) {
            if (pending) mtl_pipes_note_skipped(dc->pipes);
            else mtl_frames_note_dropped(d);
        }
        return;
    }
    [enc setRenderPipelineState:pso];
    [enc setDepthStencilState:get_dss(d, c->dss)];
    if (c->stencil_on) [enc setStencilReferenceValue:c->stencil_ref];
    [enc setCullMode:c->cull == 1 ? MTLCullModeNone : c->cull == 2 ? MTLCullModeFront : MTLCullModeBack];
    [enc setTriangleFillMode:c->fill == 2 ? MTLTriangleFillModeLines : MTLTriangleFillModeFill];
    [enc setBlendColorRed:c->blend[0] green:c->blend[1] blue:c->blend[2] alpha:c->blend[3]];
    [enc setDepthBias:c->bias slopeScale:c->slope clamp:0.0f];
    if (e->in_aura) {
        /* The game's viewport and scissor, scaled down to the small target. */
        const cmd_pass *p = e->pass;
        double sx = (double)dc->aura_ct.width / p->width, sy = (double)dc->aura_ct.height / p->height;
        [enc setViewport:(MTLViewport){c->vp[0] * sx, c->vp[1] * sy, c->vp[2] * sx, c->vp[3] * sy, c->vp[4], c->vp[5]}];
        NSUInteger x0 = (NSUInteger)(c->sc[0] * sx), y0 = (NSUInteger)(c->sc[1] * sy);
        NSUInteger x1 = MIN((NSUInteger)ceil((c->sc[0] + c->sc[2]) * sx), dc->aura_ct.width);
        NSUInteger y1 = MIN((NSUInteger)ceil((c->sc[1] + c->sc[3]) * sy), dc->aura_ct.height);
        if (x1 <= x0 || y1 <= y0) return;
        [enc setScissorRect:(MTLScissorRect){x0, y0, x1 - x0, y1 - y0}];
    } else if (e->in_arm) {
        /* The game's viewport and scissor at the same pixels, moved so the
           side's columns of the back buffer land on its part of the cards,
           the scissor kept to that part. */
        int k = arm - 1;
        double dx = (double)dc->arm_x[k] - (double)dc->arm_col[k][0];
        [enc setViewport:(MTLViewport){c->vp[0] + dx, c->vp[1], c->vp[2], c->vp[3], c->vp[4], c->vp[5]}];
        int64_t lo = dc->arm_x[k], hi = lo + (dc->arm_col[k][1] - dc->arm_col[k][0]);
        int64_t x0 = (int64_t)c->sc[0] + (int64_t)dx, x1 = x0 + c->sc[2];
        int64_t y0 = c->sc[1], y1 = MIN((int64_t)c->sc[1] + c->sc[3], (int64_t)dc->arm_ct.height);
        x0 = MAX(x0, lo), x1 = MIN(x1, hi);
        if (x1 <= x0 || y1 <= y0) return;
        [enc setScissorRect:(MTLScissorRect){(NSUInteger)x0, (NSUInteger)y0, (NSUInteger)(x1 - x0), (NSUInteger)(y1 - y0)}];
        dc->arm_drawn[k] = 1;
    } else {
        [enc setViewport:(MTLViewport){c->vp[0], c->vp[1], c->vp[2], c->vp[3], c->vp[4], c->vp[5]}];
        [enc setScissorRect:(MTLScissorRect){c->sc[0], c->sc[1], c->sc[2], c->sc[3]}];
    }
    for (uint32_t s = 0; s < MTL_MAX_STREAMS; s++)
        if (c->streams & (1u << s)) [enc setVertexBuffer:OBJ(c->stream[s].buf) offset:c->stream[s].off atIndex:MTL_BUF_STREAM + s];
    [enc setVertexBuffer:dc->zero offset:0 atIndex:MTL_BUF_NULL];
    for (uint32_t i = 0; i < 7; i++) {
        if (c->vcbv_mask & (1u << i)) [enc setVertexBuffer:OBJ(c->vcbv[i].buf) offset:c->vcbv[i].off atIndex:MTL_BUF_CBV + i];
        if (c->fcbv_mask & (1u << i))
            [enc setFragmentBuffer:OBJ(c->fcbv[i].buf) offset:c->fcbv[i].off atIndex:MTL_BUF_CBV + i];
    }
    id<MTLBuffer> blk = OBJ(c->block.buf);
    [enc setVertexBuffer:blk offset:c->block.off + BLOCK_SPEC atIndex:MTL_BUF_SPEC];
    [enc setFragmentBuffer:blk offset:c->block.off + BLOCK_SPEC atIndex:MTL_BUF_SPEC];
    [enc setVertexBuffer:blk offset:c->block.off + BLOCK_VPUSH atIndex:MTL_BUF_PUSH];
    [enc setFragmentBuffer:blk offset:c->block.off + BLOCK_PPUSH atIndex:MTL_BUF_PUSH];
    [enc setVertexBuffer:dc->heap offset:0 atIndex:MTL_BUF_HEAP];
    [enc setFragmentBuffer:dc->heap offset:0 atIndex:MTL_BUF_HEAP];
    /* Every vertex stage reads the correction, so it is bound for every
       draw: identity unless this replay is an eye and the draw is world.
       Stage A's eye projection is the game's own, so only the eye's
       offset moves the image. */
    mtl_stereo_fix fix;
    if (e->in_aura) {
        /* The game's own screen, as the flat game draws it, shrunk so the
           target holds aura_reach screens. */
        mtl_stereo_identity(&fix);
        fix.x[0] = fix.y[0] = 1.0f / aura_reach();
    } else if (e->in_arm) {
        /* The game's own screen: the viewport alone places it on the card. */
        mtl_stereo_identity(&fix);
    } else if (e->stereo && e->fv.on) {
        /* Full view: world draws into the eye's own view from the eye's
           own position, HUD and 2D onto the head-locked panel, and the
           rest as recorded: offscreen draws keep the game's projection,
           and a screen pass reads this eye's own copy. The check keeps
           the game's projection in both eyes. */
        int eye = e->eye ? 1 : 0;
        if (e->fv.game_eye) mtl_stereo_identity(&fix);
        else if (c->stereo_a > 0.0f && c->stereo_sky) {
            /* The 3D sky is drawn from a camera at its own small scale, so
               the eye's offset is the world's divided by it (0: at
               infinity); the whole offset made the city a near model
               that swam with the head. */
            float t[3];
            for (int i = 0; i < 3; i++) t[i] = e->fv.eye_t[eye][i] * e->fv.sky_scale;
            mtl_stereo_correction3(&c->stereo_p, &e->fv.eye_p[eye], t, &fix);
        } else if (c->stereo_a > 0.0f && c->stereo_screen)
            mtl_stereo_panel_correction(&e->fv.eye_p[eye], e->fv.eye_t[eye], e->hud_z, e->fv.aura[0], e->fv.aura[1],
                                        &fix);
        else if (c->stereo_a > 0.0f) mtl_stereo_correction3(&c->stereo_p, &e->fv.eye_p[eye], e->fv.eye_t[eye], &fix);
        else if (c->stereo_hud && fv_cover(c))
            /* A tint or overlay over the whole view (a discipline's, a
               fade) fills each eye rather than shrinking onto the panel. */
            mtl_stereo_identity(&fix);
        else if (c->stereo_hud)
            mtl_stereo_panel_correction(&e->fv.eye_p[eye], e->fv.eye_t[eye], e->hud_z, e->fv.panel[0], e->fv.panel[1],
                                        &fix);
        else mtl_stereo_identity(&fix);
    } else if (e->stereo && c->stereo_a > 0.0f) {
        mtl_stereo_proj p = {c->stereo_a, 0.0f, 1.0f, 0.0f};
        mtl_stereo_correction(&p, &p, e->eye_x, &fix);
    } else if (e->stereo && c->stereo_hud) {
        mtl_stereo_hud_correction(e->screen_a, e->eye_x, e->hud_z, &fix);
    } else {
        mtl_stereo_identity(&fix);
    }
    [enc setVertexBytes:&fix length:sizeof fix atIndex:MTL_BUF_STEREO];
    for (uint32_t i = 0; i < c->ntex; i++) {
        const tex_ref *t = &c->tex[i];
        for (uint32_t v = 0; v < MTL_TEX_VIEWS; v++) {
            if (t->slot[v] == MTL_TEX_NONE) continue;
            id<MTLTexture> tex = OBJ(v == 0 ? swap_tex(e, t->view[0]) : t->view[v]);
            if (t->fragment) [enc setFragmentTexture:tex atIndex:t->slot[v]];
            else [enc setVertexTexture:tex atIndex:t->slot[v]];
        }
    }
    if (c->call == CALL_ARRAYS)
        [enc drawPrimitives:(MTLPrimitiveType)c->prim vertexStart:c->start vertexCount:c->count instanceCount:c->instances];
    else if (c->call == CALL_INDEXED)
        [enc drawIndexedPrimitives:(MTLPrimitiveType)c->prim
                        indexCount:c->count
                         indexType:(MTLIndexType)c->index_type
                       indexBuffer:OBJ(c->ib.buf)
                 indexBufferOffset:c->ib.off
                     instanceCount:c->instances
                        baseVertex:c->base
                      baseInstance:0];
}

static void emit_readback(mtl_emit *e, const cmd_readback *c)
{
    /* Once per list: a second replay would only write the same bytes,
       and the readback belongs to the frame the game asked about. */
    if (e->second) return;
    mtl_emit_close(e);
    id<MTLBlitCommandEncoder> bl = [CB(e) blitCommandEncoder];
    [bl copyFromTexture:OBJ(c->src)
                     sourceSlice:c->slice
                     sourceLevel:c->level
                    sourceOrigin:MTLOriginMake(0, 0, 0)
                      sourceSize:MTLSizeMake(c->width, c->height, 1)
                        toBuffer:OBJ(c->dst)
               destinationOffset:0
          destinationBytesPerRow:c->pitch
        destinationBytesPerImage:(NSUInteger)c->pitch * c->rows];
    [bl endEncoding];
}

static void emit_upload(mtl_emit *e, const cmd_upload *c)
{
    mtl_emit_close(e);
    id<MTLBlitCommandEncoder> bl = [CB(e) blitCommandEncoder];
    [bl copyFromBuffer:OBJ(c->src)
               sourceOffset:0
          sourceBytesPerRow:c->row_bytes
        sourceBytesPerImage:(NSUInteger)c->row_bytes * c->rows
                 sourceSize:MTLSizeMake(c->width, c->height, 1)
                  toTexture:OBJ(swap_tex(e, c->dst))
           destinationSlice:c->slice
           destinationLevel:c->level
          destinationOrigin:MTLOriginMake(c->x, c->y, 0)];
    [bl endEncoding];
}

static void emit_stretch(mtl_device *d, mtl_emit *e, const cmd_stretch *c)
{
    mtl_emit_close(e);
    __unsafe_unretained MtlDrawCache *dc = CACHE(d);
    id<MTLTexture> st = OBJ(swap_tex(e, c->src)), dt = OBJ(swap_tex(e, c->dst));
    NSNumber *key = @(0x200000000ull | dt.pixelFormat);
    id<MTLRenderPipelineState> pso = dc->clear_psos[key];
    if (!pso) {
        id<MTLLibrary> lib = util_library(d);
        MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
        pd.vertexFunction = [lib newFunctionWithName:@"blit_vs"];
        pd.fragmentFunction = [lib newFunctionWithName:@"blit_fs"];
        pd.colorAttachments[0].pixelFormat = dt.pixelFormat;
        NSError *err = nil;
        pso = [MTLD(d) newRenderPipelineStateWithDescriptor:pd error:&err];
        if (!pso) {
            mtl_log("StretchRect pipeline failed: %s", err.localizedDescription.UTF8String);
            return;
        }
        dc->clear_psos[key] = pso;
    }
    id<MTLTexture> view = st;
    if (c->src_level || c->src_slice || st.textureType != MTLTextureType2D)
        view = [st newTextureViewWithPixelFormat:st.pixelFormat
                                     textureType:MTLTextureType2D
                                          levels:NSMakeRange(c->src_level, 1)
                                          slices:NSMakeRange(c->src_slice, 1)];
    int point = c->filter == 1;
    if (!dc->stretch_samp[point]) {
        MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
        sd.minFilter = sd.magFilter = point ? MTLSamplerMinMagFilterNearest : MTLSamplerMinMagFilterLinear;
        dc->stretch_samp[point] = [MTLD(d) newSamplerStateWithDescriptor:sd];
    }
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = dt;
    rp.colorAttachments[0].level = c->dst_level;
    rp.colorAttachments[0].slice = c->dst_slice;
    rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> enc = [CB(e) renderCommandEncoderWithDescriptor:rp];
    const mtl_rect *s = &c->s, *t = &c->t;
    float r[4] = {(float)s->left / view.width, (float)s->top / view.height, (float)(s->right - s->left) / view.width,
                  (float)(s->bottom - s->top) / view.height};
    [enc setRenderPipelineState:pso];
    [enc setViewport:(MTLViewport){t->left, t->top, t->right - t->left, t->bottom - t->top, 0, 1}];
    [enc setVertexBytes:r length:sizeof r atIndex:0];
    [enc setFragmentTexture:view atIndex:0];
    [enc setFragmentSamplerState:dc->stretch_samp[point] atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc endEncoding];
}

static void emit_fill(mtl_device *d, mtl_emit *e, const cmd_fill *c)
{
    mtl_emit_close(e);
    uint32_t color = c->color;
    id<MTLTexture> t = OBJ(swap_tex(e, c->tex));
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = t;
    rp.colorAttachments[0].level = c->level;
    rp.colorAttachments[0].slice = c->slice;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    rp.colorAttachments[0].loadAction = c->whole ? MTLLoadActionClear : MTLLoadActionLoad;
    rp.colorAttachments[0].clearColor = MTLClearColorMake(((color >> 16) & 255) / 255.0, ((color >> 8) & 255) / 255.0,
                                                          (color & 255) / 255.0, (color >> 24) / 255.0);
    id<MTLRenderCommandEncoder> enc = [CB(e) renderCommandEncoderWithDescriptor:rp];
    id<MTLRenderPipelineState> pso = c->whole ? nil : clear_pipeline(d, t.pixelFormat, 0, D3DCLEAR_TARGET);
    if (pso) {
        float col[4] = {((color >> 16) & 255) / 255.0f, ((color >> 8) & 255) / 255.0f, (color & 255) / 255.0f,
                        (color >> 24) / 255.0f};
        float z = 0;
        const mtl_rect *r = &c->r;
        [enc setRenderPipelineState:pso];
        [enc setViewport:(MTLViewport){0, 0, c->lw, c->lh, 0, 1}];
        [enc setScissorRect:(MTLScissorRect){(NSUInteger)r->left, (NSUInteger)r->top, (NSUInteger)(r->right - r->left),
                                             (NSUInteger)(r->bottom - r->top)}];
        [enc setVertexBytes:&z length:sizeof z atIndex:0];
        [enc setFragmentBytes:col length:sizeof col atIndex:0];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }
    [enc endEncoding];
}

/* The full view shot's per command description (frame.m opens the file):
   one line per command as this eye replays it, textures named by address
   so a draw's inputs can be matched to the passes that wrote them. */
static void dump_tex(FILE *f, const char *what, void *p)
{
    id<MTLTexture> t = OBJ(p);
    if (!t) {
        fprintf(f, " %s=none", what);
        return;
    }
    fprintf(f, " %s=%p:%lux%lu:fmt%lu%s%s", what, p, (unsigned long)t.width, (unsigned long)t.height,
            (unsigned long)t.pixelFormat, (t.usage & MTLTextureUsageRenderTarget) ? ":rt" : "",
            t.label.length ? [[NSString stringWithFormat:@":%@", t.label] UTF8String] : "");
}

static void dump_cmd(mtl_emit *e, const cmd_head *h)
{
    FILE *f = e->dump;
    uint32_t n = e->dump_cmd++;
    switch (h->type) {
    case CMD_PASS: {
        const cmd_pass *p = (const cmd_pass *)h;
        fprintf(f, "%u pass %ux%u clear 0x%x z %.3f", n, p->width, p->height, p->clear, p->z);
        dump_tex(f, "color", swap_tex(e, p->ct));
        dump_tex(f, "depth", p->dt ? swap_tex(e, p->dt) : NULL);
        break;
    }
    case CMD_END_PASS: fprintf(f, "%u end", n); break;
    case CMD_CLEAR: {
        const cmd_clear *c = (const cmd_clear *)h;
        fprintf(f, "%u clear flags 0x%x color 0x%08x z %.3f rects %u %ux%u", n, c->flags, c->color, c->z, c->n, c->width,
                c->height);
        break;
    }
    case CMD_DRAW: {
        const cmd_draw *c = (const cmd_draw *)h;
        const char *cls = c->stereo_a > 0.0f ? (c->stereo_sky ? "sky" : c->stereo_screen ? "screen" : "world") : c->stereo_hud ? (c->fv_full == 1 ? "overlay" : "hud") : "other";
        const mtl_stereo_proj_full *p = &c->stereo_p;
        fprintf(f, "%u draw %u %s stereo_a %.4f a %.4f b %.4f cx %.4f cy %.4f A %.4f vp %.0f,%.0f %.0fx%.0f z %.3f-%.3f",
                n, e->dump_draw++, cls, c->stereo_a, p->a, p->b, p->cx, p->cy, p->A, c->vp[0], c->vp[1], c->vp[2], c->vp[3],
                c->vp[4], c->vp[5]);
        fprintf(f, " vs %s ps %s prim %u count %u streams 0x%x dss 0x%llx cull %u", c->vs && c->vs->sc ? mtl_sc_hash(c->vs->sc) : "ff",
                c->ps && c->ps->sc ? mtl_sc_hash(c->ps->sc) : "ff", c->prim, c->count, c->streams, (unsigned long long)c->dss,
                c->cull);
        if (c->stereo_hud && c->stereo_a <= 0.0f) {
            float bb[4];
            if (cov_box(c, bb)) {
                fprintf(f, " bb %.3f %.3f %.3f %.3f", bb[0], bb[1], bb[2], bb[3]);
                int side = c->fv_full == 1 ? 0 : mtl_stereo_arm_side(bb);
                if (side) fprintf(f, " arm %s%s", side == 1 ? "L" : "R", e->fv.arm[side - 1] ? "" : " (on the panel)");
            } else fprintf(f, " bb ?");
        }
        for (uint32_t i = 0; i < c->ntex; i++) {
            char what[24];
            void *tp = swap_tex(e, c->tex[i].view[0]);
            id<MTLTexture> tt = OBJ(tp);
            if (tt && tt.width == 1 && tt.height == 1) continue; /* the unbound slots' dummy */
            snprintf(what, sizeof what, "%s%u", c->tex[i].fragment ? "ps" : "vs", c->tex[i].binding);
            dump_tex(f, what, tp);
        }
        break;
    }
    case CMD_READBACK: fprintf(f, "%u readback", n); break;
    case CMD_UPLOAD: {
        const cmd_upload *c = (const cmd_upload *)h;
        fprintf(f, "%u upload", n);
        dump_tex(f, "dst", c->dst);
        break;
    }
    case CMD_STRETCH: {
        const cmd_stretch *c = (const cmd_stretch *)h;
        fprintf(f, "%u stretch", n);
        dump_tex(f, "src", swap_tex(e, c->src));
        dump_tex(f, "dst", swap_tex(e, c->dst));
        break;
    }
    case CMD_FILL: {
        const cmd_fill *c = (const cmd_fill *)h;
        fprintf(f, "%u fill color 0x%08x", n, c->color);
        dump_tex(f, "tex", swap_tex(e, c->tex));
        break;
    }
    case CMD_PRESENT: fprintf(f, "%u present", n); break;
    default: fprintf(f, "%u command %u", n, h->type); break;
    }
    fputc('\n', f);
}

void mtl_emit_run(mtl_device *d, mtl_emit *e, const uint8_t *cmds, size_t from, size_t to)
{
    for (size_t at = from; at < to;) {
        const cmd_head *h = (const cmd_head *)(cmds + at);
        if (e->dump) dump_cmd(e, h);
        /* The cards' pass stays open only across card draws; a command that
           opens, ends or presents a pass needs the eye's pass no more. */
        if (e->in_arm && !(h->type == CMD_DRAW && arm_side(d, e, (const cmd_draw *)h)))
            arm_leave(e, h->type != CMD_PASS && h->type != CMD_END_PASS && h->type != CMD_PRESENT);
        switch (h->type) {
        case CMD_PASS: emit_pass(e, (const cmd_pass *)h); break;
        case CMD_END_PASS: mtl_emit_close(e); break;
        case CMD_DRAW: emit_draw(d, e, (const cmd_draw *)h); break;
        case CMD_CLEAR: emit_clear(d, e, (const cmd_clear *)h); break;
        case CMD_READBACK: emit_readback(e, (const cmd_readback *)h); break;
        case CMD_UPLOAD: emit_upload(e, (const cmd_upload *)h); break;
        case CMD_STRETCH: emit_stretch(d, e, (const cmd_stretch *)h); break;
        case CMD_FILL: emit_fill(d, e, (const cmd_fill *)h); break;
        case CMD_PRESENT:
            mtl_emit_close(e);
            if (!e->second) {
                const cmd_present *c = (const cmd_present *)h;
                memcpy(e->pose, c->pose, sizeof e->pose);
                /* An eye's replay presents once the other eye is drawn
                   too, which is after this list has run again (frame.m). */
                if (e->stereo) {
                    e->pend_present = 1;
                    e->pend_back = c->back;
                    e->pend_ns = c->presented_ns;
                } else {
                    mtl_emit_present(d, e, c->back, NULL, c->presented_ns);
                }
            }
            break;
        default: mtl_log("unknown recorded command %u", h->type); return;
        }
        at += h->bytes;
    }
    /* Inline mode replays a list in pieces as it is recorded: the next
       piece goes on in the eye's pass. */
    if (e->in_arm) arm_leave(e, 1);
}

void mtl_fill_draw(void)
{
    void **v = mtl_vt_device;
#define SET(m, f) v[SLOT_IDirect3DDevice9_##m] = (void *)(f)
    SET(Clear, mtl_clear);
    SET(StretchRect, dev_stretch);
    SET(DrawPrimitive, dev_dp);
    SET(DrawIndexedPrimitive, dev_dip);
    SET(DrawPrimitiveUP, dev_dp_up);
    SET(DrawIndexedPrimitiveUP, dev_dip_up);
    SET(SetTexture, dev_set_texture);
    SET(GetTexture, dev_get_texture);
    SET(SetSamplerState, dev_set_ss);
    SET(GetSamplerState, dev_get_ss);
    SET(SetTextureStageState, dev_set_tss);
    SET(GetTextureStageState, dev_get_tss);
    SET(SetStreamSource, dev_stream);
    SET(GetStreamSource, dev_get_stream);
    SET(SetStreamSourceFreq, dev_stream_freq);
    SET(SetIndices, dev_set_indices);
    SET(GetIndices, dev_get_indices);
    SET(SetVertexDeclaration, dev_set_decl);
    SET(GetVertexDeclaration, dev_get_decl);
    SET(SetFVF, dev_set_fvf);
    SET(GetFVF, dev_get_fvf);
    SET(SetVertexShader, dev_set_vs);
    SET(GetVertexShader, dev_get_vs);
    SET(SetPixelShader, dev_set_ps);
    SET(GetPixelShader, dev_get_ps);
    SET(SetVertexShaderConstantF, dev_vs_f);
    SET(GetVertexShaderConstantF, dev_vs_f_get);
    SET(SetVertexShaderConstantI, dev_vs_i);
    SET(GetVertexShaderConstantI, dev_vs_i_get);
    SET(SetVertexShaderConstantB, dev_vs_b);
    SET(GetVertexShaderConstantB, dev_vs_b_get);
    SET(SetPixelShaderConstantF, dev_ps_f);
    SET(GetPixelShaderConstantF, dev_ps_f_get);
    SET(SetPixelShaderConstantI, dev_ps_i);
    SET(GetPixelShaderConstantI, dev_ps_i_get);
    SET(SetPixelShaderConstantB, dev_ps_b);
    SET(GetPixelShaderConstantB, dev_ps_b_get);
    SET(SetClipPlane, dev_set_clip);
    SET(GetClipPlane, dev_get_clip);
    SET(SetScissorRect, dev_set_scissor);
    SET(GetScissorRect, dev_get_scissor);
    SET(SetTransform, dev_set_transform);
    SET(GetTransform, dev_get_transform);
    SET(MultiplyTransform, dev_mul_transform);
    SET(SetMaterial, dev_set_material);
    SET(GetMaterial, dev_get_material);
    SET(SetLight, dev_set_light);
    SET(GetLight, dev_get_light);
    SET(LightEnable, dev_light_enable);
    SET(GetLightEnable, dev_get_light_enable);
    SET(ColorFill, dev_color_fill);
    SET(UpdateSurface, dev_update_surface);
    SET(UpdateTexture, dev_update_texture);
#undef SET
}

/* For the headless test: DXVK's two fixed-function ubershaders through
   SPIRV-Cross and the Metal compiler, which is the route an FF draw would
   take. Returns how many compiled (0 to 2). */
D9_EXPORT int metal_d3d9_ff_selfcheck(void)
{
    int ok = 0;
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        for (int w = 0; w < 2; w++) {
            char err[1024];
            const char *msl = mtl_sc_ff_msl(w, err, sizeof err);
            if (!msl) {
                mtl_log("FF %s: %s", w ? "FS" : "VS", err);
                continue;
            }
            NSError *e = nil;
            id<MTLLibrary> lib = [dev newLibraryWithSource:@(msl) options:nil error:&e];
            id<MTLFunction> fn = [lib newFunctionWithName:@"main0"];
            if (fn) ok++;
            else mtl_log("FF %s did not compile: %s", w ? "FS" : "VS", e.localizedDescription.UTF8String);
        }
    }
    return ok;
}

/* For the headless test: the pipeline cache's counters (pipelines.h,
   MTL_PIPES_STAT_*), and a wait until its warm queue is idle. */
D9_EXPORT void metal_d3d9_pipeline_stats(void *dev, uint64_t out[MTL_PIPES_STAT_N])
{
    mtl_device *d = dev;
    mtl_pipes_stats(CACHE(d)->pipes, out);
}

D9_EXPORT void metal_d3d9_pipeline_wait_idle(void *dev)
{
    mtl_device *d = dev;
    mtl_pipes_wait_idle(CACHE(d)->pipes);
}

/* For the headless test: every pipeline built from now on fails, as one a
   GPU rejects would. Pipelines already made are kept. */
D9_EXPORT void metal_d3d9_pipeline_fail(void *dev, int on)
{
    mtl_device *d = dev;
    mtl_pipes_set_fail(CACHE(d)->pipes, on);
}
