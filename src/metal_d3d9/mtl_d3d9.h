/*
 * Shared declarations for the Metal D3D9 backend.
 *
 * The porter's bridge calls a host D3D9 object through its vtable with the
 * plain arm64 C ABI, the way it calls DXVK: the object pointer is first,
 * 32-bit DWORD arguments stay 32-bit, pointers are 64-bit, and the one float
 * argument (Clear's Z) travels in a float register. Every interface here is
 * a struct whose first member is that vtable pointer, so an object address
 * is an interface pointer.
 *
 * Slot order is not restated: it comes from mtl_d3d9_slots.h, which the
 * build generates from the porter's own null device tables, the one list
 * the bridge itself resolves method names against.
 */
#ifndef MTL_D3D9_H
#define MTL_D3D9_H

/* Palm vertex pairs kept per hand per frame (draw.m, arm_hands). */
#define MTL_ARM_PAIRS 192

#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

#include "mtl_d3d9_slots.h"
#include "stereo.h"

#define D9_EXPORT __attribute__((visibility("default")))

typedef int32_t HRESULT;

#define D3D_OK 0
#define S_FALSE 1
#define E_OUTOFMEMORY ((HRESULT)0x8007000Eu)
#define D3DERR_INVALIDCALL ((HRESULT)0x8876086Cu)
#define D3DERR_NOTAVAILABLE ((HRESULT)0x8876086Au)
#define D3DERR_NOTFOUND ((HRESULT)0x88760866u)
#define D3DERR_OUTOFVIDEOMEMORY ((HRESULT)0x8876017Cu)

#define D3DFMT_UNKNOWN 0u
#define D3DFMT_R8G8B8 20u
#define D3DFMT_A8R8G8B8 21u
#define D3DFMT_X8R8G8B8 22u
#define D3DFMT_R5G6B5 23u
#define D3DFMT_P8 41u
#define D3DFMT_D16_LOCKABLE 70u
#define D3DFMT_D32F_LOCKABLE 82u
#define D3DFMT_VERTEXDATA 100u
#define D3DFMT_DXT1 0x31545844u
#define D3DFMT_INDEX16 101u
#define D3DFMT_INDEX32 102u

#define D3DPOOL_DEFAULT 0u
#define D3DPOOL_MANAGED 1u
#define D3DPOOL_SYSTEMMEM 2u
#define D3DPOOL_SCRATCH 3u

#define D3DUSAGE_RENDERTARGET 0x1u
#define D3DUSAGE_DEPTHSTENCIL 0x2u
#define D3DUSAGE_AUTOGENMIPMAP 0x400u

#define D3DRTYPE_SURFACE 1u
#define D3DRTYPE_TEXTURE 3u
#define D3DRTYPE_CUBETEXTURE 5u
#define D3DRTYPE_VERTEXBUFFER 6u
#define D3DRTYPE_INDEXBUFFER 7u

#define D3DCLEAR_TARGET 0x1u
#define D3DCLEAR_ZBUFFER 0x2u
#define D3DCLEAR_STENCIL 0x4u

#define D3DPRESENT_INTERVAL_IMMEDIATE 0x80000000u

/* Host-side D3D9 structures, in the layouts the bridge repacks into: the
   guest's 32-bit pointers widened to 64 bits, everything else as in the
   SDK headers. The bridge asserts the same sizes on its side. */
typedef struct {
    uint32_t BackBufferWidth, BackBufferHeight, BackBufferFormat, BackBufferCount, MultiSampleType, MultiSampleQuality,
        SwapEffect;
    void *hDeviceWindow; /* an SDL_Window *, as for DXVK's SDL3 window system */
    int32_t Windowed, EnableAutoDepthStencil;
    uint32_t AutoDepthStencilFormat, Flags, FullScreen_RefreshRateInHz, PresentationInterval;
} mtl_present_params;

typedef struct {
    int32_t Pitch;
    void *pBits;
} mtl_locked_rect;

typedef struct {
    char Driver[512], Description[512], DeviceName[32];
    int64_t DriverVersion;
    uint32_t VendorId, DeviceId, SubSysId, Revision;
    uint8_t DeviceIdentifier[16];
    uint32_t WHQLLevel;
} mtl_adapter_id;

typedef struct {
    uint32_t Format, Type, Usage, Pool, MultiSampleType, MultiSampleQuality, Width, Height;
} mtl_surface_desc;

typedef struct {
    uint32_t Width, Height, RefreshRate, Format;
} mtl_display_mode;

typedef struct {
    uint32_t X, Y, Width, Height;
    float MinZ, MaxZ;
} mtl_viewport;

typedef struct {
    int32_t left, top, right, bottom;
} mtl_rect;

_Static_assert(sizeof(mtl_present_params) == 64, "host D3DPRESENT_PARAMETERS is 64 bytes");
_Static_assert(offsetof(mtl_present_params, hDeviceWindow) == 32, "hDeviceWindow at 32");
_Static_assert(sizeof(mtl_locked_rect) == 16, "host D3DLOCKED_RECT is 16 bytes");
_Static_assert(sizeof(mtl_adapter_id) == 1104, "host D3DADAPTER_IDENTIFIER9 is 1104 bytes");
_Static_assert(sizeof(mtl_surface_desc) == 32, "D3DSURFACE_DESC is 32 bytes");
_Static_assert(sizeof(mtl_viewport) == 24, "D3DVIEWPORT9 is 24 bytes");

/* Slot indices, one enum per interface, from the generated lists. */
#define MTL_SLOT_ENUM(i, m, n) SLOT_##i##_##m = n,
enum { MTL_SLOTS_IDirect3D9(MTL_SLOT_ENUM) };
enum { MTL_SLOTS_IDirect3DDevice9(MTL_SLOT_ENUM) };
enum { MTL_SLOTS_IDirect3DSurface9(MTL_SLOT_ENUM) };
enum { MTL_SLOTS_IDirect3DTexture9(MTL_SLOT_ENUM) };
enum { MTL_SLOTS_IDirect3DCubeTexture9(MTL_SLOT_ENUM) };
enum { MTL_SLOTS_IDirect3DVertexBuffer9(MTL_SLOT_ENUM) };
enum { MTL_SLOTS_IDirect3DIndexBuffer9(MTL_SLOT_ENUM) };
enum { MTL_SLOTS_IDirect3DVertexDeclaration9(MTL_SLOT_ENUM) };
enum { MTL_SLOTS_IDirect3DVertexShader9(MTL_SLOT_ENUM) };
enum { MTL_SLOTS_IDirect3DPixelShader9(MTL_SLOT_ENUM) };
enum { MTL_SLOTS_IDirect3DQuery9(MTL_SLOT_ENUM) };

typedef enum {
    MTL_K_D3D,
    MTL_K_DEVICE,
    MTL_K_SURFACE,
    MTL_K_TEXTURE,
    MTL_K_CUBE,
    MTL_K_VB,
    MTL_K_IB,
    MTL_K_DECL,
    MTL_K_VS,
    MTL_K_PS,
    MTL_K_QUERY,
    MTL_K_N
} mtl_kind;

struct mtl_device;

/* Common head of every object. refs is the public count that AddRef and
   Release return; the bridge drops its proxy when Release returns 0, so
   that count must reach 0 exactly when the guest lets go. priv counts the
   device's own bindings (render target, depth surface, back buffer), which
   keep the memory alive past that point without the guest seeing them, as
   DXVK's private references do. A subresource surface has a container and
   forwards both counts to it. */
typedef struct mtl_obj {
    void *const *vtbl;
    mtl_kind kind;
    uint32_t refs, priv;
    struct mtl_obj *container;
    struct mtl_device *dev;
} mtl_obj;

typedef struct mtl_surface {
    mtl_obj o;
    uint32_t format, usage, pool, width, height;
    uint32_t pitch, rows; /* bytes per row (or per row of 4x4 blocks), and how many */
    uint8_t *mem;         /* CPU backing, allocated zeroed on first use */
    int locked, no_cpu;   /* no_cpu: a depth surface or the back buffer, never locked */
    void *tex;            /* id<MTLTexture>: the back buffer lives on the GPU */
    void *buf;            /* id<MTLBuffer>: a system memory readback target, mem is its contents */
    uint64_t pending_serial; /* the recorded list whose readback writes buf, 0 when none is outstanding */
    uint8_t dirty;        /* a texture level written since its last upload */
} mtl_surface;

typedef struct mtl_texture {
    mtl_obj o;
    uint32_t format, usage, pool, width, height, levels, faces;
    uint32_t priority, lod, filter;
    uint64_t bytes; /* counted against the video memory budget if pool is DEFAULT */
    mtl_surface *sub; /* faces * levels surfaces, face major */
    void *gtex;       /* id<MTLTexture>, made on first bind (gpu.m) */
    uint64_t last_use; /* command buffer serial of the last draw that read it */
    uint32_t dirty, warned;
    uint32_t update_dirty; /* written since the last UpdateTexture read it as a source */
} mtl_texture;

#define MTL_BUF_FREE 4

typedef struct mtl_buffer {
    mtl_obj o;
    uint32_t length, usage, format_or_fvf, pool, priority;
    uint8_t *mem;      /* the contents of gbuf */
    void *gbuf;        /* id<MTLBuffer>, shared storage */
    uint64_t last_use; /* serial of the last draw that read it */
    void *free_buf[MTL_BUF_FREE]; /* retired storage, reused once its serial completes */
    uint64_t free_serial[MTL_BUF_FREE];
    uint32_t nfree;
} mtl_buffer;

typedef struct mtl_blob {
    mtl_obj o;
    uint32_t bytes, version; /* declaration bytes or shader bytecode bytes */
    uint8_t *data;
    uint64_t uid;       /* never reused, unlike the address: pipeline keys use it */
    struct mtl_sc *sc;  /* shaders: DXVK's analysis, and the MSL once translated */
    void *fn;           /* id<MTLFunction>, compiled at the first draw */
    int failed;         /* translation or compilation failed; draws with it are dropped */
    int has_post;       /* declaration with POSITIONT, which DXVK draws with the FF VS */
    uint32_t decl_flags, tc_mask; /* declarations: MTL_DECL_* (shader.h), and texcoord sizes, 3 bits each */
    uint64_t held_serial; /* shaders: the list that last took a hold on it (frame.m), so a list holds it once */
} mtl_blob;

typedef struct mtl_query {
    mtl_obj o;
    uint32_t type;
} mtl_query;

typedef struct mtl_d3d9 {
    mtl_obj o;
} mtl_d3d9;

#define MTL_MAX_RT 4
#define MTL_NXF 18 /* transforms kept: view, projection, texture 0-7, world 0-7 (MTL_XF_*, shader.h) */

/* One SetLight slot: the D3DLIGHT9 bytes as the engine passed them. */
typedef struct {
    uint8_t light[104];
    uint8_t enabled;
} mtl_light;
#define MTL_NSAMP 21 /* DXVK texture bindings: 0-15 pixel, 17-20 vertex */

/* Everything the draw path reads, as the engine set it. */
typedef struct {
    struct mtl_texture *tex[MTL_NSAMP];
    uint32_t samp[MTL_NSAMP][14];
    int32_t samp_heap[MTL_NSAMP]; /* draw.m: each binding's sampler heap index, -1 until a draw looks it up again */
    uint32_t tss[8][33];
    struct mtl_buffer *vb[16];
    uint32_t vb_off[16], vb_stride[16], freq[16];
    struct mtl_buffer *ib;
    struct mtl_blob *decl, *vs, *ps;
    uint32_t fvf;
    float vsf[256 * 4];
    int32_t vsi[16 * 4];
    uint32_t vsb[16];
    float psf[224 * 4];
    int32_t psi[16 * 4];
    uint32_t psb[16];
    uint32_t vs_changed_floats;
    float clip[6][4];
    mtl_rect scissor;
    float xf[MTL_NXF][16]; /* D3DMATRIX order, as SetTransform passes them */
    uint8_t material[68];  /* D3DMATERIAL9 */
    mtl_light *lights;     /* grown by SetLight and LightEnable, as DXVK's list */
    uint32_t nlights;
} mtl_state;

/* The video memory budget DXVK runs under in the bridge's DXVK_CONFIG
   (d3d9.maxAvailableMemory = 1024), counted the way the null device counts
   it: textures in the default pool. */
#define MTL_VIDMEM_CAP (1024ull << 20)

/* What this GPU can do, read once from the MTLDevice when the device is
   created (gpu.m, mtl_caps_query). Every choice that differs between
   GPUs reads this, never the platform, so the simulator, the headset and
   a Mac each take the path their GPU allows. RT_METAL_FORCE_LIMITS=1
   pretends to the least of each, so the constrained paths run on a Mac. */
typedef struct {
    int bc;              /* BC1 to BC3 sample natively; else DXT is decoded on upload */
    int packed16;        /* B5G6R5, BGR5A1 and ABGR4 exist; else they are widened on upload */
    int archive;         /* MTLBinaryArchive may be created; else pipelines are compiled each run */
    int forced;          /* RT_METAL_FORCE_LIMITS=1 */
    uint32_t ab_tier;    /* argument buffers tier, 1 or 2 */
    uint32_t max_tex;    /* texture argument table entries per stage */
    uint32_t max_ab_samplers; /* samplers one argument buffer may hold */
} mtl_gpu_caps;

#define MTL_EYE_COPIES 4

typedef struct mtl_device {
    mtl_obj o;
    mtl_d3d9 *d3d;
    void *mtl, *queue, *layer, *view, *window; /* MTLDevice, MTLCommandQueue, CAMetalLayer, SDL_MetalView, SDL_Window */
    mtl_gpu_caps caps;                            /* gpu.m: fixed at creation */
    struct mtl_frames *fq;                     /* frame.m: the list being recorded, the queue, the render thread */
    void *clear_pso;                           /* unused since the draw path: see draw.m clear pipelines */
    int pass_open;                             /* the recorded list has a render pass open on these attachments */
    void *pass_rt, *pass_ds;                   /* (MTLTextures, kept alive by the list) */
    uint32_t pass_level, pass_slice, pass_pf;
    uint32_t pend_clear;                       /* D3DCLEAR flags waiting for the next pass's load actions */
    uint32_t pend_color, pend_stencil;
    float pend_z;
    void *draw;                                /* draw.m's caches: pipelines, samplers, uploads */
    mtl_state st;
    uint64_t serial;                           /* of the list being recorded; starts at 1 */
    _Atomic uint64_t completed;                /* highest serial whose command buffer the GPU has finished */
    uint64_t renames, tex_renames, tex_uploads, ff_draws, dropped_draws, shaders_built, shaders_cached;
    mtl_present_params pp;
    mtl_surface *back, *auto_ds, *rt[MTL_MAX_RT], *ds;
    mtl_viewport vp;
    uint32_t rs[256];
    uint32_t cursor, flags;
    uint64_t vidmem;
    uint64_t draws, clears, presents, free_drops;
    /* draw.m: stereo's classing of draws (stereo.h), on the game thread.
       stereo_a is the last world view projection's clip x per unit, which
       a model view projection draw borrows. */
    float stereo_a;
    mtl_stereo_proj_full stereo_p; /* and its whole projection, which full view borrows the same way */
    uint64_t stereo_class[7]; /* MTL_STEREO_CLASSES */
    /* The frame's world draws' a, taken at its Present and logged when
       it changes (stereo_world_logs so far); the eye planes are sized by
       the session's a, not this (stereo.h). Game thread only. */
    mtl_stereo_tally stereo_tally;
    float stereo_world_a;
    uint32_t stereo_world_logs;
    uint32_t stereo_seen; /* bit c: class c has been logged once */
    /* draw.m, mtl_fv_scan: the back buffer's world draws in the list being
       recorded (offsets into it, the depth clear group each fell in, and
       its a), read at each submit to find the 3D sky and the view model.
       Game thread only. */
    struct mtl_fv_draw *fv_draws;
    uint32_t fv_n, fv_cap, fv_group;
    /* Where the camera of the last world draw that showed it was (a view
       projection's rows or a fixed-function view), for the model view
       projection draws after it, which cannot show it. */
    float fv_cam[3];
    int fv_cam_ok;
    int fv_vm_ever, fv_vm_state, fv_vm_run;
    uint32_t fv_sky_logs, fv_vm_logs, fv_screen_logs;
    /* The third person aim (mtl_fv_aim): the frame's main view projection
       and viewport from the scan of whichever list drew it, and where the
       open list's Present is. Game thread only. */
    float fv_aim_rows[4][4], fv_aim_vp[6];
    int fv_aim_ok;
    /* The arms' hands from their own draws (draw.m, arm_hands): the arm
       textures the last scan found, this frame's bind pose hands from the
       runtime (asked at the frame's first arm draw), and per hand the
       pairs of a palm vertex's bind place and where the frame drew it in
       the world. Game thread only. */
    void *fv_arm_tex[2];
    /* Every texture the arms have been drawn with this session: the arms
       keep their materials whatever the weapon, so a draw on one of them
       with skinning is the arms even in a frame whose scan has not run
       yet, or with another shader (a muzzle flash's light). */
    void *fv_arm_known[4];
    int fv_arm_known_n;
    int fv_arm_bind_asked, fv_arm_bind_mask;
    float fv_arm_bind[32];
    float fv_arm_a[2][MTL_ARM_PAIRS][3], fv_arm_b[2][MTL_ARM_PAIRS][3];
    int fv_arm_n[2];
    /* The first arm draw's own view projection, c8 to c11: the hands are
       handed on through it so they land where full view shows the arms,
       which have their own field of view. */
    float fv_arm_rows[4][4];
    int fv_arm_rows_ok;
    /* The frame's first view model draw's rows and depth range (mtl_fv_scan),
       for the scene depth's view model pixels. */
    float fv_vm_rows[4][4], fv_vm_vp[2];
    int fv_vm_ok;
    uint32_t fv_present_off;
    int fv_present_set;
    /* Textures the game copies the back buffer into (StretchRect), each
       retained: in stereo the right eye replays into twins of them, so an
       eye never reads the other's picture (Celerity's trail lays last
       frame's copy over this one). Appended by the recording thread,
       eye_copy_n published last; never shrinks. */
    void *eye_copy[MTL_EYE_COPIES];
    uint32_t eye_copy_n;
    int force_clear;
    uint32_t force_color;
    struct mtl_device *live_next;              /* device.m: the devices metal_d3d9_shutdown reaches */
} mtl_device;

/* vtables.c */
extern void *mtl_vt_d3d9[MTL_NSLOTS_IDirect3D9];
extern void *mtl_vt_device[MTL_NSLOTS_IDirect3DDevice9];
extern void *mtl_vt_surface[MTL_NSLOTS_IDirect3DSurface9];
extern void *mtl_vt_texture[MTL_NSLOTS_IDirect3DTexture9];
extern void *mtl_vt_cube[MTL_NSLOTS_IDirect3DCubeTexture9];
extern void *mtl_vt_vb[MTL_NSLOTS_IDirect3DVertexBuffer9];
extern void *mtl_vt_ib[MTL_NSLOTS_IDirect3DIndexBuffer9];
extern void *mtl_vt_decl[MTL_NSLOTS_IDirect3DVertexDeclaration9];
extern void *mtl_vt_vs[MTL_NSLOTS_IDirect3DVertexShader9];
extern void *mtl_vt_ps[MTL_NSLOTS_IDirect3DPixelShader9];
extern void *mtl_vt_query[MTL_NSLOTS_IDirect3DQuery9];
void mtl_vtables_build(void);
void mtl_fill_d3d9(void);    /* adapter.m */
void mtl_fill_device(void);  /* device.m */
void mtl_fill_resources(void); /* resources.c */

/* adapter.m: logging, the runtime's stop hook, formats */
void mtl_log(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void mtl_unexpected(const char *method) __attribute__((noreturn));
uint32_t mtl_fmt_bits(uint32_t f);
int mtl_fmt_dxt(uint32_t f);
uint64_t mtl_fmt_pitch(uint32_t f, uint32_t w);
uint64_t mtl_fmt_rows(uint32_t f, uint32_t h);
void mtl_caps(uint32_t out[76]);
HRESULT mtl_display_mode_of(mtl_display_mode *m);

/* refcounting, resources.c */
mtl_obj *mtl_root(mtl_obj *o);
uint32_t mtl_addref(mtl_obj *o);
uint32_t mtl_release(mtl_obj *o);
void mtl_hold(mtl_obj *o);
void mtl_unhold(mtl_obj *o);
void mtl_obj_init(mtl_obj *o, void *const *vtbl, mtl_kind kind, mtl_device *dev);

/* resources.c: creation, used by the device */
mtl_surface *mtl_surface_new(mtl_device *d, uint32_t w, uint32_t h, uint32_t fmt, uint32_t pool, uint32_t usage);
uint8_t *mtl_surface_mem(mtl_surface *s);
HRESULT mtl_texture_new(mtl_device *d, int cube, uint32_t w, uint32_t h, uint32_t levels, uint32_t usage, uint32_t fmt,
                        uint32_t pool, mtl_texture **out);
HRESULT mtl_buffer_new(mtl_device *d, mtl_kind kind, uint32_t len, uint32_t usage, uint32_t fmt_or_fvf, uint32_t pool,
                       mtl_buffer **out);
HRESULT mtl_blob_new(mtl_device *d, mtl_kind kind, const void *src, mtl_blob **out);
HRESULT mtl_query_new(mtl_device *d, uint32_t type, mtl_query **out);

/* device.m: the GPU side resources.c needs */
void mtl_surface_gpu_free(mtl_surface *s);
void mtl_surface_sync(mtl_surface *s);
int mtl_surface_readback_buffer(mtl_surface *s);
void mtl_device_destroy(mtl_device *d);

/* gpu.m */
int mtl_busy(mtl_device *d, uint64_t serial);
int mtl_buffer_gpu_init(mtl_buffer *b);
void mtl_buffer_lock_prepare(mtl_buffer *b, uint32_t flags);
void mtl_buffer_gpu_free(mtl_buffer *b);
void mtl_texture_gpu_free(mtl_texture *t);
int mtl_format_is_depth(uint32_t f);
uint32_t mtl_color_pixel_format(const mtl_device *d, uint32_t f); /* an MTLPixelFormat */
uint32_t mtl_depth_pixel_format(void);

/* frame.m: recorded lists. The game thread records each D3D9 frame as a
   list of commands holding everything by value; a list is closed by
   Present or by a readback that must be waited for, and is then replayed
   into a command buffer, on the render thread or (RT_METAL_INLINE=1) on
   the game thread as each command is recorded. A list's serial is the one
   resources compare against d->completed. */
int mtl_frames_init(mtl_device *d);
void mtl_frames_destroy(mtl_device *d);  /* drains, stops the render thread, frees the lists */
void *mtl_rec_cmd(mtl_device *d, uint32_t type, uint32_t bytes); /* a zeroed command in the open list */
void mtl_rec_end(mtl_device *d);                /* inline mode: encode what was recorded so far */
void mtl_rec_keep(mtl_device *d, void *objc);   /* the list retains an Objective-C object until it completes */
void mtl_rec_hold(mtl_device *d, mtl_blob *b);  /* and holds a shader the replay compiles from */
uint64_t mtl_submit(mtl_device *d, int present); /* closes the open list; returns its serial */
void mtl_wait_serial(mtl_device *d, uint64_t serial); /* closes the list if it is the open one, then waits */
void mtl_drain(mtl_device *d);
int mtl_threaded(const mtl_device *d);
int mtl_frames_shot_soon(const mtl_device *d); /* a full view shot may take the frame being recorded */
uint8_t *mtl_rec_base(mtl_device *d); /* the open list's commands, for an offset into it (moves as it grows) */

/* draw.m */
void mtl_fill_draw(void);
/* At each submit, before the list leaves the game thread: marks the 3D
   sky's world draws in it (full view puts them at infinity) and notes
   whether the frame drew a view model. cmds is the list's commands. */
void mtl_fv_scan(mtl_device *d, uint8_t *cmds, int present);
/* After the scan of the list with the Present: the third person aim into
   its Present (draw.m). */
void mtl_fv_aim(mtl_device *d, uint8_t *cmds);
int mtl_draw_init(mtl_device *d);
void mtl_draw_destroy(mtl_device *d);
void mtl_draw_finish(mtl_device *d); /* saves the pipeline cache and logs its summary, keeping the device usable */
void mtl_end_pass(mtl_device *d);  /* applies pending clears and ends the open encoder */
HRESULT mtl_clear(mtl_device *d, uint32_t count, const mtl_rect *rects, uint32_t flags, uint32_t color, float z,
                  uint32_t stencil);
void mtl_state_release(mtl_device *d);
void mtl_state_reset(mtl_device *d); /* Reset: unbinds everything and restores the creation defaults */
void mtl_blob_gpu_free(mtl_blob *b);
uint64_t mtl_next_uid(void);

/* adapter.m -> device.m */
HRESULT mtl_device_create(mtl_d3d9 *d3d, uint32_t adapter, uint32_t type, void *focus, uint32_t flags,
                          mtl_present_params *pp, mtl_device **out);

/* Exports. Direct3DCreate9 is the entry DXVK exports, with the same
   signature; metal_d3d9_set_host is ours, so the runtime's log and stop
   carry our messages. Before it is called, both go to stderr. */
D9_EXPORT void *Direct3DCreate9(uint32_t sdk_version);
D9_EXPORT void metal_d3d9_set_host(void (*log)(const char *msg), void (*stop)(const char *msg));

/* For the process exit: the game never releases its device when it
   quits, so what a release would leave on disk (the pipeline archive)
   and in the log is done here for every live device instead. Devices
   stay usable and may still be released, which then saves only what is
   new since; calling it again is harmless. */
D9_EXPORT void metal_d3d9_shutdown(void);

#endif
