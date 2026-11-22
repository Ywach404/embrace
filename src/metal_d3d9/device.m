/*
 * IDirect3DDevice9 on Metal: the device, a swap chain on a CAMetalLayer,
 * Present, render and depth targets, and the readback the bridge's
 * capture hooks use (GetBackBuffer, CreateOffscreenPlainSurface,
 * GetRenderTargetData, LockRect). Clear, the draws and the state they
 * read are in draw.m.
 *
 * The back buffer is a private BGRA8 texture, which is X8R8G8B8 byte for
 * byte, so a readback needs no swizzle. Passes, clears and readback blits
 * are recorded into the open list (frame.m) that Present closes. Each
 * list has a serial; resources remember the serial of their last use, and
 * the completion handler publishes the highest finished one, which is how
 * gpu.m and the upload ring know what the GPU may still be reading.
 *
 * Every call arrives on the guest main thread, which is the process main
 * thread, so the layer is created and configured in place. On visionOS it
 * is not: the host app owns the main thread and the guest runs on a
 * thread of its own, so the view, the layer's properties and the window
 * size are handled on the main thread (on_main), as UIKit requires, and
 * the game thread waits for each. The drawable
 * is taken and presented where the list is replayed: on the render
 * thread, or here in inline mode. Reset and release drain the render
 * thread before they touch the layer or the targets.
 *
 * RT_METAL_CLEAR=RRGGBB forces the colour of every Clear of a render
 * target, so the first captured frame shows a Clear and its readback
 * rather than memory that happens to be zero.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#include <TargetConditionals.h>

#include "mtl_objc.h"

#if TARGET_OS_VISION
#include <SDL3/SDL_error.h>
#include <SDL3/SDL_metal.h>
#include <SDL3/SDL_video.h>
#endif
#include <dlfcn.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#define MTLD(d) ((__bridge id<MTLDevice>)(d)->mtl)
#define QUEUE(d) ((__bridge id<MTLCommandQueue>)(d)->queue)
#define TEX(s) ((__bridge id<MTLTexture>)(s)->tex)
#define LAYER(d) ((__bridge CAMetalLayer *)(d)->layer)

static void cf_drop(void **p)
{
    if (*p) CFRelease(*p);
    *p = NULL;
}

/* SDL is the executable's; it is looked up, not linked, so this library
   never brings a second libSDL3 into the process (the bridge's image check
   counts them) and loads without SDL in the headless test. */
#if TARGET_OS_VISION
/* On visionOS SDL and this backend are both linked into the app, and an
   app's executable need not export SDL's symbols (and dead stripping may
   drop those only a lookup would reach), so dlsym cannot be relied on. The
   same names resolve here to direct references instead, which keeps them
   linked and leaves every call site as it is on the Mac. */
static void *sdl_fn(const char *name)
{
    static const struct {
        const char *name;
        void *fn;
    } k[] = {
        {"SDL_GetWindowSizeInPixels", (void *)SDL_GetWindowSizeInPixels},
        {"SDL_Metal_CreateView", (void *)SDL_Metal_CreateView},
        {"SDL_Metal_GetLayer", (void *)SDL_Metal_GetLayer},
        {"SDL_Metal_DestroyView", (void *)SDL_Metal_DestroyView},
        {"SDL_GetError", (void *)SDL_GetError},
    };
    for (size_t i = 0; i < sizeof k / sizeof k[0]; i++)
        if (!strcmp(k[i].name, name)) return k[i].fn;
    mtl_log("%s is not one of the SDL functions linked for visionOS", name);
    return NULL;
}

/* Runs b on the main thread and waits. The host's main thread never waits
   for the game's, so this cannot deadlock; on the main thread already it
   runs in place. Only setup, reset and release come here: nextDrawable
   and present stay on the render thread, which Metal allows. */
static void on_main(void (^b)(void))
{
    if (pthread_main_np()) b();
    else dispatch_sync(dispatch_get_main_queue(), b);
}
#else
static void *sdl_fn(const char *name)
{
    void *f = dlsym(RTLD_DEFAULT, name);
    if (!f) mtl_log("%s not found in the process", name);
    return f;
}
#endif

void mtl_surface_gpu_free(mtl_surface *s)
{
    /* A readback still in flight into buf is safe: its list keeps buf. */
    s->pending_serial = 0;
    cf_drop(&s->tex);
    cf_drop(&s->buf);
}

/* Waits for the readback into a surface: the list that holds it is
   closed if it is still being recorded, and replayed to its end. This is
   the drain the capture hooks rely on to read exactly the frame they
   asked for. */
void mtl_surface_sync(mtl_surface *s)
{
    if (!s->pending_serial) return;
    uint64_t serial = s->pending_serial;
    s->pending_serial = 0;
    mtl_wait_serial(s->o.dev, serial);
}

/* Gives a system memory surface its memory as a shared MTLBuffer, so a
   readback can blit straight into it and LockRect only waits. */
int mtl_surface_readback_buffer(mtl_surface *s)
{
    if (s->buf) return 1;
    if (s->mem || !s->o.dev) return 0;
    @autoreleasepool {
        id<MTLBuffer> b = [MTLD(s->o.dev) newBufferWithLength:(NSUInteger)s->pitch * s->rows + 16
                                                      options:MTLResourceStorageModeShared];
        if (!b) return 0;
        memset(b.contents, 0, b.length);
        s->mem = b.contents;
        s->buf = (void *)CFBridgingRetain(b);
    }
    return 1;
}

static mtl_surface *gpu_target(mtl_device *d, uint32_t w, uint32_t h, uint32_t fmt)
{
    mtl_surface *s = mtl_surface_new(d, w, h, fmt, D3DPOOL_DEFAULT, D3DUSAGE_RENDERTARGET);
    if (!s) return NULL;
    @autoreleasepool {
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                      width:w
                                                                                     height:h
                                                                                  mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModePrivate;
        id<MTLTexture> t = [MTLD(d) newTextureWithDescriptor:td];
        if (!t) {
            free(s);
            return NULL;
        }
        s->tex = (void *)CFBridgingRetain(t);
    }
    s->no_cpu = 1;
    /* Held by the device, not by the guest: GetBackBuffer hands out public
       references on top, and the last one the guest releases returns 0. */
    mtl_hold(&s->o);
    return s;
}

static mtl_surface *depth_target(mtl_device *d, uint32_t w, uint32_t h, uint32_t fmt)
{
    mtl_surface *s = mtl_surface_new(d, w, h, fmt, D3DPOOL_DEFAULT, D3DUSAGE_DEPTHSTENCIL);
    if (!s) return NULL;
    if (!mtl_surface_make_target(d, s)) {
        free(s);
        return NULL;
    }
    mtl_hold(&s->o);
    return s;
}

static void set_rt(mtl_device *d, uint32_t i, mtl_surface *s)
{
    if (d->rt[i] != s) mtl_end_pass(d);
    mtl_hold(s ? &s->o : NULL);
    mtl_unhold(d->rt[i] ? &d->rt[i]->o : NULL);
    d->rt[i] = s;
    if (!i && s) d->vp = (mtl_viewport){0, 0, s->width, s->height, 0.0f, 1.0f};
}

static void set_ds(mtl_device *d, mtl_surface *s)
{
    if (d->ds != s) mtl_end_pass(d);
    mtl_hold(s ? &s->o : NULL);
    mtl_unhold(d->ds ? &d->ds->o : NULL);
    d->ds = s;
}

/* Fills in what the game left to the driver (a zero size takes the
   window's pixel size, an unknown format X8R8G8B8, a zero count one
   buffer) and checks what the backend supports: one 32-bit back buffer. */
static HRESULT settle_params(mtl_device *d, mtl_present_params *pp)
{
    if (!pp->BackBufferWidth || !pp->BackBufferHeight) {
        int w = 0, h = 0;
        void (*size)(void *, int *, int *) = d->window ? sdl_fn("SDL_GetWindowSizeInPixels") : NULL;
#if TARGET_OS_VISION
        /* SDL reads the size from the window's UIView. */
        __block int bw = 0, bh = 0;
        if (size) on_main(^{
            size(d->window, &bw, &bh);
        });
        w = bw;
        h = bh;
#else
        if (size) size(d->window, &w, &h);
#endif
        if (!pp->BackBufferWidth) pp->BackBufferWidth = w > 0 ? (uint32_t)w : 640;
        if (!pp->BackBufferHeight) pp->BackBufferHeight = h > 0 ? (uint32_t)h : 480;
    }
    if (pp->BackBufferFormat == D3DFMT_UNKNOWN) pp->BackBufferFormat = D3DFMT_X8R8G8B8;
    if (!pp->BackBufferCount) pp->BackBufferCount = 1;
    if (pp->BackBufferFormat != D3DFMT_X8R8G8B8 && pp->BackBufferFormat != D3DFMT_A8R8G8B8) {
        mtl_log("back buffer format %u is not X8R8G8B8 or A8R8G8B8", pp->BackBufferFormat);
        return D3DERR_NOTAVAILABLE;
    }
    if (pp->MultiSampleType > 1) {
        mtl_log("multisample type %u requested; the backend renders single sampled", pp->MultiSampleType);
        pp->MultiSampleType = 0;
        pp->MultiSampleQuality = 0;
    }
    return D3D_OK;
}

static void configure_layer_here(mtl_device *d)
{
    if (!d->layer) return;
    CAMetalLayer *l = LAYER(d);
    l.drawableSize = CGSizeMake(d->pp.BackBufferWidth, d->pp.BackBufferHeight);
#if !TARGET_OS_VISION
    /* macOS only: on visionOS the system compositor paces every drawable. */
    l.displaySyncEnabled = d->pp.PresentationInterval != D3DPRESENT_INTERVAL_IMMEDIATE;
#endif
}

/* A layer property set off the main thread opens an implicit transaction
   there that UIKit's compositing does not expect, so on visionOS the
   drawable size goes through the main thread like the rest of the setup. */
static void configure_layer(mtl_device *d)
{
#if TARGET_OS_VISION
    on_main(^{
        configure_layer_here(d);
    });
#else
    configure_layer_here(d);
#endif
}

static int attach_layer_here(mtl_device *d)
{
    void *(*create)(void *) = sdl_fn("SDL_Metal_CreateView");
    void *(*get)(void *) = sdl_fn("SDL_Metal_GetLayer");
    if (!create || !get) return -1;
    d->view = create(d->window);
    void *layer = d->view ? get(d->view) : NULL;
    if (!layer) {
        const char *(*err)(void) = sdl_fn("SDL_GetError");
        mtl_log("no CAMetalLayer on the window: %s", err ? err() : "unknown");
        return -1;
    }
    d->layer = (void *)CFBridgingRetain((__bridge CAMetalLayer *)layer);
    CAMetalLayer *l = LAYER(d);
    l.device = MTLD(d);
    l.pixelFormat = MTLPixelFormatBGRA8Unorm;
    /* Present blits the back buffer into the drawable, which a
       framebuffer-only drawable does not allow. */
    l.framebufferOnly = NO;
    l.maximumDrawableCount = 3;
    l.contentsGravity = kCAGravityResizeAspect;
    configure_layer_here(d);
    return 0;
}

/* SDL_Metal_CreateView adds a UIView to the window on visionOS, and
   SDL_GetError is per thread, so the whole attach is one hop. */
static int attach_layer(mtl_device *d)
{
#if TARGET_OS_VISION
    __block int r = -1;
    on_main(^{
        r = attach_layer_here(d);
    });
    return r;
#else
    return attach_layer_here(d);
#endif
}

static HRESULT make_targets(mtl_device *d)
{
    mtl_surface *bb = gpu_target(d, d->pp.BackBufferWidth, d->pp.BackBufferHeight, d->pp.BackBufferFormat);
    if (!bb) return E_OUTOFMEMORY;
    mtl_surface *ds = NULL;
    if (d->pp.EnableAutoDepthStencil) {
        ds = depth_target(d, d->pp.BackBufferWidth, d->pp.BackBufferHeight, d->pp.AutoDepthStencilFormat);
        if (!ds) {
            mtl_unhold(&bb->o);
            return E_OUTOFMEMORY;
        }
    }
    mtl_surface *old_bb = d->back, *old_ds = d->auto_ds;
    d->back = bb;
    d->auto_ds = ds;
    for (uint32_t i = 1; i < MTL_MAX_RT; i++) set_rt(d, i, NULL);
    set_rt(d, 0, bb);
    set_ds(d, ds);
    mtl_unhold(old_bb ? &old_bb->o : NULL);
    mtl_unhold(old_ds ? &old_ds->o : NULL);
    return D3D_OK;
}

static void read_force_clear(mtl_device *d)
{
    const char *v = getenv("RT_METAL_CLEAR");
    char *end = NULL;
    unsigned long c = v ? strtoul(v, &end, 16) : 0;
    d->force_clear = v && *v && end && !*end && strlen(v) == 6;
    d->force_color = 0xFF000000u | (uint32_t)c;
    if (v && !d->force_clear) mtl_log("RT_METAL_CLEAR=%s ignored: it takes six hex digits, RRGGBB", v);
    if (d->force_clear) mtl_log("RT_METAL_CLEAR: every render target clear is 0x%08X", d->force_color);
}

/* Live devices, for metal_d3d9_shutdown. The lock is held across the
   whole shutdown, so a release on another thread waits for it rather
   than freeing a device it is draining. */
static pthread_mutex_t s_live_mu = PTHREAD_MUTEX_INITIALIZER;
static mtl_device *s_live;

static void live_add(mtl_device *d)
{
    pthread_mutex_lock(&s_live_mu);
    d->live_next = s_live;
    s_live = d;
    pthread_mutex_unlock(&s_live_mu);
}

static void live_remove(mtl_device *d)
{
    pthread_mutex_lock(&s_live_mu);
    for (mtl_device **p = &s_live; *p; p = &(*p)->live_next)
        if (*p == d) {
            *p = d->live_next;
            break;
        }
    d->live_next = NULL;
    pthread_mutex_unlock(&s_live_mu);
}

void metal_d3d9_shutdown(void)
{
    pthread_mutex_lock(&s_live_mu);
    unsigned n = 0;
    for (mtl_device *d = s_live; d; d = d->live_next, n++) {
        /* Pipelines are made where lists are replayed, so the render
           thread must have replayed everything before the cache can know
           what this run used. */
        if (d->fq && d->draw) mtl_drain(d);
        mtl_draw_finish(d);
    }
    pthread_mutex_unlock(&s_live_mu);
    mtl_log("shutdown: %u live device%s finished", n, n == 1 ? "" : "s");
    mtl_immersive_report();
}

HRESULT mtl_device_create(mtl_d3d9 *d3d, uint32_t adapter, uint32_t type, void *focus, uint32_t flags,
                          mtl_present_params *pp, mtl_device **out)
{
    (void)adapter;
    (void)type;
    mtl_device *d = calloc(1, sizeof *d);
    if (!d) return E_OUTOFMEMORY;
    mtl_obj_init(&d->o, mtl_vt_device, MTL_K_DEVICE, d);
    mtl_addref(&d->o);
    d->d3d = d3d;
    d->flags = flags;
    d->window = pp->hDeviceWindow ? pp->hDeviceWindow : focus;
    HRESULT hr = settle_params(d, pp);
    if (hr) {
        free(d);
        return hr;
    }
    d->pp = *pp;
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        if (!dev || !q) {
            mtl_log("no Metal device or command queue");
            free(d);
            return D3DERR_NOTAVAILABLE;
        }
        d->mtl = (void *)CFBridgingRetain(dev);
        d->queue = (void *)CFBridgingRetain(q);
        mtl_caps_query(dev, &d->caps);
        d->serial = 1;
        hr = mtl_frames_init(d) && mtl_draw_init(d) ? D3D_OK : E_OUTOFMEMORY;
        if (!hr) hr = make_targets(d);
        if (!hr && d->window && attach_layer(d)) hr = D3DERR_NOTAVAILABLE;
        if (hr) {
            mtl_device_destroy(d);
            return hr;
        }
        mtl_log("device %ux%u format %u windowed %d interval 0x%X auto depth %u on %s, %s", pp->BackBufferWidth,
                pp->BackBufferHeight, pp->BackBufferFormat, pp->Windowed, pp->PresentationInterval,
                pp->EnableAutoDepthStencil ? pp->AutoDepthStencilFormat : 0, dev.name.UTF8String,
                d->layer ? "CAMetalLayer on the SDL window" : "no window (offscreen)");
    }
    read_force_clear(d);
    mtl_hold(&d3d->o);
    live_add(d);
    *out = d;
    return D3D_OK;
}

void mtl_device_destroy(mtl_device *d)
{
    /* Out of the list first, so the exit hook never reaches a device
       that is going away. A device whose creation failed was never in it. */
    live_remove(d);
    if (d->fq && d->draw) mtl_drain(d);
    mtl_log("device released: clears %llu draws %llu presents %llu, %llu without a drawable",
            (unsigned long long)d->clears, (unsigned long long)d->draws, (unsigned long long)d->presents,
            (unsigned long long)mtl_frames_no_drawable(d));
    /* Everything recorded has run: the render thread stops here, before
       any target, layer or cache it replays into goes. */
    mtl_frames_destroy(d);
    mtl_state_release(d);
    for (uint32_t i = 0; i < MTL_MAX_RT; i++) set_rt(d, i, NULL);
    set_ds(d, NULL);
    mtl_surface *bb = d->back, *ds = d->auto_ds;
    d->back = d->auto_ds = NULL;
    mtl_unhold(bb ? &bb->o : NULL);
    mtl_unhold(ds ? &ds->o : NULL);
    cf_drop(&d->layer);
    if (d->view) {
        void (*destroy)(void *) = sdl_fn("SDL_Metal_DestroyView");
#if TARGET_OS_VISION
        /* Removes the view from its UIKit window. */
        void *view = d->view;
        if (destroy) on_main(^{
            destroy(view);
        });
#else
        if (destroy) destroy(d->view);
#endif
        d->view = NULL;
    }
    cf_drop(&d->clear_pso);
    mtl_draw_destroy(d);
    cf_drop(&d->queue);
    cf_drop(&d->mtl);
    mtl_d3d9 *d3d = d->d3d;
    free(d);
    if (d3d) mtl_unhold(&d3d->o);
}

/* Refcounting and the trivial device methods */

static uint32_t dev_addref(mtl_device *d) { return mtl_addref(&d->o); }
static uint32_t dev_release(mtl_device *d) { return mtl_release(&d->o); }
static HRESULT dev_ok(mtl_device *d) { (void)d; return D3D_OK; }

static uint32_t dev_avail_mem(mtl_device *d)
{
    uint64_t left = MTL_VIDMEM_CAP > d->vidmem ? MTL_VIDMEM_CAP - d->vidmem : 0;
    return (uint32_t)(left & ~0xFFFFFull);
}

static HRESULT dev_get_d3d(mtl_device *d, mtl_d3d9 **pp)
{
    if (!pp) return D3DERR_INVALIDCALL;
    mtl_addref(&d->d3d->o);
    *pp = d->d3d;
    return D3D_OK;
}

static HRESULT dev_caps(mtl_device *d, uint32_t *caps)
{
    (void)d;
    if (!caps) return D3DERR_INVALIDCALL;
    mtl_caps(caps);
    return D3D_OK;
}

static HRESULT dev_display_mode(mtl_device *d, uint32_t swap, mtl_display_mode *m)
{
    (void)d;
    if (swap || !m) return D3DERR_INVALIDCALL;
    return mtl_display_mode_of(m);
}

static int32_t dev_showcursor(mtl_device *d, int32_t show)
{
    int32_t old = (int32_t)d->cursor;
    d->cursor = show != 0;
    return old;
}

static HRESULT dev_reset(mtl_device *d, mtl_present_params *pp)
{
    if (!pp) return D3DERR_INVALIDCALL;
    HRESULT hr = settle_params(d, pp);
    if (hr) return hr;
    mtl_drain(d); /* the render thread is idle before the layer and targets change */
    @autoreleasepool {
        mtl_present_params old = d->pp;
        d->pp = *pp;
        hr = make_targets(d);
        if (hr) {
            d->pp = old;
            return hr;
        }
        configure_layer(d);
    }
    /* D3D9 Reset returns every state to its creation default and unbinds
       what was bound, as DXVK's ResetState does; the targets and viewport
       were set to the new back buffer above. */
    mtl_state_reset(d);
    mtl_log("Reset %ux%u format %u windowed %d", pp->BackBufferWidth, pp->BackBufferHeight, pp->BackBufferFormat,
            pp->Windowed);
    return D3D_OK;
}

static HRESULT dev_backbuffer(mtl_device *d, uint32_t swap, uint32_t i, uint32_t type, mtl_surface **pp)
{
    (void)type;
    if (!pp) return D3DERR_INVALIDCALL;
    *pp = NULL;
    if (swap || i || !d->back) return D3DERR_INVALIDCALL;
    mtl_addref(&d->back->o);
    *pp = d->back;
    return D3D_OK;
}

static void dev_gamma(mtl_device *d, uint32_t swap, uint32_t flags, const void *ramp)
{
    (void)d;
    (void)swap;
    (void)flags;
    (void)ramp;
}

/* Present: blit the back buffer into the layer's next drawable and show
   it. Pacing is the layer's own: nextDrawable waits while all three
   drawables are in flight, and displaySyncEnabled follows the game's
   presentation interval. With no window (the headless test) the list is
   only replayed and committed. This runs where the list is replayed.
   In immersive mode the frame goes to the compositor's ring, and also to
   the layer while the host's window is still up, so the window shows the
   game until the space has shown it (space.h). Once the window is
   dismissed the layer is left alone: its scene is gone, and nextDrawable
   on a layer nobody displays blocks for up to a second. */
void mtl_emit_present(mtl_device *d, mtl_emit *e, void *back, void *right, uint64_t presented_ns)
{
    if (mtl_immersive_on()) {
        mtl_immersive_present(d, e, back, right, presented_ns);
        if (!mtl_immersive_window_too()) return;
    }
    if (!d->layer || !back) return;
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)e->cb;
    id<CAMetalDrawable> dr = [LAYER(d) nextDrawable];
    if (!dr) {
        mtl_frames_note_no_drawable(d);
        return;
    }
    id<MTLTexture> bb = (__bridge id<MTLTexture>)back, out = dr.texture;
    id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
    [bl copyFromTexture:bb
             sourceSlice:0
             sourceLevel:0
            sourceOrigin:MTLOriginMake(0, 0, 0)
              sourceSize:MTLSizeMake(MIN(bb.width, out.width), MIN(bb.height, out.height), 1)
               toTexture:out
        destinationSlice:0
        destinationLevel:0
       destinationOrigin:MTLOriginMake(0, 0, 0)];
    [bl endEncoding];
    [cb presentDrawable:dr];
}

/* Present closes the frame's list and returns: the render thread replays
   it and presents. The game thread waits here only when two frames are
   already in flight. */
static HRESULT dev_present(mtl_device *d, const mtl_rect *src, const mtl_rect *dst, void *hwnd, const void *dirty)
{
    (void)dirty;
    static int noted;
    if ((src || dst || (hwnd && hwnd != d->window)) && !noted++)
        mtl_log("Present with source or destination rectangles or another window: the backend presents the whole back buffer");
    mtl_rec_present(d, d->back ? d->back->tex : NULL);
    mtl_submit(d, 1);
    d->presents++;
    return D3D_OK;
}

/* Readback: the capture hooks' path. A back buffer source is blitted into
   the destination's shared buffer in the list being recorded; the
   destination's LockRect waits for that list, closing it first if it is
   still open, so the bridge's three-deep luma ring does not stall a
   frame and a capture still reads exactly the frame it asked for. */
static HRESULT dev_rtdata(mtl_device *d, mtl_surface *src, mtl_surface *dst)
{
    if (!src || !dst || dst->pool != D3DPOOL_SYSTEMMEM || dst->locked) return D3DERR_INVALIDCALL;
    if (src->width != dst->width || src->height != dst->height || mtl_fmt_bits(src->format) != mtl_fmt_bits(dst->format))
        return D3DERR_INVALIDCALL;
    mtl_surface_sync(dst);
    uint32_t level, slice;
    id<MTLTexture> stex = mtl_surface_texture(src, &level, &slice);
    if (!stex) {
        uint8_t *s = mtl_surface_mem(src), *m = mtl_surface_mem(dst);
        if (!s || !m) return D3DERR_INVALIDCALL;
        for (uint32_t y = 0; y < dst->rows; y++)
            memcpy(m + (size_t)y * dst->pitch, s + (size_t)y * src->pitch, MIN(src->pitch, dst->pitch));
        return D3D_OK;
    }
    @autoreleasepool {
        id<MTLBuffer> staging = nil;
        id<MTLBuffer> out = mtl_surface_readback_buffer(dst) ? (__bridge id<MTLBuffer>)dst->buf : nil;
        if (!out) {
            staging = [MTLD(d) newBufferWithLength:(NSUInteger)dst->pitch * dst->rows options:MTLResourceStorageModeShared];
            if (!staging) return E_OUTOFMEMORY;
            out = staging;
        }
        mtl_rec_readback(d, (__bridge void *)stex, level, slice, src->width, src->height, (__bridge void *)out, dst->pitch,
                         dst->rows);
        if (staging) {
            mtl_wait_serial(d, d->serial);
            uint8_t *m = mtl_surface_mem(dst);
            if (!m) return E_OUTOFMEMORY;
            memcpy(m, staging.contents, (size_t)dst->pitch * dst->rows);
        } else {
            dst->pending_serial = d->serial;
            /* Inline mode commits here, as it always has. */
            if (!mtl_threaded(d)) mtl_submit(d, 0);
        }
    }
    return D3D_OK;
}

/* Creation */

static HRESULT dev_create_texture(mtl_device *d, uint32_t w, uint32_t h, uint32_t levels, uint32_t usage, uint32_t fmt,
                                  uint32_t pool, mtl_texture **pp, void *shared)
{
    (void)shared;
    if (!pp) return D3DERR_INVALIDCALL;
    return mtl_texture_new(d, 0, w, h, levels, usage, fmt, pool, pp);
}

static HRESULT dev_create_cube(mtl_device *d, uint32_t edge, uint32_t levels, uint32_t usage, uint32_t fmt, uint32_t pool,
                               mtl_texture **pp, void *shared)
{
    (void)shared;
    if (!pp) return D3DERR_INVALIDCALL;
    return mtl_texture_new(d, 1, edge, edge, levels, usage, fmt, pool, pp);
}

static HRESULT dev_create_vb(mtl_device *d, uint32_t len, uint32_t usage, uint32_t fvf, uint32_t pool, mtl_buffer **pp,
                             void *shared)
{
    (void)shared;
    if (!pp) return D3DERR_INVALIDCALL;
    return mtl_buffer_new(d, MTL_K_VB, len, usage, fvf, pool, pp);
}

static HRESULT dev_create_ib(mtl_device *d, uint32_t len, uint32_t usage, uint32_t fmt, uint32_t pool, mtl_buffer **pp,
                             void *shared)
{
    (void)shared;
    if (!pp) return D3DERR_INVALIDCALL;
    return mtl_buffer_new(d, MTL_K_IB, len, usage, fmt, pool, pp);
}

static HRESULT surface_out(mtl_surface *s, mtl_surface **pp)
{
    *pp = s;
    if (!s) return E_OUTOFMEMORY;
    mtl_addref(&s->o);
    return D3D_OK;
}

static HRESULT dev_create_rt(mtl_device *d, uint32_t w, uint32_t h, uint32_t fmt, uint32_t ms, uint32_t msq,
                             int32_t lockable, mtl_surface **pp, void *shared)
{
    (void)ms;
    (void)msq;
    (void)lockable;
    (void)shared;
    if (!pp) return D3DERR_INVALIDCALL;
    mtl_surface *s = mtl_surface_new(d, w, h, fmt, D3DPOOL_DEFAULT, D3DUSAGE_RENDERTARGET);
    if (s && !mtl_surface_make_target(d, s)) {
        free(s);
        s = NULL;
    }
    return surface_out(s, pp);
}

static HRESULT dev_create_ds(mtl_device *d, uint32_t w, uint32_t h, uint32_t fmt, uint32_t ms, uint32_t msq,
                             int32_t discard, mtl_surface **pp, void *shared)
{
    (void)ms;
    (void)msq;
    (void)discard;
    (void)shared;
    if (!pp) return D3DERR_INVALIDCALL;
    mtl_surface *s = mtl_surface_new(d, w, h, fmt, D3DPOOL_DEFAULT, D3DUSAGE_DEPTHSTENCIL);
    if (s && !mtl_surface_make_target(d, s)) {
        free(s);
        s = NULL;
    }
    return surface_out(s, pp);
}

static HRESULT dev_offscreen(mtl_device *d, uint32_t w, uint32_t h, uint32_t fmt, uint32_t pool, mtl_surface **pp,
                             void *shared)
{
    (void)shared;
    if (!pp) return D3DERR_INVALIDCALL;
    mtl_surface *s = mtl_surface_new(d, w, h, fmt, pool, 0);
    if (s && pool == D3DPOOL_SYSTEMMEM && !mtl_fmt_dxt(fmt) && !mtl_surface_readback_buffer(s)) {
        free(s); /* not handed out yet, and nothing GPU side was made */
        s = NULL;
    }
    return surface_out(s, pp);
}

static HRESULT dev_create_decl(mtl_device *d, const void *elems, mtl_blob **pp)
{
    if (!pp) return D3DERR_INVALIDCALL;
    return mtl_blob_new(d, MTL_K_DECL, elems, pp);
}

static HRESULT dev_create_vs(mtl_device *d, const void *tokens, mtl_blob **pp)
{
    if (!pp) return D3DERR_INVALIDCALL;
    return mtl_blob_new(d, MTL_K_VS, tokens, pp);
}

static HRESULT dev_create_ps(mtl_device *d, const void *tokens, mtl_blob **pp)
{
    if (!pp) return D3DERR_INVALIDCALL;
    return mtl_blob_new(d, MTL_K_PS, tokens, pp);
}

static HRESULT dev_create_query(mtl_device *d, uint32_t type, mtl_query **pp)
{
    if (!pp) return D3D_OK; /* a support check: every type is answered */
    return mtl_query_new(d, type, pp);
}

/* Targets and viewport */

static HRESULT dev_set_rt(mtl_device *d, uint32_t i, mtl_surface *s)
{
    if (i >= MTL_MAX_RT || (!i && !s)) return D3DERR_INVALIDCALL;
    set_rt(d, i, s);
    return D3D_OK;
}

static HRESULT dev_get_rt(mtl_device *d, uint32_t i, mtl_surface **pp)
{
    if (!pp || i >= MTL_MAX_RT) return D3DERR_INVALIDCALL;
    *pp = d->rt[i];
    if (!d->rt[i]) return D3DERR_NOTFOUND;
    mtl_addref(&d->rt[i]->o);
    return D3D_OK;
}

static HRESULT dev_set_ds(mtl_device *d, mtl_surface *s)
{
    set_ds(d, s);
    return D3D_OK;
}

static HRESULT dev_get_ds(mtl_device *d, mtl_surface **pp)
{
    if (!pp) return D3DERR_INVALIDCALL;
    *pp = d->ds;
    if (!d->ds) return D3DERR_NOTFOUND;
    mtl_addref(&d->ds->o);
    return D3D_OK;
}

static HRESULT dev_set_viewport(mtl_device *d, const mtl_viewport *vp)
{
    if (!vp) return D3DERR_INVALIDCALL;
    d->vp = *vp;
    return D3D_OK;
}

static HRESULT dev_get_viewport(mtl_device *d, mtl_viewport *vp)
{
    if (!vp) return D3DERR_INVALIDCALL;
    *vp = d->vp;
    return D3D_OK;
}

static HRESULT dev_set_rs(mtl_device *d, uint32_t state, uint32_t value)
{
    if (state < 256) d->rs[state] = value;
    return D3D_OK;
}

static HRESULT dev_get_rs(mtl_device *d, uint32_t state, uint32_t *value)
{
    if (!value || state >= 256) return D3DERR_INVALIDCALL;
    *value = d->rs[state];
    return D3D_OK;
}

void mtl_fill_device(void)
{
    void **v = mtl_vt_device;
#define SET(m, f) v[SLOT_IDirect3DDevice9_##m] = (void *)(f)
    SET(AddRef, dev_addref);
    SET(Release, dev_release);
    SET(TestCooperativeLevel, dev_ok);
    SET(GetAvailableTextureMem, dev_avail_mem);
    SET(EvictManagedResources, dev_ok);
    SET(GetDirect3D, dev_get_d3d);
    SET(GetDeviceCaps, dev_caps);
    SET(GetDisplayMode, dev_display_mode);
    SET(ShowCursor, dev_showcursor);
    SET(Reset, dev_reset);
    SET(Present, dev_present);
    SET(GetBackBuffer, dev_backbuffer);
    SET(SetGammaRamp, dev_gamma);
    SET(CreateTexture, dev_create_texture);
    SET(CreateCubeTexture, dev_create_cube);
    SET(CreateVertexBuffer, dev_create_vb);
    SET(CreateIndexBuffer, dev_create_ib);
    SET(CreateRenderTarget, dev_create_rt);
    SET(CreateDepthStencilSurface, dev_create_ds);
    SET(GetRenderTargetData, dev_rtdata);
    SET(CreateOffscreenPlainSurface, dev_offscreen);
    SET(SetRenderTarget, dev_set_rt);
    SET(GetRenderTarget, dev_get_rt);
    SET(SetDepthStencilSurface, dev_set_ds);
    SET(GetDepthStencilSurface, dev_get_ds);
    SET(BeginScene, dev_ok);
    SET(EndScene, dev_ok);
    SET(SetViewport, dev_set_viewport);
    SET(GetViewport, dev_get_viewport);
    SET(SetRenderState, dev_set_rs);
    SET(GetRenderState, dev_get_rs);
    SET(CreateVertexDeclaration, dev_create_decl);
    SET(CreateVertexShader, dev_create_vs);
    SET(CreatePixelShader, dev_create_ps);
    SET(CreateQuery, dev_create_query);
#undef SET
    mtl_fill_draw();
}
