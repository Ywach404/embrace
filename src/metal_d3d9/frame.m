/*
 * Recorded lists and the render thread.
 *
 * The game thread does not encode Metal. Every D3D9 call that would draw,
 * clear, blit or present appends a command to the open list instead, with
 * what it needs captured by value: pipeline and depth keys, constants and
 * push data already written into the upload ring, and the exact storage
 * of every buffer and texture it reads, retained by the list. Present
 * closes the list and hands it to a render thread, which replays it into
 * a command buffer, takes the next drawable and presents. The game thread
 * waits only at Present (when two frames are already in flight) and when
 * a readback has to be read: never inside a draw or a lock, so the frame
 * trace's draws column stays time spent recording.
 *
 * A list is replayable more than once: nothing in it refers to device
 * state, and everything it reads stays as it was until its serial
 * completes, because a lock or upload that would change storage a list
 * still reads moves to new storage instead (gpu.m). The test switch that
 * replays each list twice, into the back buffer and into a twin, checks
 * that claim; stereo needs it per eye.
 *
 * Serials are the ones resources already compare against: d->serial
 * names the list being recorded, and a list's command buffer publishes
 * its serial to d->completed when the GPU finishes it. "Recorded but not
 * yet replayed" is simply a serial above completed, so the rename, ring
 * and free-list rules hold unchanged with a queue in between.
 *
 * Lists are retired on the game thread: the objects they retained are
 * released there and the device objects they held are let go there, since
 * mtl_unhold may destroy and only the game thread may do that. A retired
 * list's memory is reused, so memory stays bounded by the frames in
 * flight.
 *
 * RT_METAL_INLINE=1 keeps everything on the game thread: each command is
 * replayed as soon as it is recorded, into a command buffer the game
 * thread commits, as the backend did before the render thread existed.
 * It is the comparison and the fallback.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "mtl_objc.h"

#include <mach/mach_time.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define QUEUE(d) ((__bridge id<MTLCommandQueue>)(d)->queue)
#define MTLD(d) ((__bridge id<MTLDevice>)(d)->mtl)

/* Frames the game may run ahead of the GPU: two in a window, where the
   layer's drawables pace the game and a second frame keeps the GPU fed;
   one while immersive, where the compositor paces itself and every frame
   queued in front of the GPU only adds a frame to the age it is shown at
   (immersive.m). The array keeps room for the deeper of the two. */
#define IN_FLIGHT 2
#define IN_FLIGHT_IMMERSIVE 1

typedef struct mtl_seg {
    struct mtl_seg *next_q;   /* the render thread's queue */
    struct mtl_seg *next_sub; /* submitted, waiting to be retired */
    uint8_t *cmd;
    size_t len, cap, emitted;
    void **keep;
    uint32_t nkeep, capkeep;
    mtl_blob **hold;
    uint32_t nhold, caphold;
    uint64_t serial;
    int twice;
    int present; /* the list ends with the frame's Present */
    int fullview; /* the frame this list starts is meant for the eyes' own views (game thread's choice) */
} mtl_seg;

typedef struct mtl_frames {
    pthread_mutex_t mu;
    pthread_cond_t work, done;
    pthread_t thread;
    int threaded, running, stop;
    mtl_seg *cur;
    mtl_seg *q_head, *q_tail;
    mtl_seg *sub_head, *sub_tail;
    mtl_seg *spare;
    uint32_t nsegs, nsub;
    uint64_t replayed; /* the last list the render thread has finished reading, set under mu */
    uint64_t presented[IN_FLIGHT];
    mtl_emit inl;          /* inline mode's replay, into inl.cb */
    int twice;             /* test: replay every list a second time into the twin */
    void *twin[2];         /* id<MTLTexture>: colour and depth twins of the back buffer and its depth */
    /* Stereo: each frame is replayed once per eye, the left into the back
       buffer as recorded and the right into these twins of it and its
       depth surface. Whether a frame is per eye is decided at its first
       list and holds to its Present, on the thread that replays, so a
       switch never splits a frame between the two. */
    void *right[2];
    void *copy_src[MTL_EYE_COPIES], *copy_twin[MTL_EYE_COPIES]; /* the right eye's twins of d->eye_copy */
    int stereo, mid_frame;
    float left_x, right_x; /* the eyes' offsets, latched with the choice for the whole frame */
    float screen_a;        /* the a the frame's eye images are shown with, latched with them */
    mtl_fullview_frame fv; /* full view's eyes, latched with them; fv.on only with stereo */
    uint64_t stereo_frames, fullview_frames;
    /* The full view shot (RT_FULLVIEW_SHOT=after,count,gap in full view
       frames): both eyes, the same frame replayed once more as the game
       drew it (the mono reference), and every command of the left eye
       described, for finding what full view does to a draw on the
       headset without anyone describing it. */
    int shot_cfg, shot_active;
    uint32_t shot_left, shot_gap, shot_n;
    uint64_t shot_next;
    FILE *shot_dump;
    void *shot_mono[2];
    char shot_base[1024];
    /* Statistics, logged when the device is released. The game thread's
       time in draws is the bridge's to measure; these are the parts that
       moved off it, and what the game thread still waits for. */
    uint64_t lists, frames, flushes, waits, wait_ns, wait_max, pace_waits, pace_ns, pace_max;
    int depth; /* the in-flight depth the last frame was submitted with */
    _Atomic uint64_t emit_ns, emit_max, gpu_ns, gpu_max, dropped, no_drawable;
} mtl_frames;

static uint64_t now_ns(void)
{
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    return mach_absolute_time() * tb.numer / tb.denom;
}

static void atomic_max(_Atomic uint64_t *m, uint64_t v)
{
    uint64_t old = atomic_load_explicit(m, memory_order_relaxed);
    while (v > old && !atomic_compare_exchange_weak_explicit(m, &old, v, memory_order_relaxed, memory_order_relaxed)) {
    }
}

int mtl_threaded(const mtl_device *d) { return d->fq && d->fq->threaded; }

void *mtl_frames_right_depth(const mtl_device *d) { return d->fq ? d->fq->right[1] : NULL; }

/* Whether a frame being recorded now may be the one shot: the game thread
   runs a few frames ahead of the replay that opens the shot, so from a
   few frames before it until it is taken. Read from the game thread, the
   counters the render thread moves loosely: a frame early or late costs
   only a box the dump prints as unknown. */
int mtl_frames_shot_soon(const mtl_device *d)
{
    const mtl_frames *f = d->fq;
    if (!f || !f->threaded || f->shot_cfg != 1) return 0;
    if (!__atomic_load_n(&f->shot_left, __ATOMIC_RELAXED)) return 0;
    return __atomic_load_n(&f->fullview_frames, __ATOMIC_RELAXED) + 4 >= __atomic_load_n(&f->shot_next, __ATOMIC_RELAXED);
}

uint8_t *mtl_rec_base(mtl_device *d) { return d->fq->cur->cmd; }

/* Lists */

static mtl_seg *seg_new(mtl_frames *f)
{
    mtl_seg *s = f->spare;
    if (s) {
        f->spare = s->next_q;
        s->next_q = NULL;
        return s;
    }
    s = calloc(1, sizeof *s);
    if (s) f->nsegs++;
    return s;
}

static void *grow(void *p, uint32_t *cap, size_t elem, uint32_t need)
{
    if (need <= *cap) return p;
    uint32_t n = *cap ? *cap : 64;
    while (n < need) n *= 2;
    void *q = realloc(p, n * elem);
    if (!q) return NULL;
    *cap = n;
    return q;
}

void *mtl_rec_cmd(mtl_device *d, uint32_t type, uint32_t bytes)
{
    mtl_frames *f = d->fq;
    mtl_seg *s = f->cur;
    /* Inline mode replays each finished command before the next one is
       appended, so an encoder sees draws in the same order and at the
       same moment as before the lists existed. */
    if (!f->threaded) mtl_rec_end(d);
    bytes = (bytes + 15u) & ~15u;
    if (s->len + bytes > s->cap) {
        size_t n = s->cap ? s->cap : 64 << 10;
        while (n < s->len + bytes) n *= 2;
        uint8_t *m = realloc(s->cmd, n);
        if (!m) {
            mtl_log("out of memory recording a command list (%zu bytes)", n);
            abort();
        }
        s->cmd = m;
        s->cap = n;
    }
    uint8_t *c = s->cmd + s->len;
    memset(c, 0, bytes);
    ((uint32_t *)c)[0] = type;
    ((uint32_t *)c)[1] = bytes;
    s->len += bytes;
    return c;
}

void mtl_rec_end(mtl_device *d)
{
    mtl_frames *f = d->fq;
    if (!f || f->threaded) return;
    mtl_seg *s = f->cur;
    if (s->emitted == s->len) return;
    /* Not timed: inline encoding is exactly what the bridge's draws
       column measures, and a clock read per call would only add to it. */
    @autoreleasepool {
        if (!f->inl.cb) f->inl.cb = (void *)CFBridgingRetain([QUEUE(d) commandBuffer]);
        mtl_emit_run(d, &f->inl, s->cmd, s->emitted, s->len);
    }
    s->emitted = s->len;
}

void mtl_rec_keep(mtl_device *d, void *objc)
{
    if (!objc) return;
    mtl_seg *s = d->fq->cur;
    void **k = grow(s->keep, &s->capkeep, sizeof *k, s->nkeep + 1);
    if (!k) {
        mtl_log("out of memory keeping a list's objects");
        abort();
    }
    s->keep = k;
    s->keep[s->nkeep++] = (void *)CFRetain(objc);
}

void mtl_rec_hold(mtl_device *d, mtl_blob *b)
{
    if (!b || b->held_serial == d->serial) return;
    mtl_seg *s = d->fq->cur;
    mtl_blob **h = grow(s->hold, &s->caphold, sizeof *h, s->nhold + 1);
    if (!h) {
        mtl_log("out of memory holding a list's shaders");
        abort();
    }
    s->hold = h;
    mtl_hold(&b->o);
    b->held_serial = d->serial;
    s->hold[s->nhold++] = b;
}

/* Lets go of every list the GPU has finished: game thread only. A list
   is reused only once the render thread has also said, under the lock,
   that it is done reading it; completion already implies that, through
   Metal's own ordering, but this edge is one a race detector can see. */
static void retire(mtl_device *d)
{
    mtl_frames *f = d->fq;
    if (!f->sub_head) return;
    uint64_t done = atomic_load_explicit(&d->completed, memory_order_acquire);
    if (f->threaded) {
        pthread_mutex_lock(&f->mu);
        if (f->replayed < done) done = f->replayed;
        pthread_mutex_unlock(&f->mu);
    }
    while (f->sub_head && f->sub_head->serial <= done) {
        mtl_seg *s = f->sub_head;
        f->sub_head = s->next_sub;
        if (!f->sub_head) f->sub_tail = NULL;
        f->nsub--;
        for (uint32_t i = 0; i < s->nkeep; i++) CFRelease(s->keep[i]);
        for (uint32_t i = 0; i < s->nhold; i++) mtl_unhold(&s->hold[i]->o);
        s->nkeep = s->nhold = 0;
        s->len = s->emitted = 0;
        s->serial = 0;
        s->next_sub = NULL;
        s->next_q = f->spare;
        f->spare = s;
    }
}

/* The command buffer of one list: published as completed when the GPU is
   done with it. Called once per list, in serial order, so completed only
   grows. */
static void commit(mtl_device *d, id<MTLCommandBuffer> cb, uint64_t serial)
{
    mtl_frames *f = d->fq;
    _Atomic uint64_t *done = &d->completed;
    [cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
        double g = b.GPUEndTime - b.GPUStartTime;
        uint64_t gns = g > 0 ? (uint64_t)(g * 1e9) : 0;
        atomic_fetch_add_explicit(&f->gpu_ns, gns, memory_order_relaxed);
        atomic_max(&f->gpu_max, gns);
        pthread_mutex_lock(&f->mu);
        atomic_store_explicit(done, serial, memory_order_release);
        pthread_cond_broadcast(&f->done);
        pthread_mutex_unlock(&f->mu);
    }];
    [cb commit];
}

static id<MTLTexture> twin_like(mtl_device *d, mtl_surface *s);

/* The a the eye images are shown with: the session's, from the back
   buffer's shape (stereo.h, mtl_stereo_screen_a), the same for every
   frame whatever its own camera, so a zoom reads as a zoom and a 2D
   frame covers the plane a world frame does. */
static float back_aspect(mtl_device *d)
{
    return d->back && d->back->height ? (float)d->back->width / (float)d->back->height : 4.0f / 3.0f;
}

static float screen_a(mtl_device *d) { return mtl_stereo_screen_a_for(back_aspect(d)); }

/* Full view's eyes for a frame: only when the frame is per eye at all
   and the game meant it for them; fv.on stays 0 otherwise, and the frame
   goes on stereo's plane. */
static void latch_fullview(mtl_device *d, int stereo, int wanted, mtl_fullview_frame *fv)
{
    if (stereo && wanted) mtl_fullview_latch(back_aspect(d), fv);
    else memset(fv, 0, sizeof *fv);
}

/* The right eye's targets, matching the back buffer and its depth as they
   are now (a Reset may have changed them). 0 when there is no back buffer
   or they could not be made, and the frame is then replayed once. */
static int right_eye_ready(mtl_device *d, mtl_frames *f)
{
    if (!d->back || !d->back->tex) return 0;
    id<MTLTexture> bb = (__bridge id<MTLTexture>)d->back->tex;
    id<MTLTexture> ds = d->auto_ds && d->auto_ds->tex ? (__bridge id<MTLTexture>)d->auto_ds->tex : nil;
    id<MTLTexture> c = (__bridge id<MTLTexture>)f->right[0], z = (__bridge id<MTLTexture>)f->right[1];
    if (c && c.width == bb.width && c.height == bb.height && c.pixelFormat == bb.pixelFormat &&
        (!ds ? !z : z && z.width == ds.width && z.height == ds.height && z.pixelFormat == ds.pixelFormat))
        return 1;
    for (int i = 0; i < 2; i++)
        if (f->right[i]) {
            CFRelease(f->right[i]);
            f->right[i] = NULL;
        }
    c = twin_like(d, d->back);
    z = twin_like(d, d->auto_ds);
    if (!c || (ds && !z)) return 0;
    c.label = @"right eye";
    z.label = @"right eye depth";
    f->right[0] = (void *)CFBridgingRetain(c);
    f->right[1] = z ? (void *)CFBridgingRetain(z) : NULL;
    mtl_log("stereo: right eye targets %lux%lu", (unsigned long)c.width, (unsigned long)c.height);
    return 1;
}

/* The second eye's replay of a list the first eye has just replayed into
   the same command buffer: the same commands with the back buffer and its
   depth swapped for the right eye's, the right eye's offset, and none of
   what happens once (readbacks, Present). Then the frame's Present, if
   the first eye held one, with both eyes. */
static void replay_right_eye(mtl_device *d, mtl_frames *f, void *cb, const uint8_t *cmd, size_t len, mtl_emit *left)
{
    mtl_emit e2 = {.cb = cb,
                   .second = 1,
                   .stereo = 1,
                   .eye_x = f->right_x,
                   .screen_a = left->screen_a,
                   .hud_z = left->hud_z,
                   .fv = left->fv,
                   .eye = 1};
    e2.swap_from[0] = d->back ? d->back->tex : NULL;
    e2.swap_from[1] = d->auto_ds ? d->auto_ds->tex : NULL;
    e2.swap_to[0] = f->right[0];
    e2.swap_to[1] = f->right[1];
    /* Each back buffer copy the game keeps, twinned for this eye. */
    uint32_t n = __atomic_load_n(&d->eye_copy_n, __ATOMIC_ACQUIRE);
    for (uint32_t k = 0; k < n; k++) {
        void *src = d->eye_copy[k];
        if (f->copy_src[k] != src) {
            if (f->copy_twin[k]) CFRelease(f->copy_twin[k]);
            id<MTLTexture> t = (__bridge id<MTLTexture>)src;
            MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:t.pixelFormat
                                                                                          width:t.width
                                                                                         height:t.height
                                                                                      mipmapped:NO];
            td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
            td.storageMode = MTLStorageModePrivate;
            id<MTLTexture> twin = [MTLD(d) newTextureWithDescriptor:td];
            twin.label = @"right eye copy";
            f->copy_twin[k] = twin ? (void *)CFBridgingRetain(twin) : NULL;
            f->copy_src[k] = src;
        }
        e2.swap_from[2 + k] = src;
        e2.swap_to[2 + k] = f->copy_twin[k];
    }
    mtl_emit_run(d, &e2, cmd, 0, len);
    mtl_emit_close(&e2);
    if (left->pend_present) {
        /* Into this command buffer: inline mode has already taken its own
           out of the first eye's emitter by now. */
        void *own = left->cb;
        left->cb = cb;
        left->pend_present = 0;
        mtl_emit_present(d, left, left->pend_back, f->right[0], left->pend_ns);
        left->cb = own;
        f->stereo_frames++;
        if (left->fv.on) f->fullview_frames++;
    }
}

/* The shots' folder: RT_FULLVIEW_SHOT_DIR, else shots/ beside the
   overlay's cache (RT_METAL_CACHE_DIR is <overlay>/cache/metal), which is
   where the bridge's own captures go. */
static int shot_dir(char *out, size_t n)
{
    const char *dir = getenv("RT_FULLVIEW_SHOT_DIR");
    if (dir && *dir) return snprintf(out, n, "%s", dir) < (int)n;
    const char *cache = getenv("RT_METAL_CACHE_DIR");
    if (!cache || !*cache) return 0;
    char tmp[1024];
    snprintf(tmp, sizeof tmp, "%s", cache);
    for (int up = 0; up < 2; up++) {
        size_t l = strlen(tmp);
        while (l > 1 && tmp[l - 1] == '/') tmp[--l] = 0;
        char *slash = strrchr(tmp, '/');
        if (!slash || slash == tmp) return 0;
        *slash = 0;
    }
    return snprintf(out, n, "%s/shots", tmp) < (int)n;
}

static void shot_config(mtl_frames *f)
{
    if (f->shot_cfg) return;
    f->shot_cfg = -1;
    const char *v = getenv("RT_FULLVIEW_SHOT");
    if (!v || !*v) return;
    unsigned after = 600, count = 1, gap = 120;
    sscanf(v, "%u,%u,%u", &after, &count, &gap);
    if (!count) return;
    f->shot_cfg = 1;
    f->shot_next = after;
    f->shot_left = count;
    f->shot_gap = gap ? gap : 1;
    mtl_log("full view shot: %u after %u full view frames, %u apart", count, after, f->shot_gap);
}

/* At a frame's first list: whether this frame is shot, and if so its
   mono target and the description file, opened before the left eye. */
static void shot_begin(mtl_device *d, mtl_frames *f)
{
    shot_config(f);
    if (f->shot_cfg != 1 || !f->shot_left || !f->stereo || !f->fv.on || f->fullview_frames < f->shot_next) return;
    char dir[1024];
    if (!shot_dir(dir, sizeof dir)) {
        mtl_log("full view shot: no folder (RT_FULLVIEW_SHOT_DIR or RT_METAL_CACHE_DIR)");
        f->shot_left = 0;
        return;
    }
    mkdir(dir, 0755);
    id<MTLTexture> mono = (__bridge id<MTLTexture>)f->shot_mono[0];
    id<MTLTexture> bb = d->back ? (__bridge id<MTLTexture>)d->back->tex : nil;
    if (!bb) return;
    if (!mono || mono.width != bb.width || mono.height != bb.height) {
        for (int i = 0; i < 2; i++)
            if (f->shot_mono[i]) {
                CFRelease(f->shot_mono[i]);
                f->shot_mono[i] = NULL;
            }
        id<MTLTexture> c = twin_like(d, d->back), z = twin_like(d, d->auto_ds);
        if (!c) return;
        f->shot_mono[0] = (void *)CFBridgingRetain(c);
        f->shot_mono[1] = z ? (void *)CFBridgingRetain(z) : NULL;
    }
    snprintf(f->shot_base, sizeof f->shot_base, "%s/fv%u-f%llu", dir, f->shot_n, (unsigned long long)f->fullview_frames);
    char path[1100];
    snprintf(path, sizeof path, "%s.txt", f->shot_base);
    f->shot_dump = fopen(path, "w");
    if (!f->shot_dump) {
        mtl_log("full view shot: cannot write %s", path);
        f->shot_left = 0;
        return;
    }
    const mtl_fullview_frame *fv = &f->fv;
    fprintf(f->shot_dump, "# full view frame %llu, back %lux%lu fmt %lu, back %p right %p mono %p\n",
            (unsigned long long)f->fullview_frames, (unsigned long)bb.width, (unsigned long)bb.height,
            (unsigned long)bb.pixelFormat, d->back->tex, f->right[0], f->shot_mono[0]);
    for (int i = 0; i < 2; i++)
        fprintf(f->shot_dump, "# eye %d: a %.4f cx %.4f b %.4f cy %.4f t %.4f %.4f %.4f tan %.3f %.3f %.3f %.3f\n", i,
                fv->eye_p[i].a, fv->eye_p[i].cx, fv->eye_p[i].b, fv->eye_p[i].cy, fv->eye_t[i][0], fv->eye_t[i][1],
                fv->eye_t[i][2], fv->tan[i][0], fv->tan[i][1], fv->tan[i][2], fv->tan[i][3]);
    fprintf(f->shot_dump, "# panel %.3f %.3f sky_scale %.4f hud_z %.3f game_eye %d\n", fv->panel[0], fv->panel[1],
            fv->sky_scale, mtl_stereo_hud_z(), fv->game_eye);
    f->shot_active = 1;
}

/* One image, as a binary PPM:
   the back buffer's BGRA, or RGBA, bytes with the alpha dropped. */
static void shot_write(const char *path, const uint8_t *px, size_t w, size_t h, size_t pitch, int bgra)
{
    FILE *o = fopen(path, "wb");
    if (!o) return;
    fprintf(o, "P6\n%zu %zu\n255\n", w, h);
    uint8_t *row = malloc(w * 3);
    for (size_t y = 0; row && y < h; y++) {
        const uint8_t *s = px + y * pitch;
        for (size_t x = 0; x < w; x++) {
            row[3 * x + 0] = s[4 * x + (bgra ? 2 : 0)];
            row[3 * x + 1] = s[4 * x + 1];
            row[3 * x + 2] = s[4 * x + (bgra ? 0 : 2)];
        }
        fwrite(row, 3, w, o);
    }
    free(row);
    fclose(o);
}

/* After the frame's last list has replayed into cb: copy the three images
   out and write them, with the description, once the GPU is done. */
static void shot_finish(mtl_device *d, mtl_frames *f, id<MTLCommandBuffer> cb)
{
    f->shot_active = 0;
    FILE *dump = f->shot_dump;
    f->shot_dump = NULL;
    f->shot_n++;
    f->shot_left--;
    f->shot_next = f->fullview_frames + f->shot_gap;
    id<MTLTexture> tex[3] = {(__bridge id<MTLTexture>)d->back->tex, (__bridge id<MTLTexture>)f->right[0],
                             (__bridge id<MTLTexture>)f->shot_mono[0]};
    static const char *names[3] = {"left", "right", "mono"};
    MTLPixelFormat pf = tex[0].pixelFormat;
    int bgra = pf == MTLPixelFormatBGRA8Unorm || pf == MTLPixelFormatBGRA8Unorm_sRGB;
    if (!bgra && pf != MTLPixelFormatRGBA8Unorm && pf != MTLPixelFormatRGBA8Unorm_sRGB) {
        mtl_log("full view shot: back buffer format %lu is not 8 bit; description only", (unsigned long)pf);
        fclose(dump);
        return;
    }
    size_t w = tex[0].width, h = tex[0].height, pitch = w * 4;
    id<MTLBuffer> buf[3];
    id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
    for (int i = 0; i < 3; i++) {
        buf[i] = [MTLD(d) newBufferWithLength:pitch * h options:MTLResourceStorageModeShared];
        if (!buf[i] || !tex[i]) continue;
        [bl copyFromTexture:tex[i]
                         sourceSlice:0
                         sourceLevel:0
                        sourceOrigin:MTLOriginMake(0, 0, 0)
                          sourceSize:MTLSizeMake(w, h, 1)
                            toBuffer:buf[i]
                   destinationOffset:0
              destinationBytesPerRow:pitch
            destinationBytesPerImage:pitch * h];
    }
    /* The arm HUD's cards as this frame drew them, each side on its own,
       colour premultiplied over black. */
    mtl_arm_frame arm;
    mtl_arm_last(d, &arm);
    id<MTLTexture> at = (__bridge id<MTLTexture>)arm.tex;
    id<MTLBuffer> abuf[2] = {nil, nil};
    size_t aw[2] = {0, 0}, ah = arm.h;
    for (int k = 0; k < 2 && at && at.pixelFormat == pf; k++) {
        aw[k] = arm.w[k];
        if (!aw[k] || !ah) continue;
        abuf[k] = [MTLD(d) newBufferWithLength:aw[k] * 4 * ah options:MTLResourceStorageModeShared];
        if (!abuf[k]) continue;
        [bl copyFromTexture:at
                         sourceSlice:0
                         sourceLevel:0
                        sourceOrigin:MTLOriginMake(arm.x[k], 0, 0)
                          sourceSize:MTLSizeMake(aw[k], ah, 1)
                            toBuffer:abuf[k]
                   destinationOffset:0
              destinationBytesPerRow:aw[k] * 4
            destinationBytesPerImage:aw[k] * 4 * ah];
    }
    fprintf(dump, "# arm HUD cards: left %s %zux%zu, right %s %zux%zu\n", arm.drawn[0] ? "drawn" : "not drawn", aw[0], ah,
            arm.drawn[1] ? "drawn" : "not drawn", aw[1], ah);
    [bl endEncoding];
    NSString *base = @(f->shot_base);
    id<MTLBuffer> b0 = buf[0], b1 = buf[1], b2 = buf[2], a0 = abuf[0], a1 = abuf[1];
    size_t aw0 = aw[0], aw1 = aw[1];
    [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
        (void)done;
        id<MTLBuffer> bs[3] = {b0, b1, b2};
        for (int i = 0; i < 3; i++) {
            if (!bs[i]) continue;
            NSString *p = [NSString stringWithFormat:@"%@-%s.ppm", base, names[i]];
            shot_write(p.UTF8String, bs[i].contents, w, h, pitch, bgra);
        }
        id<MTLBuffer> as[2] = {a0, a1};
        const size_t aws[2] = {aw0, aw1};
        for (int k = 0; k < 2; k++) {
            if (!as[k]) continue;
            NSString *p = [NSString stringWithFormat:@"%@-arm%s.ppm", base, k ? "R" : "L"];
            shot_write(p.UTF8String, as[k].contents, aws[k], ah, aws[k] * 4, bgra);
        }
        fclose(dump);
        mtl_log("full view shot written: %s (left, right, mono, armL, armR, txt)", base.UTF8String);
    }];
}

/* Replays a closed list into a fresh command buffer and commits it. The
   list is not touched after commit: its completion lets the game thread
   retire it at any moment from then on. */
static void run_list(mtl_device *d, mtl_seg *s)
{
    mtl_frames *f = d->fq;
    uint64_t t0 = now_ns(), serial = s->serial;
    id<MTLCommandBuffer> cb = [QUEUE(d) commandBuffer];
    if (!f->mid_frame) {
        f->stereo = mtl_stereo_on() && right_eye_ready(d, f);
        f->left_x = mtl_stereo_eye_x(0);
        f->right_x = mtl_stereo_eye_x(1);
        f->screen_a = screen_a(d);
        latch_fullview(d, f->stereo, s->fullview, &f->fv);
        shot_begin(d, f);
    }
    mtl_emit e = {.cb = (__bridge void *)cb,
                  .dequeued_ns = t0,
                  .stereo = f->stereo,
                  .eye_x = f->stereo ? f->left_x : 0.0f,
                  .screen_a = f->screen_a,
                  .hud_z = mtl_stereo_hud_z(),
                  .fv = f->fv,
                  .dump = f->shot_active ? f->shot_dump : NULL};
    if (e.dump) fprintf(e.dump, "# list %llu, %s\n", (unsigned long long)serial, s->present ? "with the Present" : "mid frame");
    mtl_emit_run(d, &e, s->cmd, 0, s->len);
    mtl_emit_close(&e);
    f->mid_frame = !s->present;
    if (f->stereo) {
        replay_right_eye(d, f, (__bridge void *)cb, s->cmd, s->len, &e);
        if (f->shot_active) {
            /* The mono reference: the same list once more as the game
               drew it, into the shot's own twins. */
            mtl_emit e3 = {.cb = (__bridge void *)cb, .second = 1};
            e3.swap_from[0] = d->back ? d->back->tex : NULL;
            e3.swap_from[1] = d->auto_ds ? d->auto_ds->tex : NULL;
            e3.swap_to[0] = f->shot_mono[0];
            e3.swap_to[1] = f->shot_mono[1];
            mtl_emit_run(d, &e3, s->cmd, 0, s->len);
            mtl_emit_close(&e3);
            if (s->present) shot_finish(d, f, cb);
        }
    } else if (s->twice && f->twin[0]) {
        mtl_emit e2 = {.cb = (__bridge void *)cb, .second = 1};
        e2.swap_from[0] = d->back ? d->back->tex : NULL;
        e2.swap_from[1] = d->auto_ds ? d->auto_ds->tex : NULL;
        e2.swap_to[0] = f->twin[0];
        e2.swap_to[1] = f->twin[1];
        mtl_emit_run(d, &e2, s->cmd, 0, s->len);
        mtl_emit_close(&e2);
    }
    uint64_t dt = now_ns() - t0;
    atomic_fetch_add_explicit(&f->emit_ns, dt, memory_order_relaxed);
    atomic_max(&f->emit_max, dt);
    commit(d, cb, serial);
    mtl_immersive_committed(now_ns());
    pthread_mutex_lock(&f->mu);
    f->replayed = serial;
    pthread_mutex_unlock(&f->mu);
}

static void *render_main(void *arg)
{
    mtl_device *d = arg;
    mtl_frames *f = d->fq;
    pthread_setname_np("metal_d3d9 render");
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    for (;;) {
        pthread_mutex_lock(&f->mu);
        while (!f->q_head && !f->stop) pthread_cond_wait(&f->work, &f->mu);
        mtl_seg *s = f->q_head;
        if (s) {
            f->q_head = s->next_q;
            if (!f->q_head) f->q_tail = NULL;
            s->next_q = NULL;
        }
        pthread_mutex_unlock(&f->mu);
        if (!s) break;
        @autoreleasepool {
            run_list(d, s);
        }
    }
    return NULL;
}

static void wait_done(mtl_device *d, uint64_t serial, uint64_t *count, uint64_t *total, uint64_t *max)
{
    mtl_frames *f = d->fq;
    if (atomic_load_explicit(&d->completed, memory_order_acquire) >= serial) return;
    uint64_t t0 = now_ns();
    pthread_mutex_lock(&f->mu);
    while (atomic_load_explicit(&d->completed, memory_order_acquire) < serial) pthread_cond_wait(&f->done, &f->mu);
    pthread_mutex_unlock(&f->mu);
    uint64_t dt = now_ns() - t0;
    (*count)++;
    *total += dt;
    if (dt > *max) *max = dt;
}

uint64_t mtl_submit(mtl_device *d, int present)
{
    mtl_frames *f = d->fq;
    /* A list ends with its pass closed and pending clears applied, so the
       next list starts from what this one stored. */
    mtl_end_pass(d);
    mtl_fv_scan(d, f->cur->cmd, present);
    if (present) mtl_fv_aim(d, f->cur->cmd);
    mtl_seg *s = f->cur, *next = seg_new(f);
    if (!next) {
        mtl_log("out of memory starting a command list");
        abort();
    }
    s->serial = d->serial++;
    s->twice = f->twice;
    s->present = present;
    /* Only the frame's first list is read for it: the game thread's
       choice, made from what the runtime said before this frame began
       (metal_d3d9_immersive_view_play), the same choice its field of view
       was drawn with. */
    s->fullview = mtl_fullview_wanted();
    if (present) {
        mtl_fullview_note_aspect(back_aspect(d));
        /* The world's own field of view no longer sizes anything; it is
           logged when it changes while in stereo (a dialogue close up, a
           menu's 3D piece), a few times a run, so a device log says what
           the plane showed. */
        float a = mtl_stereo_tally_take(&d->stereo_tally, 0.0f);
        if (a > 0.0f && fabsf(a - d->stereo_world_a) > 0.01f * a) {
            d->stereo_world_a = a;
            if (mtl_stereo_on() && d->stereo_world_logs < 24) {
                d->stereo_world_logs++;
                float s_a = screen_a(d);
                mtl_log("stereo: the world now drawn %.1f degrees across (a %.4f), shown %.2fx on the session's plane "
                        "(a %.4f)%s",
                        2.0 * atan(1.0 / a) * 180.0 / M_PI, a, a / s_a, s_a,
                        d->stereo_world_logs == 24 ? "; no more of these" : "");
            }
        }
    }
    f->cur = next;
    f->lists++;
    if (present) f->frames++;
    else f->flushes++;
    if (f->sub_tail) f->sub_tail->next_sub = s;
    else f->sub_head = s;
    f->sub_tail = s;
    f->nsub++;
    if (!f->threaded) {
        @autoreleasepool {
            if (!f->inl.cb) f->inl.cb = (void *)CFBridgingRetain([QUEUE(d) commandBuffer]);
            if (s->emitted != s->len) mtl_emit_run(d, &f->inl, s->cmd, s->emitted, s->len);
            mtl_emit_close(&f->inl);
            s->emitted = s->len;
            id<MTLCommandBuffer> cb = CFBridgingRelease(f->inl.cb);
            f->inl.cb = NULL;
            if (f->inl.stereo) {
                replay_right_eye(d, f, (__bridge void *)cb, s->cmd, s->len, &f->inl);
            } else if (s->twice && f->twin[0]) {
                mtl_emit e2 = {.cb = (__bridge void *)cb, .second = 1};
                e2.swap_from[0] = d->back ? d->back->tex : NULL;
                e2.swap_from[1] = d->auto_ds ? d->auto_ds->tex : NULL;
                e2.swap_to[0] = f->twin[0];
                e2.swap_to[1] = f->twin[1];
                mtl_emit_run(d, &e2, s->cmd, 0, s->len);
                mtl_emit_close(&e2);
            }
            commit(d, cb, s->serial);
            mtl_immersive_committed(now_ns());
            /* Inline mode replays the first eye as the game records, so
               the next frame's choice is made here, at this one's end. */
            if (present) {
                f->inl.stereo = mtl_stereo_on() && right_eye_ready(d, f);
                f->inl.eye_x = f->inl.stereo ? mtl_stereo_eye_x(0) : 0.0f;
                f->right_x = mtl_stereo_eye_x(1);
                f->inl.screen_a = screen_a(d);
                f->inl.hud_z = mtl_stereo_hud_z();
                /* From what the runtime said at the last frame's end, one
                   frame behind its own choice, as the eyes' offsets are. */
                latch_fullview(d, f->inl.stereo, mtl_fullview_wanted(), &f->inl.fv);
            }
        }
    } else {
        pthread_mutex_lock(&f->mu);
        if (f->q_tail) f->q_tail->next_q = s;
        else f->q_head = s;
        f->q_tail = s;
        pthread_cond_signal(&f->work);
        pthread_mutex_unlock(&f->mu);
        if (present) {
            /* At most depth frames between the game and the GPU: the wait
               for the oldest is the game's pacing when it runs ahead, and
               it lands in Present, where the bridge times it. The depth
               follows immersive mode as it is turned on and off; the
               history keeps the last IN_FLIGHT frames either way. */
            int depth = mtl_immersive_on() ? IN_FLIGHT_IMMERSIVE : IN_FLIGHT;
            f->depth = depth;
            uint64_t oldest = f->presented[0];
            memmove(f->presented, f->presented + 1, sizeof f->presented - sizeof f->presented[0]);
            f->presented[IN_FLIGHT - 1] = s->serial;
            uint64_t wait_for = depth >= IN_FLIGHT ? oldest : f->presented[IN_FLIGHT - 1 - depth];
            if (wait_for) wait_done(d, wait_for, &f->pace_waits, &f->pace_ns, &f->pace_max);
        }
    }
    /* After the frame is queued and the in-flight wait, so the cap's wait
       falls between frames and never in front of this one's GPU work. */
    if (present && mtl_immersive_on()) mtl_immersive_pace();
    retire(d);
    return s->serial;
}

void mtl_wait_serial(mtl_device *d, uint64_t serial)
{
    if (!serial) return;
    mtl_frames *f = d->fq;
    if (serial >= d->serial) mtl_submit(d, 0); /* still in the open list */
    wait_done(d, serial, &f->waits, &f->wait_ns, &f->wait_max);
    retire(d);
}

void mtl_drain(mtl_device *d)
{
    if (!d->fq) return;
    mtl_wait_serial(d, d->serial);
}

int mtl_frames_init(mtl_device *d)
{
    mtl_frames *f = calloc(1, sizeof *f);
    if (!f) return 0;
    pthread_mutex_init(&f->mu, NULL);
    pthread_cond_init(&f->work, NULL);
    pthread_cond_init(&f->done, NULL);
    const char *v = getenv("RT_METAL_INLINE");
    f->threaded = !(v && atoi(v) > 0);
    f->depth = mtl_immersive_on() ? IN_FLIGHT_IMMERSIVE : IN_FLIGHT;
    if (f->threaded) shot_config(f);
    d->fq = f;
    f->cur = seg_new(f);
    if (!f->cur) return 0;
    if (f->threaded) {
        pthread_attr_t a;
        pthread_attr_init(&a);
        pthread_attr_set_qos_class_np(&a, QOS_CLASS_USER_INTERACTIVE, 0);
        f->running = !pthread_create(&f->thread, &a, render_main, d);
        pthread_attr_destroy(&a);
        if (!f->running) {
            mtl_log("render thread did not start: encoding on the game thread");
            f->threaded = 0;
        }
    }
    mtl_log("command lists: %s", f->threaded ? "replayed on a render thread, two frames in flight at most (one "
                                               "while immersive)"
                                             : "encoded on the game thread at each call (RT_METAL_INLINE)");
    return 1;
}

static void free_segs(mtl_seg *s)
{
    while (s) {
        mtl_seg *n = s->next_q;
        free(s->cmd);
        free(s->keep);
        free(s->hold);
        free(s);
        s = n;
    }
}

void mtl_frames_destroy(mtl_device *d)
{
    mtl_frames *f = d->fq;
    if (!f) return;
    if (d->draw) mtl_drain(d);
    if (f->running) {
        pthread_mutex_lock(&f->mu);
        f->stop = 1;
        pthread_cond_signal(&f->work);
        pthread_mutex_unlock(&f->mu);
        pthread_join(f->thread, NULL);
        f->running = 0;
    }
    retire(d);
    mtl_emit_close(&f->inl);
    if (f->inl.cb) CFRelease(f->inl.cb);
    for (int i = 0; i < 2; i++) {
        if (f->twin[i]) CFRelease(f->twin[i]);
        if (f->right[i]) CFRelease(f->right[i]);
        if (f->shot_mono[i]) CFRelease(f->shot_mono[i]);
    }
    if (f->shot_dump) fclose(f->shot_dump);
    if (f->stereo_frames)
        mtl_log("stereo: %llu frames replayed once per eye, %llu of them into the eyes' own views (full view)",
                (unsigned long long)f->stereo_frames, (unsigned long long)f->fullview_frames);
    double fr = f->frames ? (double)f->frames : 1.0, ls = f->lists ? (double)f->lists : 1.0;
    mtl_log("command lists: %llu (%llu frames, %llu closed early for a readback) on %s; replay %.3f ms a list (max "
            "%.2f, not timed inline), GPU %.3f ms a list (max %.2f); game thread waited %llu times for in-flight frames (%.2f ms a "
            "frame, max %.2f) and %llu times for readbacks (%.2f ms total, max %.2f); %u lists allocated; %llu "
            "draws dropped at replay",
            (unsigned long long)f->lists, (unsigned long long)f->frames, (unsigned long long)f->flushes,
            f->threaded ? "the render thread" : "the game thread", atomic_load(&f->emit_ns) / 1e6 / ls,
            atomic_load(&f->emit_max) / 1e6, atomic_load(&f->gpu_ns) / 1e6 / ls, atomic_load(&f->gpu_max) / 1e6,
            (unsigned long long)f->pace_waits, f->pace_ns / 1e6 / fr, f->pace_max / 1e6, (unsigned long long)f->waits,
            f->wait_ns / 1e6, f->wait_max / 1e6, f->nsegs, (unsigned long long)atomic_load(&f->dropped));
    if (f->cur) {
        f->cur->next_q = f->spare;
        f->spare = f->cur;
        f->cur = NULL;
    }
    /* Every submitted list has completed and been retired above. */
    free_segs(f->spare);
    pthread_mutex_destroy(&f->mu);
    pthread_cond_destroy(&f->work);
    pthread_cond_destroy(&f->done);
    free(f);
    d->fq = NULL;
}

/* Counted by the replay, read at release (after the render thread has
   stopped, or from the one thread in inline mode). */
void mtl_frames_note_dropped(mtl_device *d) { atomic_fetch_add_explicit(&d->fq->dropped, 1, memory_order_relaxed); }
void mtl_frames_note_no_drawable(mtl_device *d)
{
    atomic_fetch_add_explicit(&d->fq->no_drawable, 1, memory_order_relaxed);
}
uint64_t mtl_frames_no_drawable(const mtl_device *d) { return d->fq ? atomic_load(&d->fq->no_drawable) : 0; }

/* Test hooks, exported for the headless test only. */

static id<MTLTexture> twin_like(mtl_device *d, mtl_surface *s)
{
    if (!s || !s->tex) return nil;
    id<MTLTexture> t = (__bridge id<MTLTexture>)s->tex;
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:t.pixelFormat
                                                                                  width:t.width
                                                                                 height:t.height
                                                                              mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModePrivate;
    return [MTLD(d) newTextureWithDescriptor:td];
}

/* Every list closed from now on is replayed twice into one command
   buffer: as recorded, then with the back buffer and its depth surface
   swapped for twins of the same size. Returns 0 if the twins could not be
   made. */
D9_EXPORT int metal_d3d9_replay_twice(void *dev, int on)
{
    mtl_device *d = dev;
    mtl_frames *f = d->fq;
    mtl_drain(d);
    @autoreleasepool {
        for (int i = 0; i < 2; i++)
            if (f->twin[i]) {
                CFRelease(f->twin[i]);
                f->twin[i] = NULL;
            }
        f->twice = 0;
        if (!on) return 1;
        id<MTLTexture> c = twin_like(d, d->back), z = twin_like(d, d->auto_ds);
        if (!c) return 0;
        f->twin[0] = (void *)CFBridgingRetain(c);
        f->twin[1] = z ? (void *)CFBridgingRetain(z) : NULL;
        f->twice = 1;
    }
    return 1;
}

/* The twin colour target's pixels, after every recorded list has run. */
D9_EXPORT int metal_d3d9_twin_read(void *dev, uint32_t *out, uint32_t w, uint32_t h)
{
    mtl_device *d = dev;
    mtl_frames *f = d->fq;
    mtl_drain(d);
    if (!f->twin[0]) return 0;
    @autoreleasepool {
        id<MTLTexture> t = (__bridge id<MTLTexture>)f->twin[0];
        if (t.width != w || t.height != h) return 0;
        id<MTLBuffer> b = [MTLD(d) newBufferWithLength:(NSUInteger)w * h * 4 options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = [QUEUE(d) commandBuffer];
        id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
        [bl copyFromTexture:t
                         sourceSlice:0
                         sourceLevel:0
                        sourceOrigin:MTLOriginMake(0, 0, 0)
                          sourceSize:MTLSizeMake(w, h, 1)
                            toBuffer:b
                   destinationOffset:0
              destinationBytesPerRow:w * 4
            destinationBytesPerImage:(NSUInteger)w * h * 4];
        [bl endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        memcpy(out, b.contents, (size_t)w * h * 4);
    }
    return 1;
}

/* What the stress test watches for growth: lists allocated, the bytes and
   retained objects they have room for, lists not yet retired, upload ring
   chunks, and whether a render thread runs; then the in-flight depth the
   last frame was submitted with. */
uint32_t mtl_draw_ring_chunks(mtl_device *d);
D9_EXPORT void metal_d3d9_frame_stats(void *dev, uint64_t out[7])
{
    mtl_device *d = dev;
    mtl_frames *f = d->fq;
    uint64_t bytes = 0, keep = 0;
    for (mtl_seg *s = f->spare; s; s = s->next_q) {
        bytes += s->cap;
        keep += s->capkeep;
    }
    for (mtl_seg *s = f->sub_head; s; s = s->next_sub) {
        bytes += s->cap;
        keep += s->capkeep;
    }
    bytes += f->cur->cap;
    keep += f->cur->capkeep;
    out[0] = f->nsegs;
    out[1] = bytes;
    out[2] = keep;
    out[3] = f->nsub;
    out[4] = mtl_draw_ring_chunks(d);
    out[5] = (uint64_t)f->threaded;
    out[6] = (uint64_t)f->depth;
}
