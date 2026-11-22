/*
 * GPU copies of the CPU-backed resources: vertex and index buffers in
 * shared MTLBuffers, textures as MTLTextures made on first bind.
 *
 * The engine locks its eight dynamic vertex and index buffers about 350
 * times a frame, with DISCARD at each wrap and NOOVERWRITE in between.
 * A buffer's CPU memory is therefore the contents of a shared MTLBuffer,
 * written in place, and a lock that could touch bytes the GPU has not
 * read yet (DISCARD, or no flag at all) moves the buffer to fresh storage
 * first, the way DXVK renames. The old storage stays alive in every
 * recorded list that reads it (the list retains it, frame.m) and comes
 * back to the buffer's free list once its serial has completed.
 *
 * Textures keep their CPU memory as the master copy, since the bridge
 * bounces every lock through it. Unlock marks the level dirty; the next
 * bind uploads dirty levels in place when no recorded list can still
 * read the texture, or rebuilds the whole texture when one can, so a
 * draw already recorded never sees the new contents.
 *
 * "Can still read" is a serial above d->completed, which covers lists
 * waiting for the render thread as well as those on the GPU, so the same
 * rules give every list the contents of record time.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "mtl_objc.h"
#include "texconv.h"

#include <TargetConditionals.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MTLD(d) ((__bridge id<MTLDevice>)(d)->mtl)

#define D3DLOCK_READONLY 0x10u
#define D3DLOCK_DISCARD 0x2000u
#define D3DLOCK_NOOVERWRITE 0x1000u

static void cf_drop(void **p)
{
    if (*p) CFRelease(*p);
    *p = NULL;
}

int mtl_busy(mtl_device *d, uint64_t serial)
{
    return serial && serial > atomic_load_explicit(&d->completed, memory_order_acquire);
}

/* Buffers */

static id<MTLBuffer> new_storage(mtl_buffer *b)
{
    mtl_device *d = b->o.dev;
    for (uint32_t i = 0; i < b->nfree; i++) {
        if (mtl_busy(d, b->free_serial[i])) continue;
        id<MTLBuffer> g = CFBridgingRelease(b->free_buf[i]);
        b->nfree--;
        b->free_buf[i] = b->free_buf[b->nfree];
        b->free_serial[i] = b->free_serial[b->nfree];
        return g;
    }
    return [MTLD(d) newBufferWithLength:(NSUInteger)b->length + 16 options:MTLResourceStorageModeShared];
}

int mtl_buffer_gpu_init(mtl_buffer *b)
{
    @autoreleasepool {
        id<MTLBuffer> g = new_storage(b);
        if (!g) return 0;
        memset(g.contents, 0, g.length);
        b->gbuf = (void *)CFBridgingRetain(g);
        b->mem = g.contents;
    }
    return 1;
}

/* Moves the buffer to storage no submitted or encoded draw reads. Keep
   says whether the old bytes carry over (a lock without DISCARD). */
static void rename_buffer(mtl_buffer *b, int keep)
{
    @autoreleasepool {
        id<MTLBuffer> g = new_storage(b);
        if (!g) return; /* out of memory: write in place, as before */
        if (keep) memcpy(g.contents, b->mem, b->length);
        if (b->nfree < MTL_BUF_FREE) {
            b->free_buf[b->nfree] = b->gbuf;
            b->free_serial[b->nfree] = b->last_use;
            b->nfree++;
        } else {
            CFRelease(b->gbuf);
            b->o.dev->free_drops++;
        }
        b->gbuf = (void *)CFBridgingRetain(g);
        b->mem = g.contents;
        b->last_use = 0;
        b->o.dev->renames++;
    }
}

void mtl_buffer_lock_prepare(mtl_buffer *b, uint32_t flags)
{
    if (!b->gbuf || (flags & (D3DLOCK_READONLY | D3DLOCK_NOOVERWRITE))) return;
    mtl_device *d = b->o.dev;
    if (b->last_use == d->serial || mtl_busy(d, b->last_use)) rename_buffer(b, !(flags & D3DLOCK_DISCARD));
}

void mtl_buffer_gpu_free(mtl_buffer *b)
{
    for (uint32_t i = 0; i < b->nfree; i++) cf_drop(&b->free_buf[i]);
    b->nfree = 0;
    cf_drop(&b->gbuf);
    b->mem = NULL;
}

/* Capabilities */

void mtl_caps_query(id<MTLDevice> dev, mtl_gpu_caps *c)
{
    memset(c, 0, sizeof *c);
    const char *force = getenv("RT_METAL_FORCE_LIMITS");
    c->forced = force && atoi(force) > 0;
    if (@available(macOS 11.0, iOS 16.4, *)) c->bc = dev.supportsBCTextureCompression;
    /* Apple's simulator GPU refuses B5G6R5, BGR5A1, A1BGR5 and ABGR4 and
       Metal has no query for that list ("Developing Metal apps that run
       in Simulator" says to conditionalize on the simulator for exactly
       these); Macs without an Apple GPU lack them by family. */
    c->packed16 = [dev supportsFamily:MTLGPUFamilyApple1];
#if TARGET_OS_SIMULATOR
    c->packed16 = 0;
#endif
    c->ab_tier = dev.argumentBuffersSupport == MTLArgumentBuffersTier2 ? 2 : 1;
    /* Texture argument table entries per stage, from the Metal feature
       set tables: 31 on Apple2 and Apple3 (which the simulator reports
       itself as), 96 on Apple4 and Apple5, 128 from Apple6 and on Macs. */
    int mac2 = 0;
#if TARGET_OS_OSX
    /* Intel Macs: Mac2 is the only family they report, and it is
       deprecated only for its replacement (Apple7), which they lack. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    mac2 = [dev supportsFamily:MTLGPUFamilyMac2];
#pragma clang diagnostic pop
#endif
    c->max_tex = [dev supportsFamily:MTLGPUFamilyApple6] || mac2 ? 128 : [dev supportsFamily:MTLGPUFamilyApple4] ? 96 : 31;
    c->max_ab_samplers = (uint32_t)dev.maxArgumentBufferSamplerCount;
    /* The pipeline archive's loader asserts ("Target device architecture
       is nil") on a device that reports no architecture, the simulator
       GPU among them, so there is no archive call at all without one. */
    NSString *arch = nil;
    if (@available(macOS 14.0, iOS 17.0, *)) arch = ((MTLArchitecture *)(id)dev.architecture).name;
    c->archive = arch.length > 0;
    if (c->forced) {
        c->bc = c->packed16 = c->archive = 0;
        c->ab_tier = 1;
        c->max_tex = 31;
        if (c->max_ab_samplers > 16) c->max_ab_samplers = 16;
    }
    char line[512];
    snprintf(line, sizeof line,
             "gpu %s%s: BC textures %s, 16-bit packed formats %s, argument buffers tier %u, %u textures per stage, %u "
             "samplers per argument buffer, pipeline archive %s%s",
             dev.name.UTF8String, arch.length ? [NSString stringWithFormat:@" (%@)", arch].UTF8String : "",
             c->bc ? "sampled" : "decoded on upload", c->packed16 ? "native" : "widened on upload", c->ab_tier, c->max_tex,
             c->max_ab_samplers, c->archive ? "yes" : "no (pipelines compiled each run)",
             c->forced ? " (RT_METAL_FORCE_LIMITS=1)" : "");
    /* Once per process, unless a later device differs. */
    static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    static char last[512];
    pthread_mutex_lock(&mu);
    int fresh = strcmp(last, line) != 0;
    if (fresh) memcpy(last, line, sizeof last);
    pthread_mutex_unlock(&mu);
    if (fresh) mtl_log("%s", line);
}

/* Formats */

typedef struct {
    MTLPixelFormat pf;
    MTLTextureSwizzleChannels sw;
    int ok;
    int conv; /* uploads go through mtl_conv_level into RGBA8 */
} fmt_map;

#define SW(r, g, b, a) MTLTextureSwizzleChannelsMake(MTLTextureSwizzle##r, MTLTextureSwizzle##g, MTLTextureSwizzle##b, MTLTextureSwizzle##a)

/* D3D9 formats name their channels from the most significant bit, and the
   packed Metal formats that match byte for byte are the ones MoltenVK pairs
   with the Vulkan format DXVK picks for each (d3d9_format.cpp). Formats
   with fewer channels read their missing ones the way D3D9 does: luminance
   replicated into RGB, alpha 1 where there is none. */
static fmt_map native_format(uint32_t f)
{
    switch (f) {
    case 21: return (fmt_map){MTLPixelFormatBGRA8Unorm, SW(Red, Green, Blue, Alpha), 1, 0};
    case 22: return (fmt_map){MTLPixelFormatBGRA8Unorm, SW(Red, Green, Blue, One), 1, 0};
    case 32: return (fmt_map){MTLPixelFormatRGBA8Unorm, SW(Red, Green, Blue, Alpha), 1, 0};  /* A8B8G8R8 */
    case 33: return (fmt_map){MTLPixelFormatRGBA8Unorm, SW(Red, Green, Blue, One), 1, 0};    /* X8B8G8R8 */
    case 23: return (fmt_map){MTLPixelFormatB5G6R5Unorm, SW(Red, Green, Blue, One), 1, 0};
    case 24: return (fmt_map){MTLPixelFormatBGR5A1Unorm, SW(Red, Green, Blue, One), 1, 0};   /* X1R5G5B5 */
    case 25: return (fmt_map){MTLPixelFormatBGR5A1Unorm, SW(Red, Green, Blue, Alpha), 1, 0}; /* A1R5G5B5 */
    case 26: return (fmt_map){MTLPixelFormatABGR4Unorm, SW(Green, Blue, Alpha, Red), 1, 0};  /* A4R4G4B4 */
    case 28: return (fmt_map){MTLPixelFormatA8Unorm, SW(Zero, Zero, Zero, Alpha), 1, 0};     /* A8 */
    case 50: return (fmt_map){MTLPixelFormatR8Unorm, SW(Red, Red, Red, One), 1, 0};          /* L8 */
    case 51: return (fmt_map){MTLPixelFormatRG8Unorm, SW(Red, Red, Red, Green), 1, 0};       /* A8L8 */
    case 60: return (fmt_map){MTLPixelFormatRG8Snorm, SW(Red, Green, One, One), 1, 0};       /* V8U8 */
    case 63: return (fmt_map){MTLPixelFormatRGBA8Snorm, SW(Red, Green, Blue, Alpha), 1, 0};  /* Q8W8V8U8 */
    case 36: return (fmt_map){MTLPixelFormatRGBA16Unorm, SW(Red, Green, Blue, Alpha), 1, 0}; /* A16B16G16R16 */
    case 111: return (fmt_map){MTLPixelFormatR16Float, SW(Red, One, One, One), 1, 0};        /* R16F */
    case 112: return (fmt_map){MTLPixelFormatRG16Float, SW(Red, Green, One, One), 1, 0};     /* G16R16F */
    case 113: return (fmt_map){MTLPixelFormatRGBA16Float, SW(Red, Green, Blue, Alpha), 1, 0};
    case 114: return (fmt_map){MTLPixelFormatR32Float, SW(Red, One, One, One), 1, 0};
    case 115: return (fmt_map){MTLPixelFormatRG32Float, SW(Red, Green, One, One), 1, 0};
    case 116: return (fmt_map){MTLPixelFormatRGBA32Float, SW(Red, Green, Blue, Alpha), 1, 0};
    case 0x31545844u: return (fmt_map){MTLPixelFormatBC1_RGBA, SW(Red, Green, Blue, Alpha), 1, 0};
    case 0x32545844u: case 0x33545844u: return (fmt_map){MTLPixelFormatBC2_RGBA, SW(Red, Green, Blue, Alpha), 1, 0};
    case 0x34545844u: case 0x35545844u: return (fmt_map){MTLPixelFormatBC3_RGBA, SW(Red, Green, Blue, Alpha), 1, 0};
    default: return (fmt_map){MTLPixelFormatInvalid, SW(Red, Green, Blue, Alpha), 0, 0};
    }
}

static int is_dxt(uint32_t f) { return f >= 0x31545844u && f <= 0x35545844u && (f & 0xFFu) == 0x44u; }
static int is_packed16(uint32_t f) { return f >= 23 && f <= 26; }

/* The format a texture is made in on this GPU. What it cannot sample is
   decoded or widened to RGBA8 on upload, whose channels are already in
   D3D9's order with alpha 1 where the format has none. */
static fmt_map map_format(const mtl_gpu_caps *c, uint32_t f)
{
    fmt_map m = native_format(f);
    if (m.ok && ((is_dxt(f) && !c->bc) || (is_packed16(f) && !c->packed16)) && mtl_conv_known(f))
        return (fmt_map){MTLPixelFormatRGBA8Unorm, SW(Red, Green, Blue, Alpha), 1, 1};
    return m;
}

int mtl_format_is_depth(uint32_t f) { return f >= 70 && f <= 83; }

uint32_t mtl_color_pixel_format(const mtl_device *d, uint32_t f)
{
    /* Render targets: the back buffer and the engine's one A8R8G8B8 target
       are BGRA8; other colour targets use their sampled format. */
    fmt_map m = map_format(&d->caps, f);
    return m.ok ? (uint32_t)m.pf : (uint32_t)MTLPixelFormatBGRA8Unorm;
}

/* Every D3D9 depth format becomes D32 float with S8: Apple GPUs have no
   D24S8, and one format keeps the pipeline keys small. */
uint32_t mtl_depth_pixel_format(void) { return (uint32_t)MTLPixelFormatDepth32Float_Stencil8; }

/* Textures */

static uint32_t level_dim(uint32_t v, uint32_t l)
{
    v >>= l;
    return v ? v : 1;
}

static id<MTLTexture> make_texture(mtl_texture *t)
{
    mtl_device *d = t->o.dev;
    int rt = (t->usage & D3DUSAGE_RENDERTARGET) != 0;
    fmt_map m = map_format(&d->caps, t->format);
    if (!m.ok) {
        if (!t->warned++) mtl_log("texture format %u has no Metal format; it samples as black", t->format);
        return nil;
    }
    MTLTextureDescriptor *td = [MTLTextureDescriptor new];
    td.textureType = t->faces == 6 ? MTLTextureTypeCube : MTLTextureType2D;
    td.pixelFormat = m.pf;
    td.width = t->width;
    td.height = t->height;
    td.mipmapLevelCount = t->levels;
    td.usage = MTLTextureUsageShaderRead | (rt ? MTLTextureUsageRenderTarget : 0);
    td.storageMode = rt ? MTLStorageModePrivate : MTLStorageModeShared;
    /* A swizzle may not be set on a texture that is rendered to; the one
       render target format the engine makes (A8R8G8B8) needs none. */
    if (!rt) td.swizzle = m.sw;
    id<MTLTexture> g = [MTLD(d) newTextureWithDescriptor:td];
    if (!g) mtl_log("texture %ux%u format %u levels %u: no MTLTexture", t->width, t->height, t->format, t->levels);
    return g;
}

/* One level's CPU memory into g, decoded first when the GPU cannot sample
   its format. The game's memory keeps what it wrote. */
static void upload_mem(id<MTLTexture> g, const mtl_gpu_caps *c, uint32_t fmt, uint32_t level, uint32_t slice, uint32_t w,
                       uint32_t h, const void *mem, uint32_t pitch, uint32_t rows)
{
    MTLRegion r = MTLRegionMake2D(0, 0, w, h);
    if (map_format(c, fmt).conv) {
        uint8_t *px = malloc((size_t)w * h * 4);
        if (!px) return;
        mtl_conv_level(fmt, mem, pitch, w, h, px);
        [g replaceRegion:r mipmapLevel:level slice:slice withBytes:px bytesPerRow:(NSUInteger)w * 4 bytesPerImage:(NSUInteger)w * h * 4];
        free(px);
        return;
    }
    [g replaceRegion:r mipmapLevel:level slice:slice withBytes:mem bytesPerRow:pitch bytesPerImage:(NSUInteger)pitch * rows];
}

static void upload_level(id<MTLTexture> g, mtl_texture *t, uint32_t face, uint32_t level)
{
    mtl_surface *s = &t->sub[face * t->levels + level];
    if (!s->mem) return; /* never locked: the new texture is already zero */
    upload_mem(g, &t->o.dev->caps, t->format, level, face, level_dim(t->width, level), level_dim(t->height, level),
               s->mem, s->pitch, s->rows);
}

id<MTLTexture> mtl_texture_from_mem(mtl_device *d, uint32_t fmt, uint32_t w, uint32_t h, const void *mem, uint32_t pitch,
                                    uint32_t rows)
{
    fmt_map m = map_format(&d->caps, fmt);
    if (!m.ok) return nil;
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:m.pf width:w height:h mipmapped:NO];
    td.storageMode = MTLStorageModeShared;
    td.swizzle = m.sw;
    id<MTLTexture> g = [MTLD(d) newTextureWithDescriptor:td];
    if (g) upload_mem(g, &d->caps, fmt, 0, 0, w, h, mem, pitch, rows);
    return g;
}

/* Test-only export: one level through the decoder, as an upload would. */
D9_EXPORT void metal_d3d9_conv_level(uint32_t fmt, const void *src, uint32_t pitch, uint32_t w, uint32_t h, void *rgba)
{
    mtl_conv_level(fmt, src, pitch, w, h, rgba);
}

id<MTLTexture> mtl_texture_gpu(mtl_texture *t)
{
    mtl_device *d = t->o.dev;
    if (t->usage & D3DUSAGE_RENDERTARGET) {
        if (!t->gtex) {
            id<MTLTexture> g = make_texture(t);
            if (g) t->gtex = (void *)CFBridgingRetain(g);
        }
        return (__bridge id<MTLTexture>)t->gtex;
    }
    if (t->gtex && !t->dirty) return (__bridge id<MTLTexture>)t->gtex;
    @autoreleasepool {
        int fresh = !t->gtex || t->last_use == d->serial || mtl_busy(d, t->last_use);
        id<MTLTexture> g = fresh ? make_texture(t) : (__bridge id<MTLTexture>)t->gtex;
        if (!g) {
            t->dirty = 0;
            return nil;
        }
        for (uint32_t f = 0; f < t->faces; f++)
            for (uint32_t l = 0; l < t->levels; l++) {
                mtl_surface *s = &t->sub[f * t->levels + l];
                if (fresh || s->dirty) upload_level(g, t, f, l);
                s->dirty = 0;
            }
        if (fresh) {
            if (t->gtex) d->tex_renames++;
            cf_drop(&t->gtex);
            t->gtex = (void *)CFBridgingRetain(g);
            t->last_use = 0;
        }
        t->dirty = 0;
        d->tex_uploads++;
    }
    return (__bridge id<MTLTexture>)t->gtex;
}

void mtl_texture_gpu_free(mtl_texture *t) { cf_drop(&t->gtex); }
