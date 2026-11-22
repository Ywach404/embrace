/*
 * Headless test of the Metal D3D9 backend, through the same door the
 * bridge uses: dlopen, Direct3DCreate9, then calls through vtable slots by
 * index with the plain C ABI. No window is created; the device renders to
 * its back buffer texture only.
 *
 * Usage: metal_d3d9_test LIB [METHODS_FILE|-] [SHADER_DIR]
 *
 * METHODS_FILE lists "Interface::Method" names, one per line (the methods
 * a run reached); each must resolve to an implementation, not a stop.
 *
 * SHADER_DIR is the S2b probe's input set (build/s2-shaders/in: the game's
 * own shader combos as .dxso files and index.json). With it the test also
 * draws offscreen with real game shaders (unlitgeneric, vs_1_1 and ps_1_1)
 * over synthetic geometry and checks pixels: a DXT1 textured quad, alpha
 * blend, alpha test, depth test, and a render target texture sampled back;
 * and it links every game pixel shader with a vertex shader of its family
 * into a Metal pipeline, through a fresh shader cache and then from it.
 * With it too, the mixed draws: DXVK's fixed-function vertex shader with
 * a game pixel shader and the reverse, and every game shader linked
 * against the fixed-function stage on the other side. And pipelines
 * across runs: a set of game shader pipelines drawn with an empty cache,
 * then on fresh devices from the binary archive the first one wrote (none
 * may be compiled cold), with the archive damaged or written for another
 * identity (detected and rebuilt), and in skip mode.
 *
 * Every scene the backend draws two ways, encoded on the game thread at
 * each call (RT_METAL_INLINE=1) or recorded into lists a render thread
 * replays, is also run on a device of each mode and the frames compared
 * bit for bit: game shader and fixed-function quads, DISCARD and plain
 * relocks between draws, readbacks in the middle of a frame, a list
 * replayed twice into a twin target, and a stress run of 300 frames in
 * flight checked frame by frame the way the bridge's luma ring reads
 * them, with lists, ring chunks and the process footprint watched for
 * growth. The whole suite also runs under either mode as the environment
 * sets it.
 *
 * RT_METAL_FORCE_LIMITS=1 runs everything as a GPU with no BC sampling,
 * no 16-bit packed formats, argument buffers tier 1, a 31-entry texture
 * table and no pipeline archive would (the simulator's): DXT and 16-bit
 * textures go through the upload decoder, and the pipeline tests check
 * that nothing is archived instead of that the archive serves. Either
 * way the decoder is checked texel by texel on known blocks, a pipeline
 * forced to fail must drop its draws without holding up the frame, and
 * every game shader's MSL must use texture indices 0 to 30 only.
 *
 * Without SHADER_DIR it still draws through the fixed-function path:
 * a lit, textured quad with transforms, a POSITIONT quad, an FVF draw,
 * cube maps, every texture format the Mac hub runs created, ColorFill,
 * UpdateSurface (the movie path's sequence), UpdateTexture, the video
 * memory probe, and Reset back to default state.
 */
#include "immersive.h"
#include "mtl_d3d9_slots.h"
#include "stereo.h"

#include <math.h>
#include <zlib.h>

#include <dirent.h>
#include <dlfcn.h>
#include <mach/mach.h>
#include <stdint.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MTL_SLOT_ENUM(i, m, n) SLOT_##i##_##m = n,
enum { MTL_SLOTS_IDirect3D9(MTL_SLOT_ENUM) };
enum { MTL_SLOTS_IDirect3DDevice9(MTL_SLOT_ENUM) };
enum { MTL_SLOTS_IDirect3DSurface9(MTL_SLOT_ENUM) };
enum { MTL_SLOTS_IDirect3DTexture9(MTL_SLOT_ENUM) };
enum { MTL_SLOTS_IDirect3DCubeTexture9(MTL_SLOT_ENUM) };
enum { MTL_SLOTS_IDirect3DVertexBuffer9(MTL_SLOT_ENUM) };
enum { MTL_SLOTS_IDirect3DIndexBuffer9(MTL_SLOT_ENUM) };

typedef int32_t HR;

typedef struct {
    uint32_t BackBufferWidth, BackBufferHeight, BackBufferFormat, BackBufferCount, MultiSampleType, MultiSampleQuality,
        SwapEffect;
    void *hDeviceWindow;
    int32_t Windowed, EnableAutoDepthStencil;
    uint32_t AutoDepthStencilFormat, Flags, FullScreen_RefreshRateInHz, PresentationInterval;
} present_params;

typedef struct {
    int32_t Pitch;
    void *pBits;
} locked_rect;

typedef struct {
    int32_t l, t, r, b;
} rect;

#define VT(obj, slot, T) ((T)((*(void ***)(obj))[slot]))

static int s_fail;

#define CHECK(cond, ...) \
    do { \
        if (cond) { \
            printf("PASS "); \
        } else { \
            printf("FAIL "); \
            s_fail++; \
        } \
        printf(__VA_ARGS__); \
        printf("\n"); \
    } while (0)

/* Reads the back buffer the way the bridge's rt_gfx_readback does:
   GetBackBuffer, GetDesc, CreateOffscreenPlainSurface in system memory,
   GetRenderTargetData, LockRect. Returns a malloc'd copy, w * h pixels. */
static uint32_t *readback(void *dev, uint32_t *w, uint32_t *h)
{
    void *bb = NULL, *sys = NULL;
    uint32_t desc[8];
    locked_rect lr;
    uint32_t *px = NULL;
    HR hr = VT(dev, SLOT_IDirect3DDevice9_GetBackBuffer, HR (*)(void *, uint32_t, uint32_t, uint32_t, void **))(
        dev, 0, 0, 0, &bb);
    if (!hr) hr = VT(bb, SLOT_IDirect3DSurface9_GetDesc, HR (*)(void *, uint32_t *))(bb, desc);
    if (!hr)
        hr = VT(dev, SLOT_IDirect3DDevice9_CreateOffscreenPlainSurface,
                HR (*)(void *, uint32_t, uint32_t, uint32_t, uint32_t, void **, void *))(dev, desc[6], desc[7], desc[0], 2,
                                                                                       &sys, NULL);
    if (!hr) hr = VT(dev, SLOT_IDirect3DDevice9_GetRenderTargetData, HR (*)(void *, void *, void *))(dev, bb, sys);
    if (!hr)
        hr = VT(sys, SLOT_IDirect3DSurface9_LockRect, HR (*)(void *, locked_rect *, const void *, uint32_t))(sys, &lr,
                                                                                                         NULL, 0x10);
    if (!hr) {
        *w = desc[6];
        *h = desc[7];
        px = malloc((size_t)*w * *h * 4);
        for (uint32_t y = 0; px && y < *h; y++)
            memcpy(px + (size_t)y * *w, (uint8_t *)lr.pBits + (size_t)y * (size_t)lr.Pitch, (size_t)*w * 4);
        VT(sys, SLOT_IDirect3DSurface9_UnlockRect, HR (*)(void *))(sys);
    }
    if (sys) VT(sys, SLOT_IDirect3DSurface9_Release, uint32_t (*)(void *))(sys);
    if (bb) VT(bb, SLOT_IDirect3DSurface9_Release, uint32_t (*)(void *))(bb);
    if (hr) printf("readback hr 0x%08X\n", (uint32_t)hr);
    return px;
}

static uint32_t count_rgb(const uint32_t *px, uint32_t n, uint32_t rgb)
{
    uint32_t k = 0;
    for (uint32_t i = 0; i < n; i++) k += (px[i] & 0x00FFFFFFu) == rgb;
    return k;
}

static void *make_device(void *d3d, uint32_t w, uint32_t h)
{
    present_params pp;
    memset(&pp, 0, sizeof pp);
    pp.BackBufferWidth = w;
    pp.BackBufferHeight = h;
    pp.BackBufferFormat = 22;
    pp.BackBufferCount = 1;
    pp.SwapEffect = 1;
    pp.Windowed = 1;
    pp.EnableAutoDepthStencil = 1;
    pp.AutoDepthStencilFormat = 75;
    pp.PresentationInterval = 0x80000000u;
    void *dev = NULL;
    HR hr = VT(d3d, SLOT_IDirect3D9_CreateDevice,
               HR (*)(void *, uint32_t, uint32_t, void *, uint32_t, present_params *, void **))(d3d, 0, 1, NULL, 0x52, &pp,
                                                                                               &dev);
    CHECK(!hr && dev, "CreateDevice %ux%u without a window: hr 0x%08X", w, h, (uint32_t)hr);
    return hr ? NULL : dev;
}

static HR clear(void *dev, uint32_t n, const rect *r, uint32_t flags, uint32_t color)
{
    return VT(dev, SLOT_IDirect3DDevice9_Clear, HR (*)(void *, uint32_t, const void *, uint32_t, uint32_t, float, uint32_t))(
        dev, n, r, flags, color, 1.0f, 0);
}

static void methods_check(const char *path, int (*impl)(const char *, const char *))
{
    FILE *f = fopen(path, "r");
    if (!f) {
        CHECK(0, "methods file %s readable", path);
        return;
    }
    char line[256];
    uint32_t n = 0, ok = 0;
    while (fgets(line, sizeof line, f)) {
        line[strcspn(line, "\r\n")] = 0;
        char *sep = strstr(line, "::");
        if (!sep) continue; /* Direct3DCreate9 is the export itself */
        *sep = 0;
        n++;
        int r = impl(line, sep + 2);
        if (r == 1) ok++;
        else printf("  %s::%s %s\n", line, sep + 2, r ? "is not a slot" : "is a stop");
    }
    fclose(f);
    CHECK(n && ok == n, "%u of %u methods in %s are implemented", ok, n, path);
}

/* Drawing with real game shaders. */

typedef struct {
    float x, y, z, u, v;
} vtx;

static const char *s_shader_dir;

static void *load_shader(void *dev, const char *model, const char *hash)
{
    char path[1024];
    snprintf(path, sizeof path, "%s/%s/%s.dxso", s_shader_dir, model, hash);
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    static uint8_t buf[1 << 20];
    size_t n = fread(buf, 1, sizeof buf, f);
    fclose(f);
    memset(buf + n, 0, 8);
    void *sh = NULL;
    int ps = model[0] == 'p';
    HR hr = VT(dev, ps ? SLOT_IDirect3DDevice9_CreatePixelShader : SLOT_IDirect3DDevice9_CreateVertexShader,
               HR (*)(void *, const void *, void **))(dev, buf, &sh);
    return hr ? NULL : sh;
}

static void rs(void *dev, uint32_t state, uint32_t value)
{
    VT(dev, SLOT_IDirect3DDevice9_SetRenderState, HR (*)(void *, uint32_t, uint32_t))(dev, state, value);
}

static void vs_const(void *dev, uint32_t reg, float a, float b, float c, float d)
{
    float v[4] = {a, b, c, d};
    VT(dev, SLOT_IDirect3DDevice9_SetVertexShaderConstantF, HR (*)(void *, uint32_t, const float *, uint32_t))(dev, reg,
                                                                                                             v, 1);
}

/* unlitgeneric's vertex shader: position through c48-c50 (model) then
   c8-c11 (view projection) with w from c0.y, texcoord through c42-c43,
   colour from c38. Identity everywhere, so positions are clip space. */
static void unlit_constants(void *dev, float r, float g, float b, float a)
{
    vs_const(dev, 0, 0, 1, 2, 0.5f);
    for (int i = 0; i < 4; i++) vs_const(dev, 8 + i, i == 0, i == 1, i == 2, i == 3);
    for (int i = 0; i < 3; i++) vs_const(dev, 48 + i, i == 0, i == 1, i == 2, i == 3);
    vs_const(dev, 42, 1, 0, 0, 0);
    vs_const(dev, 43, 0, 1, 0, 0);
    vs_const(dev, 38, r, g, b, a);
}

/* A quad in clip space, wound clockwise on screen (D3D9's front face). */
static void quad(void *dev, void *vb, float x0, float y0, float x1, float y1, float z)
{
    vtx q[4] = {{x0, y1, z, 0, 0}, {x1, y1, z, 1, 0}, {x1, y0, z, 1, 1}, {x0, y0, z, 0, 1}};
    void *p = NULL;
    VT(vb, SLOT_IDirect3DVertexBuffer9_Lock, HR (*)(void *, uint32_t, uint32_t, void **, uint32_t))(vb, 0, 0, &p, 0x2000);
    if (p) memcpy(p, q, sizeof q);
    VT(vb, SLOT_IDirect3DVertexBuffer9_Unlock, HR (*)(void *))(vb);
    VT(dev, SLOT_IDirect3DDevice9_DrawIndexedPrimitive,
       HR (*)(void *, uint32_t, int32_t, uint32_t, uint32_t, uint32_t, uint32_t))(dev, 4, 0, 0, 4, 0, 2);
}

static uint32_t chan(uint32_t px, int shift) { return (px >> shift) & 255; }
static int near(uint32_t px, uint32_t r, uint32_t g, uint32_t b, uint32_t tol)
{
    int dr = (int)chan(px, 16) - (int)r, dg = (int)chan(px, 8) - (int)g, db = (int)chan(px, 0) - (int)b;
    return abs(dr) <= (int)tol && abs(dg) <= (int)tol && abs(db) <= (int)tol;
}

static void draw_tests(void *d3d)
{
    void *dev = make_device(d3d, 64, 64);
    if (!dev) return;
    void *vs = load_shader(dev, "vs_1_1", "78dc66bc16bac6b1"); /* unlitgeneric combo 0 */
    void *ps = load_shader(dev, "ps_1_1", "371500f12c2f6512"); /* unlitgeneric: texture * colour, alpha test, fog */
    CHECK(vs && ps, "game shaders unlitgeneric vs_1_1 78dc66bc16bac6b1 and ps_1_1 371500f12c2f6512 created");
    if (!vs || !ps) return;

    static const uint8_t k_decl[] = {0, 0, 0, 0, 2, 0, 0, 0,  /* stream 0 offset 0 FLOAT3 POSITION0 */
                                     0, 0, 12, 0, 1, 0, 5, 0, /* stream 0 offset 12 FLOAT2 TEXCOORD0 */
                                     0xFF, 0, 0, 0, 17, 0, 0, 0};
    void *decl = NULL, *vb = NULL, *ib = NULL;
    VT(dev, SLOT_IDirect3DDevice9_CreateVertexDeclaration, HR (*)(void *, const void *, void **))(dev, k_decl, &decl);
    VT(dev, SLOT_IDirect3DDevice9_CreateVertexBuffer, HR (*)(void *, uint32_t, uint32_t, uint32_t, uint32_t, void **, void *))(
        dev, 4 * sizeof(vtx), 0x208, 0, 0, &vb, NULL); /* DYNAMIC | WRITEONLY, as the engine's */
    VT(dev, SLOT_IDirect3DDevice9_CreateIndexBuffer, HR (*)(void *, uint32_t, uint32_t, uint32_t, uint32_t, void **, void *))(
        dev, 12, 0x208, 101, 0, &ib, NULL);
    uint16_t *ix = NULL;
    VT(ib, SLOT_IDirect3DIndexBuffer9_Lock, HR (*)(void *, uint32_t, uint32_t, void **, uint32_t))(ib, 0, 0, (void **)&ix, 0);
    static const uint16_t k_ix[6] = {0, 1, 2, 0, 2, 3};
    if (ix) memcpy(ix, k_ix, sizeof k_ix);
    VT(ib, SLOT_IDirect3DIndexBuffer9_Unlock, HR (*)(void *))(ib);

    /* DXT1 8x8, one level, four solid blocks: red, green / blue, white. */
    void *tex = NULL;
    VT(dev, SLOT_IDirect3DDevice9_CreateTexture,
       HR (*)(void *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, void **, void *))(dev, 8, 8, 1, 0,
                                                                                                  0x31545844u, 1, &tex, NULL);
    locked_rect lr = {0, NULL};
    VT(tex, SLOT_IDirect3DTexture9_LockRect, HR (*)(void *, uint32_t, locked_rect *, const void *, uint32_t))(tex, 0, &lr,
                                                                                                          NULL, 0);
    static const uint16_t k_c565[4] = {0xF800, 0x07E0, 0x001F, 0xFFFF};
    for (int b = 0; lr.pBits && b < 4; b++) {
        uint8_t *blk = (uint8_t *)lr.pBits + (b / 2) * lr.Pitch + (b % 2) * 8;
        memcpy(blk, &k_c565[b], 2);
        memcpy(blk + 2, &k_c565[b], 2);
        memset(blk + 4, 0, 4); /* every texel is colour 0 */
    }
    VT(tex, SLOT_IDirect3DTexture9_UnlockRect, HR (*)(void *, uint32_t))(tex, 0);

    /* A white DXT1 4x4, for the blend and depth checks. */
    void *white = NULL;
    VT(dev, SLOT_IDirect3DDevice9_CreateTexture,
       HR (*)(void *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, void **, void *))(dev, 4, 4, 1, 0,
                                                                                                  0x31545844u, 1, &white, NULL);
    VT(white, SLOT_IDirect3DTexture9_LockRect, HR (*)(void *, uint32_t, locked_rect *, const void *, uint32_t))(white, 0, &lr,
                                                                                                            NULL, 0);
    if (lr.pBits) memset(lr.pBits, 0xFF, 4), memset((uint8_t *)lr.pBits + 4, 0, 4);
    VT(white, SLOT_IDirect3DTexture9_UnlockRect, HR (*)(void *, uint32_t))(white, 0);

    VT(dev, SLOT_IDirect3DDevice9_SetVertexDeclaration, HR (*)(void *, void *))(dev, decl);
    VT(dev, SLOT_IDirect3DDevice9_SetVertexShader, HR (*)(void *, void *))(dev, vs);
    VT(dev, SLOT_IDirect3DDevice9_SetPixelShader, HR (*)(void *, void *))(dev, ps);
    VT(dev, SLOT_IDirect3DDevice9_SetStreamSource, HR (*)(void *, uint32_t, void *, uint32_t, uint32_t))(dev, 0, vb, 0,
                                                                                                       sizeof(vtx));
    VT(dev, SLOT_IDirect3DDevice9_SetIndices, HR (*)(void *, void *))(dev, ib);
    HR (*set_tex)(void *, uint32_t, void *) = VT(dev, SLOT_IDirect3DDevice9_SetTexture, HR (*)(void *, uint32_t, void *));
    HR (*set_ss)(void *, uint32_t, uint32_t, uint32_t) =
        VT(dev, SLOT_IDirect3DDevice9_SetSamplerState, HR (*)(void *, uint32_t, uint32_t, uint32_t));
    set_ss(dev, 0, 5, 1); /* point filtering, so block edges stay sharp */
    set_ss(dev, 0, 6, 1);
    set_ss(dev, 0, 1, 3); /* clamp */
    set_ss(dev, 0, 2, 3);

    /* 1. DXT1 textured quad, with the default CCW cull: the clockwise
       quad is front facing and must draw. */
    set_tex(dev, 0, tex);
    unlit_constants(dev, 1, 1, 1, 1);
    clear(dev, 0, NULL, 0x7, 0xFF000000u);
    quad(dev, vb, -1, -1, 1, 1, 0.5f);
    uint32_t w = 0, h = 0, *px = readback(dev, &w, &h);
    int ok = px && near(px[16 * 64 + 16], 255, 0, 0, 2) && near(px[16 * 64 + 48], 0, 255, 0, 2) &&
             near(px[48 * 64 + 16], 0, 0, 255, 2) && near(px[48 * 64 + 48], 255, 255, 255, 2);
    CHECK(ok, "DXT1 textured quad (game shaders, default cull): quadrants 0x%06X 0x%06X 0x%06X 0x%06X (want FF0000 "
              "00FF00 0000FF FFFFFF)",
          px ? px[16 * 64 + 16] & 0xFFFFFF : 0, px ? px[16 * 64 + 48] & 0xFFFFFF : 0, px ? px[48 * 64 + 16] & 0xFFFFFF : 0,
          px ? px[48 * 64 + 48] & 0xFFFFFF : 0);
    free(px);

    /* 2. Alpha blend: red at alpha 0.5 over a blue clear, SRCALPHA and
       INVSRCALPHA, over the left half only. */
    rs(dev, 22, 1); /* CULL_NONE */
    set_tex(dev, 0, white);
    clear(dev, 0, NULL, 0x7, 0xFF0000FFu);
    rs(dev, 27, 1);
    rs(dev, 19, 5);
    rs(dev, 20, 6);
    unlit_constants(dev, 1, 0, 0, 0.5f);
    quad(dev, vb, -1, -1, 0, 1, 0.5f);
    rs(dev, 27, 0);
    px = readback(dev, &w, &h);
    ok = px && near(px[32 * 64 + 16], 128, 0, 128, 2) && near(px[32 * 64 + 48], 0, 0, 255, 0);
    CHECK(ok, "alpha blend 50%% red over blue: 0x%06X (want about 800080), untouched 0x%06X (want 0000FF)",
          px ? px[32 * 64 + 16] & 0xFFFFFF : 0, px ? px[32 * 64 + 48] & 0xFFFFFF : 0);
    free(px);

    /* 3. Alpha test, in the pixel shader through the spec data: alpha 0.25
       against GREATER 0x80 is discarded, alpha 0.75 passes. */
    rs(dev, 15, 1);
    rs(dev, 24, 0x80);
    rs(dev, 25, 5);
    unlit_constants(dev, 1, 1, 0, 0.25f);
    quad(dev, vb, -1, -1, 0, 1, 0.5f);
    unlit_constants(dev, 0, 1, 0, 0.75f);
    quad(dev, vb, 0, -1, 1, 1, 0.5f);
    rs(dev, 15, 0);
    px = readback(dev, &w, &h);
    ok = px && near(px[32 * 64 + 16], 128, 0, 128, 2) && near(px[32 * 64 + 48], 0, 255, 0, 0);
    CHECK(ok, "alpha test GREATER 0x80: alpha 0.25 discarded (0x%06X, want about 800080), alpha 0.75 drawn (0x%06X)",
          px ? px[32 * 64 + 16] & 0xFFFFFF : 0, px ? px[32 * 64 + 48] & 0xFFFFFF : 0);
    free(px);

    /* 4. Depth: green at z 0.5 on the left, then red at z 0.75 over all,
       LESSEQUAL. Red must lose on the left and win on the right. */
    clear(dev, 0, NULL, 0x7, 0xFF000000u);
    unlit_constants(dev, 0, 1, 0, 1);
    quad(dev, vb, -1, -1, 0, 1, 0.5f);
    unlit_constants(dev, 1, 0, 0, 1);
    quad(dev, vb, -1, -1, 1, 1, 0.75f);
    px = readback(dev, &w, &h);
    ok = px && near(px[32 * 64 + 16], 0, 255, 0, 0) && near(px[32 * 64 + 48], 255, 0, 0, 0);
    CHECK(ok, "depth test LESSEQUAL: near green kept 0x%06X, far red drawn where nothing is nearer 0x%06X",
          px ? px[32 * 64 + 16] & 0xFFFFFF : 0, px ? px[32 * 64 + 48] & 0xFFFFFF : 0);
    free(px);
    /* and with the test off the far quad covers everything */
    rs(dev, 7, 0);
    quad(dev, vb, -1, -1, 1, 1, 0.75f);
    rs(dev, 7, 1);
    px = readback(dev, &w, &h);
    CHECK(px && near(px[32 * 64 + 16], 255, 0, 0, 0), "depth test off: far red covers the near green (0x%06X)",
          px ? px[32 * 64 + 16] & 0xFFFFFF : 0);
    free(px);

    /* 5. A render target texture, as the menu's 1024 A8R8G8B8 target:
       cleared through SetRenderTarget on its level, then sampled. */
    void *rtt = NULL, *lvl = NULL, *bb = NULL;
    VT(dev, SLOT_IDirect3DDevice9_CreateTexture,
       HR (*)(void *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, void **, void *))(dev, 32, 32, 1, 1, 21, 0,
                                                                                                  &rtt, NULL);
    if (rtt) VT(rtt, SLOT_IDirect3DTexture9_GetSurfaceLevel, HR (*)(void *, uint32_t, void **))(rtt, 0, &lvl);
    VT(dev, SLOT_IDirect3DDevice9_GetRenderTarget, HR (*)(void *, uint32_t, void **))(dev, 0, &bb);
    HR hs = VT(dev, SLOT_IDirect3DDevice9_SetRenderTarget, HR (*)(void *, uint32_t, void *))(dev, 0, lvl);
    clear(dev, 0, NULL, 0x1, 0xFF20C040u);
    VT(dev, SLOT_IDirect3DDevice9_SetRenderTarget, HR (*)(void *, uint32_t, void *))(dev, 0, bb);
    set_tex(dev, 0, rtt);
    unlit_constants(dev, 1, 1, 1, 1);
    quad(dev, vb, -1, -1, 1, 1, 0.5f);
    px = readback(dev, &w, &h);
    CHECK(!hs && px && near(px[32 * 64 + 32], 0x20, 0xC0, 0x40, 1),
          "SetRenderTarget to a texture level, Clear, sample it back: 0x%06X (want 20C040)",
          px ? px[32 * 64 + 32] & 0xFFFFFF : 0);
    free(px);
    if (lvl) VT(lvl, SLOT_IDirect3DSurface9_Release, uint32_t (*)(void *))(lvl);
    if (bb) VT(bb, SLOT_IDirect3DSurface9_Release, uint32_t (*)(void *))(bb);

    /* 6. Dynamic buffer renaming: DISCARD locks between draws in one
       pass must not change what earlier draws read. */
    clear(dev, 0, NULL, 0x7, 0xFF000000u);
    set_tex(dev, 0, white);
    unlit_constants(dev, 0, 0, 1, 1);
    quad(dev, vb, -1, -1, 0, 1, 0.5f);
    unlit_constants(dev, 1, 1, 0, 1);
    quad(dev, vb, 0, -1, 1, 1, 0.5f);
    px = readback(dev, &w, &h);
    CHECK(px && near(px[32 * 64 + 16], 0, 0, 255, 0) && near(px[32 * 64 + 48], 255, 255, 0, 0),
          "two DISCARD-locked quads in one pass: 0x%06X and 0x%06X (want 0000FF, FFFF00)",
          px ? px[32 * 64 + 16] & 0xFFFFFF : 0, px ? px[32 * 64 + 48] & 0xFFFFFF : 0);
    free(px);

    /* 7. Mixed stages. DXVK's fixed-function vertex shader feeding the
       game's pixel shader: no lighting and no vertex colour make the
       diffuse white, identity transforms keep positions in clip space,
       so the quad shows the texture as test 1 did. */
    set_tex(dev, 0, tex);
    rs(dev, 137, 0); /* LIGHTING off */
    VT(dev, SLOT_IDirect3DDevice9_SetVertexShader, HR (*)(void *, void *))(dev, NULL);
    clear(dev, 0, NULL, 0x7, 0xFF000000u);
    quad(dev, vb, -1, -1, 1, 1, 0.5f);
    px = readback(dev, &w, &h);
    ok = px && near(px[16 * 64 + 16], 255, 0, 0, 2) && near(px[16 * 64 + 48], 0, 255, 0, 2) &&
         near(px[48 * 64 + 16], 0, 0, 255, 2) && near(px[48 * 64 + 48], 255, 255, 255, 2);
    CHECK(ok, "fixed-function VS with the game's ps_1_1: quadrants 0x%06X 0x%06X 0x%06X 0x%06X (want FF0000 00FF00 "
              "0000FF FFFFFF)",
          px ? px[16 * 64 + 16] & 0xFFFFFF : 0, px ? px[16 * 64 + 48] & 0xFFFFFF : 0, px ? px[48 * 64 + 16] & 0xFFFFFF : 0,
          px ? px[48 * 64 + 48] & 0xFFFFFF : 0);
    free(px);

    /* and the game's vertex shader feeding the fixed-function pixel
       shader: stage 0's default MODULATE of texture and diffuse, with the
       vertex shader's colour (c38) yellow. */
    VT(dev, SLOT_IDirect3DDevice9_SetVertexShader, HR (*)(void *, void *))(dev, vs);
    VT(dev, SLOT_IDirect3DDevice9_SetPixelShader, HR (*)(void *, void *))(dev, NULL);
    unlit_constants(dev, 1, 1, 0, 1);
    clear(dev, 0, NULL, 0x7, 0xFF404040u);
    quad(dev, vb, -1, -1, 1, 1, 0.5f);
    px = readback(dev, &w, &h);
    ok = px && near(px[16 * 64 + 16], 255, 0, 0, 2) && near(px[16 * 64 + 48], 0, 255, 0, 2) &&
         near(px[48 * 64 + 16], 0, 0, 0, 2) && near(px[48 * 64 + 48], 255, 255, 0, 2);
    CHECK(ok, "game vs_1_1 with the fixed-function PS (texture times yellow): quadrants 0x%06X 0x%06X 0x%06X 0x%06X "
              "(want FF0000 00FF00 000000 FFFF00)",
          px ? px[16 * 64 + 16] & 0xFFFFFF : 0, px ? px[16 * 64 + 48] & 0xFFFFFF : 0, px ? px[48 * 64 + 16] & 0xFFFFFF : 0,
          px ? px[48 * 64 + 48] & 0xFFFFFF : 0);
    free(px);

    set_tex(dev, 0, NULL);
    VT(dev, SLOT_IDirect3DDevice9_SetVertexShader, HR (*)(void *, void *))(dev, NULL);
    void *objs[] = {vs, ps, decl, vb, ib, tex, white, rtt};
    for (unsigned i = 0; i < sizeof objs / sizeof objs[0]; i++)
        if (objs[i]) VT(objs[i], 2, uint32_t (*)(void *))(objs[i]);
    uint32_t cd = VT(dev, SLOT_IDirect3DDevice9_Release, uint32_t (*)(void *))(dev);
    CHECK(cd == 0, "device with drawing state released: %u", cd);
}

/* Fixed function, formats, copies and Reset, all without game shaders. */

static void release(void *o)
{
    if (o) VT(o, 2, uint32_t (*)(void *))(o);
}

static void *new_texture(void *dev, uint32_t w, uint32_t h, uint32_t levels, uint32_t fmt, uint32_t pool, HR *hr_out)
{
    void *t = NULL;
    HR hr = VT(dev, SLOT_IDirect3DDevice9_CreateTexture,
               HR (*)(void *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, void **, void *))(
        dev, w, h, levels, 0, fmt, pool, &t, NULL);
    if (hr_out) *hr_out = hr;
    return hr ? NULL : t;
}

static void *new_cube(void *dev, uint32_t edge, uint32_t fmt)
{
    void *t = NULL;
    HR hr = VT(dev, SLOT_IDirect3DDevice9_CreateCubeTexture,
               HR (*)(void *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, void **, void *))(dev, edge, 1, 0, fmt, 1,
                                                                                                &t, NULL);
    return hr ? NULL : t;
}

/* Fills level 0 of a texture (face < 0) or one cube face with one texel
   or block value, repeated. */
static void fill(void *t, int face, const void *texel, uint32_t texel_bytes, uint32_t row_units, uint32_t rows,
                 uint32_t flags)
{
    locked_rect lr = {0, NULL};
    if (face < 0)
        VT(t, SLOT_IDirect3DTexture9_LockRect, HR (*)(void *, uint32_t, locked_rect *, const void *, uint32_t))(t, 0, &lr,
                                                                                                            NULL, flags);
    else
        VT(t, SLOT_IDirect3DCubeTexture9_LockRect,
           HR (*)(void *, uint32_t, uint32_t, locked_rect *, const void *, uint32_t))(t, (uint32_t)face, 0, &lr, NULL, flags);
    for (uint32_t y = 0; lr.pBits && y < rows; y++)
        for (uint32_t x = 0; x < row_units; x++)
            memcpy((uint8_t *)lr.pBits + (size_t)y * lr.Pitch + (size_t)x * texel_bytes, texel, texel_bytes);
    if (face < 0) VT(t, SLOT_IDirect3DTexture9_UnlockRect, HR (*)(void *, uint32_t))(t, 0);
    else VT(t, SLOT_IDirect3DCubeTexture9_UnlockRect, HR (*)(void *, uint32_t, uint32_t))(t, (uint32_t)face, 0);
}

static void tss(void *dev, uint32_t stage, uint32_t type, uint32_t value)
{
    VT(dev, SLOT_IDirect3DDevice9_SetTextureStageState, HR (*)(void *, uint32_t, uint32_t, uint32_t))(dev, stage, type,
                                                                                                     value);
}

static void set_xf(void *dev, uint32_t t, const float *m)
{
    VT(dev, SLOT_IDirect3DDevice9_SetTransform, HR (*)(void *, uint32_t, const float *))(dev, t, m);
}

static void *set_decl(void *dev, const uint8_t *elems)
{
    void *decl = NULL;
    VT(dev, SLOT_IDirect3DDevice9_CreateVertexDeclaration, HR (*)(void *, const void *, void **))(dev, elems, &decl);
    VT(dev, SLOT_IDirect3DDevice9_SetVertexDeclaration, HR (*)(void *, void *))(dev, decl);
    return decl;
}

static HR draw_up(void *dev, uint32_t prims, const void *v, uint32_t stride)
{
    return VT(dev, SLOT_IDirect3DDevice9_DrawPrimitiveUP, HR (*)(void *, uint32_t, uint32_t, const void *, uint32_t))(
        dev, 4, prims, v, stride);
}

/* A pretransformed quad, two triangles, with an extra float3 per vertex
   (texcoord, or ignored). */
typedef struct {
    float x, y, z, rhw, u, v, w;
} tvtx;

static void screen_quad(void *dev, float x0, float y0, float x1, float y1, float u, float v, float w)
{
    tvtx q[6] = {{x0, y0, 0.5f, 1, u, v, w}, {x1, y0, 0.5f, 1, u, v, w}, {x1, y1, 0.5f, 1, u, v, w},
                 {x0, y0, 0.5f, 1, u, v, w}, {x1, y1, 0.5f, 1, u, v, w}, {x0, y1, 0.5f, 1, u, v, w}};
    draw_up(dev, 2, q, sizeof q[0]);
}

static uint32_t at(const uint32_t *px, uint32_t w, uint32_t x, uint32_t y) { return px ? px[y * w + x] & 0xFFFFFF : 0; }

static void ff_tests(void *d3d)
{
    void *dev = make_device(d3d, 64, 64);
    if (!dev) return;
    HR (*set_tex)(void *, uint32_t, void *) = VT(dev, SLOT_IDirect3DDevice9_SetTexture, HR (*)(void *, uint32_t, void *));
    HR (*set_ss)(void *, uint32_t, uint32_t, uint32_t) =
        VT(dev, SLOT_IDirect3DDevice9_SetSamplerState, HR (*)(void *, uint32_t, uint32_t, uint32_t));
    uint32_t w = 0, h = 0, *px;
    int ok;
    rs(dev, 22, 1); /* CULL_NONE: winding is the draw tests' business */
    set_ss(dev, 0, 5, 1);
    set_ss(dev, 0, 6, 1);
    set_ss(dev, 0, 1, 3);
    set_ss(dev, 0, 2, 3);

    /* 1. Lit and textured, fixed function on both stages: one white
       directional light along +z on normals facing the viewer, a material
       diffuse of (0.5, 0.25, 1), stage 0's default MODULATE with the
       quadrant DXT1 texture. Identity transforms keep clip space. */
    void *tex = new_texture(dev, 8, 8, 1, 0x31545844u, 1, NULL);
    static const uint16_t k_c565[4] = {0xF800, 0x07E0, 0x001F, 0xFFFF};
    if (tex) {
        locked_rect lr = {0, NULL};
        VT(tex, SLOT_IDirect3DTexture9_LockRect, HR (*)(void *, uint32_t, locked_rect *, const void *, uint32_t))(
            tex, 0, &lr, NULL, 0);
        for (int b = 0; lr.pBits && b < 4; b++) {
            uint8_t *blk = (uint8_t *)lr.pBits + (b / 2) * lr.Pitch + (b % 2) * 8;
            memcpy(blk, &k_c565[b], 2);
            memcpy(blk + 2, &k_c565[b], 2);
            memset(blk + 4, 0, 4);
        }
        VT(tex, SLOT_IDirect3DTexture9_UnlockRect, HR (*)(void *, uint32_t))(tex, 0);
    }
    static const uint8_t k_lit_decl[] = {0, 0, 0,  0, 2, 0, 0, 0, /* FLOAT3 POSITION0 */
                                         0, 0, 12, 0, 2, 0, 3, 0, /* FLOAT3 NORMAL0 */
                                         0, 0, 24, 0, 1, 0, 5, 0, /* FLOAT2 TEXCOORD0 */
                                         0xFF, 0, 0, 0, 17, 0, 0, 0};
    void *lit_decl = set_decl(dev, k_lit_decl);
    struct {
        uint32_t type;
        float diffuse[4], specular[4], ambient[4], pos[3], dir[3], range, falloff, att0, att1, att2, theta, phi;
    } light = {3, {1, 1, 1, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0}, {0, 0, 1}, 0, 0, 0, 0, 0, 0, 0};
    float material[17] = {0.5f, 0.25f, 1.0f, 1.0f}; /* diffuse; ambient, specular, emissive and power zero */
    HR hl = VT(dev, SLOT_IDirect3DDevice9_SetLight, HR (*)(void *, uint32_t, const void *))(dev, 0, &light);
    HR he = VT(dev, SLOT_IDirect3DDevice9_LightEnable, HR (*)(void *, uint32_t, int32_t))(dev, 0, 1);
    HR hm = VT(dev, SLOT_IDirect3DDevice9_SetMaterial, HR (*)(void *, const void *))(dev, material);
    set_tex(dev, 0, tex);
    typedef struct {
        float x, y, z, nx, ny, nz, u, v;
    } lvtx;
#define LIT_QUAD(x0, y0, x1, y1) \
    { \
        lvtx q[6] = {{x0, y1, 0.5f, 0, 0, -1, 0, 0}, {x1, y1, 0.5f, 0, 0, -1, 1, 0}, {x1, y0, 0.5f, 0, 0, -1, 1, 1}, \
                     {x0, y1, 0.5f, 0, 0, -1, 0, 0}, {x1, y0, 0.5f, 0, 0, -1, 1, 1}, {x0, y0, 0.5f, 0, 0, -1, 0, 1}}; \
        draw_up(dev, 2, q, sizeof q[0]); \
    }
    clear(dev, 0, NULL, 0x7, 0xFF404040u);
    LIT_QUAD(-1.0f, -1.0f, 1.0f, 1.0f);
    px = readback(dev, &w, &h);
    ok = !hl && !he && !hm && px && near(at(px, w, 16, 16), 128, 0, 0, 2) && near(at(px, w, 48, 16), 0, 64, 0, 2) &&
         near(at(px, w, 16, 48), 0, 0, 255, 2) && near(at(px, w, 48, 48), 128, 64, 255, 2);
    CHECK(ok, "fixed-function lit textured quad: quadrants 0x%06X 0x%06X 0x%06X 0x%06X (want 800000 004000 0000FF "
              "8040FF)",
          at(px, w, 16, 16), at(px, w, 48, 16), at(px, w, 16, 48), at(px, w, 48, 48));
    free(px);

    /* 2. Transform order: world scales x by a half, view moves x by
       -0.5. D3D applies world then view, so x in [0, 1] lands in
       [-0.5, 0]; the other order would give [-0.25, 0.25]. */
    float world[16] = {0.5f, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    float view[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -0.5f, 0, 0, 1};
    float ident[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    set_xf(dev, 256, world);
    set_xf(dev, 2, view);
    clear(dev, 0, NULL, 0x7, 0xFF404040u);
    LIT_QUAD(0.0f, -1.0f, 1.0f, 1.0f);
    px = readback(dev, &w, &h);
    /* x = -0.4 is pixel 19, x = 0.1 is pixel 35 */
    ok = px && !near(at(px, w, 19, 16), 0x40, 0x40, 0x40, 0) && near(at(px, w, 35, 16), 0x40, 0x40, 0x40, 0) &&
         near(at(px, w, 8, 16), 0x40, 0x40, 0x40, 0);
    CHECK(ok, "world then view: drawn at x -0.4 (0x%06X), clear at 0.1 (0x%06X) and -0.75 (0x%06X)", at(px, w, 19, 16),
          at(px, w, 35, 16), at(px, w, 8, 16));
    free(px);
    set_xf(dev, 256, ident);
    set_xf(dev, 2, ident);

    /* 3. The light turned around lights nothing: black, with no ambient. */
    light.dir[2] = -1;
    VT(dev, SLOT_IDirect3DDevice9_SetLight, HR (*)(void *, uint32_t, const void *))(dev, 0, &light);
    clear(dev, 0, NULL, 0x7, 0xFF404040u);
    LIT_QUAD(-1.0f, -1.0f, 1.0f, 1.0f);
    px = readback(dev, &w, &h);
    CHECK(px && near(at(px, w, 48, 48), 0, 0, 0, 1), "light from behind: 0x%06X (want 000000)", at(px, w, 48, 48));
    free(px);

    /* 3b. Linear fog, red, from 0 to 1, over an unlit white quad at depth
       0.5: half fogged, about FF8080. First per pixel (table mode; the
       identity projection is orthographic, so it reads z), then per
       vertex, which also carries the fog value across the stages. */
    float fstart = 0.0f, fend = 1.0f;
    uint32_t bits;
    set_tex(dev, 0, NULL);
    rs(dev, 137, 0); /* LIGHTING off: no vertex colour, so diffuse is white */
    rs(dev, 28, 1);  /* FOGENABLE */
    rs(dev, 34, 0x00FF0000u);
    memcpy(&bits, &fstart, 4);
    rs(dev, 36, bits);
    memcpy(&bits, &fend, 4);
    rs(dev, 37, bits);
    uint32_t fog_px[2];
    for (int mode = 0; mode < 2; mode++) {
        rs(dev, 35, mode ? 0 : 3);  /* FOGTABLEMODE */
        rs(dev, 140, mode ? 3 : 0); /* FOGVERTEXMODE */
        clear(dev, 0, NULL, 0x7, 0xFF000000u);
        LIT_QUAD(-1.0f, -1.0f, 1.0f, 1.0f);
        px = readback(dev, &w, &h);
        fog_px[mode] = at(px, w, 32, 32);
        free(px);
    }
    rs(dev, 28, 0);
    rs(dev, 137, 1);
    CHECK(near(fog_px[0], 255, 128, 128, 2) && near(fog_px[1], 255, 128, 128, 2),
          "linear fog half way: per pixel 0x%06X, per vertex 0x%06X (want about FF8080)", fog_px[0], fog_px[1]);
#undef LIT_QUAD
    set_tex(dev, 0, NULL);
    release(lit_decl);

    /* 4. POSITIONT: window coordinates through the viewport, top left at
       0, 0; no texture, so stage 0 is off and the vertex colour shows. */
    static const uint8_t k_post_decl[] = {0, 0, 0, 0, 3, 0, 9, 0,  /* FLOAT4 POSITIONT */
                                          0, 0, 16, 0, 4, 0, 10, 0, /* D3DCOLOR COLOR0 */
                                          0xFF, 0, 0, 0, 17, 0, 0, 0};
    void *post_decl = set_decl(dev, k_post_decl);
    typedef struct {
        float x, y, z, rhw;
        uint32_t c;
    } cvtx;
    cvtx pq[6] = {{0, 0, 0.5f, 1, 0xFF20C040u},   {32, 0, 0.5f, 1, 0xFF20C040u}, {32, 16, 0.5f, 1, 0xFF20C040u},
                  {0, 0, 0.5f, 1, 0xFF20C040u},   {32, 16, 0.5f, 1, 0xFF20C040u}, {0, 16, 0.5f, 1, 0xFF20C040u}};
    clear(dev, 0, NULL, 0x7, 0xFF000000u);
    draw_up(dev, 2, pq, sizeof pq[0]);
    px = readback(dev, &w, &h);
    ok = px && near(at(px, w, 2, 2), 0x20, 0xC0, 0x40, 1) && near(at(px, w, 30, 14), 0x20, 0xC0, 0x40, 1) &&
         at(px, w, 34, 4) == 0 && at(px, w, 4, 18) == 0 && at(px, w, 4, 60) == 0;
    CHECK(ok, "POSITIONT quad at 0,0 to 32,16: inside 0x%06X 0x%06X, right 0x%06X, below 0x%06X, bottom 0x%06X",
          at(px, w, 2, 2), at(px, w, 30, 14), at(px, w, 34, 4), at(px, w, 4, 18), at(px, w, 4, 60));
    free(px);
    release(post_decl);

    /* 5. The same through an FVF, XYZRHW | DIFFUSE. */
    HR hf = VT(dev, SLOT_IDirect3DDevice9_SetFVF, HR (*)(void *, uint32_t))(dev, 0x44);
    for (int i = 0; i < 6; i++) {
        pq[i].x += 32;
        pq[i].y += 48;
        pq[i].c = 0xFF8040FFu;
    }
    draw_up(dev, 2, pq, sizeof pq[0]);
    px = readback(dev, &w, &h);
    CHECK(!hf && px && near(at(px, w, 40, 56), 0x80, 0x40, 0xFF, 1) && near(at(px, w, 2, 2), 0x20, 0xC0, 0x40, 1),
          "FVF XYZRHW|DIFFUSE quad: 0x%06X (want 8040FF), earlier quad kept 0x%06X", at(px, w, 40, 56), at(px, w, 2, 2));
    free(px);

    /* 6. Cube maps, sampled by direction: one face colour each, in the
       formats the Mac hub runs made cube maps in (X8R8G8B8, DXT1, DXT5,
       Q8W8V8U8). Stage 0 selects the texture. */
    static const uint8_t k_tex_decl[] = {0, 0, 0,  0, 3, 0, 9, 0, /* FLOAT4 POSITIONT */
                                         0, 0, 16, 0, 2, 0, 5, 0, /* FLOAT3 TEXCOORD0 */
                                         0xFF, 0, 0, 0, 17, 0, 0, 0};
    void *tex_decl = set_decl(dev, k_tex_decl);
    tss(dev, 0, 1, 2); /* COLOROP SELECTARG1 */
    tss(dev, 0, 2, 2); /* COLORARG1 TEXTURE */
    tss(dev, 0, 4, 2); /* ALPHAOP SELECTARG1 */
    tss(dev, 0, 5, 2);
    static const float k_dir[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    static const uint32_t k_face[6] = {0xFF0000, 0x00FF00, 0x0000FF, 0xFFFF00, 0x00FFFF, 0xFF00FF};
    static const uint16_t k_face565[6] = {0xF800, 0x07E0, 0x001F, 0xFFE0, 0x07FF, 0xF81F};
    /* Q8W8V8U8 is signed: U 0x7F reads 1.0 into red, V into green, W into blue. */
    static const uint8_t k_faceq[6][4] = {{0x7F, 0, 0, 0x7F},    {0, 0x7F, 0, 0x7F},    {0, 0, 0x7F, 0x7F},
                                          {0x7F, 0x7F, 0, 0x7F}, {0, 0x7F, 0x7F, 0x7F}, {0x7F, 0, 0x7F, 0x7F}};
    const uint32_t fmts[4] = {22, 0x31545844u, 0x35545844u, 63};
    const char *names[4] = {"X8R8G8B8", "DXT1", "DXT5", "Q8W8V8U8"};
    clear(dev, 0, NULL, 0x7, 0xFF000000u);
    void *cubes[4] = {NULL, NULL, NULL, NULL};
    for (int f = 0; f < 4; f++) {
        cubes[f] = new_cube(dev, 4, fmts[f]);
        for (int face = 0; cubes[f] && face < 6; face++) {
            if (f == 0) {
                uint32_t c = 0xFF000000u | k_face[face];
                fill(cubes[f], face, &c, 4, 4, 4, 0);
            } else if (f == 1) {
                uint8_t blk[8] = {0};
                memcpy(blk, &k_face565[face], 2);
                memcpy(blk + 2, &k_face565[face], 2);
                fill(cubes[f], face, blk, 8, 1, 1, 0);
            } else if (f == 2) {
                uint8_t blk[16] = {0xFF, 0xFF, 0, 0, 0, 0, 0, 0}; /* alpha 255 everywhere */
                memcpy(blk + 8, &k_face565[face], 2);
                memcpy(blk + 10, &k_face565[face], 2);
                fill(cubes[f], face, blk, 16, 1, 1, 0);
            } else {
                fill(cubes[f], face, k_faceq[face], 4, 4, 4, 0);
            }
        }
        set_tex(dev, 0, cubes[f]);
        for (int face = 0; face < 6; face++)
            screen_quad(dev, (float)(face * 10), (float)(f * 16), (float)(face * 10 + 8), (float)(f * 16 + 8), k_dir[face][0],
                        k_dir[face][1], k_dir[face][2]);
    }
    set_tex(dev, 0, NULL);
    px = readback(dev, &w, &h);
    for (int f = 0; f < 4; f++) {
        int good = cubes[f] != NULL;
        char got[96] = {0};
        for (int face = 0; face < 6; face++) {
            uint32_t p = at(px, w, (uint32_t)(face * 10 + 4), (uint32_t)(f * 16 + 4));
            good &= px && near(p, k_face[face] >> 16, (k_face[face] >> 8) & 255, k_face[face] & 255, 2);
            snprintf(got + strlen(got), sizeof got - strlen(got), " %06X", p);
        }
        CHECK(good, "cube %s sampled by direction +X -X +Y -Y +Z -Z:%s", names[f], got);
    }
    free(px);
    for (int f = 0; f < 4; f++) release(cubes[f]);

    /* 7. The 2D formats the Mac hub runs created: A8R8G8B8, X8R8G8B8,
       R5G6B5 in the default pool at 256x256 (the format DXVK failed one
       creation of), Q8W8V8U8, DXT1, DXT5. */
    static const uint8_t k_2d_decl[] = {0, 0, 0,  0, 3, 0, 9, 0, /* FLOAT4 POSITIONT */
                                        0, 0, 16, 0, 1, 0, 5, 0, /* FLOAT2 TEXCOORD0 */
                                        0xFF, 0, 0, 0, 17, 0, 0, 0};
    void *decl2 = set_decl(dev, k_2d_decl);
    release(tex_decl);
    clear(dev, 0, NULL, 0x7, 0xFF000000u);
    struct {
        uint32_t fmt, pool, size;
        uint8_t texel[16];
        uint32_t texel_bytes, units, rows;
        uint32_t want;
    } k2[6] = {
        {21, 1, 8, {0x40, 0x80, 0xC0, 0xFF}, 4, 8, 8, 0xC08040},
        {22, 1, 8, {0x10, 0x20, 0x30, 0x00}, 4, 8, 8, 0x302010},
        {23, 0, 256, {0x08, 0xFC}, 2, 256, 256, 0xFF8242}, /* R 31, G 32, B 8 */
        {63, 1, 8, {0x7F, 0x40, 0x00, 0x7F}, 4, 8, 8, 0xFF8100},
        {0x31545844u, 1, 8, {0x1F, 0x00, 0x1F, 0x00}, 8, 2, 2, 0x0000FF},
        {0x35545844u, 1, 8, {0xFF, 0xFF, 0, 0, 0, 0, 0, 0, 0xE0, 0x07, 0xE0, 0x07}, 16, 2, 2, 0x00FF00},
    };
    void *t2[6] = {NULL};
    for (int i = 0; i < 6; i++) {
        t2[i] = new_texture(dev, k2[i].size, k2[i].size, 1, k2[i].fmt, k2[i].pool, NULL);
        if (t2[i]) fill(t2[i], -1, k2[i].texel, k2[i].texel_bytes, k2[i].units, k2[i].rows, 0);
        set_tex(dev, 0, t2[i]);
        screen_quad(dev, (float)(i * 10), 0, (float)(i * 10 + 8), 8, 0.5f, 0.5f, 0);
    }
    set_tex(dev, 0, NULL);
    px = readback(dev, &w, &h);
    for (int i = 0; i < 6; i++) {
        uint32_t p = at(px, w, (uint32_t)(i * 10 + 4), 4), want = k2[i].want;
        CHECK(t2[i] && near(p, want >> 16, (want >> 8) & 255, want & 255, 2), "2D format %u pool %u %ux%u sampled: 0x%06X "
              "(want %06X)", k2[i].fmt, k2[i].pool, k2[i].size, k2[i].size, p, want);
    }
    free(px);
    for (int i = 0; i < 6; i++) release(t2[i]);

    /* 8. ColorFill and UpdateSurface, in the order the bridge's movie
       blit uses them: a frame written into a system memory surface,
       UpdateSurface into a default pool offscreen surface, ColorFill of
       the back buffer to black, StretchRect of the frame onto it. */
    void *bb = NULL, *sys = NULL, *surf = NULL;
    VT(dev, SLOT_IDirect3DDevice9_GetBackBuffer, HR (*)(void *, uint32_t, uint32_t, uint32_t, void **))(dev, 0, 0, 0, &bb);
    HR (*offscreen)(void *, uint32_t, uint32_t, uint32_t, uint32_t, void **, void *) =
        VT(dev, SLOT_IDirect3DDevice9_CreateOffscreenPlainSurface,
           HR (*)(void *, uint32_t, uint32_t, uint32_t, uint32_t, void **, void *));
    offscreen(dev, 16, 16, 22, 2, &sys, NULL);
    offscreen(dev, 16, 16, 22, 0, &surf, NULL);
    HR (*color_fill)(void *, void *, const rect *, uint32_t) =
        VT(dev, SLOT_IDirect3DDevice9_ColorFill, HR (*)(void *, void *, const rect *, uint32_t));
    HR (*update_surface)(void *, void *, const rect *, void *, const int32_t *) =
        VT(dev, SLOT_IDirect3DDevice9_UpdateSurface, HR (*)(void *, void *, const rect *, void *, const int32_t *));
    HR (*stretch)(void *, void *, const rect *, void *, const rect *, uint32_t) =
        VT(dev, SLOT_IDirect3DDevice9_StretchRect, HR (*)(void *, void *, const rect *, void *, const rect *, uint32_t));
    locked_rect lr = {0, NULL};
    if (sys) VT(sys, SLOT_IDirect3DSurface9_LockRect, HR (*)(void *, locked_rect *, const void *, uint32_t))(sys, &lr, NULL, 0);
    for (uint32_t y = 0; lr.pBits && y < 16; y++)
        for (uint32_t x = 0; x < 16; x++) ((uint32_t *)((uint8_t *)lr.pBits + y * lr.Pitch))[x] = x < 8 ? 0xFFFF0000u : 0xFF0000FFu;
    if (sys) VT(sys, SLOT_IDirect3DSurface9_UnlockRect, HR (*)(void *))(sys);
    clear(dev, 0, NULL, 0x7, 0xFF808080u);
    HR hu = update_surface(dev, sys, NULL, surf, NULL);
    HR hc = color_fill(dev, bb, NULL, 0xFF000000u);
    rect dst = {0, 0, 32, 32};
    HR hs = stretch(dev, surf, NULL, bb, &dst, 1);
    px = readback(dev, &w, &h);
    ok = !hu && !hc && !hs && px && at(px, w, 8, 16) == 0xFF0000 && at(px, w, 24, 16) == 0x0000FF &&
         at(px, w, 48, 48) == 0 && at(px, w, 40, 8) == 0;
    CHECK(ok, "movie blit (UpdateSurface, ColorFill, StretchRect): frame 0x%06X 0x%06X, filled 0x%06X 0x%06X", at(px, w, 8, 16),
          at(px, w, 24, 16), at(px, w, 48, 48), at(px, w, 40, 8));
    free(px);

    /* A partial ColorFill of the back buffer, and one of the CPU surface;
       a source rectangle moved to a destination point. */
    rect fr = {40, 40, 56, 48};
    hc = color_fill(dev, bb, &fr, 0xFF00FF00u);
    rect cr = {0, 0, 8, 8};
    HR hc2 = color_fill(dev, surf, &cr, 0xFFFFFF00u);
    rect sr = {8, 0, 16, 8};
    int32_t pt[2] = {0, 8};
    hu = update_surface(dev, sys, &sr, surf, pt);
    hs = stretch(dev, surf, NULL, bb, &dst, 1);
    px = readback(dev, &w, &h);
    ok = !hc && !hc2 && !hu && !hs && px && at(px, w, 44, 44) == 0x00FF00 && at(px, w, 44, 52) == 0 &&
         at(px, w, 38, 44) == 0 && at(px, w, 8, 8) == 0xFFFF00 && at(px, w, 8, 24) == 0x0000FF && at(px, w, 24, 24) == 0x0000FF;
    CHECK(ok, "partial ColorFill on the GPU (0x%06X, edge 0x%06X 0x%06X) and on CPU memory (0x%06X), UpdateSurface rect "
              "to a point (0x%06X)",
          at(px, w, 44, 44), at(px, w, 44, 52), at(px, w, 38, 44), at(px, w, 8, 8), at(px, w, 8, 24));
    free(px);

    /* UpdateSurface straight into the back buffer (a GPU-only target). */
    int32_t pt2[2] = {40, 40};
    hu = update_surface(dev, sys, NULL, bb, pt2);
    px = readback(dev, &w, &h);
    CHECK(!hu && px && at(px, w, 44, 44) == 0xFF0000 && at(px, w, 52, 44) == 0x0000FF && at(px, w, 38, 44) == 0,
          "UpdateSurface into the back buffer at 40,40: 0x%06X 0x%06X, outside 0x%06X", at(px, w, 44, 44), at(px, w, 52, 44),
          at(px, w, 38, 44));
    free(px);
    release(surf);

    /* 9. UpdateTexture, system memory to default pool, then sampled. A
       lock dirties the source; a lock with NO_DIRTY_UPDATE does not, and
       the copy after it changes nothing, as in D3D9. */
    void *src = new_texture(dev, 8, 8, 2, 21, 2, NULL), *dtex = new_texture(dev, 8, 8, 2, 21, 0, NULL);
    HR (*update_texture)(void *, void *, void *) =
        VT(dev, SLOT_IDirect3DDevice9_UpdateTexture, HR (*)(void *, void *, void *));
    uint32_t yellow = 0xFFFFFF00u, cyan = 0xFF00FFFFu, magenta = 0xFFFF00FFu;
    uint32_t got[3] = {0};
    HR hu3[3] = {-1, -1, -1};
    const uint32_t *vals[3] = {&yellow, &cyan, &magenta};
    for (int i = 0; i < 3 && src && dtex; i++) {
        fill(src, -1, vals[i], 4, 8, 8, i == 2 ? 0x8000u : 0);
        hu3[i] = update_texture(dev, src, dtex);
        set_tex(dev, 0, dtex);
        clear(dev, 0, NULL, 0x7, 0xFF000000u);
        screen_quad(dev, 0, 0, 16, 16, 0.5f, 0.5f, 0);
        px = readback(dev, &w, &h);
        got[i] = at(px, w, 8, 8);
        free(px);
    }
    set_tex(dev, 0, NULL);
    CHECK(!hu3[0] && !hu3[1] && !hu3[2] && got[0] == 0xFFFF00 && got[1] == 0x00FFFF && got[2] == 0x00FFFF,
          "UpdateTexture: 0x%06X, after a relock 0x%06X, after a NO_DIRTY_UPDATE lock 0x%06X (want FFFF00 00FFFF 00FFFF)",
          got[0], got[1], got[2]);

    /* and UpdateSurface into a default pool texture's level */
    void *lvl = NULL;
    if (dtex) VT(dtex, SLOT_IDirect3DTexture9_GetSurfaceLevel, HR (*)(void *, uint32_t, void **))(dtex, 0, &lvl);
    void *sys8 = NULL;
    offscreen(dev, 8, 8, 21, 2, &sys8, NULL);
    if (sys8) VT(sys8, SLOT_IDirect3DSurface9_LockRect, HR (*)(void *, locked_rect *, const void *, uint32_t))(sys8, &lr, NULL, 0);
    for (uint32_t y = 0; sys8 && lr.pBits && y < 8; y++)
        for (uint32_t x = 0; x < 8; x++) ((uint32_t *)((uint8_t *)lr.pBits + y * lr.Pitch))[x] = 0xFF123456u;
    if (sys8) VT(sys8, SLOT_IDirect3DSurface9_UnlockRect, HR (*)(void *))(sys8);
    hu = update_surface(dev, sys8, NULL, lvl, NULL);
    set_tex(dev, 0, dtex);
    screen_quad(dev, 0, 0, 16, 16, 0.5f, 0.5f, 0);
    set_tex(dev, 0, NULL);
    px = readback(dev, &w, &h);
    CHECK(!hu && px && at(px, w, 8, 8) == 0x123456, "UpdateSurface into a default pool texture level, sampled: 0x%06X",
          at(px, w, 8, 8));
    free(px);
    release(lvl);
    release(sys8);
    release(src);
    release(dtex);
    release(sys);
    release(bb);
    release(decl2);

    /* 10. The engine's video memory probe: 256x256 R5G6B5 textures in the
       default pool until creation fails, as in the Mac hub logs (7,847
       made, one failed). The budget is DXVK's 1 GiB, so it ends near
       8,192 with D3DERR_OUTOFVIDEOMEMORY, and freeing them gives it back. */
    enum { PROBE_MAX = 9000 };
    void **probe = calloc(PROBE_MAX, sizeof *probe);
    uint32_t made = 0;
    HR last = 0;
    while (probe && made < PROBE_MAX) {
        void *t = new_texture(dev, 256, 256, 1, 23, 0, &last);
        if (!t) break;
        probe[made++] = t;
    }
    for (uint32_t i = 0; i < made; i++) release(probe[i]);
    free(probe);
    HR again = 0;
    void *one = new_texture(dev, 256, 256, 1, 23, 0, &again);
    release(one);
    CHECK((uint32_t)last == 0x8876017Cu && made > 7000 && made <= 8192 && !again,
          "video memory probe: %u R5G6B5 256x256 default textures, then 0x%08X; after freeing, a new one 0x%08X", made,
          (uint32_t)last, (uint32_t)again);

    /* 11. Reset returns state to its defaults and unbinds. */
    void *keep_tex = new_texture(dev, 4, 4, 1, 21, 1, NULL);
    set_tex(dev, 0, keep_tex);
    rs(dev, 22, 1);
    set_xf(dev, 2, view);
    tss(dev, 0, 1, 2);
    present_params rp;
    memset(&rp, 0, sizeof rp);
    rp.BackBufferWidth = 64;
    rp.BackBufferHeight = 64;
    rp.BackBufferFormat = 22;
    rp.Windowed = 1;
    rp.EnableAutoDepthStencil = 1;
    rp.AutoDepthStencilFormat = 75;
    rp.PresentationInterval = 0x80000000u;
    HR hr = VT(dev, SLOT_IDirect3DDevice9_Reset, HR (*)(void *, present_params *))(dev, &rp);
    uint32_t cull = 0, op = 0;
    VT(dev, SLOT_IDirect3DDevice9_GetRenderState, HR (*)(void *, uint32_t, uint32_t *))(dev, 22, &cull);
    VT(dev, SLOT_IDirect3DDevice9_GetTextureStageState, HR (*)(void *, uint32_t, uint32_t, uint32_t *))(dev, 0, 1, &op);
    void *bound = (void *)1;
    VT(dev, SLOT_IDirect3DDevice9_GetTexture, HR (*)(void *, uint32_t, void **))(dev, 0, &bound);
    float m[16];
    VT(dev, SLOT_IDirect3DDevice9_GetTransform, HR (*)(void *, uint32_t, float *))(dev, 2, m);
    uint32_t kept = keep_tex ? VT(keep_tex, 2, uint32_t (*)(void *))(keep_tex) : 99;
    CHECK(!hr && cull == 3 && op == 4 && bound == NULL && !memcmp(m, ident, sizeof m) && kept == 0,
          "Reset restores defaults: cull %u, stage 0 colour op %u, texture %p, view identity %d, unbound texture freed "
          "(Release %u)",
          cull, op, bound, !memcmp(m, ident, sizeof m), kept);

    release(tex);
    uint32_t cd = VT(dev, SLOT_IDirect3DDevice9_Release, uint32_t (*)(void *))(dev);
    CHECK(cd == 0, "device after the fixed-function tests released: %u", cd);
}

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

/* Inline against threaded. Each scene runs on a device that encodes at
   each call on the game thread (RT_METAL_INLINE=1) and on one that
   records lists for the render thread, and the frames must match bit for
   bit: the render thread may only move work, never change a pixel. */

static void *s_lib;

static uint64_t frame_stat(void *dev, int i)
{
    void (*stats)(void *, uint64_t *) = (void (*)(void *, uint64_t *))dlsym(s_lib, "metal_d3d9_frame_stats");
    uint64_t out[7] = {0};
    if (stats) stats(dev, out);
    return out[i];
}

/* A device in the given mode, whatever RT_METAL_INLINE the suite itself
   runs under; the mode is checked, so the comparison cannot quietly be
   inline against inline. */
static void *mode_device(void *d3d, int threaded, uint32_t w, uint32_t h)
{
    const char *was = getenv("RT_METAL_INLINE");
    char keep[32] = {0};
    if (was) snprintf(keep, sizeof keep, "%s", was);
    if (threaded) unsetenv("RT_METAL_INLINE");
    else setenv("RT_METAL_INLINE", "1", 1);
    void *dev = make_device(d3d, w, h);
    if (was) setenv("RT_METAL_INLINE", keep, 1);
    else unsetenv("RT_METAL_INLINE");
    if (dev && frame_stat(dev, 5) != (uint64_t)threaded) {
        CHECK(0, "device made in %s mode reports the other", threaded ? "threaded" : "inline");
        release(dev);
        return NULL;
    }
    return dev;
}

static void same_frames(const char *what, uint32_t *const a[], uint32_t *const b[], int n, uint32_t w, uint32_t h)
{
    uint32_t diff = 0, first = 0, fa = 0, fb = 0;
    int missing = 0;
    for (int k = 0; k < n; k++) {
        if (!a[k] || !b[k]) {
            missing = 1;
            continue;
        }
        for (uint32_t i = 0; i < w * h; i++)
            if (a[k][i] != b[k][i]) {
                if (!diff++) {
                    first = i;
                    fa = a[k][i];
                    fb = b[k][i];
                }
            }
    }
    CHECK(!missing && !diff,
          "%s: inline and threaded identical over %d frame(s) (%u pixels differ, first at %u,%u: 0x%08X against 0x%08X)",
          what, n, diff, w ? first % w : 0, w ? first / w : 0, fa, fb);
}

/* The game-shader scenes' objects, made on each device. */
typedef struct {
    void *vs, *ps, *decl, *vb, *ib, *tex, *white;
} kit;

static int kit_make(void *dev, kit *k)
{
    memset(k, 0, sizeof *k);
    k->vs = load_shader(dev, "vs_1_1", "78dc66bc16bac6b1");
    k->ps = load_shader(dev, "ps_1_1", "371500f12c2f6512");
    if (!k->vs || !k->ps) return 0;
    static const uint8_t k_decl[] = {0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 12, 0, 1, 0, 5, 0, 0xFF, 0, 0, 0, 17, 0, 0, 0};
    VT(dev, SLOT_IDirect3DDevice9_CreateVertexDeclaration, HR (*)(void *, const void *, void **))(dev, k_decl, &k->decl);
    VT(dev, SLOT_IDirect3DDevice9_CreateVertexBuffer, HR (*)(void *, uint32_t, uint32_t, uint32_t, uint32_t, void **, void *))(
        dev, 4 * sizeof(vtx), 0x208, 0, 0, &k->vb, NULL);
    VT(dev, SLOT_IDirect3DDevice9_CreateIndexBuffer, HR (*)(void *, uint32_t, uint32_t, uint32_t, uint32_t, void **, void *))(
        dev, 12, 0x208, 101, 0, &k->ib, NULL);
    uint16_t *ix = NULL;
    VT(k->ib, SLOT_IDirect3DIndexBuffer9_Lock, HR (*)(void *, uint32_t, uint32_t, void **, uint32_t))(k->ib, 0, 0, (void **)&ix, 0);
    static const uint16_t k_ix[6] = {0, 1, 2, 0, 2, 3};
    if (ix) memcpy(ix, k_ix, sizeof k_ix);
    VT(k->ib, SLOT_IDirect3DIndexBuffer9_Unlock, HR (*)(void *))(k->ib);
    k->tex = new_texture(dev, 8, 8, 1, 0x31545844u, 1, NULL);
    static const uint16_t k_c565[4] = {0xF800, 0x07E0, 0x001F, 0xFFFF};
    locked_rect lr = {0, NULL};
    if (k->tex)
        VT(k->tex, SLOT_IDirect3DTexture9_LockRect, HR (*)(void *, uint32_t, locked_rect *, const void *, uint32_t))(
            k->tex, 0, &lr, NULL, 0);
    for (int b = 0; lr.pBits && b < 4; b++) {
        uint8_t *blk = (uint8_t *)lr.pBits + (b / 2) * lr.Pitch + (b % 2) * 8;
        memcpy(blk, &k_c565[b], 2);
        memcpy(blk + 2, &k_c565[b], 2);
        memset(blk + 4, 0, 4);
    }
    if (k->tex) VT(k->tex, SLOT_IDirect3DTexture9_UnlockRect, HR (*)(void *, uint32_t))(k->tex, 0);
    k->white = new_texture(dev, 4, 4, 1, 0x31545844u, 1, NULL);
    lr.pBits = NULL;
    if (k->white)
        VT(k->white, SLOT_IDirect3DTexture9_LockRect, HR (*)(void *, uint32_t, locked_rect *, const void *, uint32_t))(
            k->white, 0, &lr, NULL, 0);
    if (lr.pBits) memset(lr.pBits, 0xFF, 4), memset((uint8_t *)lr.pBits + 4, 0, 4);
    if (k->white) VT(k->white, SLOT_IDirect3DTexture9_UnlockRect, HR (*)(void *, uint32_t))(k->white, 0);
    VT(dev, SLOT_IDirect3DDevice9_SetVertexDeclaration, HR (*)(void *, void *))(dev, k->decl);
    VT(dev, SLOT_IDirect3DDevice9_SetVertexShader, HR (*)(void *, void *))(dev, k->vs);
    VT(dev, SLOT_IDirect3DDevice9_SetPixelShader, HR (*)(void *, void *))(dev, k->ps);
    VT(dev, SLOT_IDirect3DDevice9_SetStreamSource, HR (*)(void *, uint32_t, void *, uint32_t, uint32_t))(dev, 0, k->vb, 0,
                                                                                                       sizeof(vtx));
    VT(dev, SLOT_IDirect3DDevice9_SetIndices, HR (*)(void *, void *))(dev, k->ib);
    HR (*set_ss)(void *, uint32_t, uint32_t, uint32_t) =
        VT(dev, SLOT_IDirect3DDevice9_SetSamplerState, HR (*)(void *, uint32_t, uint32_t, uint32_t));
    set_ss(dev, 0, 5, 1);
    set_ss(dev, 0, 6, 1);
    set_ss(dev, 0, 1, 3);
    set_ss(dev, 0, 2, 3);
    return k->decl && k->vb && k->ib && k->tex && k->white;
}

static void kit_free(void *dev, kit *k)
{
    VT(dev, SLOT_IDirect3DDevice9_SetTexture, HR (*)(void *, uint32_t, void *))(dev, 0, NULL);
    void *o[] = {k->vs, k->ps, k->decl, k->vb, k->ib, k->tex, k->white};
    for (unsigned i = 0; i < sizeof o / sizeof o[0]; i++) release(o[i]);
}

static void present(void *dev)
{
    VT(dev, SLOT_IDirect3DDevice9_Present, HR (*)(void *, const void *, const void *, void *, const void *))(dev, NULL, NULL,
                                                                                                         NULL, NULL);
}

static void set_tex0(void *dev, void *t)
{
    VT(dev, SLOT_IDirect3DDevice9_SetTexture, HR (*)(void *, uint32_t, void *))(dev, 0, t);
}

/* The scenes. Each draws on the device it is given and returns its
   readbacks in out[] (up to four). */

static int scene_dxt1(void *dev, uint32_t *out[])
{
    kit k;
    uint32_t w, h;
    if (!kit_make(dev, &k)) return 0;
    set_tex0(dev, k.tex);
    unlit_constants(dev, 1, 1, 1, 1);
    clear(dev, 0, NULL, 0x7, 0xFF000000u);
    quad(dev, k.vb, -1, -1, 1, 1, 0.5f);
    present(dev);
    out[0] = readback(dev, &w, &h);
    kit_free(dev, &k);
    return 1;
}

static int scene_discard(void *dev, uint32_t *out[])
{
    kit k;
    uint32_t w, h;
    if (!kit_make(dev, &k)) return 0;
    rs(dev, 22, 1);
    set_tex0(dev, k.white);
    clear(dev, 0, NULL, 0x7, 0xFF000000u);
    unlit_constants(dev, 0, 0, 1, 1);
    quad(dev, k.vb, -1, -1, 0, 1, 0.5f);
    unlit_constants(dev, 1, 1, 0, 1);
    quad(dev, k.vb, 0, -1, 1, 1, 0.5f);
    present(dev);
    out[0] = readback(dev, &w, &h);
    kit_free(dev, &k);
    return 1;
}

/* The fixed-function quads below: position and one texcoord in clip
   space, lighting off, so stage 0's MODULATE gives the texel exactly. */
typedef struct {
    float x, y, z, u, v;
} ffv;

static void *ff_setup(void *dev)
{
    static const uint8_t k_decl[] = {0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 12, 0, 1, 0, 5, 0, 0xFF, 0, 0, 0, 17, 0, 0, 0};
    void *decl = set_decl(dev, k_decl);
    rs(dev, 22, 1);
    rs(dev, 137, 0);
    HR (*set_ss)(void *, uint32_t, uint32_t, uint32_t) =
        VT(dev, SLOT_IDirect3DDevice9_SetSamplerState, HR (*)(void *, uint32_t, uint32_t, uint32_t));
    set_ss(dev, 0, 5, 1);
    set_ss(dev, 0, 6, 1);
    return decl;
}

/* Writes a strip quad at vertex v of a buffer with the given lock flags. */
static void put_quad(void *vb, uint32_t v, float x0, float y0, float x1, float y1, uint32_t flags)
{
    ffv q[4] = {{x0, y0, 0.5f, 0, 1}, {x0, y1, 0.5f, 0, 0}, {x1, y0, 0.5f, 1, 1}, {x1, y1, 0.5f, 1, 0}};
    void *p = NULL;
    VT(vb, SLOT_IDirect3DVertexBuffer9_Lock, HR (*)(void *, uint32_t, uint32_t, void **, uint32_t))(
        vb, v * (uint32_t)sizeof(ffv), sizeof q, &p, flags);
    if (p) memcpy(p, q, sizeof q);
    VT(vb, SLOT_IDirect3DVertexBuffer9_Unlock, HR (*)(void *))(vb);
}

static void draw_strip(void *dev, uint32_t first)
{
    VT(dev, SLOT_IDirect3DDevice9_DrawPrimitive, HR (*)(void *, uint32_t, uint32_t, uint32_t))(dev, 5, first, 2);
}

static void solid(void *t, uint32_t argb, uint32_t flags) { fill(t, -1, &argb, 4, 4, 4, flags); }

static void *ff_vb(void *dev, uint32_t verts, uint32_t usage)
{
    void *vb = NULL;
    VT(dev, SLOT_IDirect3DDevice9_CreateVertexBuffer, HR (*)(void *, uint32_t, uint32_t, uint32_t, uint32_t, void **, void *))(
        dev, verts * (uint32_t)sizeof(ffv), usage, 0, 0, &vb, NULL);
    VT(dev, SLOT_IDirect3DDevice9_SetStreamSource, HR (*)(void *, uint32_t, void *, uint32_t, uint32_t))(dev, 0, vb, 0,
                                                                                                       sizeof(ffv));
    return vb;
}

static int scene_ff_lit(void *dev, uint32_t *out[])
{
    uint32_t w, h;
    rs(dev, 22, 1);
    HR (*set_ss)(void *, uint32_t, uint32_t, uint32_t) =
        VT(dev, SLOT_IDirect3DDevice9_SetSamplerState, HR (*)(void *, uint32_t, uint32_t, uint32_t));
    set_ss(dev, 0, 5, 1);
    set_ss(dev, 0, 6, 1);
    set_ss(dev, 0, 1, 3);
    set_ss(dev, 0, 2, 3);
    void *tex = new_texture(dev, 8, 8, 1, 0x31545844u, 1, NULL);
    static const uint16_t k_c565[4] = {0xF800, 0x07E0, 0x001F, 0xFFFF};
    locked_rect lr = {0, NULL};
    if (tex)
        VT(tex, SLOT_IDirect3DTexture9_LockRect, HR (*)(void *, uint32_t, locked_rect *, const void *, uint32_t))(tex, 0, &lr,
                                                                                                            NULL, 0);
    for (int b = 0; lr.pBits && b < 4; b++) {
        uint8_t *blk = (uint8_t *)lr.pBits + (b / 2) * lr.Pitch + (b % 2) * 8;
        memcpy(blk, &k_c565[b], 2);
        memcpy(blk + 2, &k_c565[b], 2);
        memset(blk + 4, 0, 4);
    }
    if (tex) VT(tex, SLOT_IDirect3DTexture9_UnlockRect, HR (*)(void *, uint32_t))(tex, 0);
    static const uint8_t k_lit_decl[] = {0, 0, 0,  0, 2, 0, 0, 0, 0, 0, 12, 0, 2, 0, 3, 0,
                                         0, 0, 24, 0, 1, 0, 5, 0, 0xFF, 0, 0, 0, 17, 0, 0, 0};
    void *decl = set_decl(dev, k_lit_decl);
    struct {
        uint32_t type;
        float diffuse[4], specular[4], ambient[4], pos[3], dir[3], range, falloff, att0, att1, att2, theta, phi;
    } light = {3, {1, 1, 1, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0}, {0, 0, 1}, 0, 0, 0, 0, 0, 0, 0};
    float material[17] = {0.5f, 0.25f, 1.0f, 1.0f};
    VT(dev, SLOT_IDirect3DDevice9_SetLight, HR (*)(void *, uint32_t, const void *))(dev, 0, &light);
    VT(dev, SLOT_IDirect3DDevice9_LightEnable, HR (*)(void *, uint32_t, int32_t))(dev, 0, 1);
    VT(dev, SLOT_IDirect3DDevice9_SetMaterial, HR (*)(void *, const void *))(dev, material);
    set_tex0(dev, tex);
    typedef struct {
        float x, y, z, nx, ny, nz, u, v;
    } lvtx;
    lvtx q[6] = {{-1, 1, 0.5f, 0, 0, -1, 0, 0}, {1, 1, 0.5f, 0, 0, -1, 1, 0}, {1, -1, 0.5f, 0, 0, -1, 1, 1},
                 {-1, 1, 0.5f, 0, 0, -1, 0, 0}, {1, -1, 0.5f, 0, 0, -1, 1, 1}, {-1, -1, 0.5f, 0, 0, -1, 0, 1}};
    clear(dev, 0, NULL, 0x7, 0xFF404040u);
    draw_up(dev, 2, q, sizeof q[0]);
    present(dev);
    out[0] = readback(dev, &w, &h);
    set_tex0(dev, NULL);
    release(tex);
    release(decl);
    return 1;
}

/* A lock with no flags on a buffer and on a texture a draw of this frame
   already reads: the first draw must keep the old contents. */
static int scene_relock(void *dev, uint32_t *out[])
{
    uint32_t w, h;
    void *decl = ff_setup(dev);
    void *vb = ff_vb(dev, 4, 0);
    void *tex = new_texture(dev, 4, 4, 1, 21, 1, NULL);
    set_tex0(dev, tex);
    clear(dev, 0, NULL, 0x7, 0xFF000000u);
    put_quad(vb, 0, -1, 0, 0, 1, 0);
    solid(tex, 0xFFC03010u, 0);
    draw_strip(dev, 0);
    put_quad(vb, 0, 0, -1, 1, 0, 0); /* the same bytes, rewritten in place */
    solid(tex, 0xFF2080F0u, 0);
    draw_strip(dev, 0);
    present(dev);
    out[0] = readback(dev, &w, &h);
    uint32_t *px = out[0];
    CHECK(px && at(px, w, 16, 16) == 0xC03010 && at(px, w, 48, 48) == 0x2080F0 && at(px, w, 48, 16) == 0 &&
              at(px, w, 16, 48) == 0,
          "plain relock of a buffer and a texture between two draws (%s): first quad 0x%06X, second 0x%06X, untouched "
          "0x%06X 0x%06X",
          frame_stat(dev, 5) ? "threaded" : "inline", at(px, w, 16, 16), at(px, w, 48, 48), at(px, w, 48, 16),
          at(px, w, 16, 48));
    set_tex0(dev, NULL);
    release(tex);
    release(vb);
    release(decl);
    return 1;
}

/* Readbacks between draws, with no Present: each must hold exactly the
   draws before it, as the capture hooks expect. */
static int scene_midframe(void *dev, uint32_t *out[])
{
    uint32_t w, h;
    void *decl = ff_setup(dev);
    void *vb = ff_vb(dev, 8, 0x208);
    void *tex = new_texture(dev, 4, 4, 1, 21, 1, NULL);
    solid(tex, 0xFF00FF80u, 0);
    set_tex0(dev, tex);
    clear(dev, 0, NULL, 0x7, 0xFF101010u);
    put_quad(vb, 0, -1, 0, 0, 1, 0x2000);
    draw_strip(dev, 0);
    out[0] = readback(dev, &w, &h);
    put_quad(vb, 4, 0, -1, 1, 0, 0x1000);
    draw_strip(dev, 4);
    out[1] = readback(dev, &w, &h);
    solid(tex, 0xFFFF4000u, 0);
    put_quad(vb, 0, 0, 0, 1, 1, 0x2000);
    draw_strip(dev, 0);
    present(dev);
    out[2] = readback(dev, &w, &h);
    CHECK(out[0] && out[1] && out[2] && at(out[0], w, 16, 16) == 0x00FF80 && at(out[0], w, 48, 48) == 0x101010 &&
              at(out[1], w, 48, 48) == 0x00FF80 && at(out[1], w, 48, 16) == 0x101010 &&
              at(out[2], w, 48, 16) == 0xFF4000 && at(out[2], w, 16, 16) == 0x00FF80,
          "readbacks in the middle of a frame (%s): after one draw 0x%06X 0x%06X, after two 0x%06X 0x%06X, at Present "
          "0x%06X 0x%06X",
          frame_stat(dev, 5) ? "threaded" : "inline", at(out[0], w, 16, 16), at(out[0], w, 48, 48), at(out[1], w, 48, 48),
          at(out[1], w, 48, 16), at(out[2], w, 48, 16), at(out[2], w, 16, 16));
    set_tex0(dev, NULL);
    release(tex);
    release(vb);
    release(decl);
    return 3;
}

static void compare_scene(void *d3d, const char *what, int (*scene)(void *, uint32_t *[]))
{
    uint32_t *px[2][4] = {{0}};
    int n = 0;
    for (int threaded = 0; threaded < 2; threaded++) {
        void *dev = mode_device(d3d, threaded, 64, 64);
        if (!dev) return;
        n = scene(dev, px[threaded]);
        uint32_t cd = VT(dev, SLOT_IDirect3DDevice9_Release, uint32_t (*)(void *))(dev);
        if (cd) CHECK(0, "%s: device released with %u references left", what, cd);
    }
    if (n) same_frames(what, px[0], px[1], n, 64, 64);
    else CHECK(0, "%s: the scene could not be set up", what);
    for (int t = 0; t < 2; t++)
        for (int k = 0; k < 4; k++) free(px[t][k]);
}

/* A list replayed twice, into the back buffer and into a twin: the twin
   must match the back buffer. The second replay skips the readback and
   the Present in the list, which belong to the frame once; everything
   else (passes, clears, draws, uploads) runs again from what the list
   holds, so a list that read device state or storage changed after it
   was recorded would differ here. */
static void replay_twice_test(void *d3d)
{
    int (*twice)(void *, int) = (int (*)(void *, int))dlsym(s_lib, "metal_d3d9_replay_twice");
    int (*twin)(void *, uint32_t *, uint32_t, uint32_t) =
        (int (*)(void *, uint32_t *, uint32_t, uint32_t))dlsym(s_lib, "metal_d3d9_twin_read");
    if (!twice || !twin) {
        CHECK(0, "replay-twice test hooks exported");
        return;
    }
    for (int threaded = 0; threaded < 2; threaded++) {
        void *dev = mode_device(d3d, threaded, 64, 64);
        if (!dev) return;
        int on = twice(dev, 1);
        void *decl = ff_setup(dev);
        void *vb = ff_vb(dev, 16, 0x208);
        void *tex = new_texture(dev, 4, 4, 1, 21, 1, NULL);
        set_tex0(dev, tex);
        uint32_t w = 0, h = 0, *a = NULL, *b = malloc(64 * 64 * 4);
        int ok = on && b;
        for (int f = 0; f < 3 && ok; f++) {
            clear(dev, 0, NULL, 0x7, 0xFF000000u | (uint32_t)(0x30 * f));
            solid(tex, 0xFF000000u | (uint32_t)(0x400000 * (f + 1)) | 0x00A000, 0);
            put_quad(vb, 0, -1, -1, 0, 0, 0x2000);
            draw_strip(dev, 0);
            solid(tex, 0xFF0000FFu | (uint32_t)(0x100 * 0x20 * (f + 1)), 0);
            put_quad(vb, 4, 0, 0, 1, 1, 0x1000);
            draw_strip(dev, 4);
            present(dev);
        }
        a = readback(dev, &w, &h);
        ok = ok && a && twin(dev, b, 64, 64);
        uint32_t diff = 0, first = 0;
        for (uint32_t i = 0; ok && i < 64 * 64; i++)
            if (a[i] != b[i] && !diff++) first = i;
        CHECK(ok && !diff && at(a, 64, 16, 48) == 0xC0A000 && at(a, 64, 48, 16) == 0x0060FF,
              "a list replayed twice (%s) into the back buffer and a twin: %u pixels differ (first %u), back buffer "
              "0x%06X 0x%06X",
              threaded ? "threaded" : "inline", diff, first, at(a, 64, 16, 48), at(a, 64, 48, 16));
        free(a);
        free(b);
        twice(dev, 0);
        set_tex0(dev, NULL);
        release(tex);
        release(vb);
        release(decl);
        release(dev);
    }
}

static uint64_t footprint(void)
{
    task_vm_info_data_t vi;
    mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vi, &n) != KERN_SUCCESS) return 0;
    return vi.phys_footprint;
}

static uint32_t stress_color(uint32_t frame, uint32_t k)
{
    return 0xFF000000u | ((frame * 37u + k * 50u) & 0xFF) << 16 | ((frame * 11u + k * 90u) & 0xFF) << 8 | (k * 60u + 15u);
}

/* Many frames in flight with every kind of change between draws: a
   dynamic buffer written with DISCARD, NOOVERWRITE and no flag, a texture
   relocked twice a frame, a second texture, and a movie-style
   UpdateSurface plus StretchRect. Each frame's back buffer is read the
   way the bridge's luma ring reads it (GetRenderTargetData every frame,
   LockRect two frames later), so the queue is never drained to check,
   and every frame's four quadrants must be exactly that frame's colours.
   Lists, ring chunks and the process footprint are taken after a warm-up
   and at the end and must not grow. */
static void stress_test(void *d3d, int threaded, int frames)
{
    void *dev = mode_device(d3d, threaded, 64, 64);
    if (!dev) return;
    void *decl = ff_setup(dev);
    void *vb = ff_vb(dev, 64, 0x208);
    void *t1 = new_texture(dev, 4, 4, 1, 21, 1, NULL), *t2 = new_texture(dev, 4, 4, 1, 21, 1, NULL);
    void *bb = NULL, *movie_sys = NULL, *movie = NULL, *ring[3] = {NULL};
    VT(dev, SLOT_IDirect3DDevice9_GetBackBuffer, HR (*)(void *, uint32_t, uint32_t, uint32_t, void **))(dev, 0, 0, 0, &bb);
    HR (*offscreen)(void *, uint32_t, uint32_t, uint32_t, uint32_t, void **, void *) =
        VT(dev, SLOT_IDirect3DDevice9_CreateOffscreenPlainSurface,
           HR (*)(void *, uint32_t, uint32_t, uint32_t, uint32_t, void **, void *));
    offscreen(dev, 16, 16, 22, 2, &movie_sys, NULL);
    offscreen(dev, 16, 16, 22, 0, &movie, NULL);
    for (int k = 0; k < 3; k++) offscreen(dev, 64, 64, 22, 2, &ring[k], NULL);
    HR (*update_surface)(void *, void *, const rect *, void *, const int32_t *) =
        VT(dev, SLOT_IDirect3DDevice9_UpdateSurface, HR (*)(void *, void *, const rect *, void *, const int32_t *));
    HR (*stretch)(void *, void *, const rect *, void *, const rect *, uint32_t) =
        VT(dev, SLOT_IDirect3DDevice9_StretchRect, HR (*)(void *, void *, const rect *, void *, const rect *, uint32_t));
    HR (*rtdata)(void *, void *, void *) = VT(dev, SLOT_IDirect3DDevice9_GetRenderTargetData, HR (*)(void *, void *, void *));
    uint32_t bad = 0, checked = 0, first_bad = 0;
    uint64_t warm_fp = 0, warm_lists = 0, warm_chunks = 0;
    double draw_s = 0;
    uint32_t ndraws = 0;
    for (int f = 0; f < frames; f++) {
        uint32_t c1 = stress_color((uint32_t)f, 0), c2 = stress_color((uint32_t)f, 1), c3 = stress_color((uint32_t)f, 2),
                 c4 = stress_color((uint32_t)f, 3);
        clear(dev, 0, NULL, 0x7, 0xFF000000u);
        double t0;
        set_tex0(dev, t1);
        solid(t1, c1, 0);
        put_quad(vb, 0, -1, 0, 0, 1, 0x2000);
        t0 = now_s();
        draw_strip(dev, 0);
        draw_s += now_s() - t0;
        solid(t1, c2, 0);
        put_quad(vb, 4, 0, 0, 1, 1, 0x1000);
        t0 = now_s();
        draw_strip(dev, 4);
        draw_s += now_s() - t0;
        set_tex0(dev, t2);
        solid(t2, c3, 0);
        put_quad(vb, 8, -1, -1, 0, 0, 0);
        t0 = now_s();
        draw_strip(dev, 8);
        draw_s += now_s() - t0;
        ndraws += 3;
        locked_rect lr = {0, NULL};
        VT(movie_sys, SLOT_IDirect3DSurface9_LockRect, HR (*)(void *, locked_rect *, const void *, uint32_t))(movie_sys, &lr,
                                                                                                            NULL, 0);
        for (uint32_t y = 0; lr.pBits && y < 16; y++)
            for (uint32_t x = 0; x < 16; x++) ((uint32_t *)((uint8_t *)lr.pBits + y * lr.Pitch))[x] = c4;
        VT(movie_sys, SLOT_IDirect3DSurface9_UnlockRect, HR (*)(void *))(movie_sys);
        update_surface(dev, movie_sys, NULL, movie, NULL);
        rect dst = {32, 32, 64, 64};
        stretch(dev, movie, NULL, bb, &dst, 1);
        /* The luma ring: this frame's back buffer into one surface, the
           one from two frames ago locked and checked. */
        rtdata(dev, bb, ring[f % 3]);
        if (f >= 2) {
            int old = (f + 1) % 3;
            uint32_t o = (uint32_t)f - 2;
            HR hl = VT(ring[old], SLOT_IDirect3DSurface9_LockRect, HR (*)(void *, locked_rect *, const void *, uint32_t))(
                ring[old], &lr, NULL, 0x10);
            if (!hl) {
                const uint32_t *px = lr.pBits;
                uint32_t p = (uint32_t)lr.Pitch / 4;
                int good = (px[16 * p + 16] & 0xFFFFFF) == (stress_color(o, 0) & 0xFFFFFF) &&
                           (px[16 * p + 48] & 0xFFFFFF) == (stress_color(o, 1) & 0xFFFFFF) &&
                           (px[48 * p + 16] & 0xFFFFFF) == (stress_color(o, 2) & 0xFFFFFF) &&
                           (px[48 * p + 48] & 0xFFFFFF) == (stress_color(o, 3) & 0xFFFFFF);
                if (!good && !bad++) first_bad = o;
                checked++;
                VT(ring[old], SLOT_IDirect3DSurface9_UnlockRect, HR (*)(void *))(ring[old]);
            } else {
                bad++;
            }
        }
        present(dev);
        if (f == 49) {
            warm_fp = footprint();
            warm_lists = frame_stat(dev, 0);
            warm_chunks = frame_stat(dev, 4);
        }
    }
    uint64_t end_fp = footprint(), lists = frame_stat(dev, 0), chunks = frame_stat(dev, 4);
    uint32_t w, h, *px = readback(dev, &w, &h); /* drains */
    uint64_t pending = frame_stat(dev, 3);
    int64_t grow = (int64_t)end_fp - (int64_t)warm_fp;
    CHECK(!bad && checked == (uint32_t)frames - 2,
          "stress (%s): %d frames with DISCARD, NOOVERWRITE and plain locks, texture relocks, UpdateSurface and "
          "StretchRect: %u of %u frames exact when read two frames later (first wrong: %d)",
          threaded ? "threaded" : "inline", frames, checked - bad, checked, bad ? (int)first_bad : -1);
    /* Threaded mode bounds the frames in flight, so its lists stop at the
       count the warm-up reached; inline mode is paced only by the GPU, as
       it always was, so its count may still move by one or two. */
#if defined(__has_feature) && __has_feature(thread_sanitizer)
    grow = 0; /* the race detector's own shadow memory grows with the run */
#endif
    CHECK((threaded ? lists == warm_lists : lists <= 8) && chunks == warm_chunks && pending == 0 && warm_fp &&
              grow < (4 << 20),
          "stress (%s): lists allocated %llu (after warm-up %llu), ring chunks %llu (%llu), not retired after a drain "
          "%llu, footprint %.1f MB after warm-up, %+.2f MB at the end",
          threaded ? "threaded" : "inline", (unsigned long long)lists, (unsigned long long)warm_lists,
          (unsigned long long)chunks, (unsigned long long)warm_chunks, (unsigned long long)pending, warm_fp / 1048576.0,
          grow / 1048576.0);
    printf("INFO stress (%s): %.2f us of game thread time per DrawPrimitive (%u draws)\n", threaded ? "threaded" : "inline",
           ndraws ? draw_s * 1e6 / ndraws : 0.0, ndraws);
    free(px);
    set_tex0(dev, NULL);
    void *o[] = {t1, t2, movie_sys, movie, ring[0], ring[1], ring[2], bb, vb, decl};
    for (unsigned i = 0; i < sizeof o / sizeof o[0]; i++) release(o[i]);
    uint32_t cd = VT(dev, SLOT_IDirect3DDevice9_Release, uint32_t (*)(void *))(dev);
    CHECK(cd == 0, "stress device (%s) released: %u", threaded ? "threaded" : "inline", cd);
}

/* Game thread time per draw through the game shaders, the closest the
   headless test gets to the trace's draws column. */
static void draw_cost(void *d3d, int threaded)
{
    void *dev = mode_device(d3d, threaded, 64, 64);
    if (!dev) return;
    kit k;
    if (kit_make(dev, &k)) {
        set_tex0(dev, k.tex);
        unlit_constants(dev, 1, 1, 1, 1);
        double spent = 0;
        uint32_t n = 0;
        const char *nf = getenv("METAL_TEST_DRAW_FRAMES"); /* longer, for a profiler */
        int frames = nf && atoi(nf) > 0 ? atoi(nf) : 200;
        for (int f = 0; f < frames; f++) {
            clear(dev, 0, NULL, 0x7, 0xFF000000u);
            for (int i = 0; i < 100; i++) {
                double t0 = now_s();
                VT(dev, SLOT_IDirect3DDevice9_DrawIndexedPrimitive,
                   HR (*)(void *, uint32_t, int32_t, uint32_t, uint32_t, uint32_t, uint32_t))(dev, 4, 0, 0, 4, 0, 2);
                if (f >= 20) spent += now_s() - t0, n++;
            }
            present(dev);
        }
        printf("INFO game shaders (%s): %.2f us of game thread time per DrawIndexedPrimitive (%u draws, 100 a frame)\n",
               threaded ? "threaded" : "inline", n ? spent * 1e6 / n : 0.0, n);
    }
    kit_free(dev, &k);
    release(dev);
}

/* Pipelines across runs (pipelines.h): a set of real game shader pairs
   drawn through the D3D9 door on a device with an empty cache, then on
   fresh devices with the archive the first one wrote, then with that
   archive damaged or written for another identity. */

enum { PS_OPEN, PS_RECORDED, PS_ARCHIVE, PS_COLD, PS_BY_WARM, PS_BY_REPLAY, PS_WAITS, PS_WAIT_US, PS_WAIT_MAX_US,
       PS_SKIPPED, PS_FN_WARM, PS_FN_REPLAY, PS_WARM_DONE, PS_USED, PS_WARM_US, PS_FAILED, PS_N };

static void pipe_stats(void *dev, uint64_t out[PS_N])
{
    void (*f)(void *, uint64_t *) = (void (*)(void *, uint64_t *))dlsym(s_lib, "metal_d3d9_pipeline_stats");
    memset(out, 0, PS_N * sizeof *out);
    if (f) f(dev, out);
}

static void pipe_wait_idle(void *dev)
{
    void (*f)(void *) = (void (*)(void *))dlsym(s_lib, "metal_d3d9_pipeline_wait_idle");
    if (f) f(dev);
}

static int name_cmp(const void *a, const void *b) { return strcmp(a, b); }

/* The first n shader hashes of a model, in name order, so every pass
   draws the same set. */
static int list_shaders(const char *model, char (*out)[17], int n)
{
    char dir[1024];
    snprintf(dir, sizeof dir, "%s/%s", s_shader_dir, model);
    DIR *d = opendir(dir);
    if (!d) return 0;
    char (*all)[17] = NULL;
    int k = 0, cap = 0;
    struct dirent *e;
    while ((e = readdir(d))) {
        size_t l = strlen(e->d_name);
        if (l != 21 || strcmp(e->d_name + 16, ".dxso")) continue;
        if (k == cap) {
            cap = cap ? 2 * cap : 256;
            all = realloc(all, (size_t)cap * sizeof *all);
        }
        memcpy(all[k], e->d_name, 16);
        all[k++][16] = 0;
    }
    closedir(d);
    qsort(all, (size_t)k, sizeof *all, name_cmp);
    if (k > n) k = n;
    memcpy(out, all, (size_t)k * sizeof *all);
    free(all);
    return k;
}

#define PSO_PS 72
#define PSO_MIXED 8

typedef struct {
    char ps_model[PSO_PS][8], ps_hash[PSO_PS][17];
    int nps;
} pso_set;

static void pso_set_make(pso_set *s)
{
    static const struct {
        const char *model;
        int n;
    } k_models[] = {{"ps_2_0", 40}, {"ps_1_1", 24}, {"ps_1_4", 8}};
    s->nps = 0;
    for (unsigned m = 0; m < sizeof k_models / sizeof k_models[0]; m++) {
        char names[64][17];
        int n = list_shaders(k_models[m].model, names, k_models[m].n);
        for (int i = 0; i < n && s->nps < PSO_PS; i++, s->nps++) {
            snprintf(s->ps_model[s->nps], sizeof s->ps_model[0], "%s", k_models[m].model);
            memcpy(s->ps_hash[s->nps], names[i], 17);
        }
    }
}

/* Pipelines one pass asks for, at most: every pixel shader behind two
   game vertex shaders, the first PSO_MIXED behind the fixed-function
   vertex shader, and the fixed-function pair. */
static int pso_pipelines(const pso_set *s, int limit)
{
    int nps = limit && limit < s->nps ? limit : s->nps;
    return 2 * nps + (nps < PSO_MIXED ? nps : PSO_MIXED) + 1;
}

typedef struct {
    double secs, close_secs; /* the draws, and the device Release that writes the archive */
    uint64_t st[PS_N];
    uint32_t dev_refs;
} pso_result;

/* One pass on a new device. limit 0 draws the whole set. warm: wait for
   the warm queue before the first draw. extra: one more draw with alpha
   blending, a pipeline no earlier pass made. keep: the device is not
   released but handed back, the way the game leaves its device at exit. */
static int pso_pass_k(void *d3d, const pso_set *s, int limit, int warm, int extra, pso_result *r, void **keep)
{
    memset(r, 0, sizeof *r);
    void *dev = make_device(d3d, 64, 64);
    if (!dev) return 0;
    kit k;
    int ok = kit_make(dev, &k);
    void *vs[2] = {k.vs, load_shader(dev, "vs_2_0", "00b6b64952cc4128")};
    int nps = limit && limit < s->nps ? limit : s->nps;
    void *ps[PSO_PS] = {0};
    for (int i = 0; i < nps; i++) {
        ps[i] = load_shader(dev, s->ps_model[i], s->ps_hash[i]);
        ok &= ps[i] != NULL;
    }
    ok &= vs[1] != NULL;
    if (ok) {
        set_tex0(dev, k.tex);
        unlit_constants(dev, 1, 1, 1, 1);
        if (warm) pipe_wait_idle(dev);
        HR (*set_vs)(void *, void *) = VT(dev, SLOT_IDirect3DDevice9_SetVertexShader, HR (*)(void *, void *));
        HR (*set_ps)(void *, void *) = VT(dev, SLOT_IDirect3DDevice9_SetPixelShader, HR (*)(void *, void *));
        double t0 = now_s();
        clear(dev, 0, NULL, 0x7, 0xFF000000u);
        int n = 0;
        for (int v = 0; v < 3; v++)
            for (int i = 0; i < nps; i++) {
                if (v == 2 && i >= PSO_MIXED) break;
                set_vs(dev, v < 2 ? vs[v] : NULL);
                set_ps(dev, ps[i]);
                quad(dev, k.vb, -1, -1, 1, 1, 0.5f);
                if (++n % 16 == 0) present(dev);
            }
        set_vs(dev, NULL);
        set_ps(dev, NULL);
        quad(dev, k.vb, -1, -1, 1, 1, 0.5f);
        if (extra) {
            set_vs(dev, vs[0]);
            set_ps(dev, ps[0]);
            rs(dev, 27, 1); /* ALPHABLENDENABLE, SRCALPHA, INVSRCALPHA */
            rs(dev, 19, 5);
            rs(dev, 20, 6);
            quad(dev, k.vb, -1, -1, 1, 1, 0.5f);
            rs(dev, 27, 0);
        }
        present(dev);
        uint32_t w, h;
        free(readback(dev, &w, &h)); /* waits until the render thread has replayed it all */
        r->secs = now_s() - t0;
        set_vs(dev, k.vs);
        set_ps(dev, k.ps);
        pipe_stats(dev, r->st);
    }
    for (int i = 0; i < nps; i++) release(ps[i]);
    release(vs[1]);
    kit_free(dev, &k);
    if (keep) {
        *keep = dev;
        return ok;
    }
    double t1 = now_s();
    r->dev_refs = VT(dev, SLOT_IDirect3DDevice9_Release, uint32_t (*)(void *))(dev);
    r->close_secs = now_s() - t1;
    return ok;
}

static int pso_pass_x(void *d3d, const pso_set *s, int limit, int warm, int extra, pso_result *r)
{
    return pso_pass_k(d3d, s, limit, warm, extra, r, NULL);
}

static int pso_pass(void *d3d, const pso_set *s, int limit, int warm, pso_result *r)
{
    return pso_pass_x(d3d, s, limit, warm, 0, r);
}

static void pso_report(const char *what, const pso_result *r)
{
    const uint64_t *s = r->st;
    printf("INFO %s: %.3f s for the draws; %llu pipelines used, %llu from the archive, %llu cold; warm queue made %llu "
           "(%llu functions, done %llu, %.1f ms), replay thread made %llu (%llu functions); replay thread waited %llu "
           "times, %.1f ms in all, longest %.1f ms\n",
           what, r->secs, (unsigned long long)s[PS_USED], (unsigned long long)s[PS_ARCHIVE],
           (unsigned long long)s[PS_COLD], (unsigned long long)s[PS_BY_WARM], (unsigned long long)s[PS_FN_WARM],
           (unsigned long long)s[PS_WARM_DONE], s[PS_WARM_US] / 1000.0, (unsigned long long)s[PS_BY_REPLAY],
           (unsigned long long)s[PS_FN_REPLAY], (unsigned long long)s[PS_WAITS], s[PS_WAIT_US] / 1000.0,
           s[PS_WAIT_MAX_US] / 1000.0);
}

/* Overwrites len bytes at the middle of a file, the damage Metal itself
   does not notice. */
static int damage_middle(const char *path, size_t len)
{
    FILE *f = fopen(path, "r+b");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    if (size < (long)(2 * len)) {
        fclose(f);
        return 0;
    }
    fseek(f, size / 2, SEEK_SET);
    for (size_t i = 0; i < len; i++) fputc((int)(0xA5 ^ i), f);
    fclose(f);
    return 1;
}

static int truncate_half(const char *path)
{
    struct stat sb;
    return stat(path, &sb) == 0 && truncate(path, sb.st_size / 2) == 0;
}

/* The exit hook's log, counted: the pipeline summary must appear once
   however many times the device is finished. */
static int s_summaries, s_saves;

static void count_log(const char *msg)
{
    if (!strncmp(msg, "pipelines: ", 11) && strstr(msg, " used; ")) s_summaries++;
    if (!strncmp(msg, "pipelines: saved ", 17)) s_saves++;
    fprintf(stderr, "metal_d3d9: %s\n", msg);
}

static int same_time(const struct stat *a, const struct stat *b)
{
    return a->st_mtimespec.tv_sec == b->st_mtimespec.tv_sec && a->st_mtimespec.tv_nsec == b->st_mtimespec.tv_nsec;
}

/* The game quits without releasing its device: metal_d3d9_shutdown, the
   bridge's exit hook, must leave what a release would have. */
static void shutdown_test(void *d3d, const pso_set *set, const char *argv0)
{
    void (*shutdown)(void) = (void (*)(void))dlsym(s_lib, "metal_d3d9_shutdown");
    void (*host)(void (*)(const char *), void (*)(const char *)) =
        (void (*)(void (*)(const char *), void (*)(const char *)))dlsym(s_lib, "metal_d3d9_set_host");
    CHECK(shutdown != NULL, "exports metal_d3d9_shutdown");
    if (!shutdown || !host) return;
    char dir[1024], tag[256] = {0}, archive[1400], keys[1400];
    snprintf(dir, sizeof dir, "%s.pso-XXXXXX", argv0);
    if (!mkdtemp(dir)) {
        CHECK(0, "shutdown cache directory made");
        return;
    }
    setenv("RT_METAL_CACHE_DIR", dir, 1);
    int small = 12, want = pso_pipelines(set, small);
    s_summaries = s_saves = 0;
    host(count_log, NULL);
    pso_result a, b;
    void *dev = NULL;
    int ok = pso_pass_k(d3d, set, small, 0, 0, &a, &dev);
    int summaries_before = s_summaries;
    shutdown();
    int saves_first = s_saves, summaries_first = s_summaries;
    DIR *dd = opendir(dir);
    for (struct dirent *e; dd && (e = readdir(dd));)
        if (e->d_name[0] != '.') snprintf(tag, sizeof tag, "%s", e->d_name);
    if (dd) closedir(dd);
    snprintf(archive, sizeof archive, "%s/%s/pipelines.bin", dir, tag);
    snprintf(keys, sizeof keys, "%s/%s/pipelines.keys", dir, tag);
    struct stat sa, sk;
    int have = stat(archive, &sa) == 0 && sa.st_size > 0 && stat(keys, &sk) == 0 && sk.st_size > 0;
    CHECK(ok && dev && a.st[PS_COLD] == (uint64_t)want && have && saves_first == 1 && summaries_before == 0 &&
              summaries_first == 1,
          "device never released, metal_d3d9_shutdown: archive and keys written (%s), %d save and %d summary (want 1 "
          "and 1) for %llu cold pipelines",
          have ? "yes" : "no", saves_first, summaries_first - summaries_before, (unsigned long long)a.st[PS_COLD]);

    ok = pso_pass(d3d, set, small, 0, &b);
    CHECK(ok && b.dev_refs == 0 && b.st[PS_OPEN] == 2 && b.st[PS_RECORDED] == (uint64_t)want && b.st[PS_COLD] == 0 &&
              b.st[PS_ARCHIVE] == (uint64_t)want,
          "and a fresh device loads what shutdown wrote: open state %llu (want 2), %llu cold (want 0), %llu of %d from "
          "the archive",
          (unsigned long long)b.st[PS_OPEN], (unsigned long long)b.st[PS_COLD], (unsigned long long)b.st[PS_ARCHIVE],
          want);

    int summaries_mid = s_summaries, saves_mid = s_saves;
    shutdown();
    uint32_t refs = dev ? VT(dev, SLOT_IDirect3DDevice9_Release, uint32_t (*)(void *))(dev) : 1;
    struct stat sa2, sk2;
    int same = stat(archive, &sa2) == 0 && stat(keys, &sk2) == 0 && same_time(&sa, &sa2) && same_time(&sk, &sk2);
    CHECK(refs == 0 && same && s_saves == saves_mid && s_summaries == summaries_mid,
          "shutdown again, then Release of the kept device (returns %u): files untouched (%s), %d more saves and %d more "
          "summaries (want 0 and 0)",
          refs, same ? "yes" : "no", s_saves - saves_mid, s_summaries - summaries_mid);
    shutdown(); /* with no device left, nothing to reach */
    host(NULL, NULL);
}

static void skip_mode_test(void *d3d, const char *argv0)
{
    /* Skip mode: a draw whose functions or pipeline are not ready is
       left out of its frame, and drawn once the warm queue has made
       them. Functions go first, so a cold draw is skipped twice: once
       for its functions, once for its pipeline. */
    char dir2[1024];
    snprintf(dir2, sizeof dir2, "%s.pso-XXXXXX", argv0);
    if (mkdtemp(dir2)) {
        setenv("RT_METAL_CACHE_DIR", dir2, 1);
        setenv("RT_METAL_PSO_SKIP", "1", 1);
        void *dev = make_device(d3d, 32, 32);
        unsetenv("RT_METAL_PSO_SKIP");
        kit k;
        if (dev && kit_make(dev, &k)) {
            set_tex0(dev, k.white);
            unlit_constants(dev, 1, 0, 0, 1);
            uint32_t w = 0, h = 0, blue0 = 0, red = 0;
            int frames = 0;
            while (frames < 4 && (frames == 0 || red != w * h)) {
                if (frames) pipe_wait_idle(dev);
                clear(dev, 0, NULL, 0x7, 0xFF0000FFu);
                quad(dev, k.vb, -1, -1, 1, 1, 0.5f);
                present(dev);
                uint32_t *px = readback(dev, &w, &h);
                if (!px) break;
                if (!frames) blue0 = count_rgb(px, w * h, 0x0000FF);
                red = count_rgb(px, w * h, 0xFF0000);
                free(px);
                frames++;
            }
            uint64_t st[PS_N];
            pipe_stats(dev, st);
            CHECK(w && blue0 == w * h && red == w * h && frames == 3 && st[PS_SKIPPED] == 2 && st[PS_WAITS] == 0,
                  "RT_METAL_PSO_SKIP=1: the first frame skips the draw (%u of %u pixels still the clear), frame %d draws it "
                  "(%u red, want frame 3), %llu draws skipped, replay thread waited %llu times",
                  blue0, w * h, frames, red, (unsigned long long)st[PS_SKIPPED], (unsigned long long)st[PS_WAITS]);
            kit_free(dev, &k);
        }
        if (dev) release(dev);
    }
}

/* RT_METAL_FORCE_LIMITS=1 (or a GPU with no architecture): there is no
   archive, so every pass compiles every pipeline, nothing is warmed and
   nothing is written beside the MSL. */
static void no_archive_tests(void *d3d, const pso_set *set, const char *dir)
{
    int want = pso_pipelines(set, 0);
    pso_result a, b;
    int ok = pso_pass(d3d, set, 0, 0, &a) && pso_pass(d3d, set, 0, 0, &b);
    int files = 0;
    char path[1400];
    DIR *dd = opendir(dir);
    for (struct dirent *e; dd && (e = readdir(dd));) {
        if (e->d_name[0] == '.') continue;
        struct stat sb;
        snprintf(path, sizeof path, "%s/%s/pipelines.bin", dir, e->d_name);
        files += stat(path, &sb) == 0;
        snprintf(path, sizeof path, "%s/%s/pipelines.keys", dir, e->d_name);
        files += stat(path, &sb) == 0;
    }
    if (dd) closedir(dd);
    CHECK(ok && a.dev_refs == 0 && b.dev_refs == 0 && a.st[PS_OPEN] == 0 && b.st[PS_OPEN] == 0 &&
              a.st[PS_COLD] == (uint64_t)want && b.st[PS_COLD] == (uint64_t)want && a.st[PS_ARCHIVE] == 0 &&
              b.st[PS_ARCHIVE] == 0 && a.st[PS_FAILED] == 0 && b.st[PS_FAILED] == 0 && files == 0,
          "no pipeline archive: two passes each compile all %d pipelines cold (%llu, %llu), none failed, open state %llu, "
          "%d archive or keys files written",
          want, (unsigned long long)a.st[PS_COLD], (unsigned long long)b.st[PS_COLD], (unsigned long long)a.st[PS_OPEN],
          files);
}

static void pso_tests(void *d3d, const char *argv0)
{
    const char *was = getenv("RT_METAL_CACHE_DIR");
    char keep[1024] = {0};
    if (was) snprintf(keep, sizeof keep, "%s", was);
    char dir[1024], archive[1200], keys[1200], tag[256] = {0};
    snprintf(dir, sizeof dir, "%s.pso-XXXXXX", argv0);
    if (!mkdtemp(dir)) {
        CHECK(0, "pipeline cache directory made");
        return;
    }
    setenv("RT_METAL_CACHE_DIR", dir, 1);
    pso_set *set = calloc(1, sizeof *set);
    pso_set_make(set);
    int want = pso_pipelines(set, 0);
    pso_result a, b, c;
    const char *fl = getenv("RT_METAL_FORCE_LIMITS");
    if (fl && atoi(fl) > 0) {
        no_archive_tests(d3d, set, dir);
        free(set);
        skip_mode_test(d3d, argv0);
        if (was) setenv("RT_METAL_CACHE_DIR", keep, 1);
        else unsetenv("RT_METAL_CACHE_DIR");
        return;
    }

    /* 1. Cold, then from the archive racing the warm queue, then with
       the warm queue finished first. */
    int ok = pso_pass(d3d, set, 0, 0, &a);
    pso_report("cold cache", &a);
    CHECK(ok && a.dev_refs == 0 && a.st[PS_OPEN] == 1 && a.st[PS_USED] == (uint64_t)want &&
              a.st[PS_COLD] == (uint64_t)want && a.st[PS_ARCHIVE] == 0 && a.st[PS_FAILED] == 0,
          "cold cache: %llu of %d game shader pipelines compiled cold, 0 from the archive (got %llu), opened as new (%llu)",
          (unsigned long long)a.st[PS_COLD], want, (unsigned long long)a.st[PS_ARCHIVE],
          (unsigned long long)a.st[PS_OPEN]);
    DIR *dd = opendir(dir);
    for (struct dirent *e; dd && (e = readdir(dd));)
        if (e->d_name[0] != '.') snprintf(tag, sizeof tag, "%s", e->d_name);
    if (dd) closedir(dd);
    snprintf(archive, sizeof archive, "%s/%s/pipelines.bin", dir, tag);
    snprintf(keys, sizeof keys, "%s/%s/pipelines.keys", dir, tag);
    struct stat sb;
    CHECK(stat(archive, &sb) == 0 && sb.st_size > 0 && stat(keys, &sb) == 0, "archive and keys written beside the MSL (%s)",
          tag);

    ok = pso_pass(d3d, set, 0, 0, &b);
    pso_report("archive, drawing at once", &b);
    CHECK(ok && b.dev_refs == 0 && b.st[PS_OPEN] == 2 && b.st[PS_RECORDED] == (uint64_t)want && b.st[PS_COLD] == 0 &&
              b.st[PS_ARCHIVE] == (uint64_t)want && b.st[PS_FAILED] == 0,
          "fresh device with the archive, drawing at once: %llu cold (want 0), %llu of %d from the archive (%llu by the "
          "warm queue, %llu by the replay thread)",
          (unsigned long long)b.st[PS_COLD], (unsigned long long)b.st[PS_ARCHIVE], want,
          (unsigned long long)b.st[PS_BY_WARM], (unsigned long long)b.st[PS_BY_REPLAY]);

    ok = pso_pass(d3d, set, 0, 1, &c);
    pso_report("archive, warm queue finished first", &c);
    CHECK(ok && c.dev_refs == 0 && c.st[PS_COLD] == 0 && c.st[PS_BY_WARM] == (uint64_t)want &&
              c.st[PS_BY_REPLAY] == 0 && c.st[PS_FN_REPLAY] == 0 && c.st[PS_WAITS] == 0,
          "warmed before the first draw: %llu of %d pipelines made by the warm queue, replay thread built %llu and waited "
          "%llu times",
          (unsigned long long)c.st[PS_BY_WARM], want, (unsigned long long)(c.st[PS_BY_REPLAY] + c.st[PS_FN_REPLAY]),
          (unsigned long long)c.st[PS_WAITS]);
    printf("INFO pipeline timings: cold %.3f s (replay thread waited %.1f ms), archive %.3f s (%.1f ms), warmed %.3f s "
           "(%.1f ms)\n",
           a.secs, a.st[PS_WAIT_US] / 1000.0, b.secs, b.st[PS_WAIT_US] / 1000.0, c.secs, c.st[PS_WAIT_US] / 1000.0);

    /* What a quit costs when a run adds one pipeline to a known set: the
       archive is written new from every pipeline made (pipelines.m), so
       this is the per-pipeline cost of copying the ones the old archive
       already held. */
    pso_result x;
    ok = pso_pass_x(d3d, set, 0, 1, 1, &x);
    CHECK(ok && x.dev_refs == 0 && x.st[PS_COLD] == 1 && x.st[PS_USED] == (uint64_t)want + 1,
          "one new pipeline over the warmed set: %llu cold (want 1) of %llu used",
          (unsigned long long)x.st[PS_COLD], (unsigned long long)x.st[PS_USED]);
    printf("INFO device close: %.1f ms with nothing new (no write), %.1f ms rewriting the archive for %d pipelines with "
           "1 new, %.1f ms writing %d new on the cold pass\n",
           c.close_secs * 1000.0, x.close_secs * 1000.0, want + 1, a.close_secs * 1000.0, want);

    /* 2. Damage Metal would load: the hash in the keys catches it. */
    int small = 12, want_small = pso_pipelines(set, small);
    pso_result d;
    int dmg = damage_middle(archive, 256);
    ok = pso_pass(d3d, set, small, 0, &d);
    CHECK(dmg && ok && d.dev_refs == 0 && d.st[PS_OPEN] == 3 && d.st[PS_RECORDED] == 0 &&
              d.st[PS_COLD] == (uint64_t)want_small && d.st[PS_FAILED] == 0,
          "archive damaged in the middle: detected (open state %llu, want 3), rebuilt, %llu of %d pipelines compiled again",
          (unsigned long long)d.st[PS_OPEN], (unsigned long long)d.st[PS_COLD], want_small);
    ok = pso_pass(d3d, set, small, 0, &d);
    CHECK(ok && d.st[PS_OPEN] == 2 && d.st[PS_COLD] == 0 && d.st[PS_ARCHIVE] == (uint64_t)want_small,
          "and the rebuilt archive serves the next device: %llu cold, %llu from the archive",
          (unsigned long long)d.st[PS_COLD], (unsigned long long)d.st[PS_ARCHIVE]);

    /* 3. A zeroed archive header and keys cut short. */
    FILE *f = fopen(archive, "r+b");
    if (f) {
        for (int i = 0; i < 64; i++) fputc(0, f);
        fclose(f);
    }
    truncate_half(keys);
    ok = pso_pass(d3d, set, small, 0, &d);
    CHECK(ok && d.dev_refs == 0 && d.st[PS_OPEN] == 3 && d.st[PS_COLD] == (uint64_t)want_small && d.st[PS_FAILED] == 0,
          "archive header zeroed and keys cut short: detected (open state %llu, want 3), rebuilt, %llu of %d compiled",
          (unsigned long long)d.st[PS_OPEN], (unsigned long long)d.st[PS_COLD], want_small);

    /* 4. Written for another build, GPU or OS. */
    setenv("RT_METAL_PSO_IDENT", "another gpu", 1);
    ok = pso_pass(d3d, set, small, 0, &d);
    unsetenv("RT_METAL_PSO_IDENT");
    CHECK(ok && d.dev_refs == 0 && d.st[PS_OPEN] == 4 && d.st[PS_COLD] == (uint64_t)want_small && d.st[PS_FAILED] == 0,
          "archive from another identity: detected as mismatched (open state %llu, want 4), rebuilt, %llu of %d compiled",
          (unsigned long long)d.st[PS_OPEN], (unsigned long long)d.st[PS_COLD], want_small);
    ok = pso_pass(d3d, set, small, 0, &d);
    CHECK(ok && d.st[PS_OPEN] == 4 && d.st[PS_COLD] == (uint64_t)want_small,
          "and back on this identity, that one is mismatched in turn (open state %llu), rebuilt", (unsigned long long)d.st[PS_OPEN]);
    shutdown_test(d3d, set, argv0);
    free(set);

    skip_mode_test(d3d, argv0);
    if (was) setenv("RT_METAL_CACHE_DIR", keep, 1);
    else unsetenv("RT_METAL_CACHE_DIR");
}

/* The upload decoder (texconv.c), through its test export, on blocks
   whose every texel is known: endpoints of 255 and 0 make the
   interpolated values exact (170 and 85, the BC3 alpha sevenths and
   fifths). These are what a GPU without BC sampling shows. */
static int px_is(const uint8_t *p, int r, int g, int b, int a) { return p[0] == r && p[1] == g && p[2] == b && p[3] == a; }

static void conv_tests(void)
{
    void (*conv)(uint32_t, const void *, uint32_t, uint32_t, uint32_t, void *) =
        (void (*)(uint32_t, const void *, uint32_t, uint32_t, uint32_t, void *))dlsym(s_lib, "metal_d3d9_conv_level");
    CHECK(conv != NULL, "exports metal_d3d9_conv_level");
    if (!conv) return;
    uint8_t o[16 * 4];

    /* DXT1, four colours (c0 red > c1 blue), indices 0 1 2 3 in every row. */
    static const uint8_t k_bc1[8] = {0x00, 0xF8, 0x1F, 0x00, 0xE4, 0xE4, 0xE4, 0xE4};
    conv(0x31545844u, k_bc1, 8, 4, 4, o);
    CHECK(px_is(o, 255, 0, 0, 255) && px_is(o + 4, 0, 0, 255, 255) && px_is(o + 8, 170, 0, 85, 255) &&
              px_is(o + 12, 85, 0, 170, 255) && px_is(o + 60, 85, 0, 170, 255),
          "DXT1 four-colour block decodes to red, blue, 2:1 and 1:2 (texel 2 %u,%u,%u,%u)", o[8], o[9], o[10], o[11]);

    /* DXT1, three colours and transparent black (c0 blue <= c1 red). */
    static const uint8_t k_bc1t[8] = {0x1F, 0x00, 0x00, 0xF8, 0xE4, 0xE4, 0xE4, 0xE4};
    conv(0x31545844u, k_bc1t, 8, 4, 4, o);
    CHECK(px_is(o, 0, 0, 255, 255) && px_is(o + 4, 255, 0, 0, 255) && px_is(o + 8, 127, 0, 127, 255) &&
              px_is(o + 12, 0, 0, 0, 0),
          "DXT1 three-colour block: midpoint %u,%u,%u and index 3 transparent black (alpha %u)", o[8], o[9], o[10], o[15]);

    /* DXT3: explicit alpha 0, 15, then 8s; its colour block interpolates
       four colours even with c0 <= c1. */
    static const uint8_t k_bc2[16] = {0xF0, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88,
                                      0x1F, 0x00, 0x00, 0xF8, 0xE4, 0xE4, 0xE4, 0xE4};
    conv(0x33545844u, k_bc2, 16, 4, 4, o);
    CHECK(px_is(o, 0, 0, 255, 0) && px_is(o + 4, 255, 0, 0, 255) && px_is(o + 8, 85, 0, 170, 136) &&
              px_is(o + 12, 170, 0, 85, 136),
          "DXT3 block: alpha 0, 255, 136 and four colours with c0 <= c1 (texel 3 %u,%u,%u,%u)", o[12], o[13], o[14], o[15]);

    /* DXT5, a0 255 > a1 0: codes 0 to 7 on texels 0 to 7 give the six
       interpolated values; then a0 0 <= a1 255: fifths, then 0 and 255. */
    uint8_t bc3[16] = {255, 0};
    uint64_t bits = 0;
    for (int t = 0; t < 16; t++) bits |= (uint64_t)(t & 7) << (3 * t);
    for (int i = 0; i < 6; i++) bc3[2 + i] = (uint8_t)(bits >> (8 * i));
    memcpy(bc3 + 8, k_bc1, 8);
    conv(0x35545844u, bc3, 16, 4, 4, o);
    static const uint8_t k_a6[8] = {255, 0, 218, 182, 145, 109, 72, 36};
    int good = 1;
    char got[64] = {0};
    for (int t = 0; t < 8; t++) {
        good &= o[4 * t + 3] == k_a6[t];
        snprintf(got + strlen(got), sizeof got - strlen(got), " %u", o[4 * t + 3]);
    }
    good &= px_is(o, 255, 0, 0, 255) && o[4 * 8 + 3] == 255;
    CHECK(good, "DXT5 block, a0 > a1: alpha codes 0 to 7 read%s", got);
    bc3[0] = 0;
    bc3[1] = 255;
    conv(0x35545844u, bc3, 16, 4, 4, o);
    static const uint8_t k_a4[8] = {0, 255, 51, 102, 153, 204, 0, 255};
    good = 1;
    got[0] = 0;
    for (int t = 0; t < 8; t++) {
        good &= o[4 * t + 3] == k_a4[t];
        snprintf(got + strlen(got), sizeof got - strlen(got), " %u", o[4 * t + 3]);
    }
    CHECK(good, "DXT5 block, a0 <= a1: alpha codes 0 to 7 read%s", got);

    /* A 2x2 level still stores a whole block; the decode is cropped. */
    memset(o, 0xEE, sizeof o);
    conv(0x31545844u, k_bc1, 8, 2, 2, o);
    CHECK(px_is(o, 255, 0, 0, 255) && px_is(o + 4, 0, 0, 255, 255) && px_is(o + 8, 255, 0, 0, 255) &&
              px_is(o + 12, 0, 0, 255, 255) && o[16] == 0xEE,
          "DXT1 2x2 level: cropped from its block, nothing written past 2x2");

    /* The 16-bit formats a GPU may lack: R5G6B5, X1R5G5B5 and A1R5G5B5
       (alpha bit clear), A4R4G4B4. */
    static const uint16_t k16[4] = {0xF800, 0x7C00, 0x7C00, 0xF0A5};
    static const uint32_t f16[4] = {23, 24, 25, 26};
    uint8_t want16[4][4] = {{255, 0, 0, 255}, {255, 0, 0, 255}, {255, 0, 0, 0}, {0, 170, 85, 255}};
    good = 1;
    for (int i = 0; i < 4; i++) {
        conv(f16[i], &k16[i], 2, 1, 1, o);
        good &= !memcmp(o, want16[i], 4);
    }
    CHECK(good, "16-bit R5G6B5, X1R5G5B5, A1R5G5B5 and A4R4G4B4 widen to RGBA8");
}

/* A pipeline the GPU rejects drops its draws and nothing waits on it:
   the frame is presented and read back with the clear intact and the
   draws after it drawn, in the mode RT_METAL_INLINE selects. */
static void fail_pipeline_test(void *d3d, int threaded)
{
    void (*fail)(void *, int) = (void (*)(void *, int))dlsym(s_lib, "metal_d3d9_pipeline_fail");
    CHECK(fail != NULL, "exports metal_d3d9_pipeline_fail");
    if (!fail) return;
    void *dev = mode_device(d3d, threaded, 32, 32);
    if (!dev) return;
    void *t = new_texture(dev, 4, 4, 1, 21, 1, NULL);
    uint32_t red = 0xFFFF0000u;
    if (t) solid(t, red, 0);
    void *vb = ff_vb(dev, 8, 0);
    if (!t || !vb) {
        CHECK(0, "failing pipeline test set up");
        release(t);
        release(vb);
        release(dev);
        return;
    }
    void *decl = ff_setup(dev);
    set_tex0(dev, t);
    put_quad(vb, 0, -1, -1, 0, 1, 0);
    put_quad(vb, 4, 0, -1, 1, 1, 0);
    /* The left half with every new pipeline failing, the right half with
       a pipeline made before the failures began. */
    clear(dev, 0, NULL, 0x7, 0xFF0000FFu);
    draw_strip(dev, 4);
    present(dev);
    uint32_t w = 0, h = 0, *px = readback(dev, &w, &h); /* the render thread has made that pipeline */
    free(px);
    fail(dev, 1);
    rs(dev, 27, 1); /* ALPHABLENDENABLE: a pipeline no earlier draw made */
    rs(dev, 19, 2); /* ONE, ZERO: draws like no blend if it were made */
    rs(dev, 20, 1);
    double t0 = now_s();
    clear(dev, 0, NULL, 0x7, 0xFF0000FFu);
    for (int i = 0; i < 3; i++) draw_strip(dev, 0); /* the same key three times: logged once */
    rs(dev, 27, 0);
    draw_strip(dev, 4);
    present(dev);
    px = readback(dev, &w, &h);
    double secs = now_s() - t0;
    fail(dev, 0);
    uint32_t left = px ? at(px, w, w / 4, h / 2) : 0, right = px ? at(px, w, 3 * w / 4, h / 2) : 0;
    CHECK(px && left == 0x0000FF && right == 0xFF0000 && secs < 2.0,
          "%s: a pipeline that fails to build drops its draws (left 0x%06X, the clear), the frame completes in %.3f s and "
          "the next draw lands (right 0x%06X)",
          threaded ? "render thread" : "inline", left, secs, right);
    free(px);
    set_tex0(dev, NULL);
    release(t);
    release(vb);
    release(decl);
    release(dev);
}

/* Immersive mode's handoff on the GPU: Present copies the back buffer into
   the ring instead of a layer, a slot is published only once its copy has
   completed, the compositor's held slot is never written while held, and
   the counters see repeats and frames never shown. The bookkeeping alone
   is unit tested in handoff_test.c; this is the same through the device. */
static int wait_handed(void (*stats)(uint64_t[8]), uint64_t want)
{
    for (int i = 0; i < 2000; i++) {
        uint64_t st[8];
        stats(st);
        if (st[2] + st[3] >= want) return 1;
        struct timespec ts = {0, 1000000};
        nanosleep(&ts, NULL);
    }
    return 0;
}

static void immersive_tests(void *d3d)
{
    void (*enable)(int) = (void (*)(int))dlsym(s_lib, "metal_d3d9_immersive_enable");
    int (*acquire)(metal_d3d9_shown_frame *) = (int (*)(metal_d3d9_shown_frame *))dlsym(s_lib, "metal_d3d9_immersive_acquire");
    void (*release)(int) = (void (*)(int))dlsym(s_lib, "metal_d3d9_immersive_release");
    void (*stats)(uint64_t[8]) = (void (*)(uint64_t[8]))dlsym(s_lib, "metal_d3d9_immersive_stats");
    int (*read)(int, uint32_t *, uint32_t, uint32_t) =
        (int (*)(int, uint32_t *, uint32_t, uint32_t))dlsym(s_lib, "metal_d3d9_immersive_read");
    CHECK(enable && acquire && release && stats && read, "exports the immersive handoff and its test hooks");
    if (!enable || !acquire || !release || !stats || !read) return;
    /* No pacing here: the copies are waited for one by one anyway. Read
       once, when the handoff first starts. */
    setenv("RT_IMMERSIVE_MAX_HZ", "0", 1);
    enable(1);
    metal_d3d9_shown_frame f, g, g2;
    int got = acquire(&f);
    CHECK(!got && f.slot == -1, "immersive: acquire before any frame holds nothing");
    void *dev = make_device(d3d, 64, 48);
    if (!dev) return;
    HR (*present)(void *, const void *, const void *, void *, const void *) =
        VT(dev, SLOT_IDirect3DDevice9_Present, HR (*)(void *, const void *, const void *, void *, const void *));
    static const uint32_t k_col[] = {0xFF2040, 0x20FF40, 0x2040FF, 0xFFFF20, 0x20FFFF, 0xFF20FF, 0x808080};
    clear(dev, 0, NULL, 0x1, 0xFF000000u | k_col[0]);
    HR hr = present(dev, NULL, NULL, NULL, NULL);
    int ok = !hr && wait_handed(stats, 1);
    /* The compositor paces itself, so no frame waits in a queue in front
       of the GPU for it: one frame in flight, not the window's two. */
    CHECK(frame_stat(dev, 5) == 0 || frame_stat(dev, 6) == 1,
          "immersive: one frame in flight while immersive (depth %llu)", (unsigned long long)frame_stat(dev, 6));
    got = ok && acquire(&f);
    uint32_t px[64 * 48];
    int rd = got && read(f.slot, px, 64, 48);
    CHECK(got && rd && f.width == 64 && f.height == 48 && f.frame == 1 && !f.repeat && f.age_ms >= 0 &&
              f.age_ms < 2000 && count_rgb(px, 64 * 48, k_col[0]) == 64 * 48,
          "immersive: Present hands frame %llu (%ux%u, age %.2f ms) to the ring, %u of %u pixels the cleared colour",
          (unsigned long long)f.frame, f.width, f.height, f.age_ms, rd ? count_rgb(px, 64 * 48, k_col[0]) : 0, 64 * 48);
    if (!got) {
        VT(dev, SLOT_IDirect3DDevice9_Release, uint32_t (*)(void *))(dev);
        enable(0);
        return;
    }
    /* Six more frames while the compositor holds the first. */
    int handed = 1;
    for (uint32_t i = 1; i < sizeof k_col / sizeof k_col[0]; i++) {
        clear(dev, 0, NULL, 0x1, 0xFF000000u | k_col[i]);
        present(dev, NULL, NULL, NULL, NULL);
        handed &= wait_handed(stats, 1 + i);
    }
    rd = read(f.slot, px, 64, 48);
    CHECK(handed && rd && count_rgb(px, 64 * 48, k_col[0]) == 64 * 48,
          "immersive: six frames later the held slot %d still holds frame 1 (%u of %u pixels)", f.slot,
          rd ? count_rgb(px, 64 * 48, k_col[0]) : 0, 64 * 48);
    got = acquire(&g);
    rd = got && read(g.slot, px, 64, 48);
    uint32_t last = k_col[sizeof k_col / sizeof k_col[0] - 1];
    CHECK(got && rd && g.slot != f.slot && g.frame == 7 && !g.repeat && count_rgb(px, 64 * 48, last) == 64 * 48,
          "immersive: the next acquire gets the newest frame %llu in slot %d, its pixels the last colour",
          (unsigned long long)g.frame, g.slot);
    got = acquire(&g2);
    CHECK(got && g2.slot == g.slot && g2.frame == g.frame && g2.repeat,
          "immersive: with no new frame the compositor shows frame %llu again, marked a repeat",
          (unsigned long long)g2.frame);
    release(f.slot);
    release(g.slot);
    release(g2.slot);
    uint64_t st[8];
    stats(st);
    CHECK(st[2] == 7 && st[3] == 0 && st[4] == 5 && st[5] == 2 && st[6] == 1 && st[7] == 1,
          "immersive counters: %llu handed over, %llu skipped, %llu never shown, %llu shown, %llu shown again, %llu "
          "repeats",
          (unsigned long long)st[2], (unsigned long long)st[3], (unsigned long long)st[4], (unsigned long long)st[5],
          (unsigned long long)st[6], (unsigned long long)st[7]);
    VT(dev, SLOT_IDirect3DDevice9_Release, uint32_t (*)(void *))(dev);

    /* Off again: a device's Present hands nothing over. */
    enable(0);
    dev = make_device(d3d, 32, 32);
    if (!dev) return;
    clear(dev, 0, NULL, 0x1, 0xFF000000u);
    present(dev, NULL, NULL, NULL, NULL);
    CHECK(frame_stat(dev, 5) == 0 || frame_stat(dev, 6) == 2,
          "immersive off: back to two frames in flight (depth %llu)", (unsigned long long)frame_stat(dev, 6));
    /* Release drains the render thread, so a copy would be done by now. */
    VT(dev, SLOT_IDirect3DDevice9_Release, uint32_t (*)(void *))(dev);
    uint64_t st2[8];
    stats(st2);
    CHECK(st2[2] == st[2], "immersive off: Present hands nothing over (%llu then %llu)", (unsigned long long)st[2],
          (unsigned long long)st2[2]);
}


/* Stereo and the eye-identity test (stereo.h, immersive.h): frames replayed
   once per eye into the ring's two slices, on real game shaders
   (unlitgeneric) and on the fixed-function path, in both replay modes.
   With no eye offset both eyes must be the mono frame bit for bit; with a
   32 mm half IPD world quads show crossed disparity that shrinks with
   depth, by the amount the projection predicts, and the HUD and 2D draws
   the disparity of the HUD plane. Then both eye images go through the
   compositor's eye planes into sample off-axis headset views, where the
   rays to each quad must meet at the depth it was drawn at. The eye
   images, the views and their anaglyphs are written as PNGs. */

static void put_be32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v >> 24), p[1] = (uint8_t)(v >> 16), p[2] = (uint8_t)(v >> 8), p[3] = (uint8_t)v;
}

static void png_chunk(FILE *f, const char *type, const uint8_t *data, uint32_t len)
{
    uint8_t h[8];
    put_be32(h, len);
    memcpy(h + 4, type, 4);
    fwrite(h, 1, 8, f);
    if (len) fwrite(data, 1, len, f);
    uLong crc = crc32(0, (const Bytef *)type, 4);
    if (len) crc = crc32(crc, data, len);
    put_be32(h, (uint32_t)crc);
    fwrite(h, 1, 4, f);
}

/* BGRA pixels (as the back buffer reads) to an RGB PNG. */
static int write_png(const char *path, const uint32_t *px, uint32_t w, uint32_t h)
{
    size_t raw_n = (size_t)h * (1 + 3 * w);
    uint8_t *raw = malloc(raw_n);
    if (!raw) return 0;
    for (uint32_t y = 0; y < h; y++) {
        uint8_t *r = raw + (size_t)y * (1 + 3 * w);
        r[0] = 0;
        for (uint32_t x = 0; x < w; x++) {
            uint32_t c = px[y * w + x];
            r[1 + 3 * x] = (uint8_t)(c >> 16), r[2 + 3 * x] = (uint8_t)(c >> 8), r[3 + 3 * x] = (uint8_t)c;
        }
    }
    uLongf zn = compressBound(raw_n);
    uint8_t *z = malloc(zn);
    int ok = z && compress2(z, &zn, raw, raw_n, 6) == Z_OK;
    FILE *f = ok ? fopen(path, "wb") : NULL;
    if (f) {
        static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
        uint8_t ihdr[13];
        put_be32(ihdr, w);
        put_be32(ihdr + 4, h);
        ihdr[8] = 8, ihdr[9] = 2, ihdr[10] = 0, ihdr[11] = 0, ihdr[12] = 0;
        fwrite(sig, 1, 8, f);
        png_chunk(f, "IHDR", ihdr, 13);
        png_chunk(f, "IDAT", z, (uint32_t)zn);
        png_chunk(f, "IEND", NULL, 0);
        ok = fclose(f) == 0;
    } else {
        ok = 0;
    }
    free(raw);
    free(z);
    return ok;
}

#define SW 256
#define SH 192
/* The game's camera for the scene: 75 degrees across a 4:3 view, as the
   game's default, near 4 and far 4096 (the porter's perspective test). */
#define S_A 1.30322540f
#define S_B (S_A * 4.0f / 3.0f)
#define S_NEAR 4.0f
#define S_FAR 4096.0f
#define S_Z_NEAR 50.0f
#define S_Z_FAR 400.0f
#define S_HALF_IPD_M 0.032f
#define S_UNITS_PER_M (1.0f / 0.0254f)

static char s_png_dir[1024];

static int write_png_size(const char *path, const uint32_t *px, uint32_t w, uint32_t h)
{
    int ok = write_png(path, px, w, h);
    if (!ok) printf("note: could not write %s\n", path);
    return ok;
}

static void save_png(const char *name, const uint32_t *px)
{
    char path[1200];
    snprintf(path, sizeof path, "%s/stereo-%s.png", s_png_dir, name);
    if (!write_png(path, px, SW, SH)) printf("note: could not write %s\n", path);
}

/* The view projection unlitgeneric reads at c8 to c11, one dp4 row each,
   with the camera at the origin looking down +z. */
static void vp_constants(void *dev, int perspective)
{
    float q = S_FAR / (S_FAR - S_NEAR), r = -S_NEAR * q;
    if (perspective) {
        vs_const(dev, 8, S_A, 0, 0, 0);
        vs_const(dev, 9, 0, S_B, 0, 0);
        vs_const(dev, 10, 0, 0, q, r);
        vs_const(dev, 11, 0, 0, 1, 0);
    } else {
        for (int i = 0; i < 4; i++) vs_const(dev, 8 + i, i == 0, i == 1, i == 2, i == 3);
    }
}

/* A quad in view space at depth z, its centre at NDC (nx, ny), half size
   hx by hy in NDC, so its image is the same size at any depth. */
static void view_quad(void *dev, void *vb, float nx, float ny, float hx, float hy, float z)
{
    float x0 = (nx - hx) * z / S_A, x1 = (nx + hx) * z / S_A, y0 = (ny - hy) * z / S_B, y1 = (ny + hy) * z / S_B;
    vtx q[4] = {{x0, y1, z, 0, 0}, {x1, y1, z, 1, 0}, {x1, y0, z, 1, 1}, {x0, y0, z, 0, 1}};
    void *p = NULL;
    VT(vb, SLOT_IDirect3DVertexBuffer9_Lock, HR (*)(void *, uint32_t, uint32_t, void **, uint32_t))(vb, 0, 0, &p, 0x2000);
    if (p) memcpy(p, q, sizeof q);
    VT(vb, SLOT_IDirect3DVertexBuffer9_Unlock, HR (*)(void *))(vb);
    VT(dev, SLOT_IDirect3DDevice9_DrawIndexedPrimitive,
       HR (*)(void *, uint32_t, int32_t, uint32_t, uint32_t, uint32_t, uint32_t))(dev, 4, 0, 0, 4, 0, 2);
}

/* Far red world quad on the right, near green one on the left, both
   through the perspective; a blue HUD quad through an identity view
   projection with the depth test off, as the engine's 2D is drawn. */
static void scene_stereo_game(void *dev, kit *k)
{
    set_tex0(dev, k->white);
    rs(dev, 22, 1);
    clear(dev, 0, NULL, 0x7, 0xFF000000u);
    vp_constants(dev, 1);
    vs_const(dev, 0, 0, 1, 2, 0.5f);
    for (int i = 0; i < 3; i++) vs_const(dev, 48 + i, i == 0, i == 1, i == 2, i == 3);
    vs_const(dev, 42, 1, 0, 0, 0);
    vs_const(dev, 43, 0, 1, 0, 0);
    vs_const(dev, 38, 1, 0, 0, 1);
    view_quad(dev, k->vb, 0.4f, 0.0f, 0.2f, 0.25f, S_Z_FAR);
    vs_const(dev, 38, 0, 1, 0, 1);
    view_quad(dev, k->vb, -0.35f, -0.2f, 0.2f, 0.25f, S_Z_NEAR);
    rs(dev, 7, 0);
    vp_constants(dev, 0);
    vs_const(dev, 38, 0, 0, 1, 1);
    quad(dev, k->vb, -0.9f, 0.6f, -0.55f, 0.9f, 0.1f);
    rs(dev, 7, 1);
    present(dev);
}

/* The fixed-function path: a yellow world quad through SetTransform's
   perspective, then an orthographic magenta quad and a POSITIONT cyan
   one, both HUD. */
static void scene_stereo_ff(void *dev)
{
    typedef struct {
        float x, y, z;
        uint32_t c;
    } cv;
    typedef struct {
        float x, y, z, rhw;
        uint32_t c;
    } tv;
    float id[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    float q = S_FAR / (S_FAR - S_NEAR);
    float pd[16] = {S_A, 0, 0, 0, 0, S_B, 0, 0, 0, 0, q, 1, 0, 0, -S_NEAR * q, 0};
    VT(dev, SLOT_IDirect3DDevice9_SetVertexShader, HR (*)(void *, void *))(dev, NULL);
    VT(dev, SLOT_IDirect3DDevice9_SetPixelShader, HR (*)(void *, void *))(dev, NULL);
    set_tex0(dev, NULL);
    rs(dev, 22, 1);
    rs(dev, 137, 0);
    tss(dev, 0, 1, 2); /* COLOROP SELECTARG1 */
    tss(dev, 0, 2, 0); /* COLORARG1 DIFFUSE */
    tss(dev, 0, 4, 2);
    tss(dev, 0, 5, 0);
    set_xf(dev, 256, id);
    set_xf(dev, 2, id);
    set_xf(dev, 3, pd);
    VT(dev, SLOT_IDirect3DDevice9_SetFVF, HR (*)(void *, uint32_t))(dev, 0x42);
    clear(dev, 0, NULL, 0x7, 0xFF000000u);
    float z = 80.0f, x0 = 0.1f * z / S_A, x1 = 0.5f * z / S_A, y0 = -0.3f * z / S_B, y1 = 0.3f * z / S_B;
    cv w[6] = {{x0, y1, z, 0xFFFFFF00u}, {x1, y1, z, 0xFFFFFF00u}, {x1, y0, z, 0xFFFFFF00u},
               {x0, y1, z, 0xFFFFFF00u}, {x1, y0, z, 0xFFFFFF00u}, {x0, y0, z, 0xFFFFFF00u}};
    draw_up(dev, 2, w, sizeof w[0]);
    set_xf(dev, 3, id);
    rs(dev, 7, 0);
    cv o[6] = {{-0.9f, 0.9f, 0.1f, 0xFFFF00FFu}, {-0.5f, 0.9f, 0.1f, 0xFFFF00FFu}, {-0.5f, 0.6f, 0.1f, 0xFFFF00FFu},
               {-0.9f, 0.9f, 0.1f, 0xFFFF00FFu}, {-0.5f, 0.6f, 0.1f, 0xFFFF00FFu}, {-0.9f, 0.6f, 0.1f, 0xFFFF00FFu}};
    draw_up(dev, 2, o, sizeof o[0]);
    VT(dev, SLOT_IDirect3DDevice9_SetFVF, HR (*)(void *, uint32_t))(dev, 0x44);
    tv t[6] = {{20, 150, 0.5f, 1, 0xFF00FFFFu}, {80, 150, 0.5f, 1, 0xFF00FFFFu}, {80, 180, 0.5f, 1, 0xFF00FFFFu},
               {20, 150, 0.5f, 1, 0xFF00FFFFu}, {80, 180, 0.5f, 1, 0xFF00FFFFu}, {20, 180, 0.5f, 1, 0xFF00FFFFu}};
    draw_up(dev, 2, t, sizeof t[0]);
    rs(dev, 7, 1);
    present(dev);
}

/* A discipline's tint as the game draws it: a POSITIONT quad over the
   whole back buffer (violet), after a small POSITIONT HUD quad (cyan). */
static void scene_overlay_ff(void *dev)
{
    typedef struct {
        float x, y, z, rhw;
        uint32_t c;
    } tv;
    VT(dev, SLOT_IDirect3DDevice9_SetVertexShader, HR (*)(void *, void *))(dev, NULL);
    VT(dev, SLOT_IDirect3DDevice9_SetPixelShader, HR (*)(void *, void *))(dev, NULL);
    set_tex0(dev, NULL);
    rs(dev, 22, 1);
    rs(dev, 137, 0);
    tss(dev, 0, 1, 2);
    tss(dev, 0, 2, 0);
    tss(dev, 0, 4, 2);
    tss(dev, 0, 5, 0);
    rs(dev, 7, 0);
    clear(dev, 0, NULL, 0x7, 0xFF000000u);
    VT(dev, SLOT_IDirect3DDevice9_SetFVF, HR (*)(void *, uint32_t))(dev, 0x44);
    tv t[6] = {{20, 150, 0.5f, 1, 0xFF00FFFFu}, {80, 150, 0.5f, 1, 0xFF00FFFFu}, {80, 180, 0.5f, 1, 0xFF00FFFFu},
               {20, 150, 0.5f, 1, 0xFF00FFFFu}, {80, 180, 0.5f, 1, 0xFF00FFFFu}, {20, 180, 0.5f, 1, 0xFF00FFFFu}};
    draw_up(dev, 2, t, sizeof t[0]);
    float w = (float)SW, h = (float)SH;
    tv f[6] = {{0, 0, 0.5f, 1, 0xFF8040C0u}, {w, 0, 0.5f, 1, 0xFF8040C0u}, {w, h, 0.5f, 1, 0xFF8040C0u},
               {0, 0, 0.5f, 1, 0xFF8040C0u}, {w, h, 0.5f, 1, 0xFF8040C0u}, {0, h, 0.5f, 1, 0xFF8040C0u}};
    draw_up(dev, 2, f, sizeof f[0]);
    rs(dev, 7, 1);
    present(dev);
}

/* The arm HUD's gauges as the game draws them, POSITIONT: green in the
   left strip (columns 0 to 38 of 256), red in the right one (225 on), a
   blue quad across the left strip's line and a small cyan one in the
   middle, which stay on the panel. */
static void scene_arm_ff(void *dev)
{
    typedef struct {
        float x, y, z, rhw;
        uint32_t c;
    } tv;
    VT(dev, SLOT_IDirect3DDevice9_SetVertexShader, HR (*)(void *, void *))(dev, NULL);
    VT(dev, SLOT_IDirect3DDevice9_SetPixelShader, HR (*)(void *, void *))(dev, NULL);
    set_tex0(dev, NULL);
    rs(dev, 22, 1);
    rs(dev, 137, 0);
    tss(dev, 0, 1, 2);
    tss(dev, 0, 2, 0);
    tss(dev, 0, 4, 2);
    tss(dev, 0, 5, 0);
    rs(dev, 7, 0);
    clear(dev, 0, NULL, 0x7, 0xFF000000u);
    VT(dev, SLOT_IDirect3DDevice9_SetFVF, HR (*)(void *, uint32_t))(dev, 0x44);
    static const struct {
        float x0, y0, x1, y1;
        uint32_t c;
    } q[4] = {{4, 60, 30, 120, 0xFF00FF00u}, {230, 60, 252, 120, 0xFFFF0000u}, {30, 20, 60, 40, 0xFF0000FFu},
              {100, 150, 150, 180, 0xFF00FFFFu}};
    for (int i = 0; i < 4; i++) {
        tv t[6] = {{q[i].x0, q[i].y0, 0.5f, 1, q[i].c}, {q[i].x1, q[i].y0, 0.5f, 1, q[i].c},
                   {q[i].x1, q[i].y1, 0.5f, 1, q[i].c}, {q[i].x0, q[i].y0, 0.5f, 1, q[i].c},
                   {q[i].x1, q[i].y1, 0.5f, 1, q[i].c}, {q[i].x0, q[i].y1, 0.5f, 1, q[i].c}};
        draw_up(dev, 2, t, sizeof t[0]);
    }
    rs(dev, 7, 1);
    present(dev);
}

/* The 3D sky as the source engine draws it in la_hub_1: a world draw into
   the back buffer from the sky's own camera far away (the yellow quad of
   scene_stereo_ff, seen from a view 500 units over), with no depth clear
   after it, then the world's own draws from the player's camera (two
   green quads on the other side), more of them than the sky's, and a red
   sprite above the centre drawn as the game draws its sprites: _22
   negative with the view's y turned over to match. With vm, a view model
   after them: a small quad drawn with its own, narrower field of view
   (twice the a). */
static void scene_fullview_sky(void *dev, int vm)
{
    typedef struct {
        float x, y, z;
        uint32_t c;
    } cv;
    float id[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    float q = S_FAR / (S_FAR - S_NEAR);
    float pd[16] = {S_A, 0, 0, 0, 0, S_B, 0, 0, 0, 0, q, 1, 0, 0, -S_NEAR * q, 0};
    VT(dev, SLOT_IDirect3DDevice9_SetVertexShader, HR (*)(void *, void *))(dev, NULL);
    VT(dev, SLOT_IDirect3DDevice9_SetPixelShader, HR (*)(void *, void *))(dev, NULL);
    set_tex0(dev, NULL);
    rs(dev, 22, 1);
    rs(dev, 137, 0);
    tss(dev, 0, 1, 2);
    tss(dev, 0, 2, 0);
    tss(dev, 0, 4, 2);
    tss(dev, 0, 5, 0);
    set_xf(dev, 256, id);
    set_xf(dev, 2, id);
    set_xf(dev, 3, pd);
    VT(dev, SLOT_IDirect3DDevice9_SetFVF, HR (*)(void *, uint32_t))(dev, 0x42);
    clear(dev, 0, NULL, 0x7, 0xFF000000u);
    float z = 80.0f, x0 = 0.1f * z / S_A, x1 = 0.5f * z / S_A, y0 = -0.3f * z / S_B, y1 = 0.3f * z / S_B;
    const float sx = 500.0f;
    float sky_view[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -sx, 0, 0, 1};
    set_xf(dev, 2, sky_view);
    cv w[6] = {{sx + x0, y1, z, 0xFFFFFF00u}, {sx + x1, y1, z, 0xFFFFFF00u}, {sx + x1, y0, z, 0xFFFFFF00u},
               {sx + x0, y1, z, 0xFFFFFF00u}, {sx + x1, y0, z, 0xFFFFFF00u}, {sx + x0, y0, z, 0xFFFFFF00u}};
    draw_up(dev, 2, w, sizeof w[0]);
    set_xf(dev, 2, id);
    for (int k = 0; k < 2; k++) {
        float g0 = k ? 0.05f * z / S_B : y0, g1 = k ? y1 : -0.05f * z / S_B;
        cv g[6] = {{-x1, g1, z, 0xFF00FF00u}, {-x0, g1, z, 0xFF00FF00u}, {-x0, g0, z, 0xFF00FF00u},
                   {-x1, g1, z, 0xFF00FF00u}, {-x0, g0, z, 0xFF00FF00u}, {-x1, g0, z, 0xFF00FF00u}};
        draw_up(dev, 2, g, sizeof g[0]);
    }
    {
        float flip_view[16] = {1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        float flip_proj[16] = {S_A, 0, 0, 0, 0, -S_B, 0, 0, 0, 0, q, 1, 0, 0, -S_NEAR * q, 0};
        set_xf(dev, 2, flip_view);
        set_xf(dev, 3, flip_proj);
        /* World up 0.1 to 0.3 of the height's tangent: view y below, and
           _22 turns it back up. */
        float r0 = -0.05f * z / S_A, r1 = 0.05f * z / S_A, u0 = 0.1f * z / S_B, u1 = 0.3f * z / S_B;
        cv r[6] = {{r0, u1, z, 0xFFFF0000u}, {r1, u1, z, 0xFFFF0000u}, {r1, u0, z, 0xFFFF0000u},
                   {r0, u1, z, 0xFFFF0000u}, {r1, u0, z, 0xFFFF0000u}, {r0, u0, z, 0xFFFF0000u}};
        draw_up(dev, 2, r, sizeof r[0]);
        set_xf(dev, 2, id);
        set_xf(dev, 3, pd);
    }
    if (vm) {
        float pv[16] = {2.0f * S_A, 0, 0, 0, 0, 2.0f * S_B, 0, 0, 0, 0, q, 1, 0, 0, -S_NEAR * q, 0};
        set_xf(dev, 3, pv);
        float vz = 20.0f, v0 = -0.1f * vz / (2.0f * S_A), v1 = 0.1f * vz / (2.0f * S_A);
        float u0 = -0.8f * vz / (2.0f * S_B), u1 = -0.6f * vz / (2.0f * S_B);
        cv v[6] = {{v0, u1, vz, 0xFF0000FFu}, {v1, u1, vz, 0xFF0000FFu}, {v1, u0, vz, 0xFF0000FFu},
                   {v0, u1, vz, 0xFF0000FFu}, {v1, u0, vz, 0xFF0000FFu}, {v0, u0, vz, 0xFF0000FFu}};
        draw_up(dev, 2, v, sizeof v[0]);
    }
    present(dev);
}

/* A frame for the plane's size, on the fixed-function path: with `a`
   positive a yellow world quad at z 80 through a perspective of that a
   (a dialogue close up's narrower camera at twice the default), else no
   world draw at all, a full-screen grey 2D quad as a menu or the
   character sheet draws its page; then a small magenta 2D quad on top. */
static void scene_plane_probe(void *dev, float a)
{
    typedef struct {
        float x, y, z;
        uint32_t c;
    } cv;
    float id[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    VT(dev, SLOT_IDirect3DDevice9_SetVertexShader, HR (*)(void *, void *))(dev, NULL);
    VT(dev, SLOT_IDirect3DDevice9_SetPixelShader, HR (*)(void *, void *))(dev, NULL);
    set_tex0(dev, NULL);
    rs(dev, 22, 1);
    rs(dev, 137, 0);
    tss(dev, 0, 1, 2);
    tss(dev, 0, 2, 0);
    tss(dev, 0, 4, 2);
    tss(dev, 0, 5, 0);
    set_xf(dev, 256, id);
    set_xf(dev, 2, id);
    VT(dev, SLOT_IDirect3DDevice9_SetFVF, HR (*)(void *, uint32_t))(dev, 0x42);
    clear(dev, 0, NULL, 0x7, 0xFF000000u);
    if (a > 0.0f) {
        float b = a * 4.0f / 3.0f, q = S_FAR / (S_FAR - S_NEAR);
        float pd[16] = {a, 0, 0, 0, 0, b, 0, 0, 0, 0, q, 1, 0, 0, -S_NEAR * q, 0};
        set_xf(dev, 3, pd);
        float z = 80.0f, x0 = 0.1f * z / a, x1 = 0.5f * z / a, y0 = -0.3f * z / b, y1 = 0.3f * z / b;
        cv w[6] = {{x0, y1, z, 0xFFFFFF00u}, {x1, y1, z, 0xFFFFFF00u}, {x1, y0, z, 0xFFFFFF00u},
                   {x0, y1, z, 0xFFFFFF00u}, {x1, y0, z, 0xFFFFFF00u}, {x0, y0, z, 0xFFFFFF00u}};
        draw_up(dev, 2, w, sizeof w[0]);
    }
    set_xf(dev, 3, id);
    rs(dev, 7, 0);
    if (!(a > 0.0f)) {
        cv g[6] = {{-1, 1, 0.1f, 0xFF404040u}, {1, 1, 0.1f, 0xFF404040u}, {1, -1, 0.1f, 0xFF404040u},
                   {-1, 1, 0.1f, 0xFF404040u}, {1, -1, 0.1f, 0xFF404040u}, {-1, -1, 0.1f, 0xFF404040u}};
        draw_up(dev, 2, g, sizeof g[0]);
    }
    cv o[6] = {{-0.6f, 0.5f, 0.1f, 0xFFFF00FFu}, {-0.2f, 0.5f, 0.1f, 0xFFFF00FFu}, {-0.2f, 0.1f, 0.1f, 0xFFFF00FFu},
               {-0.6f, 0.5f, 0.1f, 0xFFFF00FFu}, {-0.2f, 0.1f, 0.1f, 0xFFFF00FFu}, {-0.6f, 0.1f, 0.1f, 0xFFFF00FFu}};
    draw_up(dev, 2, o, sizeof o[0]);
    rs(dev, 7, 1);
    present(dev);
}

/* The extent in x of the pixels of one colour: the first and last column
   holding one, and how many columns between hold none. */
static void extent_x(const uint32_t *px, uint32_t w, uint32_t h, uint32_t rgb, int *lo, int *hi, int *gaps)
{
    *lo = -1, *hi = -1, *gaps = 0;
    for (uint32_t x = 0; x < w; x++) {
        int any = 0;
        for (uint32_t y = 0; y < h && !any; y++)
            any = near(px[y * w + x], (rgb >> 16) & 255, (rgb >> 8) & 255, rgb & 255, 8);
        if (any) {
            if (*lo < 0) *lo = (int)x;
            else if (*hi >= 0 && (int)x > *hi + 1) *gaps += (int)x - *hi - 1;
            *hi = (int)x;
        }
    }
}

/* The mean x of the pixels of one colour, and how many there are. */
static double centroid_x(const uint32_t *px, uint32_t rgb, uint32_t *n)
{
    double sx = 0;
    uint32_t c = 0;
    for (uint32_t y = 0; y < SH; y++)
        for (uint32_t x = 0; x < SW; x++)
            if (near(px[y * SW + x], (rgb >> 16) & 255, (rgb >> 8) & 255, rgb & 255, 8)) sx += x, c++;
    *n = c;
    return c ? sx / c : -1;
}

/* The compositor's half, on the CPU: each eye image put on its eye plane
   (immersive.h) and seen through a headset view's off-axis projection,
   as ScreenCompositor draws it. The views are the measured tangents
   (1.7320 outward, 1.0000 nasal, mirrored) with a sample vertical pair. */
#define VW 1600
#define VH 1200
static void view_projection(int right, float p[16])
{
    float l = right ? 1.0f : 1.732f, r = right ? 1.732f : 1.0f, d = 1.25f, u = 1.0f;
    memset(p, 0, 16 * sizeof(float));
    p[0] = 2.0f / (l + r);
    p[8] = (r - l) / (l + r);
    p[5] = 2.0f / (u + d);
    p[9] = (u - d) / (u + d);
    p[11] = -1.0f;
    p[14] = 0.1f;
}

/* Each view pixel back through the plane's mvp to the image: the plane is
   flat, so clip (x, y, w) is a 3 x 3 map of the quad's (u, v, 1), and its
   inverse takes a view pixel's NDC to the image point it shows. Nearest
   image pixel; outside the plane the view stays clear, as on device. */
static void eye_view(const uint32_t *img, int right, float half_ipd, float tx, float ty, uint32_t *out)
{
    float p[16], v[16], m[16];
    view_projection(right, p);
    mtl_screen_identity(v);
    v[12] = right ? half_ipd : -half_ipd;
    mtl_stereo_plane_mvp(p, v, tx, ty, METAL_D3D9_STEREO_HUD_M, m);
    /* Rows x, y, w of the mvp, against the quad's x, y and 1. */
    double h[3][3] = {{m[0], m[4], m[12]}, {m[1], m[5], m[13]}, {m[3], m[7], m[15]}}, inv[3][3];
    double det = h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1]) - h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0]) +
                 h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0]);
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) {
            int r1 = (c + 1) % 3, r2 = (c + 2) % 3, c1 = (r + 1) % 3, c2 = (r + 2) % 3;
            inv[r][c] = (h[r1][c1] * h[r2][c2] - h[r1][c2] * h[r2][c1]) / det;
        }
    for (uint32_t y = 0; y < VH; y++)
        for (uint32_t x = 0; x < VW; x++) {
            double nx = (x + 0.5) / VW * 2.0 - 1.0, ny = 1.0 - (y + 0.5) / VH * 2.0;
            double q[3];
            for (int r = 0; r < 3; r++) q[r] = inv[r][0] * nx + inv[r][1] * ny + inv[r][2];
            double u = q[0] / q[2] + 0.5, w = 0.5 - q[1] / q[2];
            int ix = (int)floor(u * SW), iy = (int)floor(w * SH);
            out[y * VW + x] = q[2] > 0 && ix >= 0 && ix < SW && iy >= 0 && iy < SH ? img[iy * SW + ix] : 0;
        }
}

/* The tangent of the direction a view pixel column is in. */
static double view_tangent(int right, double cx)
{
    float p[16];
    view_projection(right, p);
    return (((cx + 0.5) / VW) * 2.0 - 1.0 + p[8]) / p[0];
}

static double view_centroid_x(const uint32_t *px, uint32_t rgb, uint32_t *n)
{
    double sx = 0;
    uint32_t c = 0;
    for (uint32_t i = 0; i < VW * VH; i++)
        if (near(px[i], (rgb >> 16) & 255, (rgb >> 8) & 255, rgb & 255, 8)) sx += i % VW, c++;
    *n = c;
    return c ? sx / c : -1;
}

/* Where the two eyes' rays to one colour meet, in metres, from each
   view's centroid; and the tangent disparity it came from. */
static double meet_m(const uint32_t *lv, const uint32_t *rv, uint32_t rgb, float half_ipd, double *disp, uint32_t *n)
{
    uint32_t nl, nr;
    double cl = view_centroid_x(lv, rgb, &nl), cr = view_centroid_x(rv, rgb, &nr);
    *n = nl < nr ? nl : nr;
    *disp = nl && nr ? view_tangent(0, cl) - view_tangent(1, cr) : 0;
    return *disp > 0 ? 2.0 * half_ipd / *disp : INFINITY;
}

typedef struct {
    void (*stats)(uint64_t[8]);
    int (*acquire)(metal_d3d9_shown_frame *);
    void (*release)(int);
    int (*read_layer)(int, uint32_t, uint32_t *, uint32_t, uint32_t);
} ring_fns;

static uint64_t handed(const ring_fns *r);

/* The ring's newest frame once `want` frames in all have been handed
   over: its layer count and its eyes (right left 0 for a mono frame). */
static uint32_t take_frame(const ring_fns *r, uint64_t want, uint32_t *left, uint32_t *right, float plane[2])
{
    if (!wait_handed(r->stats, want)) {
        printf("note: stereo frame %llu never handed over (%llu so far)\n", (unsigned long long)want,
               (unsigned long long)handed(r));
        return 0;
    }
    metal_d3d9_shown_frame f;
    if (!r->acquire(&f)) return 0;
    uint32_t layers = f.layers;
    if (plane) plane[0] = f.tan_x, plane[1] = f.tan_y;
    int ok = f.width == SW && f.height == SH && r->read_layer(f.slot, 0, left, SW, SH);
    if (layers == 2) ok &= r->read_layer(f.slot, 1, right, SW, SH);
    else memset(right, 0, SW * SH * 4);
    r->release(f.slot);
    if (!ok) printf("note: stereo frame %llu (%ux%u, %u layers) could not be read\n", (unsigned long long)f.frame, f.width,
                    f.height, layers);
    return ok ? layers : 0;
}

static uint64_t handed(const ring_fns *r)
{
    uint64_t st[8];
    r->stats(st);
    return st[2] + st[3];
}

static void stereo_tests(void *d3d, const char *argv0)
{
    void (*enable)(int) = (void (*)(int))dlsym(s_lib, "metal_d3d9_immersive_enable");
    void (*stereo)(int) = (void (*)(int))dlsym(s_lib, "metal_d3d9_stereo_enable");
    void (*ipd)(float) = (void (*)(float))dlsym(s_lib, "metal_d3d9_stereo_set_half_ipd");
    ring_fns r = {
        (void (*)(uint64_t[8]))dlsym(s_lib, "metal_d3d9_immersive_stats"),
        (int (*)(metal_d3d9_shown_frame *))dlsym(s_lib, "metal_d3d9_immersive_acquire"),
        (void (*)(int))dlsym(s_lib, "metal_d3d9_immersive_release"),
        (int (*)(int, uint32_t, uint32_t *, uint32_t, uint32_t))dlsym(s_lib, "metal_d3d9_immersive_read_layer"),
    };
    CHECK(enable && stereo && ipd && r.stats && r.acquire && r.release && r.read_layer, "exports stereo and its test hooks");
    if (!enable || !stereo || !ipd || !r.stats || !r.acquire || !r.release || !r.read_layer) return;

    const char *dir = getenv("STEREO_PNG_DIR");
    if (dir && *dir) {
        snprintf(s_png_dir, sizeof s_png_dir, "%s", dir);
    } else {
        /* build/s2-porter/test/metal_d3d9_test writes into build/s2-porter. */
        snprintf(s_png_dir, sizeof s_png_dir, "%s", argv0);
        for (int up = 0; up < 2; up++) {
            char *sl = strrchr(s_png_dir, '/');
            if (sl) *sl = 0;
            else snprintf(s_png_dir, sizeof s_png_dir, up ? ".." : ".");
        }
    }

    /* Every game vertex shader's position, as the draw path classes it. */
    if (s_shader_dir) {
        int n[3] = {0, 0, 0}, total = 0;
        const char *models[2] = {"vs_1_1", "vs_2_0"};
        for (int m = 0; m < 2; m++) {
            char path[1024];
            snprintf(path, sizeof path, "%s/%s", s_shader_dir, models[m]);
            DIR *dp = opendir(path);
            struct dirent *de;
            while (dp && (de = readdir(dp))) {
                size_t l = strlen(de->d_name);
                if (l < 5 || strcmp(de->d_name + l - 5, ".dxso")) continue;
                char fp[1400];
                snprintf(fp, sizeof fp, "%s/%s", path, de->d_name);
                FILE *f = fopen(fp, "rb");
                if (!f) continue;
                static uint32_t tk[1 << 18];
                size_t got = fread(tk, 4, sizeof tk / 4, f);
                fclose(f);
                int p = mtl_stereo_scan_pos(tk, got);
                n[p == MTL_POS_VIEWPROJ ? 0 : p == MTL_POS_MVP ? 1 : 2]++;
                total++;
            }
            if (dp) closedir(dp);
        }
        CHECK(total > 3000 && n[2] * 100 < total,
              "stereo scan of %d game vertex shaders: %d view projection c8-c11, %d model view projection c4-c7, %d "
              "neither (left alone)",
              total, n[0], n[1], n[2]);
    }

    setenv("RT_IMMERSIVE_MAX_HZ", "0", 1);
    enable(1);
    uint32_t *mono = calloc(SW * SH, 4), *l0 = calloc(SW * SH, 4), *r0 = calloc(SW * SH, 4), *l = calloc(SW * SH, 4),
             *rr = calloc(SW * SH, 4), *bb = NULL, *scratch = calloc(SW * SH, 4);
    for (int threaded = 1; threaded >= 0; threaded--) {
        const char *mode = threaded ? "threaded" : "inline";
        void *dev = mode_device(d3d, threaded, SW, SH);
        if (!dev) break;
        for (int pass = 0; pass < (s_shader_dir ? 2 : 1); pass++) {
            int ff = !s_shader_dir || pass == 1;
            const char *what = ff ? "fixed-function" : "game shaders";
            kit k;
            if (!ff && !kit_make(dev, &k)) {
                CHECK(0, "stereo: unlitgeneric kit for the game shader scene");
                continue;
            }
#define SCENE() (ff ? scene_stereo_ff(dev) : scene_stereo_game(dev, &k))
            /* A switch takes effect at a frame's start; inline mode
               decides at the previous frame's end, so a frame passes
               after each one. */
            stereo(0);
            ipd(0.0f);
            uint64_t want = handed(&r) + 2;
            clear(dev, 0, NULL, 0x1, 0xFF000000u);
            present(dev);
            SCENE();
            uint32_t lm = take_frame(&r, want, mono, scratch, NULL);
            stereo(1);
            want = handed(&r) + 2;
            clear(dev, 0, NULL, 0x1, 0xFF000000u);
            present(dev);
            SCENE();
            uint32_t li = take_frame(&r, want, l0, r0, NULL);
            uint32_t w = 0, h = 0;
            bb = readback(dev, &w, &h);
            int same_lr = !memcmp(l0, r0, SW * SH * 4), same_lm = !memcmp(l0, mono, SW * SH * 4),
                same_bb = bb && w == SW && h == SH && !memcmp(bb, mono, SW * SH * 4);
            uint32_t lit = 0;
            for (uint32_t i = 0; i < SW * SH; i++) lit += (mono[i] & 0xFFFFFF) != 0;
            CHECK(lm == 1 && li == 2 && same_lr && same_lm && same_bb && lit > 1000,
                  "eye identity (%s, %s): E_eye identity and P_eye = P_game, both eye replays bit for bit the mono "
                  "frame (layers %u then %u; left = right %d, left = mono %d, back buffer = mono %d; %u lit pixels)",
                  what, mode, lm, li, same_lr, same_lm, same_bb, lit);
            free(bb);
            bb = NULL;

            ipd(S_HALF_IPD_M);
            want = handed(&r) + 2;
            clear(dev, 0, NULL, 0x1, 0xFF000000u);
            present(dev);
            SCENE();
            float plane0[2] = {0, 0};
            uint32_t ls = take_frame(&r, want, l, rr, plane0);
            double h_units = S_HALF_IPD_M * S_UNITS_PER_M;
            /* The HUD's shift is the plane's a times e over the HUD depth,
               the same for each eye and opposite: a point that far away.
               Its quads have edges on whole pixels, so each eye's image
               of a 3.05 px shift lands on 3 or 4 px: to a pixel, not 0.75. */
            double want_hud = S_A * h_units / (METAL_D3D9_STEREO_HUD_M * S_UNITS_PER_M) * SW;
            if (!ff) {
                uint32_t nl, nr, fl, fr;
                double dn = centroid_x(l, 0x00FF00, &nl) - centroid_x(rr, 0x00FF00, &nr);
                double df = centroid_x(l, 0xFF0000, &fl) - centroid_x(rr, 0xFF0000, &fr);
                double want_n = S_A * h_units / S_Z_NEAR * SW, want_f = S_A * h_units / S_Z_FAR * SW;
                uint32_t hl_n, hr_n;
                double dh = centroid_x(l, 0x0000FF, &hl_n) - centroid_x(rr, 0x0000FF, &hr_n);
                uint32_t hud_px = hl_n < hr_n ? hl_n : hr_n;
                CHECK(ls == 2 && nl && nr && fl && fr && fabs(dn - want_n) < 0.75 && fabs(df - want_f) < 0.75 && dn > df &&
                          df > 0,
                      "stereo 32 mm (%s, %s): near quad z %.0f disparity %+.2f px (want %+.2f), far z %.0f %+.2f px "
                      "(want %+.2f): crossed, shrinking with depth",
                      what, mode, (double)S_Z_NEAR, dn, want_n, (double)S_Z_FAR, df, want_f);
                CHECK(ls == 2 && hud_px > 100 && fabs(dh - want_hud) < 1.0,
                      "stereo 32 mm (%s, %s): HUD quad (%u px) at the %.2f m HUD plane's disparity %+.2f px (want "
                      "%+.2f), between the near and far quads'",
                      what, mode, hud_px, (double)METAL_D3D9_STEREO_HUD_M, dh, want_hud);
            } else {
                uint32_t nl, nr;
                double dw = centroid_x(l, 0xFFFF00, &nl) - centroid_x(rr, 0xFFFF00, &nr);
                double want_w = S_A * h_units / 80.0 * SW;
                uint32_t m_px, c_px, m2, c2;
                double dm = centroid_x(l, 0xFF00FF, &m_px) - centroid_x(rr, 0xFF00FF, &m2);
                double dc = centroid_x(l, 0x00FFFF, &c_px) - centroid_x(rr, 0x00FFFF, &c2);
                CHECK(ls == 2 && nl && nr && fabs(dw - want_w) < 0.75 && m_px > 100 && c_px > 100 && m2 && c2 &&
                          fabs(dm - want_hud) < 1.0 && fabs(dc - want_hud) < 1.0,
                      "stereo 32 mm (%s, %s): SetTransform perspective quad z 80 disparity %+.2f px (want %+.2f); "
                      "orthographic (%u px) %+.2f px and POSITIONT (%u px) %+.2f px, the HUD plane's (want %+.2f)",
                      what, mode, dw, want_w, m_px, dm, c_px, dc, want_hud);
            }
            /* Through the headset's views: the directions each eye sees,
               and where its rays meet, against the depths the scene was
               drawn at. */
            uint32_t *lv = calloc(VW * VH, 4), *rv = calloc(VW * VH, 4);
            float tx = 1.0f / S_A, ty = (float)SH / ((float)SW * S_A);
            if (lv && rv && ls == 2) {
                eye_view(l, 0, S_HALF_IPD_M, tx, ty, lv);
                eye_view(rr, 1, S_HALF_IPD_M, tx, ty, rv);
                /* What the views must show: each colour's image disparity in
                   pixels, measured above, as a direction disparity of 2 tx
                   per image width, whatever the views' off-axis frusta;
                   to one view pixel per eye. Where the rays meet then
                   follows, and is printed against the drawn depth, which
                   the image's whole pixels only reach to a pixel. */
                double tol = 2.0 * 2.732 / VW, per_px = 2.0 * tx / SW;
                const uint32_t cols[3] = {ff ? 0xFFFF00u : 0x00FF00u, ff ? 0xFF00FFu : 0xFF0000u, ff ? 0x00FFFFu : 0x0000FFu};
                const char *names[3] = {ff ? "perspective z 80" : "near z 50", ff ? "orthographic HUD" : "far z 400",
                                        ff ? "POSITIONT HUD" : "HUD"};
                double drawn[3] = {(ff ? 80.0 : S_Z_NEAR) / S_UNITS_PER_M, ff ? METAL_D3D9_STEREO_HUD_M : S_Z_FAR / S_UNITS_PER_M,
                                   METAL_D3D9_STEREO_HUD_M};
                int good = 1;
                char line[1024];
                size_t at = 0;
                for (int k2 = 0; k2 < 3; k2++) {
                    uint32_t a1, a2, n;
                    double di = centroid_x(l, cols[k2], &a1) - centroid_x(rr, cols[k2], &a2), dv;
                    double met = meet_m(lv, rv, cols[k2], S_HALF_IPD_M, &dv, &n);
                    good &= n > 200 && a1 && a2 && fabs(dv - di * per_px) < tol && dv > 0;
                    at += (size_t)snprintf(line + at, sizeof line - at, "%s%s %.4f rad (image %.2f px = %.4f), meet %.2f m "
                                           "(drawn %.2f)", k2 ? "; " : "", names[k2], dv, di, di * per_px, met, drawn[k2]);
                }
                CHECK(good, "eye views (%s, %s) through off-axis 1.732/1.0 projections: %s", what, mode, line);
            }
            if (threaded && lv && rv && ls == 2) {
                const char *tag = ff ? "ff" : "game";
                char path[1200];
                snprintf(path, sizeof path, "%s/stereo-%s-view-left.png", s_png_dir, tag);
                write_png_size(path, lv, VW, VH);
                snprintf(path, sizeof path, "%s/stereo-%s-view-right.png", s_png_dir, tag);
                write_png_size(path, rv, VW, VH);
                for (uint32_t i = 0; i < VW * VH; i++) {
                    uint32_t a2 = lv[i], b2 = rv[i];
                    uint32_t ll = (((a2 >> 16) & 255) + ((a2 >> 8) & 255) + (a2 & 255)) / 3;
                    uint32_t lr = (((b2 >> 16) & 255) + ((b2 >> 8) & 255) + (b2 & 255)) / 3;
                    lv[i] = ll << 16 | lr << 8 | lr;
                }
                snprintf(path, sizeof path, "%s/stereo-%s-view-anaglyph.png", s_png_dir, tag);
                write_png_size(path, lv, VW, VH);
            }
            free(lv);
            free(rv);
            if (threaded) {
                const char *tag = ff ? "ff" : "game";
                char name[64];
                snprintf(name, sizeof name, "%s-mono", tag);
                save_png(name, mono);
                snprintf(name, sizeof name, "%s-identity-left", tag);
                save_png(name, l0);
                snprintf(name, sizeof name, "%s-identity-right", tag);
                save_png(name, r0);
                snprintf(name, sizeof name, "%s-left", tag);
                save_png(name, l);
                snprintf(name, sizeof name, "%s-right", tag);
                save_png(name, rr);
                /* Red from the left eye, green and blue from the right. */
                for (uint32_t i = 0; i < SW * SH; i++) {
                    uint32_t lv = l[i], rv = rr[i];
                    uint32_t lum_l = (((lv >> 16) & 255) + ((lv >> 8) & 255) + (lv & 255)) / 3;
                    uint32_t lum_r = (((rv >> 16) & 255) + ((rv >> 8) & 255) + (rv & 255)) / 3;
                    scratch[i] = lum_l << 16 | lum_r << 8 | lum_r;
                }
                snprintf(name, sizeof name, "%s-anaglyph", tag);
                save_png(name, scratch);
            }
            /* One plane for the session: a narrower camera's world frame
               and a frame with no world draw at all both show on the plane
               a normal world frame does, at the game's nominal 75 degrees
               across 4:3. The narrow world is then a zoom, its disparity
               its own a's; the 2D frame covers the whole plane in each
               eye bar the HUD shift's strip, its HUD quad once, where the
               two eyes' rays meet on the plane: one image, not two. */
            if (ff && ls == 2) {
                float want_tx = 1.0f / S_A, want_ty = (float)SH / ((float)SW * S_A);
                uint32_t *nl_img = calloc(SW * SH, 4), *nr_img = calloc(SW * SH, 4);
                uint32_t *fl_img = calloc(SW * SH, 4), *fr_img = calloc(SW * SH, 4);
                float pn[2] = {0, 0}, pf[2] = {0, 0};
                want = handed(&r) + 2;
                clear(dev, 0, NULL, 0x1, 0xFF000000u);
                present(dev);
                scene_plane_probe(dev, 2.0f * S_A);
                uint32_t ln = take_frame(&r, want, nl_img, nr_img, pn);
                want = handed(&r) + 2;
                clear(dev, 0, NULL, 0x1, 0xFF000000u);
                present(dev);
                scene_plane_probe(dev, 0.0f);
                uint32_t lf = take_frame(&r, want, fl_img, fr_img, pf);
                CHECK(ln == 2 && lf == 2 && fabsf(plane0[0] - want_tx) < 1e-5f && fabsf(plane0[1] - want_ty) < 1e-5f &&
                          pn[0] == plane0[0] && pn[1] == plane0[1] && pf[0] == plane0[0] && pf[1] == plane0[1],
                      "one plane (%s): world frame tan %.5f by %.5f (want %.5f by %.5f), a %.2f close up %.5f by %.5f, "
                      "2D only %.5f by %.5f",
                      mode, (double)plane0[0], (double)plane0[1], (double)want_tx, (double)want_ty, 2.0 * S_A,
                      (double)pn[0], (double)pn[1], (double)pf[0], (double)pf[1]);
                uint32_t a1, a2;
                double dz = centroid_x(nl_img, 0xFFFF00, &a1) - centroid_x(nr_img, 0xFFFF00, &a2);
                double want_z = 2.0 * S_A * h_units / 80.0 * SW;
                uint32_t m1, m2;
                double dm = centroid_x(nl_img, 0xFF00FF, &m1) - centroid_x(nr_img, 0xFF00FF, &m2);
                CHECK(ln == 2 && a1 && a2 && fabs(dz - want_z) < 0.75 && m1 > 100 && fabs(dm - want_hud) < 1.0,
                      "close up (%s): world quad z 80 at a %.2f disparity %+.2f px (want %+.2f, twice the default's: "
                      "a zoom), its HUD still the plane's %+.2f px (want %+.2f)",
                      mode, 2.0 * S_A, dz, want_z, dm, want_hud);
                int shift = (int)ceil(want_hud / 2.0) + 1;
                int glo[2], ghi[2], ggap[2], mlo[2], mhi[2], mgap[2];
                uint32_t gn[2], mn[2];
                const uint32_t *eye[2] = {fl_img, fr_img};
                int cover = 1;
                for (int k2 = 0; k2 < 2; k2++) {
                    extent_x(eye[k2], SW, SH, 0x404040, &glo[k2], &ghi[k2], &ggap[k2]);
                    extent_x(eye[k2], SW, SH, 0xFF00FF, &mlo[k2], &mhi[k2], &mgap[k2]);
                    centroid_x(eye[k2], 0x404040, &gn[k2]);
                    centroid_x(eye[k2], 0xFF00FF, &mn[k2]);
                    cover &= gn[k2] + mn[k2] >= (uint32_t)((SW - shift) * SH) && !ggap[k2] && !mgap[k2];
                }
                double d2 = centroid_x(fl_img, 0xFF00FF, &m1) - centroid_x(fr_img, 0xFF00FF, &m2);
                uint32_t *lv2 = calloc(VW * VH, 4), *rv2 = calloc(VW * VH, 4);
                double met = 0, dv = 0;
                uint32_t vn = 0;
                if (lv2 && rv2) {
                    eye_view(fl_img, 0, S_HALF_IPD_M, pf[0], pf[1], lv2);
                    eye_view(fr_img, 1, S_HALF_IPD_M, pf[0], pf[1], rv2);
                    met = meet_m(lv2, rv2, 0xFF00FF, S_HALF_IPD_M, &dv, &vn);
                }
                CHECK(lf == 2 && cover && mn[0] == mn[1] && mn[0] > 100 && glo[0] <= shift && ghi[1] >= SW - 1 - shift &&
                          fabs(d2 - want_hud) < 1.0 && vn > 200 && fabs(met - METAL_D3D9_STEREO_HUD_M) < 0.1,
                      "2D only (%s): the page covers each eye's image (left %d..%d, right %d..%d of %d, %u and %u "
                      "px, no gaps), one HUD quad per eye (%u and %u px) at the plane's disparity %+.2f px (want "
                      "%+.2f), rays meeting at %.2f m (the plane %.2f m): fused, not doubled",
                      mode, glo[0], ghi[0], glo[1], ghi[1], SW, gn[0], gn[1], mn[0], mn[1], d2, want_hud, met,
                      (double)METAL_D3D9_STEREO_HUD_M);
                free(lv2), free(rv2), free(nl_img), free(nr_img), free(fl_img), free(fr_img);
            }
#undef SCENE
            stereo(0);
            ipd(S_HALF_IPD_M);
            if (!ff) kit_free(dev, &k);
        }
        VT(dev, SLOT_IDirect3DDevice9_Release, uint32_t (*)(void *))(dev);
    }
    printf("stereo eye images in %s/stereo-*.png\n", s_png_dir);
    free(mono), free(l0), free(r0), free(l), free(rr), free(scratch);
    stereo(0);
    enable(0);
}

/* Full view (immersive.h, stereo.h's mtl_stereo_correction3): a frame in
   play replayed per eye into each eye's own off-axis view, on the
   fixed-function scene, in both replay modes. With the check (each eye
   keeps the game's projection at the camera) both eyes are the mono
   frame bit for bit and the slot says the game's own field. With the
   sample views each world quad and HUD quad lands on the pixel the eye's
   projection predicts: the world from the eye's own position, the HUD on
   the head-locked panel. A frame out of play is stage A's, bit for bit,
   and goes on the plane. */
static uint32_t take_shown(const ring_fns *r, uint64_t want, uint32_t *left, uint32_t *right,
                           metal_d3d9_shown_frame *out)
{
    memset(out, 0, sizeof *out);
    if (!wait_handed(r->stats, want)) {
        printf("note: full view frame %llu never handed over\n", (unsigned long long)want);
        return 0;
    }
    if (!r->acquire(out)) return 0;
    uint32_t layers = out->layers;
    int ok = out->width == SW && out->height == SH && r->read_layer(out->slot, 0, left, SW, SH);
    if (layers == 2) ok &= r->read_layer(out->slot, 1, right, SW, SH);
    else memset(right, 0, SW * SH * 4);
    r->release(out->slot);
    return ok ? layers : 0;
}

/* The mean pixel position of one colour, x from the left and y from the
   top, and how many there are. */
static void centroid_xy(const uint32_t *px, uint32_t rgb, double *cx, double *cy, uint32_t *n)
{
    double sx = 0, sy = 0;
    uint32_t c = 0;
    for (uint32_t y = 0; y < SH; y++)
        for (uint32_t x = 0; x < SW; x++)
            if (near(px[y * SW + x], (rgb >> 16) & 255, (rgb >> 8) & 255, rgb & 255, 8)) sx += x, sy += y, c++;
    *n = c;
    *cx = c ? sx / c : -1;
    *cy = c ? sy / c : -1;
}

/* The pixel centroid an NDC point is the centre of. */
static double px_x(double ndc) { return (ndc + 1.0) * 0.5 * SW - 0.5; }
static double px_y(double ndc) { return (1.0 - ndc) * 0.5 * SH - 0.5; }

static void fullview_tests(void *d3d)
{
    void (*enable)(int) = (void (*)(int))dlsym(s_lib, "metal_d3d9_immersive_enable");
    void (*stereo)(int) = (void (*)(int))dlsym(s_lib, "metal_d3d9_stereo_enable");
    void (*ipd)(float) = (void (*)(float))dlsym(s_lib, "metal_d3d9_stereo_set_half_ipd");
    void (*fv)(int) = (void (*)(int))dlsym(s_lib, "metal_d3d9_fullview_enable");
    void (*views)(const float *, const float *) =
        (void (*)(const float *, const float *))dlsym(s_lib, "metal_d3d9_fullview_set_views");
    void (*play)(int) = (void (*)(int))dlsym(s_lib, "metal_d3d9_immersive_view_play");
    void (*game_eye)(int) = (void (*)(int))dlsym(s_lib, "metal_d3d9_fullview_game_eye");
    int (*game_tan)(float *, float *) = (int (*)(float *, float *))dlsym(s_lib, "metal_d3d9_fullview_game_tan");
    void (*arm_hud)(int, int) = (void (*)(int, int))dlsym(s_lib, "metal_d3d9_fullview_arm_hud");
    int (*read_arm)(int, uint32_t *, uint32_t, uint32_t) =
        (int (*)(int, uint32_t *, uint32_t, uint32_t))dlsym(s_lib, "metal_d3d9_fullview_read_arm");
    CHECK(arm_hud && read_arm, "exports the arm HUD: fullview_arm_hud, fullview_read_arm");
    ring_fns r = {
        (void (*)(uint64_t[8]))dlsym(s_lib, "metal_d3d9_immersive_stats"),
        (int (*)(metal_d3d9_shown_frame *))dlsym(s_lib, "metal_d3d9_immersive_acquire"),
        (void (*)(int))dlsym(s_lib, "metal_d3d9_immersive_release"),
        (int (*)(int, uint32_t, uint32_t *, uint32_t, uint32_t))dlsym(s_lib, "metal_d3d9_immersive_read_layer"),
    };
    int have = enable && stereo && ipd && fv && views && play && game_eye && game_tan && r.stats && r.acquire &&
               r.release && r.read_layer;
    CHECK(have, "exports full view: enable, set_views, view_play, game_eye, game_tan");
    if (!have) return;

    /* Sample views small enough that the scene's quads land inside both:
       wider outward than towards the nose, mirrored, eyes 64 mm apart. */
    const float tan[8] = {1.0f, 0.6f, 0.7f, 0.5f, 0.6f, 1.0f, 0.7f, 0.5f};
    const float eye_m[6] = {-S_HALF_IPD_M, 0, 0, S_HALF_IPD_M, 0, 0};
    const float m = METAL_D3D9_FULLVIEW_MARGIN, aspect = (float)SW / (float)SH;
    setenv("RT_IMMERSIVE_MAX_HZ", "0", 1);
    enable(1);
    views(tan, eye_m);
    fv(1);
    ipd(S_HALF_IPD_M);
    uint32_t *mono = calloc(SW * SH, 4), *l = calloc(SW * SH, 4), *rr = calloc(SW * SH, 4), *al = calloc(SW * SH, 4),
             *ar = calloc(SW * SH, 4);
    for (int threaded = 1; threaded >= 0; threaded--) {
        const char *mode = threaded ? "threaded" : "inline";
        /* The shot (threaded only): the second full view frame, which is
           the check's, so both eyes and the mono reference are the same. */
        char shot_dir[1200];
        snprintf(shot_dir, sizeof shot_dir, "%s/fullview-shot", s_png_dir);
        if (threaded) {
            mkdir(shot_dir, 0755);
            for (int i = 0; i < 3; i++) {
                static const char *k[3] = {"left", "right", "mono"};
                char p[1300];
                snprintf(p, sizeof p, "%s/fv0-f1-%s.ppm", shot_dir, k[i]);
                unlink(p);
            }
            setenv("RT_FULLVIEW_SHOT_DIR", shot_dir, 1);
            setenv("RT_FULLVIEW_SHOT", "1,1,1", 1);
        }
        void *dev = mode_device(d3d, threaded, SW, SH);
        unsetenv("RT_FULLVIEW_SHOT");
        if (!dev) break;
        /* A switch takes effect at a frame's start (inline: at the frame
           before's end), so a cleared frame goes first each time. */
#define FRAME(shown, left, right)                                                                                      \
    do {                                                                                                               \
        uint64_t want_ = handed(&r) + 2;                                                                               \
        clear(dev, 0, NULL, 0x1, 0xFF000000u);                                                                         \
        present(dev);                                                                                                  \
        scene_stereo_ff(dev);                                                                                          \
        layers = take_shown(&r, want_, left, right, shown);                                                            \
    } while (0)
        metal_d3d9_shown_frame sm, sg, so, sa, sp;
        uint32_t layers, lm, lg, lo, la, lp;
        stereo(0);
        play(1);
        FRAME(&sm, mono, l);
        lm = layers;

        /* The check: each eye the game's projection, at the camera. */
        stereo(1);
        game_eye(1);
        FRAME(&sg, l, rr);
        lg = layers;
        float gtx = 0, gty = 0;
        int wanted = game_tan(&gtx, &gty);
        float lx = 1.0f * (1 + m), ly = 0.7f * (1 + m), want_tx = fmaxf(lx, ly * aspect), want_ty = want_tx / aspect;
        int same = !memcmp(l, mono, SW * SH * 4) && !memcmp(rr, mono, SW * SH * 4);
        if (threaded) {
            /* The shot's three images, written when the GPU finished, and
               its description with the check's world quad in it. */
            char p[3][1300], tp[1300];
            size_t n[3] = {0};
            uint8_t *img[3] = {0};
            static const char *k[3] = {"left", "right", "mono"};
            for (int i = 0; i < 3; i++) snprintf(p[i], sizeof p[i], "%s/fv0-f1-%s.ppm", shot_dir, k[i]);
            snprintf(tp, sizeof tp, "%s/fv0-f1.txt", shot_dir);
            for (int wait = 0; wait < 100 && access(p[2], R_OK); wait++) usleep(10000);
            usleep(20000);
            for (int i = 0; i < 3; i++) {
                FILE *f = fopen(p[i], "rb");
                if (!f) continue;
                img[i] = malloc(SW * SH * 3 + 64);
                n[i] = img[i] ? fread(img[i], 1, SW * SH * 3 + 64, f) : 0;
                fclose(f);
            }
            int world = 0, lines = 0;
            FILE *t = fopen(tp, "r");
            char line[2048];
            while (t && fgets(line, sizeof line, t)) {
                lines++;
                if (strstr(line, " draw ") && strstr(line, " world ")) world++;
            }
            if (t) fclose(t);
            int shot_same = n[0] > SW * SH * 3 && n[0] == n[1] && n[0] == n[2] && !memcmp(img[0], img[1], n[0]) &&
                            !memcmp(img[0], img[2], n[0]);
            CHECK(shot_same && world > 0,
                  "full view shot (%s): left, right and mono written (%zu bytes each) and the same in the check, the "
                  "description has %d lines, %d of them world draws",
                  mode, n[0], lines, world);
            for (int i = 0; i < 3; i++) free(img[i]);
        }
        CHECK(lm == 1 && lg == 2 && same && sg.fullview == 1 && sg.eye_tan[0][0] == gtx && sg.eye_tan[1][3] == gty &&
                  wanted && fabsf(gtx - want_tx) < 1e-6f && fabsf(gty - want_ty) < 1e-6f,
              "full view identity (%s): each eye the game's projection at the camera, both eye replays bit for bit "
              "the mono frame (same %d); the slot says full view at the game's field %.4f by %.4f (want %.4f by "
              "%.4f, the views' union with the margin at %.3f)",
              mode, same, (double)gtx, (double)gty, (double)want_tx, (double)want_ty, (double)aspect);

        /* The sample views: every quad where each eye's own projection
           puts it. */
        game_eye(0);
        FRAME(&so, l, rr);
        lo = layers;
        save_png(threaded ? "fullview-threaded-left" : "fullview-inline-left", l);
        save_png(threaded ? "fullview-threaded-right" : "fullview-inline-right", rr);
        const float upm = S_UNITS_PER_M, hud_z = METAL_D3D9_STEREO_HUD_M * S_UNITS_PER_M;
        const float px = tanf(METAL_D3D9_FULLVIEW_HUD_DEG * 0.5f * (float)M_PI / 180.0f), py = px / aspect;
        double worst_w = 0, worst_h = 0;
        uint32_t least = UINT32_MAX;
        int tan_ok = so.fullview == 1;
        for (int e = 0; e < 2; e++) {
            const uint32_t *img = e ? rr : l;
            float te[4];
            for (int k = 0; k < 4; k++) {
                te[k] = tan[4 * e + k] * (1 + m);
                tan_ok &= fabsf(so.eye_tan[e][k] - te[k]) < 1e-6f;
            }
            float ea = 2.0f / (te[0] + te[1]), ecx = (te[0] - te[1]) / (te[0] + te[1]);
            float eb = 2.0f / (te[2] + te[3]), ecy = (te[2] - te[3]) / (te[2] + te[3]);
            float tx = (e ? S_HALF_IPD_M : -S_HALF_IPD_M) * upm;
            /* The yellow world quad's centre, view (0.3 z / S_A, 0, 80). */
            double z = 80.0, xv = 0.3 * z / S_A;
            double wx = px_x(ea * (xv - tx) / z + ecx), wy = px_y(ecy);
            /* The magenta 2D quad, game NDC centre (-0.7, 0.75), and the
               cyan POSITIONT one, pixels 20..80 by 150..180. */
            double mx = px_x(ea * px * -0.7 + ecx - ea * tx / hud_z), my = px_y(eb * py * 0.75 + ecy);
            double cnx = 50.0 / (SW * 0.5) - 1.0, cny = 1.0 - 165.0 / (SH * 0.5);
            double cx = px_x(ea * px * cnx + ecx - ea * tx / hud_z), cy = px_y(eb * py * cny + ecy);
            double gx, gy;
            uint32_t n;
            centroid_xy(img, 0xFFFF00, &gx, &gy, &n);
            worst_w = fmax(worst_w, fmax(fabs(gx - wx), fabs(gy - wy)));
            least = n < least ? n : least;
            centroid_xy(img, 0xFF00FF, &gx, &gy, &n);
            worst_h = fmax(worst_h, fmax(fabs(gx - mx), fabs(gy - my)));
            least = n < least ? n : least;
            centroid_xy(img, 0x00FFFF, &gx, &gy, &n);
            worst_h = fmax(worst_h, fmax(fabs(gx - cx), fabs(gy - cy)));
            least = n < least ? n : least;
        }
        CHECK(lo == 2 && tan_ok && least > 50 && worst_w < 0.75 && worst_h < 1.0,
              "full view off-axis (%s): in each eye the world quad from that eye's own position, and both HUD "
              "quads on the %.0f degree panel, land where the eye's projection puts them (worst %.2f px world, "
              "%.2f px HUD, at least %u px each); the slot carries each eye's tangents with the margin (%d)",
              mode, (double)METAL_D3D9_FULLVIEW_HUD_DEG, worst_w, worst_h, least, tan_ok);

        /* A full view overlay: the tint over the whole back buffer fills
           each eye, where the panel would have shrunk it to the HUD's
           field; the small HUD quad stays on the panel under it. */
        {
            uint64_t want_ = handed(&r) + 2;
            clear(dev, 0, NULL, 0x1, 0xFF000000u);
            present(dev);
            scene_overlay_ff(dev);
            metal_d3d9_shown_frame sv;
            uint32_t lv = take_shown(&r, want_, l, rr, &sv);
            uint32_t fill[2] = {0, 0};
            for (int e = 0; e < 2; e++) {
                const uint32_t *img = e ? rr : l;
                for (uint32_t i = 0; i < SW * SH; i++)
                    if (near(img[i], 0x80, 0x40, 0xC0, 8)) fill[e]++;
            }
            save_png(threaded ? "fullview-overlay-threaded-left" : "fullview-overlay-inline-left", l);
            uint32_t all = SW * SH;
            CHECK(lv == 2 && sv.fullview == 1 && fill[0] == all && fill[1] == all,
                  "full view overlay (%s): a POSITIONT quad over the whole back buffer fills each eye (%u and %u of "
                  "%u px), not the %.0f degree panel",
                  mode, fill[0], fill[1], all, (double)METAL_D3D9_FULLVIEW_HUD_DEG);
        }

        /* The arm HUD: with both sides on, the gauges' quads leave both
           eyes for the slot's cards, each at its own pixels, while the
           quads across the line and in the middle stay on the panel. With
           the right side off its quad goes back to the panel; a frame with
           no gauge hands over no cards. */
        if (arm_hud && read_arm) {
            uint32_t aw = 39 + 31, ah = SH, *card = calloc(aw * ah, 4);
            int ok[3] = {0, 0, 0};
            uint32_t eye_g = 0, eye_r = 0, eye_b = 0, eye_c = 0, card_g = 0, card_r = 0, card_c = 0, right_r = 0;
            for (int pass = 0; pass < 3; pass++) {
                arm_hud(0, pass < 2);
                arm_hud(1, pass == 0);
                uint64_t want_ = handed(&r) + 2;
                clear(dev, 0, NULL, 0x1, 0xFF000000u);
                present(dev);
                if (pass < 2) scene_arm_ff(dev);
                else scene_overlay_ff(dev);
                metal_d3d9_shown_frame sv;
                memset(&sv, 0, sizeof sv);
                if (!wait_handed(r.stats, want_) || !r.acquire(&sv)) continue;
                int eyes = sv.layers == 2 && r.read_layer(sv.slot, 0, l, SW, SH) && r.read_layer(sv.slot, 1, rr, SW, SH);
                int cards = sv.arm_texture && read_arm(sv.slot, card, aw, ah);
                r.release(sv.slot);
                if (!eyes) continue;
                uint32_t g = 0, rd = 0, b = 0, c = 0;
                for (int e = 0; e < 2; e++)
                    for (uint32_t i = 0; i < SW * SH; i++) {
                        uint32_t v = (e ? rr : l)[i];
                        g += near(v, 0, 255, 0, 8), rd += near(v, 255, 0, 0, 8);
                        b += near(v, 0, 0, 255, 8), c += near(v, 0, 255, 255, 8);
                    }
                if (pass == 0) {
                    eye_g = g, eye_r = rd, eye_b = b, eye_c = c;
                    for (uint32_t y = 0; cards && y < ah; y++)
                        for (uint32_t x = 0; x < aw; x++) {
                            uint32_t v = card[y * aw + x];
                            int in_g = x >= 4 && x < 30 && y >= 60 && y < 120;
                            int in_r = x >= 39 + 5 && x < 39 + 27 && y >= 60 && y < 120;
                            card_g += in_g && near(v, 0, 255, 0, 8) && (v >> 24) == 255;
                            card_r += in_r && near(v, 255, 0, 0, 8) && (v >> 24) == 255;
                            card_c += near(v, 0, 255, 255, 8) || (!in_g && !in_r && (v >> 24) != 0);
                        }
                    ok[0] = cards && sv.arm_shown[0] && sv.arm_shown[1] && fabsf(sv.arm_uv[0][2] - 39.0f / aw) < 1e-6f &&
                            fabsf(sv.arm_uv[1][0] - 39.0f / aw) < 1e-6f && sv.arm_uv[1][2] == 1.0f;
                    save_png(threaded ? "fullview-arm-threaded-left" : "fullview-arm-inline-left", l);
                } else if (pass == 1) {
                    right_r = rd;
                    ok[1] = cards && sv.arm_shown[0] && !sv.arm_shown[1] && !g && rd > 0;
                } else {
                    ok[2] = sv.fullview == 1 && !sv.arm_texture && !sv.arm_shown[0] && !sv.arm_shown[1];
                }
            }
            arm_hud(0, 0);
            arm_hud(1, 0);
            free(card);
            CHECK(ok[0] && !eye_g && !eye_r && eye_b > 0 && eye_c > 0 && card_g == 26 * 60 && card_r == 22 * 60 &&
                      !card_c,
                  "full view arm HUD (%s): the gauges' quads leave both eyes (%u green, %u red px) for the cards at "
                  "their own pixels (%u of %u green, %u of %u red, %u stray), the quads across the line and in the "
                  "middle stay on the panel (%u blue, %u cyan px)",
                  mode, eye_g, eye_r, card_g, 26 * 60, card_r, 22 * 60, card_c, eye_b, eye_c);
            CHECK(ok[1] && ok[2],
                  "full view arm HUD (%s): a side whose card cannot be shown keeps its gauge on the panel (%u red px), "
                  "and a frame with no gauge hands over no cards (%d)",
                  mode, right_r, ok[2]);
        }

        /* The 3D sky: drawn before the world's depth clear, so in each eye
           from the eye's offset divided by the sky's scale (16), the
           world's quads from the whole offset. Threaded only: inline mode
           has encoded the sky before the clear that marks it. */
        if (threaded) {
            uint64_t want_ = handed(&r) + 2;
            clear(dev, 0, NULL, 0x1, 0xFF000000u);
            present(dev);
            scene_fullview_sky(dev, 0);
            metal_d3d9_shown_frame ss;
            uint32_t ls = take_shown(&r, want_, l, rr, &ss);
            save_png("fullview-sky-left", l);
            save_png("fullview-sky-right", rr);
            double worst_s = 0, worst_g = 0, worst_r = 0;
            uint32_t sn = UINT32_MAX;
            for (int e = 0; e < 2; e++) {
                const uint32_t *img = e ? rr : l;
                float te[4];
                for (int k = 0; k < 4; k++) te[k] = tan[4 * e + k] * (1 + m);
                float ea = 2.0f / (te[0] + te[1]), ecx = (te[0] - te[1]) / (te[0] + te[1]);
                float eb = 2.0f / (te[2] + te[3]), ecy = (te[2] - te[3]) / (te[2] + te[3]);
                float tx = (e ? S_HALF_IPD_M : -S_HALF_IPD_M) * upm;
                double z = 80.0, xs = 0.3 * z / S_A, xg = -0.3 * z / S_A;
                double sx = px_x(ea * (xs - tx / 16.0) / z + ecx), gxw = px_x(ea * (xg - tx) / z + ecx);
                double gx, gy;
                uint32_t n;
                centroid_xy(img, 0xFFFF00, &gx, &gy, &n);
                worst_s = fmax(worst_s, fabs(gx - sx));
                sn = n < sn ? n : sn;
                centroid_xy(img, 0x00FF00, &gx, &gy, &n);
                worst_g = fmax(worst_g, fabs(gx - gxw));
                sn = n < sn ? n : sn;
                /* The sprite: up where the game put it, not turned over. */
                double rx = px_x(ea * (0.0 - tx) / 80.0 + ecx), ry = px_y(eb * 0.2 / S_B + ecy);
                centroid_xy(img, 0xFF0000, &gx, &gy, &n);
                worst_r = fmax(worst_r, fmax(fabs(gx - rx), fabs(gy - ry)));
                sn = n < sn ? n : sn;
            }
            /* The view model's verdict: unknown until one is drawn, then
               drawn, then gone (third person) after frames without it. */
            int (*view_model)(void) = (int (*)(void))dlsym(s_lib, "metal_d3d9_fullview_view_model");
            int vm0 = view_model ? view_model() : -2, vm1 = -2, vm2 = -2;
            if (view_model) {
                for (int i = 0; i < 12; i++) scene_fullview_sky(dev, 1);
                vm1 = view_model();
                for (int i = 0; i < 12; i++) scene_fullview_sky(dev, 0);
                vm2 = view_model();
            }
            CHECK(vm0 == -1 && vm1 == 1 && vm2 == 0,
                  "full view view model (%s): not known before one is drawn (%d), drawn after frames with one (%d), "
                  "third person after frames without (%d)",
                  mode, vm0, vm1, vm2);
            CHECK(ls == 2 && sn > 20 && worst_s < 0.75 && worst_g < 0.75,
                  "full view sky (%s): the world draw from the sky's own camera, with no depth clear after it, from "
                  "a sixteenth of each eye's offset (worst %.2f px), the world's from the whole offset (worst %.2f px)",
                  mode, worst_s, worst_g);
            CHECK(ls == 2 && sn > 20 && worst_r < 0.75,
                  "full view sprite (%s): a fixed-function draw with _22 negative and the view's y turned over lands "
                  "where its projection puts it, not upside down (worst %.2f px)",
                  mode, worst_r);
        }

        /* Out of play: stage A's frame to the bit, on the plane. */
        play(0);
        FRAME(&sp, l, rr);
        lp = layers;
        fv(0);
        FRAME(&sa, al, ar);
        la = layers;
        fv(1);
        play(1);
        int stage_a = !memcmp(l, al, SW * SH * 4) && !memcmp(rr, ar, SW * SH * 4) && memcmp(l, rr, SW * SH * 4);
        CHECK(lp == 2 && la == 2 && stage_a && sp.fullview == 0 && sa.fullview == 0 && sp.tan_x == sa.tan_x &&
                  sp.tan_x > 0,
              "full view out of play (%s): both eyes stage A's to the bit (%d), on the plane (full view %d, plane "
              "tan %.4f)",
              mode, stage_a, sp.fullview, (double)sp.tan_x);
#undef FRAME
        stereo(0);
        VT(dev, SLOT_IDirect3DDevice9_Release, uint32_t (*)(void *))(dev);
    }
    free(mono), free(l), free(rr), free(al), free(ar);
    fv(0);
    play(0);
    stereo(0);
    enable(0);
}

static void mode_tests(void *d3d)
{
    if (s_shader_dir) {
        compare_scene(d3d, "DXT1 quad through game shaders", scene_dxt1);
        compare_scene(d3d, "two DISCARD quads in one pass", scene_discard);
    }
    compare_scene(d3d, "fixed-function lit quad", scene_ff_lit);
    compare_scene(d3d, "plain relock between two draws", scene_relock);
    compare_scene(d3d, "readbacks in the middle of a frame", scene_midframe);
    replay_twice_test(d3d);
    for (int t = 0; t < 2; t++) fail_pipeline_test(d3d, t);
    for (int t = 0; t < 2; t++) stress_test(d3d, t, 300);
    if (s_shader_dir)
        for (int t = 0; t < 2; t++) draw_cost(d3d, t);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s LIB [METHODS_FILE]\n", argv[0]);
        return 2;
    }
    void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        fprintf(stderr, "dlopen %s: %s\n", argv[1], dlerror());
        return 2;
    }
    s_lib = lib;
    void *(*create9)(uint32_t) = (void *(*)(uint32_t))dlsym(lib, "Direct3DCreate9");
    int (*impl)(const char *, const char *) = (int (*)(const char *, const char *))dlsym(lib, "metal_d3d9_slot_implemented");
    CHECK(create9 && impl && dlsym(lib, "metal_d3d9_set_host"), "exports Direct3DCreate9, metal_d3d9_set_host");
    if (!create9 || !impl) return 1;

    /* The methods the bridge itself calls on the host for capture, lock
       and release bookkeeping, besides what the game reaches. */
    static const char *const k_bridge[][2] = {
        {"IDirect3DDevice9", "GetBackBuffer"},  {"IDirect3DDevice9", "CreateOffscreenPlainSurface"},
        {"IDirect3DDevice9", "GetRenderTargetData"}, {"IDirect3DSurface9", "GetDesc"},
        {"IDirect3DSurface9", "LockRect"},      {"IDirect3DSurface9", "UnlockRect"},
        {"IDirect3DSurface9", "Release"},       {"IDirect3DTexture9", "LockRect"},
        {"IDirect3DTexture9", "UnlockRect"},    {"IDirect3DCubeTexture9", "LockRect"},
        {"IDirect3DCubeTexture9", "UnlockRect"}, {"IDirect3DCubeTexture9", "GetLevelCount"},
        {"IDirect3DDevice9", "Reset"},          {"IDirect3DDevice9", "GetDeviceCaps"},
    };
    uint32_t bok = 0;
    for (uint32_t i = 0; i < sizeof k_bridge / sizeof k_bridge[0]; i++) bok += impl(k_bridge[i][0], k_bridge[i][1]) == 1;
    CHECK(bok == sizeof k_bridge / sizeof k_bridge[0], "%u of %zu bridge-internal methods implemented", bok,
          sizeof k_bridge / sizeof k_bridge[0]);
    CHECK(impl("IDirect3DDevice9", "UpdateSurface") == 1 && impl("IDirect3DDevice9", "UpdateTexture") == 1 &&
              impl("IDirect3DDevice9", "ColorFill") == 1 && impl("IDirect3DDevice9", "SetLight") == 1,
          "the movie path's ColorFill and UpdateSurface, UpdateTexture and SetLight are implemented");
    CHECK(impl("IDirect3DDevice9", "QueryInterface") == 0, "unexpected slots (QueryInterface) are stops");
    if (argc > 2 && strcmp(argv[2], "-")) methods_check(argv[2], impl);
    s_shader_dir = argc > 3 ? argv[3] : NULL;

    void *d3d = create9(32);
    CHECK(d3d != NULL, "Direct3DCreate9(32)");
    if (!d3d) return 1;
    if (getenv("METAL_TEST_DRAW_FRAMES") && s_shader_dir) {
        /* Only the per-draw cost, for a profiler: METAL_TEST_DRAW_FRAMES
           frames of 100 draws, in the mode RT_METAL_INLINE selects. */
        const char *m = getenv("RT_METAL_INLINE");
        draw_cost(d3d, !(m && atoi(m) > 0));
        return 0;
    }
    uint32_t n = VT(d3d, SLOT_IDirect3D9_GetAdapterCount, uint32_t (*)(void *))(d3d);
    uint8_t id[1104];
    HR hr = VT(d3d, SLOT_IDirect3D9_GetAdapterIdentifier, HR (*)(void *, uint32_t, uint32_t, void *))(d3d, 0, 0, id);
    uint32_t vendor, device;
    memcpy(&vendor, id + 1064, 4);
    memcpy(&device, id + 1068, 4);
    CHECK(n == 1 && !hr && vendor == 0x1002, "adapter: count %u, \"%s\", vendor 0x%04X device 0x%04X", n,
          (char *)id + 512, vendor, device);
    uint32_t caps[76];
    hr = VT(d3d, SLOT_IDirect3D9_GetDeviceCaps, HR (*)(void *, uint32_t, uint32_t, void *))(d3d, 0, 1, caps);
    CHECK(!hr && caps[49] == 0xFFFE0200u && caps[51] == 0xFFFF0200u, "caps: vs 0x%08X ps 0x%08X", caps[49], caps[51]);
    HR r20 = VT(d3d, SLOT_IDirect3D9_CheckDeviceFormat,
                HR (*)(void *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t))(d3d, 0, 1, 22, 0, 3, 20);
    HR r21 = VT(d3d, SLOT_IDirect3D9_CheckDeviceFormat,
                HR (*)(void *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t))(d3d, 0, 1, 22, 1, 3, 21);
    CHECK((uint32_t)r20 == 0x8876086Au && r21 == 0, "CheckDeviceFormat: R8G8B8 texture 0x%08X, A8R8G8B8 target 0x%08X",
          (uint32_t)r20, (uint32_t)r21);

    /* Clear plus readback: the whole target, then one rectangle. */
    void *dev = make_device(d3d, 64, 48);
    if (!dev) return 1;
    hr = clear(dev, 0, NULL, 0x7, 0xFF123456u);
    uint32_t w = 0, h = 0, *px = readback(dev, &w, &h);
    CHECK(!hr && px && w == 64 && h == 48 && count_rgb(px, w * h, 0x123456) == w * h,
          "Clear 0x123456 and read back: %u of %u pixels match, first 0x%08X", px ? count_rgb(px, w * h, 0x123456) : 0,
          w * h, px ? px[0] : 0);
    free(px);
    rect r = {8, 4, 24, 20};
    hr = clear(dev, 1, &r, 0x1, 0xFFAA0000u);
    px = readback(dev, &w, &h);
    uint32_t in = px ? count_rgb(px, w * h, 0xAA0000) : 0, out = px ? count_rgb(px, w * h, 0x123456) : 0;
    CHECK(!hr && in == 16 * 16 && out == w * h - 16 * 16 && px && (px[4 * 64 + 8] & 0xFFFFFF) == 0xAA0000 &&
              (px[4 * 64 + 7] & 0xFFFFFF) == 0x123456,
          "Clear of one 16x16 rectangle: %u inside, %u outside unchanged", in, out);
    free(px);

    /* Present with no window commits and returns. */
    hr = VT(dev, SLOT_IDirect3DDevice9_Present, HR (*)(void *, const void *, const void *, void *, const void *))(
        dev, NULL, NULL, NULL, NULL);
    CHECK(!hr, "Present without a window: hr 0x%08X", (uint32_t)hr);

    /* Identity and counts: the bridge keys proxies on the host pointer and
       drops one when Release returns 0. */
    void *bb1 = NULL, *bb2 = NULL;
    VT(dev, SLOT_IDirect3DDevice9_GetBackBuffer, HR (*)(void *, uint32_t, uint32_t, uint32_t, void **))(dev, 0, 0, 0, &bb1);
    uint32_t c1 = VT(bb1, SLOT_IDirect3DSurface9_Release, uint32_t (*)(void *))(bb1);
    VT(dev, SLOT_IDirect3DDevice9_GetBackBuffer, HR (*)(void *, uint32_t, uint32_t, uint32_t, void **))(dev, 0, 0, 0, &bb2);
    uint32_t c2 = VT(bb2, SLOT_IDirect3DSurface9_Release, uint32_t (*)(void *))(bb2);
    CHECK(bb1 == bb2 && c1 == 0 && c2 == 0, "GetBackBuffer returns one object, and Release of it returns 0 (%u, %u)", c1, c2);

    /* A texture level surface counts against its texture. */
    void *tex = NULL, *lvl = NULL;
    hr = VT(dev, SLOT_IDirect3DDevice9_CreateTexture,
            HR (*)(void *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, void **, void *))(
        dev, 256, 256, 0, 0, 0x31545844u, 1, &tex, NULL);
    uint32_t levels = tex ? VT(tex, SLOT_IDirect3DTexture9_GetLevelCount, uint32_t (*)(void *))(tex) : 0;
    if (tex) VT(tex, SLOT_IDirect3DTexture9_GetSurfaceLevel, HR (*)(void *, uint32_t, void **))(tex, 1, &lvl);
    uint32_t cl = lvl ? VT(lvl, SLOT_IDirect3DSurface9_Release, uint32_t (*)(void *))(lvl) : 99;
    locked_rect lr = {0, NULL};
    HR hl = tex ? VT(tex, SLOT_IDirect3DTexture9_LockRect, HR (*)(void *, uint32_t, locked_rect *, const void *, uint32_t))(
                      tex, 0, &lr, NULL, 0)
                : -1;
    int zero = 1;
    for (int i = 0; !hl && i < 64 * 64 * 8 / 8; i++) zero &= ((uint8_t *)lr.pBits)[i] == 0;
    if (!hl) VT(tex, SLOT_IDirect3DTexture9_UnlockRect, HR (*)(void *, uint32_t))(tex, 0);
    uint32_t ct = tex ? VT(tex, SLOT_IDirect3DTexture9_Release, uint32_t (*)(void *))(tex) : 99;
    CHECK(!hr && levels == 9 && cl == 1 && !hl && lr.Pitch == 512 && zero && ct == 0,
          "DXT1 256x256: %u levels, level Release %u, pitch %d, zeroed %d, texture Release %u", levels, cl, lr.Pitch, zero,
          ct);

    /* Shader bytecode is kept whole: a ps_1_1 def whose float literals
       look like a comment token and like the end token must not cut or
       overrun the scan. Version, def (6 tokens), end: 8 tokens. */
    static const uint32_t k_ps11[] = {0xFFFF0101u, 0x00000051u, 0xA00F0000u, 0x3F7FFFFEu,
                                      0x0000FFFFu, 0x00000000u, 0x3F800000u, 0x0000FFFFu};
    void *ps = NULL;
    hr = VT(dev, SLOT_IDirect3DDevice9_CreatePixelShader, HR (*)(void *, const void *, void **))(dev, k_ps11, &ps);
    uint32_t psz = 0;
    if (ps) VT(ps, 4, HR (*)(void *, void *, uint32_t *))(ps, NULL, &psz); /* GetFunction */
    if (ps) VT(ps, 2, uint32_t (*)(void *))(ps);                           /* Release */
    CHECK(!hr && psz == sizeof k_ps11, "ps_1_1 with a def: %u bytecode bytes kept (want %zu)", psz, sizeof k_ps11);

    /* Reset to another size: the back buffer follows and Clear still reads back. */
    present_params rp;
    memset(&rp, 0, sizeof rp);
    rp.BackBufferWidth = 40;
    rp.BackBufferHeight = 30;
    rp.BackBufferFormat = 22;
    rp.Windowed = 1;
    rp.EnableAutoDepthStencil = 1;
    rp.AutoDepthStencilFormat = 75;
    rp.PresentationInterval = 0x80000000u;
    hr = VT(dev, SLOT_IDirect3DDevice9_Reset, HR (*)(void *, present_params *))(dev, &rp);
    HR hc = clear(dev, 0, NULL, 0x1, 0xFF00C0FFu);
    px = readback(dev, &w, &h);
    CHECK(!hr && !hc && px && w == 40 && h == 30 && count_rgb(px, w * h, 0x00C0FF) == w * h,
          "Reset to 40x30, Clear and read back: %ux%u, %u of %u pixels match", w, h,
          px ? count_rgb(px, w * h, 0x00C0FF) : 0, w * h);
    free(px);

    /* Resources the guest holds hold the device, as in DXVK: the bridge
       expects device Release to return the number of readback surfaces it
       still keeps, and releases those afterwards. */
    void *keep = NULL;
    VT(dev, SLOT_IDirect3DDevice9_CreateOffscreenPlainSurface,
       HR (*)(void *, uint32_t, uint32_t, uint32_t, uint32_t, void **, void *))(dev, 64, 48, 22, 2, &keep, NULL);
    uint32_t cd = VT(dev, SLOT_IDirect3DDevice9_Release, uint32_t (*)(void *))(dev);
    uint32_t ck = keep ? VT(keep, SLOT_IDirect3DSurface9_Release, uint32_t (*)(void *))(keep) : 99;
    CHECK(cd == 1 && ck == 0, "device Release with one surface alive returns %u, then the surface's Release %u", cd, ck);

    /* RT_METAL_CLEAR is read when a device is created and overrides the
       game's colour for every render target clear. */
    setenv("RT_METAL_CLEAR", "FF00FF", 1);
    dev = make_device(d3d, 32, 32);
    if (!dev) return 1;
    hr = clear(dev, 0, NULL, 0x1, 0xFF000000u);
    px = readback(dev, &w, &h);
    CHECK(!hr && px && count_rgb(px, w * h, 0xFF00FF) == w * h,
          "RT_METAL_CLEAR=FF00FF over a black Clear: %u of %u pixels are FF00FF, first 0x%08X",
          px ? count_rgb(px, w * h, 0xFF00FF) : 0, w * h, px ? px[0] : 0);
    free(px);
    VT(dev, SLOT_IDirect3DDevice9_Release, uint32_t (*)(void *))(dev);
    unsetenv("RT_METAL_CLEAR");

    conv_tests();
    ff_tests(d3d);

    if (s_shader_dir) {
        /* A fresh shader cache beside the test binary, so the first pass
           translates and the second reads it back. */
        char cache[1024];
        snprintf(cache, sizeof cache, "%s.cache-XXXXXX", argv[0]);
        if (mkdtemp(cache)) setenv("RT_METAL_CACHE_DIR", cache, 1);
        draw_tests(d3d);

        int (*ff)(void) = (int (*)(void))dlsym(lib, "metal_d3d9_ff_selfcheck");
        int nff = ff ? ff() : -1;
        CHECK(nff == 2, "DXVK fixed-function ubershaders through SPIRV-Cross and Metal: %d of 2 compile", nff);

        int (*pairs)(const char *, int, int *, int *, int *) =
            (int (*)(const char *, int, int *, int *, int *))dlsym(lib, "metal_d3d9_pair_selfcheck");
        int tried = 0, built = 0, cached = 0;
        double t0 = now_s();
        int okp = pairs ? pairs(s_shader_dir, 0, &tried, &built, &cached) : -1;
        double t1 = now_s();
        CHECK(okp > 0 && okp == tried, "every game pixel shader linked with a vertex shader of its family: %d of %d "
                                       "pipelines (%d stages translated, %d from cache, %.1f s)",
              okp, tried, built, cached, t1 - t0);
        okp = pairs ? pairs(s_shader_dir, 200, &tried, &built, &cached) : -1;
        double t2 = now_s();
        CHECK(okp == tried && cached == 2 * tried && built == 0,
              "again from the cache: %d of %d pipelines, %d stages from cache, %d translated (%.1f s)", okp, tried, cached,
              built, t2 - t1);

        int (*ff_pairs)(const char *, int *) = (int (*)(const char *, int *))dlsym(lib, "metal_d3d9_ff_pair_selfcheck");
        int ff_tried = 0;
        int okf = ff_pairs ? ff_pairs(s_shader_dir, &ff_tried) : -1;
        double t3 = now_s();
        CHECK(okf > 0 && okf == ff_tried,
              "every game shader linked with the fixed-function stage opposite it: %d of %d pipelines (%.1f s)", okf,
              ff_tried, t3 - t2);

        int (*max_slot)(int) = (int (*)(int))dlsym(lib, "metal_d3d9_selfcheck_max_slot");
        int top = max_slot ? max_slot(1) : 99;
        CHECK(top >= 0 && top <= 30,
              "highest texture index in every game shader's and the fixed-function stages' MSL: %d (the smallest "
              "texture argument table ends at 30)",
              top);

        pso_tests(d3d, argv[0]);
    }

    immersive_tests(d3d);
    stereo_tests(d3d, argv[0]);
    fullview_tests(d3d);
    mode_tests(d3d);

    uint32_t c3 = VT(d3d, SLOT_IDirect3D9_Release, uint32_t (*)(void *))(d3d);
    CHECK(c3 == 0, "IDirect3D9 Release returns %u", c3);

    printf("%s: %d failed\n", s_fail ? "FAILED" : "ALL PASS", s_fail);
    return s_fail ? 1 : 0;
}
