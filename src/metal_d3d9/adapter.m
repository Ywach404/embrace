/*
 * IDirect3D9 on Metal: the adapter, its modes, caps and format checks,
 * plus the library's logging and the two exports.
 *
 * The adapter answers so that the engine takes the same path it took on
 * DXVK in the reference runs on the Mac: the same vendor ID, shader model 2.0, and
 * the same CheckDeviceFormat refusals those runs logged. Everything else
 * follows the porter's null device, which boots the engine to its menu.
 */
#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <TargetConditionals.h>

#include "mtl_d3d9.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void (*s_host_log)(const char *);
static void (*s_host_stop)(const char *);

void metal_d3d9_set_host(void (*log)(const char *msg), void (*stop)(const char *msg))
{
    s_host_log = log;
    s_host_stop = stop;
}

void mtl_log(const char *fmt, ...)
{
    char b[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    if (s_host_log) s_host_log(b);
    else fprintf(stderr, "metal_d3d9: %s\n", b);
}

void mtl_unexpected(const char *method)
{
    char b[256];
    snprintf(b, sizeof b, "unexpected call %s: the metal backend does not implement it", method);
    if (s_host_stop) s_host_stop(b);
    fprintf(stderr, "metal_d3d9: %s\n", b);
    abort();
}

/* Bits per pixel, as the bridge's rt_pitch_of computes them for the null
   device. The bridge copies rt_pitch_of(format, width) bytes per row
   through a lock using our Pitch as the stride, so our pitch must be at
   least that; keeping the same table makes it exactly that. */
uint32_t mtl_fmt_bits(uint32_t f)
{
    switch (f) {
    case 20: return 24;
    case 23: case 24: case 25: case 26: case 29: case 30: case 51: case 60: case 61: case 80: case 81: case 70: case 111:
        return 16;
    case 28: case 41: case 50: case 3: return 8;
    case 113: case 115: case 36: return 64;
    case 116: return 128;
    case 0x31545844u: return 4;
    case 0x32545844u: case 0x33545844u: case 0x34545844u: case 0x35545844u: return 8;
    default: return 32;
    }
}

int mtl_fmt_dxt(uint32_t f) { return (f & 0x00FFFFFFu) == 0x00545844u; }

uint64_t mtl_fmt_pitch(uint32_t f, uint32_t w)
{
    if (mtl_fmt_dxt(f)) return (((uint64_t)w + 3) / 4) * (f == D3DFMT_DXT1 ? 8u : 16u);
    return ((uint64_t)w * mtl_fmt_bits(f) + 7) / 8;
}

uint64_t mtl_fmt_rows(uint32_t f, uint32_t h) { return mtl_fmt_dxt(f) ? ((uint64_t)h + 3) / 4 : h; }

static void put_f(uint32_t *c, int i, float v) { memcpy(&c[i], &v, 4); }

/* D3DCAPS9 as 76 DWORDs, indexed as in the SDK's d3d9caps.h. The base is
   the null device's table (every capability bit set, its limits), with
   three groups changed. Shader model 2.0, because the DXVK reference runs
   compiled vs_2_0 and ps_2_0 shaders and a 1.1 answer would send the
   engine down its dx8 path. Texture caps without the power-of-two and
   square-only restriction bits, which the null device's all-ones word
   also claims. And the 2.0 detail fields a 2.0 part reports. */
void mtl_caps(uint32_t c[76])
{
    memset(c, 0, 76 * 4);
    for (int i = 2; i <= 21; i++) c[i] = 0xFFFFFFFFu;
    c[0] = 1; /* DeviceType: HAL */
    /* TextureCaps: perspective, alpha, texrepeatnotscaledbysize, projected,
       cubemap, volumemap, mipmap, mipvolumemap, mipcubemap. */
    c[15] = 0x1u | 0x4u | 0x40u | 0x400u | 0x800u | 0x2000u | 0x4000u | 0x8000u | 0x10000u;
    c[22] = c[23] = 8192; /* MaxTextureWidth, MaxTextureHeight */
    c[24] = 256;          /* MaxVolumeExtent */
    c[25] = 8192;         /* MaxTextureRepeat */
    c[26] = 8192;         /* MaxTextureAspectRatio */
    c[27] = 16;           /* MaxAnisotropy */
    put_f(c, 28, 1e10f);  /* MaxVertexW */
    put_f(c, 29, -1e8f);  /* guard band */
    put_f(c, 30, -1e8f);
    put_f(c, 31, 1e8f);
    put_f(c, 32, 1e8f);
    c[34] = c[36] = c[39] = 0xFFFFFFFFu; /* StencilCaps, TextureOpCaps, VertexProcessingCaps */
    c[35] = 8;            /* FVFCaps: 8 texture coordinate sets */
    c[37] = 8;            /* MaxTextureBlendStages */
    c[38] = 8;            /* MaxSimultaneousTextures */
    c[40] = 8;            /* MaxActiveLights */
    c[41] = 6;            /* MaxUserClipPlanes */
    c[42] = 4;            /* MaxVertexBlendMatrices */
    put_f(c, 44, 64.0f);  /* MaxPointSize */
    c[45] = 0xFFFFFu;     /* MaxPrimitiveCount */
    c[46] = 0xFFFFFu;     /* MaxVertexIndex */
    c[47] = 16;           /* MaxStreams */
    c[48] = 255;          /* MaxStreamStride */
    c[49] = 0xFFFE0200u;  /* VertexShaderVersion 2.0 */
    c[50] = 256;          /* MaxVertexShaderConst */
    c[51] = 0xFFFF0200u;  /* PixelShaderVersion 2.0 */
    put_f(c, 52, 8.0f);   /* PixelShader1xMaxValue */
    c[58] = 1;            /* NumberOfAdaptersInGroup */
    c[59] = 0x3FFu;       /* DeclTypes: UBYTE4 through FLOAT16_4 */
    c[60] = 4;            /* NumSimultaneousRTs */
    c[61] = 0x03000300u;  /* StretchRectFilterCaps: point and linear, min and mag */
    c[62] = 1;            /* VS20Caps: predication */
    c[63] = 24;           /* dynamic flow control depth */
    c[64] = 32;           /* temps */
    c[65] = 4;            /* static flow control depth */
    c[66] = 0x1Fu;        /* PS20Caps */
    c[67] = 24;
    c[68] = 32;
    c[69] = 4;
    c[70] = 512;          /* instruction slots */
    c[72] = 65535;        /* MaxVShaderInstructionsExecuted */
    c[73] = 65535;        /* MaxPShaderInstructionsExecuted */
}

HRESULT mtl_display_mode_of(mtl_display_mode *m)
{
#if TARGET_OS_VISION
    /* No display to ask on the headset: the CGDisplay calls are macOS
       only. The mode is a fixed 1080p at the headset's 90 Hz, which only
       seeds the engine's mode list; the window size comes from the
       runtime's pinned desktop. */
    *m = (mtl_display_mode){1920, 1080, 90, D3DFMT_X8R8G8B8};
    return D3D_OK;
#else
    CGDirectDisplayID id = CGMainDisplayID();
    CGDisplayModeRef mode = CGDisplayCopyDisplayMode(id);
    uint32_t w = 1920, h = 1080, hz = 60;
    if (mode) {
        w = (uint32_t)CGDisplayModeGetWidth(mode);
        h = (uint32_t)CGDisplayModeGetHeight(mode);
        double r = CGDisplayModeGetRefreshRate(mode);
        if (r >= 1.0) hz = (uint32_t)(r + 0.5);
        CGDisplayModeRelease(mode);
    }
    *m = (mtl_display_mode){w, h, hz, D3DFMT_X8R8G8B8};
    return D3D_OK;
#endif
}

/* Windowed modes the menu can list. The bridge adds the engine's own sizes
   and the launch size on top, and sorts. */
static const uint32_t k_modes[][2] = {{640, 480},   {800, 600},   {1024, 768},  {1280, 720},  {1280, 800},
                                      {1280, 1024}, {1440, 900},  {1600, 900},  {1920, 1080}};
#define NMODES (sizeof k_modes / sizeof k_modes[0])

static int display_format(uint32_t f) { return f == D3DFMT_X8R8G8B8 || f == D3DFMT_R5G6B5; }

static uint32_t d9_addref(mtl_d3d9 *d) { return mtl_addref(&d->o); }
static uint32_t d9_release(mtl_d3d9 *d) { return mtl_release(&d->o); }
static uint32_t d9_adapter_count(mtl_d3d9 *d) { (void)d; return 1; }

static HRESULT d9_adapter_id(mtl_d3d9 *d, uint32_t adapter, uint32_t flags, mtl_adapter_id *out)
{
    (void)d;
    (void)flags;
    if (adapter || !out) return D3DERR_INVALIDCALL;
    memset(out, 0, sizeof *out);
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        snprintf(out->Driver, sizeof out->Driver, "metal_d3d9.dll");
        snprintf(out->Description, sizeof out->Description, "%s (Metal)", dev ? dev.name.UTF8String : "Metal");
    }
    snprintf(out->DeviceName, sizeof out->DeviceName, "\\\\.\\DISPLAY1");
    out->DriverVersion = (int64_t)0x0006000000000001ll;
    /* The vendor and device IDs of the adapter DXVK reported in the
       reference runs (AMD, Radeon RX 6700 XT), since the engine picks its settings
       by ID and the Metal path should start where the reference started. */
    out->VendorId = 0x1002;
    out->DeviceId = 0x73DF;
    /* A fixed, nonzero identifier: the bridge deletes a cached video config
       whose identifier is all zeros, as the null device writes. */
    static const uint8_t k_guid[16] = {0x6d, 0x74, 0x6c, 0x64, 0x33, 0x64, 0x39, 0x00,
                                       0x8a, 0x1e, 0x52, 0x0c, 0x01, 0x02, 0x00, 0x01};
    memcpy(out->DeviceIdentifier, k_guid, 16);
    out->WHQLLevel = 1;
    return D3D_OK;
}

static uint32_t d9_mode_count(mtl_d3d9 *d, uint32_t adapter, uint32_t fmt)
{
    (void)d;
    return !adapter && display_format(fmt) ? (uint32_t)NMODES : 0;
}

static HRESULT d9_enum_modes(mtl_d3d9 *d, uint32_t adapter, uint32_t fmt, uint32_t i, mtl_display_mode *m)
{
    (void)d;
    if (adapter || !display_format(fmt) || i >= NMODES || !m) return D3DERR_INVALIDCALL;
    *m = (mtl_display_mode){k_modes[i][0], k_modes[i][1], 60, fmt};
    return D3D_OK;
}

static HRESULT d9_display_mode(mtl_d3d9 *d, uint32_t adapter, mtl_display_mode *m)
{
    (void)d;
    if (adapter || !m) return D3DERR_INVALIDCALL;
    return mtl_display_mode_of(m);
}

static int depth_format(uint32_t f) { return f >= 70 && f <= 83; }

/* The refusals the DXVK reference runs logged (quiet-2 and hub-2): 24-bit
   R8G8B8 and paletted P8 as textures, and DXT formats as render targets.
   Depth usage is refused for colour formats. Everything else is accepted,
   as the null device accepts everything. */
static HRESULT d9_check_format(mtl_d3d9 *d, uint32_t adapter, uint32_t type, uint32_t adapter_fmt, uint32_t usage,
                               uint32_t rtype, uint32_t fmt)
{
    (void)d;
    (void)type;
    (void)adapter_fmt;
    if (adapter) return D3DERR_INVALIDCALL;
    if (rtype == D3DRTYPE_TEXTURE || rtype == D3DRTYPE_CUBETEXTURE || rtype == D3DRTYPE_SURFACE) {
        if (fmt == D3DFMT_R8G8B8 || fmt == D3DFMT_P8) return D3DERR_NOTAVAILABLE;
        if (mtl_fmt_dxt(fmt) && (usage & D3DUSAGE_RENDERTARGET)) return D3DERR_NOTAVAILABLE;
        if ((usage & D3DUSAGE_DEPTHSTENCIL) && !depth_format(fmt)) return D3DERR_NOTAVAILABLE;
    }
    return D3D_OK;
}

/* No multisampling: only the single-sample answer. */
static HRESULT d9_check_ms(mtl_d3d9 *d, uint32_t adapter, uint32_t type, uint32_t fmt, int32_t windowed, uint32_t ms,
                           uint32_t *quality)
{
    (void)d;
    (void)type;
    (void)fmt;
    (void)windowed;
    if (adapter) return D3DERR_INVALIDCALL;
    if (ms > 1) return D3DERR_NOTAVAILABLE;
    if (quality) *quality = 1;
    return D3D_OK;
}

static HRESULT d9_check_ds(mtl_d3d9 *d, uint32_t adapter, uint32_t type, uint32_t afmt, uint32_t rtfmt, uint32_t dsfmt)
{
    (void)d;
    (void)type;
    (void)afmt;
    (void)rtfmt;
    (void)dsfmt;
    return adapter ? D3DERR_INVALIDCALL : D3D_OK;
}

static HRESULT d9_caps(mtl_d3d9 *d, uint32_t adapter, uint32_t type, uint32_t *caps)
{
    (void)d;
    (void)type;
    if (adapter || !caps) return D3DERR_INVALIDCALL;
    mtl_caps(caps);
    return D3D_OK;
}

static HRESULT d9_create_device(mtl_d3d9 *d, uint32_t adapter, uint32_t type, void *focus, uint32_t flags,
                                mtl_present_params *pp, mtl_device **out)
{
    if (!out || !pp) return D3DERR_INVALIDCALL;
    *out = NULL;
    if (adapter) return D3DERR_INVALIDCALL;
    return mtl_device_create(d, adapter, type, focus, flags, pp, out);
}

void mtl_fill_d3d9(void)
{
    void **v = mtl_vt_d3d9;
    v[SLOT_IDirect3D9_AddRef] = (void *)d9_addref;
    v[SLOT_IDirect3D9_Release] = (void *)d9_release;
    v[SLOT_IDirect3D9_GetAdapterCount] = (void *)d9_adapter_count;
    v[SLOT_IDirect3D9_GetAdapterIdentifier] = (void *)d9_adapter_id;
    v[SLOT_IDirect3D9_GetAdapterModeCount] = (void *)d9_mode_count;
    v[SLOT_IDirect3D9_EnumAdapterModes] = (void *)d9_enum_modes;
    v[SLOT_IDirect3D9_GetAdapterDisplayMode] = (void *)d9_display_mode;
    v[SLOT_IDirect3D9_CheckDeviceFormat] = (void *)d9_check_format;
    v[SLOT_IDirect3D9_CheckDeviceMultiSampleType] = (void *)d9_check_ms;
    v[SLOT_IDirect3D9_CheckDepthStencilMatch] = (void *)d9_check_ds;
    v[SLOT_IDirect3D9_GetDeviceCaps] = (void *)d9_caps;
    v[SLOT_IDirect3D9_CreateDevice] = (void *)d9_create_device;
}

void *Direct3DCreate9(uint32_t sdk_version)
{
    mtl_vtables_build();
    mtl_d3d9 *d = calloc(1, sizeof *d);
    if (!d) return NULL;
    mtl_obj_init(&d->o, mtl_vt_d3d9, MTL_K_D3D, NULL);
    mtl_addref(&d->o);
    mtl_log("Direct3DCreate9(%u)", sdk_version);
    return d;
}
