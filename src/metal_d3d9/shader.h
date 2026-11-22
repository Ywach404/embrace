/*
 * The shader chain, as a C interface for the backend: D3D9 bytecode
 * through DXVK's own D3D9 front end (dxbc_spv::sm3, the legacy input and
 * binding model lowering passes) to SPIR-V, then SPIRV-Cross to MSL. The
 * implementation is C++ (shader.cpp) because DXVK's converter has no C
 * surface.
 *
 * Each stage is translated alone, without a DXVK linkage, which is the
 * mode DXVK itself uses for pipeline libraries: spec constants stay in a
 * buffer and every sm1/sm2 vertex shader writes the same twelve output
 * locations every pixel shader reads a subset of, so any pair links in a
 * Metal pipeline by name.
 *
 * The MSL resource indices are fixed here and nowhere else, so the draw
 * path binds by these numbers without reflection.
 */
#ifndef MTL_D3D9_SHADER_H
#define MTL_D3D9_SHADER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Buffer indices, both stages. */
#define MTL_BUF_HEAP 0      /* set 0: the sampler heap, a device argument buffer */
#define MTL_BUF_PUSH 8      /* push data */
#define MTL_BUF_STEREO 7    /* stereo's clip-space correction, four float4 (stereo.h); every vertex stage reads it */
#define MTL_BUF_SPEC 9      /* set 7: D3D9 spec data, 80 bytes */
#define MTL_BUF_CBV 10      /* set 1 binding b at 10 + b (b < 7) */
#define MTL_BUF_STREAM 17   /* vertex stream s at 17 + s */
#define MTL_BUF_NULL 30     /* the zero stream unmatched inputs read */
#define MTL_MAX_STREAMS 13
#define MTL_MAX_INPUTS 18   /* the fixed-function vertex shader's attribute count */

/* DXVK's CBV bindings (d3d9_shader.h CbvIndex). */
#define MTL_CBV_CLIP 0
#define MTL_CBV_VS_FF 1
#define MTL_CBV_VS_BLEND 2
#define MTL_CBV_VS_STATIC 3
#define MTL_CBV_VS_DYNAMIC 4
#define MTL_CBV_PS_SHARED 5
#define MTL_CBV_PS_STATIC 6

/* Texture indices. A binding (register for a pixel shader, 17 +
   register for a vertex shader, DXVK's numbering) has up to three views:
   2D, 3D and cube. A sampler whose type the bytecode declares (sm2 and
   up) has one; an untyped one (ps_1_x, the fixed-function stages) has
   all three, since the bound texture picks at draw time. Each stage
   numbers only the views it has, densely, view by view: all 2D views in
   binding order, then all 3D, then all cube (mtl_tex_layout). The most
   any stage needs is 24 (the fixed-function pixel stage, 8 untyped
   samplers), under the 31 entries the smallest texture argument table
   holds (the simulator's GPU); the earlier fixed bases of 0, 24 and 48
   reached index 68. View-major order gives the fixed-function stage its
   three contiguous arrays of 8 with no rule of its own. */
#define MTL_TEX_BINDINGS 21
#define MTL_TEX_VIEWS 3   /* 0 2D, 1 3D, 2 cube */
#define MTL_TEX_NONE 0xFF /* the stage has no such view */

/* view types as DXVK's analysis reports them (VkImageViewType) */
#define MTL_VIEW_2D 1u
#define MTL_VIEW_3D 2u
#define MTL_VIEW_CUBE 3u
#define MTL_VIEW_ANY 0x7FFFFFFFu

typedef struct mtl_sc mtl_sc;

typedef struct {
    int is_ps;
    uint32_t major, minor;
    uint32_t n_inputs;                  /* vertex shader: attribute i is input i */
    uint8_t in_usage[MTL_MAX_INPUTS], in_index[MTL_MAX_INPUTS]; /* D3DDECLUSAGE and usage index of each */
    uint32_t sampler_mask;              /* bit b: texture binding b is used */
    uint32_t view[MTL_TEX_BINDINGS];    /* MTL_VIEW_* per used binding */
    uint8_t tex_slot[MTL_TEX_BINDINGS][MTL_TEX_VIEWS]; /* texture index of each view, or MTL_TEX_NONE */
    uint32_t n_tex_slots;               /* texture indices the stage uses, 0 to n - 1 */
    uint32_t sampler_push_base;         /* byte offset of the first u16 heap index */
    int dynamic_floats;                 /* reads cF at CBV 4, register r at r * 16 */
    uint32_t bool_mask;                 /* boolean constants the shader reads */
    int pos_regs;                       /* vertex shader: mtl_stereo_scan_pos, the constants oPos comes from */
} mtl_sc_info;

/* The info's tex_slot and n_tex_slots from its sampler_mask and view:
   the one definition of the layout above, which the MSL and the draw
   path both follow. */
void mtl_tex_layout(mtl_sc_info *info);

/* Analysis only: cheap, done when the engine creates the shader. The
   handle keeps a copy of the bytecode. NULL with a message on failure. */
mtl_sc *mtl_sc_new(const void *code, uint32_t bytes, char *err, size_t errlen);
void mtl_sc_free(mtl_sc *s);
const mtl_sc_info *mtl_sc_get_info(const mtl_sc *s);
const char *mtl_sc_hash(const mtl_sc *s); /* 32 hex digits of SHA-256 over the bytecode */

/* MSL for the shader, from the cache directory when one is given and it
   holds this hash, else translated now and written there. The entry
   point is main0. The string belongs to the handle. *cached says which. */
const char *mtl_sc_msl(mtl_sc *s, const char *cache_dir, int *cached, char *err, size_t errlen);

/* The chain's version tag: the MSL cache's subdirectory, which anything
   derived from the MSL is filed under too. */
const char *mtl_sc_chain_tag(void);

/* The cached MSL for a bytecode hash, as mtl_sc_msl would read it, with
   no shader handle: the pipeline warm thread starts from recorded hashes
   alone. A malloc'd string, or NULL when the cache does not hold it. */
char *mtl_sc_cached_msl(const char *cache_dir, const char *hash);

/* Constant upload, by DXVK's own copy routine. f, i and b are the API
   registers (float4[256 or 224], int4[16], bool[16]). changed_floats is
   one past the highest float register the engine has set; it sizes the
   dynamically indexed buffer. Fills the static CBV (floats, then ints)
   and, for a dynamically indexed vertex shader, cF; returns the bytes
   used in each and the cFSize push value. Each buffer needs room for the
   size mtl_sc_constant_sizes gives (at most MTL_SC_MAX_CONST bytes), and
   is zeroed over that size here, so callers can pack straight into
   upload memory. */
#define MTL_SC_MAX_CONST (257 * 16 + 16 * 16 + 64)
void mtl_sc_constant_sizes(const mtl_sc *s, uint32_t changed_floats, uint32_t *static_bytes, uint32_t *dyn_bytes);
void mtl_sc_constants(const mtl_sc *s, const float *f, const int32_t *i, const uint32_t *b, uint32_t changed_floats,
                      void *static_buf, uint32_t *static_bytes, void *dyn_buf, uint32_t *dyn_bytes, uint32_t *cf_size);

/* DXVK's fixed-function ubershaders (SPIR-V built from its GLSL) through
   SPIRV-Cross, with the same binding map as the programmable stages:
   CBVs at MTL_BUF_CBV + DXVK's CbvIndex, spec data, push data, the
   sampler heap, and the eight stage textures at the texture indices a
   pixel shader's samplers 0 to 7 use. which: 0 vertex, 1 fragment.
   Safe from any thread. */
const char *mtl_sc_ff_msl(int which, char *err, size_t errlen);

/* The fixed-function vertex shader's input signature (DXVK's g_ffIsgn):
   attribute i reads usage[i] with index index[i]. */
void mtl_ff_input_signature(mtl_sc_info *out);

/* Vertex declaration facts the fixed-function shader reads, as DXVK's
   D3D9VertexDecl::Classify derives them. */
#define MTL_DECL_POSITIONT 0x01u
#define MTL_DECL_COLOR0 0x02u
#define MTL_DECL_COLOR1 0x04u
#define MTL_DECL_PSIZE 0x08u
#define MTL_DECL_FOG 0x10u
#define MTL_DECL_BLENDWEIGHT 0x20u
#define MTL_DECL_BLENDINDICES 0x40u

/* Transform slots, in DXVK's GetTransformIndex order. */
#define MTL_XF_VIEW 0
#define MTL_XF_PROJ 1
#define MTL_XF_TEX0 2
#define MTL_XF_WORLD0 10
#define MTL_XF_N 18 /* the world matrices past 7 serve only software vertex blending */

/* What the fixed-function vertex data is built from: the API state as
   the engine set it, with D3D9's own numbering for render states and
   texture stage states. lights are the D3DLIGHT9s that are enabled, in
   index order, as DXVK walks its light list. */
typedef struct {
    const float *xf;          /* MTL_XF_N matrices of 16 floats, D3DMATRIX order */
    const void *material;     /* D3DMATERIAL9, 68 bytes */
    const void *const *lights; /* D3DLIGHT9, 104 bytes each */
    uint32_t nlights;
    const uint32_t *rs;       /* 256 render states */
    const uint32_t *tss;      /* 8 stages of 33 states */
    uint32_t decl_flags, tc_mask;
    float vp_x, vp_y, vp_w, vp_h;
    int ztest;
} mtl_ff_vs_in;

/* DXVK's UpdateFixedFunctionVS: fills the CBV the ubershader reads as
   ShaderData (MTL_FF_VS_BYTES) and, for vertex blending, the eight
   world-view matrices it reads as VertexBlendData. Returns the blend
   mode (0 none, 1 matrices, 2 tween). */
#define MTL_FF_VS_BYTES 1888u
#define MTL_FF_BLEND_BYTES (8u * 64u)
uint32_t mtl_ff_vs_data(const mtl_ff_vs_in *in, void *out, void *blend_out);

/* DXVK's ComputeWNearFar test: whether fixed-function fog reads z (an
   orthographic projection) rather than w. */
int mtl_ff_fog_use_z(const float *proj);

#ifdef __cplusplus
}
#endif

#endif
