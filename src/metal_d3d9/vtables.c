/*
 * The vtables. Every slot starts as a stop function that names its method
 * and ends the run through the runtime's rt_stop, the way the null device
 * stops on a slot it has no function for. The implementation files then
 * install what they implement (mtl_fill_*). A slot nobody expected to be
 * called therefore fails loudly with its own name instead of jumping
 * through a null pointer.
 */
#include "mtl_d3d9.h"

#include <pthread.h>
#include <string.h>

#define MTL_STOP_FN(i, m, n) \
    static void stop_##i##_##m(void) { mtl_unexpected(#i "::" #m); }

MTL_SLOTS_IDirect3D9(MTL_STOP_FN)
MTL_SLOTS_IDirect3DDevice9(MTL_STOP_FN)
MTL_SLOTS_IDirect3DSurface9(MTL_STOP_FN)
MTL_SLOTS_IDirect3DTexture9(MTL_STOP_FN)
MTL_SLOTS_IDirect3DCubeTexture9(MTL_STOP_FN)
MTL_SLOTS_IDirect3DVertexBuffer9(MTL_STOP_FN)
MTL_SLOTS_IDirect3DIndexBuffer9(MTL_STOP_FN)
MTL_SLOTS_IDirect3DVertexDeclaration9(MTL_STOP_FN)
MTL_SLOTS_IDirect3DVertexShader9(MTL_STOP_FN)
MTL_SLOTS_IDirect3DPixelShader9(MTL_STOP_FN)
MTL_SLOTS_IDirect3DQuery9(MTL_STOP_FN)

void *mtl_vt_d3d9[MTL_NSLOTS_IDirect3D9];
void *mtl_vt_device[MTL_NSLOTS_IDirect3DDevice9];
void *mtl_vt_surface[MTL_NSLOTS_IDirect3DSurface9];
void *mtl_vt_texture[MTL_NSLOTS_IDirect3DTexture9];
void *mtl_vt_cube[MTL_NSLOTS_IDirect3DCubeTexture9];
void *mtl_vt_vb[MTL_NSLOTS_IDirect3DVertexBuffer9];
void *mtl_vt_ib[MTL_NSLOTS_IDirect3DIndexBuffer9];
void *mtl_vt_decl[MTL_NSLOTS_IDirect3DVertexDeclaration9];
void *mtl_vt_vs[MTL_NSLOTS_IDirect3DVertexShader9];
void *mtl_vt_ps[MTL_NSLOTS_IDirect3DPixelShader9];
void *mtl_vt_query[MTL_NSLOTS_IDirect3DQuery9];

/* The slot counts the bridge's tables had when this was written. A porter
   change that moves slots should fail the build here, not misroute calls. */
_Static_assert(MTL_NSLOTS_IDirect3D9 == 17, "IDirect3D9 has 17 slots");
_Static_assert(MTL_NSLOTS_IDirect3DDevice9 == 119, "IDirect3DDevice9 has 119 slots");
_Static_assert(MTL_NSLOTS_IDirect3DSurface9 == 17, "IDirect3DSurface9 has 17 slots");
_Static_assert(MTL_NSLOTS_IDirect3DTexture9 == 22, "IDirect3DTexture9 has 22 slots");
_Static_assert(MTL_NSLOTS_IDirect3DCubeTexture9 == 22, "IDirect3DCubeTexture9 has 22 slots");
_Static_assert(MTL_NSLOTS_IDirect3DVertexBuffer9 == 14, "IDirect3DVertexBuffer9 has 14 slots");
_Static_assert(MTL_NSLOTS_IDirect3DIndexBuffer9 == 14, "IDirect3DIndexBuffer9 has 14 slots");
_Static_assert(MTL_NSLOTS_IDirect3DVertexDeclaration9 == 5, "IDirect3DVertexDeclaration9 has 5 slots");
_Static_assert(MTL_NSLOTS_IDirect3DVertexShader9 == 5, "IDirect3DVertexShader9 has 5 slots");
_Static_assert(MTL_NSLOTS_IDirect3DPixelShader9 == 5, "IDirect3DPixelShader9 has 5 slots");
_Static_assert(MTL_NSLOTS_IDirect3DQuery9 == 8, "IDirect3DQuery9 has 8 slots");

#define MTL_STOP_ENTRY_D3D9(i, m, n) mtl_vt_d3d9[n] = (void *)stop_##i##_##m;
#define MTL_STOP_ENTRY_DEVICE(i, m, n) mtl_vt_device[n] = (void *)stop_##i##_##m;
#define MTL_STOP_ENTRY_SURFACE(i, m, n) mtl_vt_surface[n] = (void *)stop_##i##_##m;
#define MTL_STOP_ENTRY_TEXTURE(i, m, n) mtl_vt_texture[n] = (void *)stop_##i##_##m;
#define MTL_STOP_ENTRY_CUBE(i, m, n) mtl_vt_cube[n] = (void *)stop_##i##_##m;
#define MTL_STOP_ENTRY_VB(i, m, n) mtl_vt_vb[n] = (void *)stop_##i##_##m;
#define MTL_STOP_ENTRY_IB(i, m, n) mtl_vt_ib[n] = (void *)stop_##i##_##m;
#define MTL_STOP_ENTRY_DECL(i, m, n) mtl_vt_decl[n] = (void *)stop_##i##_##m;
#define MTL_STOP_ENTRY_VS(i, m, n) mtl_vt_vs[n] = (void *)stop_##i##_##m;
#define MTL_STOP_ENTRY_PS(i, m, n) mtl_vt_ps[n] = (void *)stop_##i##_##m;
#define MTL_STOP_ENTRY_QUERY(i, m, n) mtl_vt_query[n] = (void *)stop_##i##_##m;

static void build_once(void)
{
    MTL_SLOTS_IDirect3D9(MTL_STOP_ENTRY_D3D9)
    MTL_SLOTS_IDirect3DDevice9(MTL_STOP_ENTRY_DEVICE)
    MTL_SLOTS_IDirect3DSurface9(MTL_STOP_ENTRY_SURFACE)
    MTL_SLOTS_IDirect3DTexture9(MTL_STOP_ENTRY_TEXTURE)
    MTL_SLOTS_IDirect3DCubeTexture9(MTL_STOP_ENTRY_CUBE)
    MTL_SLOTS_IDirect3DVertexBuffer9(MTL_STOP_ENTRY_VB)
    MTL_SLOTS_IDirect3DIndexBuffer9(MTL_STOP_ENTRY_IB)
    MTL_SLOTS_IDirect3DVertexDeclaration9(MTL_STOP_ENTRY_DECL)
    MTL_SLOTS_IDirect3DVertexShader9(MTL_STOP_ENTRY_VS)
    MTL_SLOTS_IDirect3DPixelShader9(MTL_STOP_ENTRY_PS)
    MTL_SLOTS_IDirect3DQuery9(MTL_STOP_ENTRY_QUERY)
    mtl_fill_d3d9();
    mtl_fill_device();
    mtl_fill_resources();
}

void mtl_vtables_build(void)
{
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, build_once);
}

/* For the headless test: whether a slot holds an implementation or still
   its stop function, by interface and method name. */
#define MTL_IS_STOP(i, m, n) \
    if (!strcmp(iface, #i) && !strcmp(method, #m)) return vt[n] == (void *)stop_##i##_##m ? 0 : 1;

D9_EXPORT int metal_d3d9_slot_implemented(const char *iface, const char *method)
{
    mtl_vtables_build();
    void **vt;
#define TRY(list, table) \
    vt = table; \
    list(MTL_IS_STOP)
    TRY(MTL_SLOTS_IDirect3D9, mtl_vt_d3d9)
    TRY(MTL_SLOTS_IDirect3DDevice9, mtl_vt_device)
    TRY(MTL_SLOTS_IDirect3DSurface9, mtl_vt_surface)
    TRY(MTL_SLOTS_IDirect3DTexture9, mtl_vt_texture)
    TRY(MTL_SLOTS_IDirect3DCubeTexture9, mtl_vt_cube)
    TRY(MTL_SLOTS_IDirect3DVertexBuffer9, mtl_vt_vb)
    TRY(MTL_SLOTS_IDirect3DIndexBuffer9, mtl_vt_ib)
    TRY(MTL_SLOTS_IDirect3DVertexDeclaration9, mtl_vt_decl)
    TRY(MTL_SLOTS_IDirect3DVertexShader9, mtl_vt_vs)
    TRY(MTL_SLOTS_IDirect3DPixelShader9, mtl_vt_ps)
    TRY(MTL_SLOTS_IDirect3DQuery9, mtl_vt_query)
#undef TRY
    return -1;
}
