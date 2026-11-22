/*
 * Texture formats a GPU may not sample, converted on upload to RGBA8
 * (bytes R, G, B, A). The engine's own memory is never touched: a locked
 * DXT level still holds the compressed blocks the game wrote, and only
 * the copy that goes to the GPU is decoded (gpu.m).
 *
 * DXT1 to DXT5 are BC1 to BC3 (DXT2 and DXT4 are the premultiplied
 * variants, which D3D9 samples exactly as DXT3 and DXT5). The 16-bit
 * packed formats are the ones Apple's simulator GPU refuses.
 */
#ifndef MTL_TEXCONV_H
#define MTL_TEXCONV_H

#include <stdint.h>

/* Whether mtl_conv_level knows the D3D9 format. */
int mtl_conv_known(uint32_t fmt);

/* One w x h level of a D3D9 format, as the engine laid it out (pitch
   bytes per row, of blocks for DXT), into w * h * 4 bytes of RGBA8. */
void mtl_conv_level(uint32_t fmt, const uint8_t *src, uint32_t pitch, uint32_t w, uint32_t h, uint8_t *dst);

#endif
