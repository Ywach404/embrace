/*
 * BC1 to BC3 decoding and 16-bit widening, for GPUs that sample neither
 * (texconv.h). Written from the block layout in Microsoft's "Texture
 * Block Compression in Direct3D 11" and the D3D9 format list: two RGB565
 * endpoints and 2-bit indices per 4x4 block, with BC2 adding 4-bit
 * explicit alpha and BC3 two alpha endpoints and 3-bit indices.
 */
#include "texconv.h"

#include <string.h>

#define FMT_DXT1 0x31545844u
#define FMT_DXT2 0x32545844u
#define FMT_DXT3 0x33545844u
#define FMT_DXT4 0x34545844u
#define FMT_DXT5 0x35545844u
#define FMT_R5G6B5 23u
#define FMT_X1R5G5B5 24u
#define FMT_A1R5G5B5 25u
#define FMT_A4R4G4B4 26u

int mtl_conv_known(uint32_t fmt)
{
    switch (fmt) {
    case FMT_DXT1: case FMT_DXT2: case FMT_DXT3: case FMT_DXT4: case FMT_DXT5:
    case FMT_R5G6B5: case FMT_X1R5G5B5: case FMT_A1R5G5B5: case FMT_A4R4G4B4:
        return 1;
    default:
        return 0;
    }
}

static uint16_t le16(const uint8_t *p) { return (uint16_t)(p[0] | p[1] << 8); }

/* 5 and 6 bit channels to 8 by bit replication, so 0 and full stay exact. */
static uint8_t x5(uint32_t v) { return (uint8_t)(v << 3 | v >> 2); }
static uint8_t x6(uint32_t v) { return (uint8_t)(v << 2 | v >> 4); }

static void rgb565(uint16_t c, uint8_t out[4])
{
    out[0] = x5(c >> 11 & 31);
    out[1] = x6(c >> 5 & 63);
    out[2] = x5(c & 31);
    out[3] = 255;
}

/* The colour half of a block into 16 RGBA texels. Only DXT1 has the
   three-colour mode (c0 <= c1, index 3 transparent black); the colour
   block of DXT2 to DXT5 always interpolates four. */
static void color_block(const uint8_t *b, int dxt1, uint8_t px[16][4])
{
    uint16_t c0 = le16(b), c1 = le16(b + 2);
    uint8_t pal[4][4];
    rgb565(c0, pal[0]);
    rgb565(c1, pal[1]);
    if (dxt1 && c0 <= c1) {
        for (int i = 0; i < 3; i++) pal[2][i] = (uint8_t)((pal[0][i] + pal[1][i]) / 2);
        pal[2][3] = 255;
        memset(pal[3], 0, 4);
    } else {
        for (int i = 0; i < 3; i++) {
            pal[2][i] = (uint8_t)((2 * pal[0][i] + pal[1][i]) / 3);
            pal[3][i] = (uint8_t)((pal[0][i] + 2 * pal[1][i]) / 3);
        }
        pal[2][3] = pal[3][3] = 255;
    }
    uint32_t idx = (uint32_t)b[4] | (uint32_t)b[5] << 8 | (uint32_t)b[6] << 16 | (uint32_t)b[7] << 24;
    for (int t = 0; t < 16; t++) memcpy(px[t], pal[idx >> (2 * t) & 3], 4);
}

/* BC2: 4 bits of alpha per texel, row by row, low nibble first. */
static void explicit_alpha(const uint8_t *b, uint8_t px[16][4])
{
    for (int t = 0; t < 16; t++) {
        uint32_t a = b[t / 2] >> (4 * (t & 1)) & 15;
        px[t][3] = (uint8_t)(a * 17);
    }
}

/* BC3: two endpoints, then 48 bits of 3-bit indices. a0 > a1 gives six
   interpolated values; otherwise four, with 0 and 255 as codes 6 and 7. */
static void interp_alpha(const uint8_t *b, uint8_t px[16][4])
{
    uint32_t a0 = b[0], a1 = b[1];
    uint8_t pal[8];
    pal[0] = (uint8_t)a0;
    pal[1] = (uint8_t)a1;
    if (a0 > a1) {
        for (uint32_t i = 1; i <= 6; i++) pal[1 + i] = (uint8_t)(((7 - i) * a0 + i * a1) / 7);
    } else {
        for (uint32_t i = 1; i <= 4; i++) pal[1 + i] = (uint8_t)(((5 - i) * a0 + i * a1) / 5);
        pal[6] = 0;
        pal[7] = 255;
    }
    uint64_t bits = 0;
    for (int i = 0; i < 6; i++) bits |= (uint64_t)b[2 + i] << (8 * i);
    for (int t = 0; t < 16; t++) px[t][3] = pal[bits >> (3 * t) & 7];
}

static void dxt_level(uint32_t fmt, const uint8_t *src, uint32_t pitch, uint32_t w, uint32_t h, uint8_t *dst)
{
    uint32_t bytes = fmt == FMT_DXT1 ? 8 : 16;
    uint8_t px[16][4];
    for (uint32_t by = 0; by < (h + 3) / 4; by++)
        for (uint32_t bx = 0; bx < (w + 3) / 4; bx++) {
            const uint8_t *b = src + (size_t)by * pitch + (size_t)bx * bytes;
            if (fmt == FMT_DXT1) {
                color_block(b, 1, px);
            } else {
                color_block(b + 8, 0, px);
                if (fmt == FMT_DXT2 || fmt == FMT_DXT3) explicit_alpha(b, px);
                else interp_alpha(b, px);
            }
            /* A level under four texels still stores a whole block: crop. */
            for (uint32_t y = 0; y < 4 && by * 4 + y < h; y++)
                for (uint32_t x = 0; x < 4 && bx * 4 + x < w; x++)
                    memcpy(dst + ((size_t)(by * 4 + y) * w + bx * 4 + x) * 4, px[y * 4 + x], 4);
        }
}

static void packed_level(uint32_t fmt, const uint8_t *src, uint32_t pitch, uint32_t w, uint32_t h, uint8_t *dst)
{
    for (uint32_t y = 0; y < h; y++) {
        const uint8_t *row = src + (size_t)y * pitch;
        uint8_t *o = dst + (size_t)y * w * 4;
        for (uint32_t x = 0; x < w; x++, o += 4) {
            uint32_t v = le16(row + 2 * x);
            switch (fmt) {
            case FMT_R5G6B5:
                rgb565((uint16_t)v, o);
                break;
            case FMT_X1R5G5B5:
            case FMT_A1R5G5B5:
                o[0] = x5(v >> 10 & 31);
                o[1] = x5(v >> 5 & 31);
                o[2] = x5(v & 31);
                o[3] = fmt == FMT_X1R5G5B5 || (v & 0x8000u) ? 255 : 0;
                break;
            default: /* A4R4G4B4 */
                o[0] = (uint8_t)((v >> 8 & 15) * 17);
                o[1] = (uint8_t)((v >> 4 & 15) * 17);
                o[2] = (uint8_t)((v & 15) * 17);
                o[3] = (uint8_t)((v >> 12 & 15) * 17);
                break;
            }
        }
    }
}

void mtl_conv_level(uint32_t fmt, const uint8_t *src, uint32_t pitch, uint32_t w, uint32_t h, uint8_t *dst)
{
    if (fmt == FMT_DXT1 || fmt == FMT_DXT2 || fmt == FMT_DXT3 || fmt == FMT_DXT4 || fmt == FMT_DXT5)
        dxt_level(fmt, src, pitch, w, h, dst);
    else if (mtl_conv_known(fmt))
        packed_level(fmt, src, pitch, w, h, dst);
    else
        memset(dst, 0, (size_t)w * h * 4);
}
