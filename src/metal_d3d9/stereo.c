/*
 * Stereo's pure part: the clip-space correction, the scan of a vertex
 * shader's position constants, and the classing of draws. See stereo.h.
 */
#include "stereo.h"

#include <math.h>
#include <string.h>

void mtl_stereo_identity(mtl_stereo_fix *out)
{
    memset(out, 0, sizeof *out);
    out->x[0] = 1.0f;
    out->y[0] = 1.0f;
    out->z[0] = 1.0f;
    out->w[0] = 1.0f;
}

void mtl_stereo_correction(const mtl_stereo_proj *game, const mtl_stereo_proj *eye, float eye_x, mtl_stereo_fix *out)
{
    mtl_stereo_identity(out);
    if (!(game->a != 0.0f) || !(game->b != 0.0f)) return;
    /* With eye equal to game these come out as exactly 1 and 0, so the
       identity case writes the game's own position bits. */
    out->x[0] = eye->a / game->a;
    out->x[1] = eye->cx - out->x[0] * game->cx;
    out->x[2] = -eye->a * eye_x;
    out->y[0] = eye->b / game->b;
    out->y[1] = eye->cy - out->y[0] * game->cy;
    out->y[2] = 0.0f;
}

void mtl_stereo_hud_correction(float screen_a, float eye_x, float hud_z, mtl_stereo_fix *out)
{
    mtl_stereo_identity(out);
    if (!(screen_a > 0.0f) || !(hud_z > 0.0f) || eye_x == 0.0f) return;
    /* Scaled by w so the shift is the same in NDC for any w the draw
       writes: the engine's 2D is w = 1, but a POSITIONT draw's is 1/rhw. */
    out->x[1] = -screen_a * eye_x / hud_z;
}

void mtl_stereo_correction3(const mtl_stereo_proj_full *game, const mtl_stereo_proj *eye, const float t[3],
                            mtl_stereo_fix *out)
{
    mtl_stereo_identity(out);
    if (!(game->a != 0.0f) || !(game->b != 0.0f)) return;
    /* As in mtl_stereo_correction, the identity case comes out as exactly
       1 and 0, so P_eye = P_game with t = 0 writes the game's own bits. */
    out->x[0] = eye->a / game->a;
    out->x[1] = eye->cx - out->x[0] * game->cx;
    out->x[2] = -(eye->a * t[0] + eye->cx * t[2]);
    out->y[0] = eye->b / game->b;
    out->y[1] = eye->cy - out->y[0] * game->cy;
    out->y[2] = -(eye->b * t[1] + eye->cy * t[2]);
    out->z[2] = -game->A * t[2];
    out->w[2] = -t[2];
}

void mtl_stereo_proj_from_tangents(const float tan[4], mtl_stereo_proj *out)
{
    memset(out, 0, sizeof *out);
    float w = tan[0] + tan[1], h = tan[2] + tan[3];
    if (!(w > 0.0f) || !(h > 0.0f)) return;
    out->a = 2.0f / w;
    out->cx = (tan[0] - tan[1]) / w;
    out->b = 2.0f / h;
    out->cy = (tan[2] - tan[3]) / h;
}

int mtl_stereo_world_point(const mtl_stereo_proj_full *game, float x, float y, float half, float w, float units_per_m,
                           float out[4])
{
    memset(out, 0, 4 * sizeof *out);
    if (!(game->a > 0.0f) || !(game->b > 0.0f) || !(w > 0.0f) || !(units_per_m > 0.0f) || !isfinite(x) ||
        !isfinite(y) || !isfinite(half) || !isfinite(w))
        return 0;
    out[0] = (x - game->cx) / game->a;
    out[1] = (y - game->cy) / game->b;
    out[2] = fabsf(half) / game->a;
    out[3] = w / units_per_m;
    return 1;
}

void mtl_stereo_panel_correction(const mtl_stereo_proj *eye, const float t[3], float hud_z, float px, float py,
                                 mtl_stereo_fix *out)
{
    mtl_stereo_identity(out);
    if (!(hud_z > 0.0f) || !(px > 0.0f) || !(py > 0.0f) || !(eye->a > 0.0f) || !(eye->b > 0.0f)) return;
    /* A point at game NDC (x, y) is on the panel at (px x, py y) hud_z
       from the head; the eye at t sees it at (px x - tx / hud_z) in
       tangent, ignoring tz, which only scales the panel by a few parts
       in a thousand. */
    out->x[0] = eye->a * px;
    out->x[1] = eye->cx - eye->a * t[0] / hud_z;
    out->y[0] = eye->b * py;
    out->y[1] = eye->cy - eye->b * t[1] / hud_z;
}

float mtl_stereo_screen_a(float aspect, float deg)
{
    if (!(aspect > 0.0f)) aspect = 4.0f / 3.0f;
    if (!(deg >= 10.0f && deg <= 170.0f)) deg = MTL_STEREO_NOMINAL_DEG;
    /* The height is the 4:3 view's, the width the aspect's. */
    double tan_y = 0.75 * tan((double)deg * M_PI / 360.0);
    return (float)(1.0 / (tan_y * (double)aspect));
}

void mtl_stereo_tally_add(mtl_stereo_tally *t, float a)
{
    if (!(a > 0.0f)) return;
    int free_i = -1, least = 0;
    for (int i = 0; i < MTL_STEREO_TALLY; i++) {
        /* The same projection gives the same bits; a tolerance only keeps
           a value read back through a different path together. */
        if (t->n[i] && fabsf(t->a[i] - a) <= 1e-4f * a) {
            t->n[i]++;
            return;
        }
        if (!t->n[i] && free_i < 0) free_i = i;
        if (t->n[i] < t->n[least]) least = i;
    }
    int i = free_i >= 0 ? free_i : least;
    t->a[i] = a;
    t->n[i] = 1;
}

float mtl_stereo_tally_take(mtl_stereo_tally *t, float fallback)
{
    int best = -1;
    for (int i = 0; i < MTL_STEREO_TALLY; i++)
        if (t->n[i] && (best < 0 || t->n[i] > t->n[best])) best = i;
    float a = best >= 0 ? t->a[best] : fallback;
    memset(t, 0, sizeof *t);
    return a;
}

/* D3D9 bytecode registers (d3d9types.h's D3DSHADER_PARAM_REGISTER_TYPE). */
#define REG_TEMP 0
#define REG_CONST 2
#define REG_RASTOUT 4
#define OP_MOV 1
#define OP_DP4 9
#define OP_M4X4 20
#define OP_DEFB 0x2F
#define OP_DEFI 0x30
#define OP_DEF 0x51
#define OP_COMMENT 0xFFFE
#define OP_END 0xFFFF
#define MAX_TEMPS 32

static int reg_type(uint32_t t) { return (int)(((t >> 28) & 7u) | ((t >> 8) & 0x18u)); }
static int reg_num(uint32_t t) { return (int)(t & 0x7FFu); }

int mtl_stereo_scan_pos(const uint32_t *tk, size_t n)
{
    if (!tk || n < 2 || (tk[0] >> 16) != 0xFFFEu) return 0;
    /* For each temporary, the constant register its x and its w came
       from by dp4; -1 when unknown or overwritten by anything else. */
    int tx[MAX_TEMPS], tw[MAX_TEMPS];
    for (int i = 0; i < MAX_TEMPS; i++) tx[i] = tw[i] = -1;
    int px = -1, pw = -1;
    size_t i = 1;
    while (i < n) {
        uint32_t t = tk[i];
        uint32_t op = t & 0xFFFFu;
        if (t == OP_END) break;
        if (op == OP_COMMENT) {
            i += 1 + ((t >> 16) & 0x7FFFu);
            continue;
        }
        if (op == OP_DEF || op == OP_DEFI) {
            i += 6;
            continue;
        }
        if (op == OP_DEFB) {
            i += 3;
            continue;
        }
        /* Parameter tokens all have bit 31 set; the next instruction's
           token does not. Relative addressing tokens are parameters too. */
        size_t j = i + 1;
        uint32_t ps[8];
        int np = 0;
        while (j < n && (tk[j] & 0x80000000u)) {
            if (np < 8) ps[np++] = tk[j];
            j++;
        }
        i = j;
        if (np < 2) continue;
        uint32_t dst = ps[0];
        int dtype = reg_type(dst), dnum = reg_num(dst);
        uint32_t mask = (dst >> 16) & 0xFu;
        int is_pos = dtype == REG_RASTOUT && dnum == 0;
        int is_tmp = dtype == REG_TEMP && dnum < MAX_TEMPS;
        if (!is_pos && !is_tmp) continue;
        int cx = -1, cw = -1;
        if ((op == OP_DP4 || op == OP_M4X4) && np >= 3) {
            int c = reg_type(ps[2]) == REG_CONST ? reg_num(ps[2]) : reg_type(ps[1]) == REG_CONST ? reg_num(ps[1]) : -1;
            if (c >= 0) {
                cx = c;
                cw = op == OP_M4X4 ? c + 3 : c;
            }
        } else if (op == OP_MOV && reg_type(ps[1]) == REG_TEMP && reg_num(ps[1]) < MAX_TEMPS) {
            cx = tx[reg_num(ps[1])];
            cw = tw[reg_num(ps[1])];
        }
        int *ox = is_pos ? &px : &tx[dnum], *ow = is_pos ? &pw : &tw[dnum];
        if (op == OP_M4X4) {
            if (mask & 1u) *ox = cx;
            if (mask & 8u) *ow = cw;
        } else {
            if (mask & 1u) *ox = op == OP_DP4 || op == OP_MOV ? cx : -1;
            if (mask & 8u) *ow = op == OP_DP4 || op == OP_MOV ? cw : -1;
        }
    }
    /* x is what the correction moves, so x decides. The refract and water
       shaders build oPos.w some other way than a dp4 with c11 or c7, yet
       are world materials: left alone they would float at infinity inside
       a world with depth. A w we did see must still agree. */
    if (px == MTL_POS_VIEWPROJ && (pw < 0 || pw == MTL_POS_VIEWPROJ + 3)) return MTL_POS_VIEWPROJ;
    if (px == MTL_POS_MVP && (pw < 0 || pw == MTL_POS_MVP + 3)) return MTL_POS_MVP;
    return 0;
}

#define REG_INPUT 1
#define REG_ADDR 3
#define REG_OUTPUT 6
#define OP_DCL 0x1F
#define OP_MOVA 0x2E

/* Flow control and predication: a position behind a branch is not one a
   straight run can give. */
static int eval_flow(uint32_t op)
{
    return (op >= 25 && op <= 30) || (op >= 38 && op <= 45) || op == 0x5E || op == 0x60;
}

int mtl_vs_eval_init(mtl_vs_eval *e, const uint32_t *tk, size_t n, const float *vsf)
{
    memset(e, 0, sizeof *e);
    if (!tk || n < 2 || (tk[0] >> 16) != 0xFFFEu || !vsf) return 0;
    e->tk = tk;
    e->n = n;
    e->major = (int)((tk[0] >> 8) & 0xFFu);
    e->pos_type = REG_RASTOUT;
    e->pos_num = 0;
    memcpy(e->c, vsf, sizeof e->c);
    size_t i = 1;
    while (i < n) {
        uint32_t t = tk[i], op = t & 0xFFFFu;
        if (t == OP_END) break;
        if (op == OP_COMMENT) {
            i += 1 + ((t >> 16) & 0x7FFFu);
            continue;
        }
        if (op == OP_DEF) {
            /* Its literals are raw floats, so a negative one has bit 31
               set: the length is fixed, not read from the tokens. */
            if (i + 5 >= n) return 0;
            if (reg_type(tk[i + 1]) == REG_CONST && reg_num(tk[i + 1]) < 256) memcpy(e->c[reg_num(tk[i + 1])], &tk[i + 2], 16);
            i += 6;
            continue;
        }
        if (op == OP_DEFI) {
            i += 6;
            continue;
        }
        if (op == OP_DEFB) {
            i += 3;
            continue;
        }
        if (eval_flow(op)) return 0;
        size_t j = i + 1;
        while (j < n && (tk[j] & 0x80000000u)) j++;
        if (op == OP_DCL && j - i >= 3) {
            uint32_t u = tk[i + 1], r = tk[i + 2];
            int ty = reg_type(r), num = reg_num(r);
            if (ty == REG_INPUT && num < MTL_VS_EVAL_INPUTS) {
                e->in_mask |= 1u << num;
                e->in_usage[num] = (uint8_t)(u & 0x1Fu);
                e->in_index[num] = (uint8_t)((u >> 16) & 0xFu);
            } else if (ty == REG_OUTPUT && (u & 0x1Fu) == 0 && ((u >> 16) & 0xFu) == 0) {
                e->pos_type = REG_OUTPUT;
                e->pos_num = num;
            }
        }
        i = j;
    }
    return 1;
}

typedef struct {
    float r[MAX_TEMPS][4], a[4], pos[4];
} eval_regs;

/* A source operand at register offset k (the matrix ops' later rows),
   swizzled and modified; NaN where it cannot be read. */
static void eval_src(const mtl_vs_eval *e, const eval_regs *g, const float (*v)[4], uint32_t tok, uint32_t rel, int k,
                     float out[4])
{
    float q[4] = {NAN, NAN, NAN, NAN};
    int ty = reg_type(tok), num = reg_num(tok) + k;
    if (ty == REG_TEMP && num < MAX_TEMPS) memcpy(q, g->r[num], sizeof q);
    else if (ty == REG_INPUT && num < MTL_VS_EVAL_INPUTS) memcpy(q, v[num], sizeof q);
    else if (ty == REG_CONST) {
        /* vs_1_1 indexes by a0.x with no token for it; later versions name
           the address register's component in one. */
        if (tok & 0x2000u) {
            float at = e->major >= 2 ? g->a[(rel >> 16) & 3u] : g->a[0];
            num = isfinite(at) ? num + (int)at : -1;
        }
        if (num >= 0 && num < 256) memcpy(q, e->c[num], sizeof q);
    } else if (ty == REG_ADDR && num == 0) memcpy(q, g->a, sizeof q);
    uint32_t sw = (tok >> 16) & 0xFFu;
    for (int c = 0; c < 4; c++) out[c] = q[(sw >> (2 * c)) & 3u];
    switch ((tok >> 24) & 0xFu) {
    case 0: break;
    case 1: for (int c = 0; c < 4; c++) out[c] = -out[c]; break;
    case 11: for (int c = 0; c < 4; c++) out[c] = fabsf(out[c]); break;
    case 12: for (int c = 0; c < 4; c++) out[c] = -fabsf(out[c]); break;
    default: for (int c = 0; c < 4; c++) out[c] = NAN; break;
    }
}

/* The component a scalar op reads: vs_1_1 takes w when no replicate
   swizzle is given, and a replicate swizzle makes them all the same. */
static float eval_scalar(uint32_t tok, const float s[4]) { return ((tok >> 16) & 0xFFu) == 0xE4u ? s[3] : s[0]; }

static float dot(const float *a, const float *b, int n)
{
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

int mtl_vs_eval_pos(const mtl_vs_eval *e, const float (*v)[4], float out[4])
{
    eval_regs g;
    memset(&g, 0, sizeof g);
    for (int c = 0; c < 4; c++) g.pos[c] = g.a[c] = NAN;
    size_t i = 1, n = e->n;
    const uint32_t *tk = e->tk;
    while (i < n) {
        uint32_t t = tk[i], op = t & 0xFFFFu;
        if (t == OP_END) break;
        if (op == OP_COMMENT) {
            i += 1 + ((t >> 16) & 0x7FFFu);
            continue;
        }
        if (op == OP_DEF || op == OP_DEFI) {
            i += 6;
            continue;
        }
        if (op == OP_DEFB) {
            i += 3;
            continue;
        }
        size_t j = i + 1;
        while (j < n && (tk[j] & 0x80000000u)) j++;
        const uint32_t *p = tk + i + 1;
        size_t np = j - i - 1;
        i = j;
        if (op == OP_DCL || op == 0 || np < 2) continue;
        /* The sources, each with its address token where it has one. */
        uint32_t src[3] = {0}, rel[3] = {0};
        int ns = 0;
        for (size_t q = 1; q < np && ns < 3; ns++) {
            src[ns] = p[q++];
            if (e->major >= 2 && (src[ns] & 0x2000u) && q < np) rel[ns] = p[q++];
        }
        float s[3][4], r[4] = {NAN, NAN, NAN, NAN};
        for (int k = 0; k < ns; k++) eval_src(e, &g, v, src[k], rel[k], 0, s[k]);
        uint32_t dst = p[0];
        int rows = 0, cols = 4;
        switch (op) {
        case 1:  /* mov: into a0 it floors, as vs_1_1 defines it */
            for (int c = 0; c < 4; c++) r[c] = reg_type(dst) == REG_ADDR ? floorf(s[0][c]) : s[0][c];
            break;
        case OP_MOVA: for (int c = 0; c < 4; c++) r[c] = rintf(s[0][c]); break;
        case 2: for (int c = 0; c < 4; c++) r[c] = s[0][c] + s[1][c]; break;
        case 3: for (int c = 0; c < 4; c++) r[c] = s[0][c] - s[1][c]; break;
        case 4: for (int c = 0; c < 4; c++) r[c] = s[0][c] * s[1][c] + s[2][c]; break;
        case 5: for (int c = 0; c < 4; c++) r[c] = s[0][c] * s[1][c]; break;
        case 6: r[0] = r[1] = r[2] = r[3] = 1.0f / eval_scalar(src[0], s[0]); break;
        case 7: r[0] = r[1] = r[2] = r[3] = 1.0f / sqrtf(fabsf(eval_scalar(src[0], s[0]))); break;
        case 8: r[0] = r[1] = r[2] = r[3] = dot(s[0], s[1], 3); break;
        case 9: r[0] = r[1] = r[2] = r[3] = dot(s[0], s[1], 4); break;
        /* NaN is how an unknown travels, and fminf or a compare would
           turn it into a number, so these test for it first. */
        case 10: for (int c = 0; c < 4; c++) r[c] = isnan(s[0][c]) || isnan(s[1][c]) ? NAN : fminf(s[0][c], s[1][c]); break;
        case 11: for (int c = 0; c < 4; c++) r[c] = isnan(s[0][c]) || isnan(s[1][c]) ? NAN : fmaxf(s[0][c], s[1][c]); break;
        case 12: for (int c = 0; c < 4; c++) r[c] = isnan(s[0][c]) || isnan(s[1][c]) ? NAN : s[0][c] < s[1][c] ? 1.0f : 0.0f; break;
        case 13: for (int c = 0; c < 4; c++) r[c] = isnan(s[0][c]) || isnan(s[1][c]) ? NAN : s[0][c] >= s[1][c] ? 1.0f : 0.0f; break;
        case 14: r[0] = r[1] = r[2] = r[3] = exp2f(eval_scalar(src[0], s[0])); break;
        case 15: r[0] = r[1] = r[2] = r[3] = log2f(fabsf(eval_scalar(src[0], s[0]))); break;
        case 18: for (int c = 0; c < 4; c++) r[c] = s[0][c] * (s[1][c] - s[2][c]) + s[2][c]; break;
        case 19: for (int c = 0; c < 4; c++) r[c] = s[0][c] - floorf(s[0][c]); break;
        case 20: rows = 4; break;
        case 21: rows = 3; break;
        case 22: rows = 4, cols = 3; break;
        case 23: rows = 3, cols = 3; break;
        case 24: rows = 2, cols = 3; break;
        case 0x20: r[0] = r[1] = r[2] = r[3] = powf(fabsf(eval_scalar(src[0], s[0])), eval_scalar(src[1], s[1])); break;
        case 0x23: for (int c = 0; c < 4; c++) r[c] = fabsf(s[0][c]); break;
        case 0x24: {
            float l = 1.0f / sqrtf(dot(s[0], s[0], 3));
            for (int c = 0; c < 4; c++) r[c] = s[0][c] * l;
            break;
        }
        case 0x4E: /* expp: vs_2_0's is exp; vs_1_1's packs four values */
            if (e->major >= 2) r[0] = r[1] = r[2] = r[3] = exp2f(eval_scalar(src[0], s[0]));
            break;
        default: break; /* not known: what it writes is unknown */
        }
        /* The matrix ops: one dp per row, the rows the registers after the
           second source's. */
        for (int k = 0; k < rows; k++) {
            float m[4];
            eval_src(e, &g, v, src[1], rel[1], k, m);
            r[k] = dot(s[0], m, cols);
        }
        uint32_t mask = (dst >> 16) & 0xFu;
        int sat = (dst >> 20) & 1u, ty = reg_type(dst), num = reg_num(dst);
        float *to = NULL;
        if (ty == REG_TEMP && num < MAX_TEMPS) to = g.r[num];
        else if (ty == REG_ADDR && num == 0) to = g.a;
        else if (ty == e->pos_type && num == e->pos_num) to = g.pos;
        if (!to) continue;
        for (int c = 0; c < 4; c++)
            if (mask & (1u << c)) to[c] = sat && !isnan(r[c]) ? fminf(fmaxf(r[c], 0.0f), 1.0f) : r[c];
    }
    for (int c = 0; c < 4; c++)
        if (!isfinite(g.pos[c])) return 0;
    memcpy(out, g.pos, sizeof g.pos);
    return 1;
}

int mtl_stereo_row_is_perspective(const float w[4])
{
    float len = sqrtf(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
    return len > 1e-6f && isfinite(len);
}

float mtl_stereo_a_from_rows(const float x[4], const float w[4])
{
    if (!mtl_stereo_row_is_perspective(w)) return 0.0f;
    float wl = sqrtf(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
    float f[3] = {w[0] / wl, w[1] / wl, w[2] / wl};
    /* What is left of the x row once its forward part (cx) is taken out
       is a times the camera's right, a unit vector. */
    float d = x[0] * f[0] + x[1] * f[1] + x[2] * f[2];
    float r[3] = {x[0] - d * f[0], x[1] - d * f[1], x[2] - d * f[2]};
    float a = sqrtf(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    return isfinite(a) && a > 1e-6f ? a : 0.0f;
}

float mtl_stereo_a_from_d3d(const float m[16])
{
    /* D3DMATRIX order, row vectors: clip w = x _14 + y _24 + z _34 + _44,
       so _34 (m[11]) is non-zero exactly for a perspective. */
    if (!(fabsf(m[11]) > 1e-6f) || !(fabsf(m[0]) > 1e-6f)) return 0.0f;
    return fabsf(m[0]);
}

static float dot3(const float a[4], const float b[4]) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

int mtl_stereo_proj_from_rows(const float x[4], const float y[4], const float z[4], const float w[4],
                              mtl_stereo_proj_full *out)
{
    memset(out, 0, sizeof *out);
    float a = mtl_stereo_a_from_rows(x, w);
    if (!(a > 0.0f)) return 0;
    /* Each row is s times its term along the camera's right (or up) plus
       its forward part along the forward, and the w row is s times the
       forward, so the forward part is the row along w over |w|^2 and the
       rest is over |w|, whatever scale s the rows carry. */
    float ww = dot3(w, w), wl = sqrtf(ww);
    out->a = a / wl;
    out->cx = dot3(x, w) / ww;
    out->b = mtl_stereo_a_from_rows(y, w) / wl;
    out->cy = dot3(y, w) / ww;
    out->A = dot3(z, w) / ww;
    return 1;
}

int mtl_stereo_proj_from_d3d(const float m[16], mtl_stereo_proj_full *out)
{
    memset(out, 0, sizeof *out);
    float a = mtl_stereo_a_from_d3d(m);
    if (!(a > 0.0f)) return 0;
    /* Row vectors: x_c = x _11 + z _31, y_c = y _22 + z _32, z_c = z _33
       + _43, w_c = z _34. The game's sprites come with _22 negative and
       the view's y turned over to match, so clip y is still up: a and b
       are magnitudes, as the row path reads them, or the correction turns
       those draws (and the model view projection draws that borrow this
       projection) upside down. */
    out->a = fabsf(m[0] / m[11]);
    out->cx = m[8] / m[11];
    out->b = fabsf(m[5] / m[11]);
    out->cy = m[9] / m[11];
    out->A = m[10] / m[11];
    return 1;
}

/* Solves r0 . p = b0, r1 . p = b1, r2 . p = b2 by Cramer's rule. */
static int solve3(const float r0[3], const float r1[3], const float r2[3], const float b[3], float out[3])
{
    double m[3][3] = {{r0[0], r0[1], r0[2]}, {r1[0], r1[1], r1[2]}, {r2[0], r2[1], r2[2]}};
    double det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                 m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    double scale = 0.0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) scale = fmax(scale, fabs(m[i][j]));
    if (!(scale > 0.0) || !(fabs(det) > 1e-9 * scale * scale * scale)) return 0;
    for (int c = 0; c < 3; c++) {
        double k[3][3];
        memcpy(k, m, sizeof k);
        for (int r = 0; r < 3; r++) k[r][c] = b[r];
        double dc = k[0][0] * (k[1][1] * k[2][2] - k[1][2] * k[2][1]) - k[0][1] * (k[1][0] * k[2][2] - k[1][2] * k[2][0]) +
                    k[0][2] * (k[1][0] * k[2][1] - k[1][1] * k[2][0]);
        out[c] = (float)(dc / det);
    }
    return isfinite(out[0]) && isfinite(out[1]) && isfinite(out[2]);
}

int mtl_stereo_camera_from_rows(const float x[4], const float y[4], const float w[4], float out[3])
{
    /* The camera is where clip x, y and w are all 0. */
    const float b[3] = {-x[3], -y[3], -w[3]};
    return solve3(x, y, w, b, out);
}

int mtl_stereo_camera_from_view(const float v[16], float out[3])
{
    /* Row vectors: view = p R + T, R the upper 3x3 and T row 4; the
       camera is where view is 0, p R = -T, so R's columns dot p. */
    const float c0[3] = {v[0], v[4], v[8]}, c1[3] = {v[1], v[5], v[9]}, c2[3] = {v[2], v[6], v[10]};
    const float b[3] = {-v[12], -v[13], -v[14]};
    return solve3(c0, c1, c2, b, out);
}

int mtl_stereo_draw_camera(int ff_vs, const float *vsf, const float view[16], float out[3])
{
    /* A shader draw's camera is in c8 to c11, the view projection the
       engine keeps set beside whatever positions the vertices: a draw
       through the model view projection in c4 (props, decals, the brushes
       such a shader draws) is still seen from the current view, and only
       c8 says which one. Borrowing the camera of the last draw that used
       c8 put the world's first c4 draws after the 3D sky (la_hub_1,
       2026-11-20) on the sky's camera. */
    if (ff_vs) return view ? mtl_stereo_camera_from_view(view, out) : 0;
    if (!vsf) return 0;
    return mtl_stereo_camera_from_rows(vsf + 4 * MTL_POS_VIEWPROJ, vsf + 4 * (MTL_POS_VIEWPROJ + 1),
                                       vsf + 4 * (MTL_POS_VIEWPROJ + 3), out);
}

static int classify(const mtl_stereo_draw_in *in, float *a, float *seen_a, mtl_stereo_proj_full *p,
                    mtl_stereo_proj_full *seen);

int mtl_stereo_classify_proj(const mtl_stereo_draw_in *in, float *a, float *seen_a, mtl_stereo_proj_full *p,
                             mtl_stereo_proj_full *seen)
{
    int c = classify(in, a, seen_a, p, seen);
    /* A flat pass that reads a render target is drawing over this eye's
       image from a copy of it (a blur, a fade of the scene): it already
       has the eye's disparity and must line up with it, not the HUD's. */
    if ((c == MTL_STEREO_FLAT || c == MTL_STEREO_PRETRANSFORMED) && in->samples_rt) c = MTL_STEREO_SCREEN_PASS;
    return c;
}

int mtl_stereo_classify(const mtl_stereo_draw_in *in, float *a, float *seen_a)
{
    mtl_stereo_proj_full p, seen;
    return mtl_stereo_classify_proj(in, a, seen_a, &p, &seen);
}

int mtl_stereo_class_is_hud(int c) { return c == MTL_STEREO_FLAT || c == MTL_STEREO_PRETRANSFORMED; }

static int classify(const mtl_stereo_draw_in *in, float *a, float *seen_a, mtl_stereo_proj_full *p,
                    mtl_stereo_proj_full *seen)
{
    *a = 0.0f;
    *seen_a = 0.0f;
    memset(p, 0, sizeof *p);
    memset(seen, 0, sizeof *seen);
    if (!in->to_back_buffer) return MTL_STEREO_OFFSCREEN;
    if (in->positiont) return MTL_STEREO_PRETRANSFORMED;
    if (in->ff_vs) {
        float v = in->ff_proj ? mtl_stereo_a_from_d3d(in->ff_proj) : 0.0f;
        if (!(v > 0.0f)) return MTL_STEREO_FLAT;
        *a = *seen_a = v;
        mtl_stereo_proj_from_d3d(in->ff_proj, p);
        *seen = *p;
        return MTL_STEREO_WORLD;
    }
    if (!in->vsf || !in->pos_regs) return MTL_STEREO_UNKNOWN;
    const float *x = in->vsf + 4 * in->pos_regs, *w = in->vsf + 4 * (in->pos_regs + 3);
    if (in->pos_regs == MTL_POS_VIEWPROJ) {
        float v = mtl_stereo_a_from_rows(x, w);
        if (!(v > 0.0f)) return MTL_STEREO_FLAT;
        *a = *seen_a = v;
        mtl_stereo_proj_from_rows(x, x + 4, x + 8, w, p);
        *seen = *p;
        return MTL_STEREO_WORLD;
    }
    /* A model view projection's rows carry the model's scale, so its a
       (and full view's whole projection) is the frame's, from the view
       projection draws around it. */
    if (!mtl_stereo_row_is_perspective(w)) return MTL_STEREO_FLAT;
    if (!(in->last_a > 0.0f)) return MTL_STEREO_NO_A;
    *a = in->last_a;
    if (in->last_proj) *p = *in->last_proj;
    return MTL_STEREO_WORLD;
}

const char *mtl_stereo_class_name(int c)
{
    static const char *const k[MTL_STEREO_CLASSES] = {"world", "offscreen", "pretransformed", "flat", "unknown",
                                                      "no view projection yet", "screen pass"};
    return c >= 0 && c < MTL_STEREO_CLASSES ? k[c] : "?";
}

int mtl_stereo_covers_view(const float (*xyw)[3], uint32_t n)
{
    float lo[2] = {1e30f, 1e30f}, hi[2] = {-1e30f, -1e30f};
    for (uint32_t i = 0; i < n; i++) {
        float w = xyw[i][2];
        if (!(w > 1e-6f)) return 0;
        for (int k = 0; k < 2; k++) {
            float v = xyw[i][k] / w;
            if (v < lo[k]) lo[k] = v;
            if (v > hi[k]) hi[k] = v;
        }
    }
    return n >= 3 && lo[0] <= -0.98f && hi[0] >= 0.98f && lo[1] <= -0.98f && hi[1] >= 0.98f;
}

int mtl_stereo_arm_side(const float bb[4])
{
    if (!(bb[0] <= bb[1]) || !(bb[2] <= bb[3])) return 0;
    if (bb[1] < MTL_ARM_LEFT_MAX_X) return 1;
    if (bb[0] > MTL_ARM_RIGHT_MIN_X) return 2;
    return 0;
}

void mtl_stereo_arm_columns(int side, uint32_t w, uint32_t *x0, uint32_t *x1)
{
    if (side == 1) {
        *x0 = 0;
        *x1 = (uint32_t)ceil((MTL_ARM_LEFT_MAX_X + 1.0) * 0.5 * w);
    } else if (side == 2) {
        *x0 = (uint32_t)floor((MTL_ARM_RIGHT_MIN_X + 1.0) * 0.5 * w);
        *x1 = w;
    } else {
        *x0 = *x1 = 0;
    }
    if (*x1 > w) *x1 = w;
    if (*x0 > *x1) *x0 = *x1;
}

void mtl_aim_clip(const float r[4][4], const float p[3], float c[4])
{
    for (int i = 0; i < 4; i++) c[i] = r[i][0] * p[0] + r[i][1] * p[1] + r[i][2] * p[2] + r[i][3];
}

/* A perspective z row is A times the w row and a constant, so clip z is
   A w + B whatever the point; the eye's correction keeps that form
   (z' = z0 z + z1 w + z2, w' = (w0 + w1) w + w2), so a depth the eye
   stored gives back its w. */
int mtl_stereo_depth_terms(const float rows[4][4], const mtl_stereo_fix *f, float *A_out, float *B_out)
{
    const float *z = rows[2], *w = rows[3];
    float ww = dot3(w, w);
    if (!(ww > 0.0f)) return 0;
    float A = dot3(z, w) / ww, B = z[3] - A * w[3];
    float ws = f->w[0] + f->w[1];
    if (!(ws != 0.0f)) return 0;
    *A_out = (f->z[0] * A + f->z[1]) / ws;
    *B_out = f->z[0] * B + f->z[2] - *A_out * f->w[2];
    return 1;
}

int mtl_aim_probe(const mtl_aim_ray *r, const mtl_stereo_fix *f, mtl_aim_probes *out)
{
    out->n = out->on_image = 0;
    if (!r->ok || !(r->vp[2] > 0.0f) || !(r->vp[3] > 0.0f)) return 0;
    if (!mtl_stereo_depth_terms(r->rows, f, &out->A, &out->B)) return 0;
    float ws = f->w[0] + f->w[1];
    /* Geometric steps: the same few percent of the range at every range,
       so a wall a metre away is found as closely, for its distance, as
       one across a street. */
    float step = logf(MTL_AIM_T_FAR / MTL_AIM_T_NEAR) / (float)(MTL_AIM_SAMPLES - 1);
    for (int k = 0; k < MTL_AIM_SAMPLES; k++) {
        float t = MTL_AIM_T_NEAR * expf(step * (float)k), p[3], c[4];
        for (int i = 0; i < 3; i++) p[i] = r->eye[i] + r->dir[i] * t;
        mtl_aim_clip(r->rows, p, c);
        /* The vertex stage's own correction (shader.cpp, mtl_stereo_apply). */
        float x = f->x[0] * c[0] + f->x[1] * c[3] + f->x[2];
        float y = f->y[0] * c[1] + f->y[1] * c[3] + f->y[2];
        float we = ws * c[3] + f->w[2];
        out->t[k] = t;
        out->w[k] = we;
        out->px[k][0] = out->px[k][1] = -1.0f;
        if (!(we > 1e-3f)) continue;
        float nx = x / we, ny = y / we;
        if (!(fabsf(nx) < 1.0f) || !(fabsf(ny) < 1.0f)) continue;
        out->px[k][0] = r->vp[0] + (nx + 1.0f) * 0.5f * r->vp[2];
        out->px[k][1] = r->vp[1] + (1.0f - ny) * 0.5f * r->vp[3];
        out->on_image++;
    }
    out->n = MTL_AIM_SAMPLES;
    return 1;
}

float mtl_aim_stored_w(const mtl_aim_probes *p, const float vp[6], float stored)
{
    float range = vp[5] - vp[4];
    float z = range > 0.0f ? (stored - vp[4]) / range : stored;
    /* The clear value, the far plane, is nothing drawn there. */
    if (!(z < 1.0f)) return INFINITY;
    float w = p->B / (z - p->A);
    return w > 0.0f && isfinite(w) ? w : INFINITY;
}

/* Whether the ray stays in the surface it met at sample k (drawn at w
   s_in there): every sample past it, to MTL_AIM_HOLD of the way on and at
   least MTL_AIM_T_BODY, at or behind what is drawn at its pixel, and
   that no nearer than s_in (by 5% and 4 units, for a surface that turns
   towards the camera). A sample in front of its surface, or with nothing
   drawn there, is the ray out again, past a thin thing it only went
   behind; a nearer surface is something else in the way, the player's
   head in front of their own weapon. Samples off the image say nothing
   either way. */
static int aim_holds(const mtl_aim_probes *p, const float vp[6], const float *depth, int k, float s_in)
{
    float until = fmaxf(p->t[k] * MTL_AIM_HOLD, MTL_AIM_T_BODY);
    float floor_w = s_in - 0.05f * s_in - 4.0f;
    for (int j = k + 1; j < p->n && p->t[j] <= until; j++) {
        if (p->px[j][0] < 0.0f) continue;
        float s = mtl_aim_stored_w(p, vp, depth[j]), wj = p->w[j];
        float dw = fabsf(wj - p->w[j - 1]);
        if (!(wj + 0.01f * wj + 2.0f + 2.0f * dw >= s) || s < floor_w) return 0;
    }
    return 1;
}

/* Whether a point on the ray is inside the player's own box: within
   MTL_AIM_BODY_R across of the eye and no lower than MTL_AIM_BODY_DOWN
   under it. The world cannot be there (the player's collision keeps it
   out), so a surface there is the player's model, which the shot's trace
   never meets; the floor under the feet is lower. */
static int aim_in_body(const mtl_aim_ray *r, float t)
{
    float dx = r->dir[0] * t, dy = r->dir[1] * t, dz = r->dir[2] * t;
    return dx * dx + dy * dy < MTL_AIM_BODY_R * MTL_AIM_BODY_R && dz > -MTL_AIM_BODY_DOWN;
}

/* The hit's point into the frame through the rows, shared by the search
   and the trace. */
static int aim_place(const mtl_aim_ray *r, mtl_aim_hit *out)
{
    float c[4];
    for (int i = 0; i < 3; i++) out->p[i] = r->eye[i] + r->dir[i] * out->t;
    mtl_aim_clip(r->rows, out->p, c);
    if (!(c[3] > 1e-3f)) return 0;
    out->ndc[0] = c[0] / c[3];
    out->ndc[1] = c[1] / c[3];
    out->w = c[3];
    return 1;
}

int mtl_aim_traced(const mtl_aim_ray *r, mtl_aim_hit *out)
{
    memset(out, 0, sizeof *out);
    out->k = -1;
    out->t = MTL_AIM_T_FAR;
    if (!r->ok || !r->traced || !(r->trace_t >= 0.0f) || !isfinite(r->trace_t)) return 0;
    out->hit = r->trace_hit != 0;
    out->t = r->trace_t;
    return aim_place(r, out);
}

int mtl_aim_search(const mtl_aim_ray *r, const mtl_aim_probes *p, const float *depth, mtl_aim_hit *out)
{
    memset(out, 0, sizeof *out);
    out->k = -1;
    out->t = MTL_AIM_T_FAR;
    if (!r->ok || p->n < 2) return 0;
    /* The samples' w is linear in t, exactly: the line through the first
       two gives it anywhere on the ray. */
    float slope = (p->w[1] - p->w[0]) / (p->t[1] - p->t[0]), w_at0 = p->w[0] - slope * p->t[0];
    /* Where the ray was last seen in front of what is drawn: it cannot
       have entered a surface before there. */
    float t_seen = 0.0f;
    for (int k = 0; k < p->n; k++) {
        if (p->px[k][0] < 0.0f) continue;
        float s = mtl_aim_stored_w(p, r->vp, depth[k]), wk = p->w[k];
        /* In front of what is drawn at its pixel, or nothing is: the ray
           goes on. */
        if (!(wk >= s)) {
            t_seen = p->t[k];
            continue;
        }
        float wp = k ? p->w[k - 1] : wk, lo = fminf(wp, wk), dw = fabsf(wk - wp);
        lo -= 0.01f * lo + 2.0f + 2.0f * dw;
        float t0 = k ? p->t[k - 1] : 0.0f, t = p->t[k];
        if (fabsf(slope) > 1e-6f) t = (s - w_at0) / slope;
        if (s < lo) {
            /* Something nearer hid the ray where it entered: a surface
               it comes out behind, the wall past a pole that hid the
               crossing, at a depth the ray reached after it was last
               seen, is that entry. One it reached before is a thing
               between the camera and the ray. */
            if (!(fabsf(slope) > 1e-6f && t > t_seen && t <= p->t[k]) || aim_in_body(r, t) ||
                !aim_holds(p, r->vp, depth, k, s)) {
                out->hidden++;
                continue;
            }
        } else {
            if (aim_in_body(r, p->t[k]) || !aim_holds(p, r->vp, depth, k, s)) continue;
            t = fminf(fmaxf(t, t0), p->t[k]);
        }
        out->hit = 1;
        out->k = k;
        out->t = t;
        break;
    }
    return aim_place(r, out);
}

int mtl_rigid_fit(const float (*a)[3], const float (*b)[3], int n, float r[3][3], float t[3], float *rms)
{
    if (n < 3) return 0;
    double ca[3] = {0, 0, 0}, cb[3] = {0, 0, 0};
    for (int i = 0; i < n; i++)
        for (int k = 0; k < 3; k++) ca[k] += a[i][k], cb[k] += b[i][k];
    for (int k = 0; k < 3; k++) ca[k] /= n, cb[k] /= n;
    double s[3][3] = {{0}}, scale = 0;
    for (int i = 0; i < n; i++) {
        double pa[3], pb[3];
        for (int k = 0; k < 3; k++) pa[k] = a[i][k] - ca[k], pb[k] = b[i][k] - cb[k];
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++) s[j][k] += pa[j] * pb[k];
        scale += sqrt(pa[0] * pa[0] + pa[1] * pa[1] + pa[2] * pa[2]) * sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]);
    }
    if (!(scale > 1e-9)) return 0;
    double sxx = s[0][0], sxy = s[0][1], sxz = s[0][2], syx = s[1][0], syy = s[1][1], syz = s[1][2], szx = s[2][0],
           szy = s[2][1], szz = s[2][2];
    /* Horn's matrix, shifted by the bound on its eigenvalues so the one
       wanted, its largest, is also the largest in size. */
    double m[4][4] = {{sxx + syy + szz, syz - szy, szx - sxz, sxy - syx},
                      {syz - szy, sxx - syy - szz, sxy + syx, szx + sxz},
                      {szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy},
                      {sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz}};
    for (int k = 0; k < 4; k++) m[k][k] += scale;
    double q[4] = {1, 0.01, 0.02, 0.03};
    for (int it = 0; it < 200; it++) {
        double nq[4] = {0, 0, 0, 0};
        for (int j = 0; j < 4; j++)
            for (int k = 0; k < 4; k++) nq[j] += m[j][k] * q[k];
        double len = sqrt(nq[0] * nq[0] + nq[1] * nq[1] + nq[2] * nq[2] + nq[3] * nq[3]);
        if (!(len > 1e-30)) return 0;
        for (int k = 0; k < 4; k++) q[k] = nq[k] / len;
    }
    double w = q[0], x = q[1], y = q[2], z = q[3];
    double rr[3][3] = {{w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)},
                       {2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)},
                       {2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z}};
    double err = 0;
    for (int j = 0; j < 3; j++) {
        t[j] = (float)(cb[j] - (rr[j][0] * ca[0] + rr[j][1] * ca[1] + rr[j][2] * ca[2]));
        for (int k = 0; k < 3; k++) r[j][k] = (float)rr[j][k];
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < 3; j++) {
            double d = rr[j][0] * a[i][0] + rr[j][1] * a[i][1] + rr[j][2] * a[i][2] + t[j] - b[i][j];
            err += d * d;
        }
    if (rms) *rms = (float)sqrt(err / n);
    return 1;
}

int mtl_clip_to_world(const float rows[4][4], const float clip[4], float p[3])
{
    /* Solve rows [p 1] = clip for p: four equations, three unknowns and
       the homogeneous 1; the w the point had is clip w itself, so x, y
       and z rows with it give a 3 by 3 system. */
    double a[3][3], c[3];
    for (int i = 0; i < 3; i++) {
        int row = i == 2 ? 3 : i;
        for (int k = 0; k < 3; k++) a[i][k] = rows[row][k];
        c[i] = clip[row] - rows[row][3];
    }
    double det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
                 a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if (!(fabs(det) > 1e-12)) return 0;
    for (int k = 0; k < 3; k++) {
        double m[3][3];
        memcpy(m, a, sizeof m);
        for (int i = 0; i < 3; i++) m[i][k] = c[i];
        double dk = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        p[k] = (float)(dk / det);
    }
    return isfinite(p[0]) && isfinite(p[1]) && isfinite(p[2]);
}
