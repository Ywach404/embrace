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
