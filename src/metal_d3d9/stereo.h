/*
 * Stereo by re-projecting clip space: the pure part, with no Metal, so the
 * handoff test can check the math and the headless test can check the
 * bytecode scan against every game shader.
 *
 * The game renders one view. Each eye replays the same recorded frame, and
 * every world draw's vertex output is corrected in clip space on the way
 * out of the vertex stage. For a D3D perspective projection P_game,
 *
 *     x_c = a x_v + cx z_v,   y_c = b y_v + cy z_v,   w_c = z_v,
 *
 * so the view-space point is recoverable from clip space alone:
 * z_v = w_c, x_v = (x_c - cx w_c) / a, y_v = (y_c - cy w_c) / b. An eye at
 * view-space x = e (parallel axes, no toe-in) sees x_e = x_v - e, and its
 * own projection P_eye (a', cx', b', cy') gives
 *
 *     x' = (a'/a) x_c + (cx' - a' cx / a) w_c - a' e
 *     y' = (b'/b) y_c + (cy' - b' cy / b) w_c
 *     z' = z_c,  w' = w_c.
 *
 * Only a, cx, b and cy are needed, never near or far: the eye moves along
 * x only, so its view depth is the game's and the depth the game tests
 * against is unchanged, bit for bit. The constant term is not scaled by w
 * because every vertex position is a point, whose view-space w is 1.
 *
 * Stage A uses P_eye = P_game, which leaves x' = x_c - a e: the game's own
 * image per eye, shifted by a disparity of 2 a e / z_v in NDC. Each eye's
 * image is then a symmetric frustum about that eye's own forward axis,
 * and it is only right when shown that way: the compositor puts it on a
 * plane centred on the eye's axis, sized so a pixel at NDC x lies in the
 * direction x / a (immersive.h, mtl_stereo_plane_mvp). Stretched over the
 * headset's view it cannot fuse, because each eye's view is off-axis
 * (wider on the outside than the nose side) and differs per eye, so the
 * same pixel is not the same direction in the two.
 *
 * HUD and 2D draws carry no depth. Left alone they would sit at the same
 * NDC in both eye images, which on the per-eye planes is the same
 * direction for both eyes: at infinity, behind the world. They are put at
 * a depth z_h instead, the HUD plane, by the disparity a point there has:
 *
 *     x' = x_c - (a_s e / z_h) w_c
 *
 * which is a constant NDC shift whatever w is, with a_s the a the plane
 * is sized by (the session's, mtl_stereo_screen_a). z_h is also the
 * plane's own distance, so the HUD lies on the plane, and a frame with no
 * world draw at all (a menu, the character sheet) is one flat image there
 * in both eyes, fused rather than doubled: the same picture a mono plane
 * at z_h centred on the head would show, less a strip a_s e / z_h of NDC
 * wide that each eye loses at one edge.
 */
#ifndef MTL_STEREO_H
#define MTL_STEREO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* x' = x[0] x + x[1] w + x[2], the same for y and z, and w' = w[0] w +
   w[1] w + w[2]. The vertex stage reads it as four float4 at
   MTL_BUF_STEREO. Stage A only ever moves x and y; full view moves the
   eye along its forward axis too, which changes z and w. */
typedef struct {
    float x[4], y[4], z[4], w[4];
} mtl_stereo_fix;

/* A perspective projection's terms: x_c = a x + cx z, y_c = b y + cy z. */
typedef struct {
    float a, cx, b, cy;
} mtl_stereo_proj;

/* The same with the depth term, z_c = A z + B, which full view needs
   because its eye moves along z as well. */
typedef struct {
    float a, cx, b, cy, A;
} mtl_stereo_proj_full;

void mtl_stereo_identity(mtl_stereo_fix *out);
/* eye_x: the eye's offset along the camera's right, in game units. */
void mtl_stereo_correction(const mtl_stereo_proj *game, const mtl_stereo_proj *eye, float eye_x, mtl_stereo_fix *out);
/* A HUD draw's shift to the HUD plane: screen_a the plane's a, eye_x and
   hud_z in the same units. Identity when either is not positive. */
void mtl_stereo_hud_correction(float screen_a, float eye_x, float hud_z, mtl_stereo_fix *out);

/*
 * Full view. The eye images fill each eye's whole view instead of a
 * plane: every world draw is re-projected from the game's projection to
 * the eye's own off-axis one (from the headset view's tangents), seen
 * from an eye at t = (tx, ty, tz) in the game camera's axes (x right, y
 * up, z forward, game units). With z_v = w_c, x_v = (x_c - cx w_c) / a
 * and the same for y, the eye sees (x_v - tx, y_v - ty, z_v - tz), so
 *
 *     x' = (a'/a) x_c + (cx' - a' cx / a) w_c - a' tx - cx' tz
 *     y' = (b'/b) y_c + (cy' - b' cy / b) w_c - b' ty - cy' tz
 *     z' = z_c - A tz,   w' = w_c - tz
 *
 * keeping the game's own depth range. With the eye's projection the
 * game's and t = 0 it is the identity, bit for bit, as stage A's is.
 * Identity when the game's a or b is 0.
 */
void mtl_stereo_correction3(const mtl_stereo_proj_full *game, const mtl_stereo_proj *eye, const float t[3],
                            mtl_stereo_fix *out);
/* An eye's projection terms from its view's half-angle tangents (left,
   right, down, up, each positive outwards, as mtl_stereo_tangents gives
   them), D3D's z forward: a = 2 / (l + r), cx = (l - r) / (l + r). */
void mtl_stereo_proj_from_tangents(const float tan[4], mtl_stereo_proj *out);
/* Full view's HUD: the engine's 2D, in game NDC, onto a head-locked
   panel hud_z ahead and px by py in half tangents, as an eye at t sees it:
   x' = a' px x + (cx' - a' tx / hud_z) w, the same for y. Scaled by w for
   the same reason as mtl_stereo_hud_correction. Identity when hud_z,
   px, py or the eye's a or b is not positive. */
void mtl_stereo_panel_correction(const mtl_stereo_proj *eye, const float t[3], float hud_z, float px, float py,
                                 mtl_stereo_fix *out);

/* The frame's main perspective: the a most of its world draws used. A
   view model, a skybox or a 3D menu piece may use another. Only logged
   now: the plane is the session's (mtl_stereo_screen_a), not the frame's.
   Game thread only. */
#define MTL_STEREO_TALLY 6
typedef struct {
    float a[MTL_STEREO_TALLY];
    uint32_t n[MTL_STEREO_TALLY];
} mtl_stereo_tally;
void mtl_stereo_tally_add(mtl_stereo_tally *t, float a);
/* The most used a since the last take, or `fallback` when no world draw
   was seen, and the tally emptied. */
float mtl_stereo_tally_take(mtl_stereo_tally *t, float fallback);
/* The game's default before any world draw: 75 degrees across 4:3. */
#define MTL_STEREO_DEFAULT_A 1.30322540f
/* The game's nominal field of view, degrees across a 4:3 view, which the
   porter's g2b_view_fov and g2b_camera_fov hooks keep at that height for
   a wider view (hor+). */
#define MTL_STEREO_NOMINAL_DEG 75.0f

/* The a the eye planes are sized by for the whole session, for a back
   buffer of `aspect` (width over height): the game's own hor+ rule for
   `deg` degrees across 4:3, so at the nominal 75 it is the a play is
   drawn with, and every other frame (a dialogue close up's zoom, a menu,
   the character sheet, a load screen, a movie) shows at that same
   angular size. Sized per frame instead, a narrower camera shrank the
   window, and a 2D frame kept whatever the last world had. deg outside
   10..170 gives the nominal 75. */
float mtl_stereo_screen_a(float aspect, float deg);

/* What a vertex shader's oPos is computed from, scanned from its D3D9
   bytecode: the first of four constant registers whose dp4s (or m4x4)
   give oPos.x (and oPos.w, where it comes from a dp4 at all), directly
   or through one temporary. The game's
   shaders use c8 to c11 (view projection, after a model transform in
   c48 to c50 or none) or c4 to c7 (model view projection). 0 when it is
   neither: pretransformed passes, anything not recognised. */
#define MTL_POS_MVP 4
#define MTL_POS_VIEWPROJ 8
int mtl_stereo_scan_pos(const uint32_t *tokens, size_t n);

/* Why a draw is or is not corrected. */
enum {
    MTL_STEREO_WORLD,      /* perspective into the back buffer: corrected */
    MTL_STEREO_OFFSCREEN,  /* another render target: shadow, reflection, post input */
    MTL_STEREO_PRETRANSFORMED, /* POSITIONT vertices */
    MTL_STEREO_FLAT,       /* orthographic or identity: HUD, 2D, full-screen passes */
    MTL_STEREO_UNKNOWN,    /* a shader whose oPos the scan did not recognise */
    MTL_STEREO_NO_A,       /* model view projection with no view projection seen yet */
    MTL_STEREO_SCREEN_PASS, /* flat or POSITIONT reading a render target: a pass over this eye's own image */
    MTL_STEREO_CLASSES
};

/* a from the clip x row and the clip w row of a view projection, each as
   the four constants of a dp4 against the world position: with V rigid,
   the x row is a times the camera's right plus cx times its forward, and
   the w row is the forward. 0 when the w row has no xyz (orthographic). */
float mtl_stereo_a_from_rows(const float x_row[4], const float w_row[4]);
/* Whether a w row is perspective (has xyz), for a model view projection. */
int mtl_stereo_row_is_perspective(const float w_row[4]);
/* a from a D3DMATRIX projection as SetTransform passes it, 0 when it is
   not perspective (_34 is zero). */
float mtl_stereo_a_from_d3d(const float proj[16]);
/* A view projection's terms from its four rows as dp4 constants (c8 to
   c11), all five over the rows' common scale (|w row|): a and b the
   right and up parts, cx, cy and A the forward parts of the x, y and z
   rows. 0, with *out zeroed, when the w row is not perspective. */
int mtl_stereo_proj_from_rows(const float x_row[4], const float y_row[4], const float z_row[4],
                              const float w_row[4], mtl_stereo_proj_full *out);
/* The same from a D3DMATRIX projection: a as mtl_stereo_a_from_d3d, the
   rest over _34. */
int mtl_stereo_proj_from_d3d(const float proj[16], mtl_stereo_proj_full *out);
/* Where the camera is, in the space the draw's vertices are in: from a
   view projection's x, y and w rows (the point where all three are 0), or
   from a D3DMATRIX view. 0 when the rows or the matrix are singular. Two
   draws in one frame drawn from different cameras are different views:
   the 3D sky is drawn from its own camera at its own small scale. */
int mtl_stereo_camera_from_rows(const float x_row[4], const float y_row[4], const float w_row[4], float out[3]);
int mtl_stereo_camera_from_view(const float view[16], float out[3]);
/* The camera a draw is seen from: a fixed-function draw's view matrix, a
   shader draw's view projection in c8 to c11 whichever registers position
   its vertices (vsf is the vertex constants, 4 floats a register). */
int mtl_stereo_draw_camera(int ff_vs, const float *vsf, const float view[16], float out[3]);

typedef struct {
    int to_back_buffer;
    int positiont;
    int ff_vs;
    const float *ff_proj; /* D3DMATRIX, for a fixed-function vertex stage */
    int pos_regs;         /* mtl_stereo_scan_pos of the shader */
    const float *vsf;     /* the vertex shader's float constants, 4 per register */
    float last_a;         /* the last world view projection's a, 0 before one */
    int samples_rt;       /* a texture it samples is a render target */
    /* The last world view projection's terms, which a model view
       projection draw borrows whole for full view; NULL borrows a alone. */
    const mtl_stereo_proj_full *last_proj;
} mtl_stereo_draw_in;

/* The draw's class, with *a set for a world draw (else 0). *seen_a is set
   when the draw itself revealed the frame's a (a view projection or a
   fixed-function perspective), for the caller to keep as last_a. */
int mtl_stereo_classify(const mtl_stereo_draw_in *in, float *a, float *seen_a);
/* Both at once: *a and *seen_a exactly as mtl_stereo_classify gives them,
   for stage A, and the projection's whole terms in *p for a world draw
   (and *seen when the draw revealed them; zeroed otherwise), for full
   view. */
int mtl_stereo_classify_proj(const mtl_stereo_draw_in *in, float *a, float *seen_a, mtl_stereo_proj_full *p,
                             mtl_stereo_proj_full *seen);

const char *mtl_stereo_class_name(int c);
/* Whether a class is put on the HUD plane in each eye. */
int mtl_stereo_class_is_hud(int c);

/* Whether a HUD draw covers the whole view: its vertices' clip x, y and w
   (xyw[i][0..2]) span normalized device coordinates past +-0.98 on both
   axes. Full view fills each eye with such a draw (a discipline's tint,
   a fade, a full screen overlay) instead of shrinking it onto the panel. */
int mtl_stereo_covers_view(const float (*xyw)[3], uint32_t n);

/* The arm HUD (full view): the game's left gauge, health and the portrait
   above it, all left of x -0.74 in the game's NDC, and its right one,
   blood and the discipline disc, all right of +0.77 (la_hub_1). A HUD
   draw whose box (x0 x1 y0 y1) lies wholly past a line goes to that
   side's card at the controller: 1 left, 2 right. 0 for anything else,
   a box across a line or no box at all, which stays on the panel. */
#define MTL_ARM_LEFT_MAX_X (-0.70f)
#define MTL_ARM_RIGHT_MIN_X 0.76f
int mtl_stereo_arm_side(const float bb[4]);
/* The back buffer columns [x0, x1) a side's card holds, for a back
   buffer w pixels wide: the side's NDC strip, widened to whole pixels. */
void mtl_stereo_arm_columns(int side, uint32_t w, uint32_t *x0, uint32_t *x1);

#ifdef __cplusplus
}
#endif

#endif
