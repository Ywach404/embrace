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
/*
 * Third person aim (full view): where the player's shot lands, found from
 * the shot's own ray rather than from how the game draws its crosshair,
 * which differs per weapon. The ray (the player's eye and aim direction,
 * read from the game by the runtime at Present) is sampled at
 * MTL_AIM_SAMPLES points eye + dir t, t geometric from MTL_AIM_T_NEAR to
 * MTL_AIM_T_FAR game units, each taken through the frame's own view
 * projection (the world draws' c8 to c11) and an eye's correction to the
 * pixel it lands on in that eye's image and the depth it would have there.
 * The GPU reads the eye's depth at those pixels (draw.m, mtl_aim_gather)
 * and the search runs here on what it read.
 */
#define MTL_AIM_SAMPLES 1024
#define MTL_AIM_T_NEAR 16.0f
#define MTL_AIM_T_FAR 8192.0f
/* The GPU buffer: the samples' pixels (two floats each), then their reads. */
#define MTL_AIM_DEPTH_AT (2 * MTL_AIM_SAMPLES)
#define MTL_AIM_BUF_FLOATS (3 * MTL_AIM_SAMPLES)
/* A surface the ray meets is taken only once the ray stays behind it: to
   MTL_AIM_HOLD of the way past it (3%, a few samples) and, near the
   player, to at least MTL_AIM_T_BODY along the ray, past the player's
   own arms and weapon, which the shot's trace never meets. */
#define MTL_AIM_HOLD 1.03f
#define MTL_AIM_T_BODY 96.0f
/* The player's box about the eye, game units: its half width (the hull's
   16 and a little) and how far below the eye it reaches (the eye height,
   64, less a little so the floor at the feet is not in it). */
#define MTL_AIM_BODY_R 18.0f
#define MTL_AIM_BODY_DOWN 56.0f
typedef struct {
    int ok;
    float rows[4][4]; /* clip x, y, z and w as dp4 rows on a world point (c8 to c11) */
    float vp[6];      /* the world draws' viewport: x, y, width, height in pixels, min and max depth */
    float eye[3];     /* where the shot leaves, game world units */
    float dir[3];     /* its direction, unit length */
    int traced;       /* 1 when the engine's own shot trace gave where it lands (mtl_aim_traced) */
    int trace_hit;    /* and whether it met anything there */
    float trace_t;    /* how far along the ray, game units */
} mtl_aim_ray;
/* The first person view model's hands (the runtime's rt_vm_hands): per
   hand, right then left, its origin and points along its x and y axes in
   game world units, read at the frame's Present, and the frame's view
   projection to take them into it. mask bit 0 right, bit 1 left. */
typedef struct {
    int mask;
    float rows[4][4];
    float pts[18];
    /* The palm's middle as drawn, right then left, when the hands came
       from the arms' own draws (draw.m, arm_hands_fit); for the log's
       gap between the glove and the drawn hand. */
    float palm[6];
} mtl_vm_hands;
/* The first person view model's own view projection and the depth range
   its draws are given, from the frame's first view model draw (draw.m,
   mtl_fv_scan): its pixels in the depth buffer are turned into metres
   with these, not the world's, so a glove on the gun is hidden behind
   its grip (the scene depth, immersive.m). */
typedef struct {
    int ok;
    float rows[4][4];
    float vp[2]; /* the viewport's min and max depth */
} mtl_vm_depth;
/* A world point's clip x, y, z, w under the rows. */
void mtl_aim_clip(const float rows[4][4], const float p[3], float clip[4]);
typedef struct {
    int n;          /* samples, MTL_AIM_SAMPLES when made */
    int on_image;   /* how many land on the eye's image */
    float A, B;     /* the eye's clip z = A w + B, so a stored depth gives its w back */
    float t[MTL_AIM_SAMPLES];
    float w[MTL_AIM_SAMPLES];     /* the sample's depth along the eye's forward, game units */
    float px[MTL_AIM_SAMPLES][2]; /* its pixel in the eye's image; x -1 when it lands on none */
} mtl_aim_probes;
/* An eye's depth terms under its correction f (mtl_stereo_correction3 of
   the rows' projection): a depth it stored, z in 0 to 1, is at w = B /
   (z - A) along its forward, in the rows' units. 0 when the rows have no
   perspective w. */
int mtl_stereo_depth_terms(const float rows[4][4], const mtl_stereo_fix *f, float *A, float *B);
/* The samples for one eye, whose correction (mtl_stereo_correction3 of
   the rows' projection) is fix. 0, with out->n 0, when the ray or the
   rows give none. */
int mtl_aim_probe(const mtl_aim_ray *r, const mtl_stereo_fix *fix, mtl_aim_probes *out);
/* The depth at a sample's pixel, raw as stored, back to a w along the
   eye's forward; +inf where nothing nearer than the far plane was drawn. */
float mtl_aim_stored_w(const mtl_aim_probes *p, const float vp[6], float stored);
typedef struct {
    int hit;        /* 1 a surface on the ray, 0 none on it (sky, or past the far end): the far end */
    int k;          /* the sample the hit was found at, -1 for none */
    int hidden;     /* samples a surface between the camera and the ray hid (the player's own body) */
    float t;        /* along the ray, game units */
    float p[3];     /* the point, game world units */
    float ndc[2], w; /* and in the frame: the game's NDC and clip w */
} mtl_aim_hit;
/* The search over what the GPU read at the samples (depth[k], raw): the
   first sample at or behind the surface at its pixel, with that surface
   no nearer than the sample before it, less a slack of 1%, 2 units and
   two samples' steps in depth, is where the ray enters it. The steps
   are for a surface the ray meets at a glancing angle, the street ahead
   seen from a camera little higher than the ray, whose depth at the
   sample's pixel is nearer than the ray by up to a step or two: without
   them the crossing was found in some frames and missed in others. A
   surface nearer than that is one between the camera and the ray, as
   the player's own back is to the samples just past the face, and is
   passed over. A crossing is kept only if the samples after it stay at
   or behind what is drawn at theirs, and that no nearer than where it
   entered (MTL_AIM_HOLD, MTL_AIM_T_BODY): a wall or the street holds
   the ray, where the player's arms or weapon, or a thin pole the ray
   only passes behind, gives it back a few samples on. A crossing inside
   the player's own box (MTL_AIM_BODY_R, MTL_AIM_BODY_DOWN) is the
   player's head or neck and is never kept. The point is refined to where the ray's depth, linear
   in t, equals the surface's, between the two samples. No such sample
   gives the far end, hit 0. 0 when the ray or its point give nothing in
   the frame. */
int mtl_aim_search(const mtl_aim_ray *r, const mtl_aim_probes *p, const float *depth, mtl_aim_hit *out);
/* The engine's trace's answer as a hit (r->traced): the point trace_t
   along the ray, hit as the trace says, taken into the frame through the
   rows as the search's is; k -1 and nothing hidden, since no depth was
   read. 0 when the ray was not traced or its point gives nothing in the
   frame (behind the camera). */
int mtl_aim_traced(const mtl_aim_ray *r, mtl_aim_hit *out);
/* A point the game drew into its world, from its NDC (x, y) and clip w
   under the game's projection: as tangents right and up of the camera's
   forward (x - cx) / a and (y - cy) / b, a half extent in x the same way,
   and its depth along that forward in metres (w is the view depth for a
   perspective projection whose w row is the view's z). out: tan x, tan
   y, tan half, metres. 0 and zeros when the projection, w or the scale
   give no point. */
int mtl_stereo_world_point(const mtl_stereo_proj_full *game, float x, float y, float half, float w, float units_per_m,
                           float out[4]);
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

/* A vertex shader's oPos for one vertex, evaluated from its D3D9 bytecode
   on the CPU, for the shot dump's box of a small world draw: a sprite or
   a bone placed model part has no fixed rows to project through, so the
   draw's own arithmetic is run instead. Straight line code only (no flow
   control); an instruction it does not know poisons what it writes, so a
   position that needs one comes out unknown rather than wrong. */
#define MTL_VS_EVAL_INPUTS 16
typedef struct {
    const uint32_t *tk;
    size_t n;
    int major;
    int pos_type, pos_num;     /* the register oPos is: rastout 0, or vs_3_0's output with dcl_position */
    uint32_t in_mask;          /* bit r: v r is declared */
    uint8_t in_usage[MTL_VS_EVAL_INPUTS], in_index[MTL_VS_EVAL_INPUTS]; /* its D3DDECLUSAGE and usage index */
    float c[256][4];           /* the API's float constants with the shader's defs over them */
} mtl_vs_eval;
/* 0 when the shader is not one it can run: not a vertex shader, or with
   flow control. vsf is the API's 256 float registers. */
int mtl_vs_eval_init(mtl_vs_eval *e, const uint32_t *tokens, size_t n, const float *vsf);
/* oPos from the declared inputs v[r] (0 0 0 1 for an element the
   declaration lacks); 0 when any of its four came out unknown. */
int mtl_vs_eval_pos(const mtl_vs_eval *e, const float (*v)[4], float out[4]);

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

/* The game's view model hands from the arms' own draws: point pairs of
   where a vertex sits in the model's bind pose and where the frame drew
   it in the world, and the rigid turn and move that best takes the first
   set onto the second (Horn's closed form, through the largest
   eigenvector of his 4 by 4 matrix by power iteration). r is row major,
   b = r a + t; rms is the fit's error in the points' units. 0 with fewer
   than three points or a set too flat to turn. */
int mtl_rigid_fit(const float (*a)[3], const float (*b)[3], int n, float r[3][3], float t[3], float *rms);
/* A world point from its clip x, y, z, w under view projection rows
   (mtl_aim_clip backwards): 0 when the rows do not invert. */
int mtl_clip_to_world(const float rows[4][4], const float clip[4], float p[3]);

#ifdef __cplusplus
}
#endif

#endif
