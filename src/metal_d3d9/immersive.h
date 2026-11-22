/*
 * What a host app needs from the backend to show the game in an immersive
 * space: the switch, the latest finished game frame as a texture, and the
 * placement math for the virtual screen. Plain C, so the visionOS host
 * reads it through its bridging header and the headless test on the Mac
 * includes it as it is; the backend's own headers stay private.
 *
 * Threads. metal_d3d9_immersive_enable is called once, before the runtime
 * starts. acquire and release are the compositor's, from any one thread
 * (acquire) and from its command buffers' completions (release). Neither
 * waits for the game: acquire returns at once with whatever frame was
 * finished last, and the game's Present never waits for the compositor.
 * See handoff.h for the ring behind them.
 */
#ifndef METAL_D3D9_IMMERSIVE_H
#define METAL_D3D9_IMMERSIVE_H

#include <math.h>
#include <stdint.h>

/* One compositor frame's view of the game. texture is a borrowed
   id<MTLTexture> (BGRA8Unorm, private storage, usable through an sRGB
   texture view) that stays valid and unchanged until release(slot). A
   mono frame is a 2D texture (layers 1); a stereo frame is a 2D array of
   two (layers 2), slice 0 the left eye and slice 1 the right. */
typedef struct {
    int slot;
    void *texture;
    uint32_t width, height;
    uint64_t frame;  /* game frame number, from 1 */
    double age_ms;   /* from the game's Present of this frame to this acquire */
    int repeat;      /* already shown by an earlier compositor frame */
    uint32_t layers; /* 1 mono, 2 stereo, 0 nothing held */
    /* A stereo frame's eye images, as the backend drew them: each spans
       tan_x either side of its eye's forward axis across and tan_y up and
       down (1 / a and 1 / b of the game's projection), the eyes were
       half_ipd_m either side of the head's centre, and the HUD was put at
       hud_m. The compositor shows each image on a plane that subtends
       exactly that (mtl_stereo_plane_mvp), so every pixel lies in the
       direction it was rendered for. 0 for a mono frame. */
    float tan_x, tan_y, half_ipd_m, hud_m;
    /* The head orientation the frame was drawn for, as the runtime gave
       it before the frame's Present (metal_d3d9_immersive_view_pose):
       yaw and pitch in degrees in ARKit's world, yaw positive turning
       left from -Z and pitch positive looking up, no roll. pose_valid 0
       when the game's camera did not follow the head for this frame. */
    int pose_valid;
    float pose_yaw_deg, pose_pitch_deg;
    /* Full view (metal_d3d9_fullview_enable): the eye images fill each
       eye's view rather than a plane, each spanning its eye_tan (left,
       right, down, up, the margin for the late rotation included; [0] the
       left eye) about the head's axes. 0 for a frame on the plane: a
       menu, dialogue or sheet still comes that way while full view is on. */
    int fullview;
    float eye_tan[2][4];
    /* The arm HUD (metal_d3d9_fullview_arm_hud): the game's gauges cut out of
       a full view frame for cards at the controllers, a borrowed
       id<MTLTexture> in the frame's format held with the slot. Its colour
       is premultiplied by its alpha over a clear target, gamma encoded
       like the frame's. arm_shown[side] (0 left, 1 right) says the frame
       drew that side's card, arm_uv[side] where it is in the texture (u0,
       v0, u1, v1, top left first), at the back buffer's own pixels. NULL
       and 0 for a frame with neither. */
    void *arm_texture;
    int arm_shown[2];
    float arm_uv[2][4];
} metal_d3d9_shown_frame;

/* The HUD plane's distance ahead of the eyes, in metres: near enough to
   read as a HUD in front of the world, far enough to be comfortable to
   converge on for a whole session. */
#define METAL_D3D9_STEREO_HUD_M 1.75f

/* Present hands each frame to the ring instead of a window's layer. Set
   before the device is created; the window path is the default. A host
   with a window uses metal_d3d9_space_request below instead, which keeps
   the window's layer fed until the space has shown the game. */
void metal_d3d9_immersive_enable(int on);
int metal_d3d9_immersive_enabled(void);

/* Game frames handed over so far. The host waits for the first before it
   opens its immersive space: by then the game's SDL window exists and has
   taken the app's window scene, so the space's scene cannot be picked up
   as the window's instead. */
uint64_t metal_d3d9_immersive_frames(void);

/* The newest finished frame, held until release. Returns 0 (nothing held)
   before the first frame is finished. Each call is one compositor frame
   in the counters: every few seconds the backend logs compositor frames,
   game frames handed over, frames shown more than once and the age of
   the frames shown, and RT_COMPOSITOR_TRACE names a CSV of every call. */
int metal_d3d9_immersive_acquire(metal_d3d9_shown_frame *out);
void metal_d3d9_immersive_release(int slot);

/* A line in the runtime's log, for the host's compositor. */
void metal_d3d9_immersive_log(const char *msg);

/* The head orientation the game's next frame is drawn for (the fields
   of metal_d3d9_shown_frame of the same names): the runtime calls it
   where it writes the game's view angles from the head, before the
   frame's Present, which latches it into the frame. valid 0 says the
   camera is the game's own (no head pose, a menu before the first one).
   Called on the game thread, or on the thread its pump runs on while the
   game thread waits for it. */
void metal_d3d9_immersive_view_pose(int valid, float yaw_deg, float pitch_deg);

/* A tracked controller's (later a hand's) pose for one side, 0 left and
   1 right, as the compositor reads it each of its frames: state 0
   untracked, 1 orientation only, 2 position and orientation, 3 that at
   low accuracy; grip and aim column-major origin-from-location 4x4 in
   metres; time on ARKit's clock. Kept as the latest per side for the
   view model's re-pose to come; nothing draws from it yet (the arm HUD's
   cards are placed by the compositor, metal_d3d9_fullview_arm_hud). Any
   thread. */
void metal_d3d9_immersive_hand_pose(int side, int state, const float grip[16], const float aim[16], double time);

/* The arm HUD: whether the compositor can show one side's card (0 left,
   1 right), its controller tracked with a position; it says so every
   frame it draws. A side it cannot show keeps its gauge on the head-locked
   panel, as every side does until it is first called, out of full view,
   and with RT_ARM_HUD=0. Any thread; latched with each frame. */
void metal_d3d9_fullview_arm_hud(int side, int on);

/*
 * Stereo (the backend's stereo.h). With it on and the ring in use, each
 * game frame is replayed once per eye from the same recorded list: world
 * draws get a per-eye clip-space correction in their vertex stage, every
 * other draw (HUD, 2D, offscreen, pretransformed) is the same in both, and
 * the ring's slot holds both eyes. Stage A keeps the game's own camera and
 * projection for each eye, so the image is the game's view per eye, and
 * the world turns with the head until the game's view follows it. Any
 * thread, at any time: a switch takes effect at the next frame's start.
 * half_ipd is in metres (default 0.032), changed to game units with
 * RT_STEREO_UNITS_PER_M (default 39.37, the engine's inch).
 */
void metal_d3d9_stereo_enable(int on);
int metal_d3d9_stereo_enabled(void);
void metal_d3d9_stereo_set_half_ipd(float metres);

/*
 * Full view (stereo.h, mtl_stereo_correction3). On top of stereo, a frame
 * the game draws in play is replayed per eye into that eye's own off-axis
 * projection, from the eye's own position, so each eye image fills its
 * view; the game draws it with a field wide enough for both eyes
 * (metal_d3d9_fullview_game_tan, which the runtime writes into the view's
 * FOV), and its HUD goes on a head-locked panel. Frames out of play keep
 * stereo's plane. Any thread; a change applies from the next frame.
 *
 *   set_views     the two headset views as the compositor sees them: tan
 *                 the left view's tangents then the right's (left, right,
 *                 down, up), eye_m both eyes' positions in the device's
 *                 space in metres (x, y, z each, left eye first). Until
 *                 it is called, the defaults below.
 *   game_tan      the tangents across and up the game's own symmetric
 *                 projection should span: the union of both eyes' views
 *                 with the margin, widened to the back buffer's shape.
 *                 1 when the next frame is a full view one (on, stereo
 *                 on, in play), else 0 with the values still filled.
 *   view_play     the runtime, once a game frame where it writes the
 *                 view: whether the game is in play (no menu, dialogue or
 *                 sheet), which the next Present latches.
 *   game_eye      a check: each eye keeps the game's projection and
 *                 position (both eye images the game's frame), and the
 *                 compositor shows them at the game's field.
 *
 * RT_FULLVIEW_MARGIN (default 0.06) widens every tangent for the late
 * rotation to turn into; RT_FULLVIEW_HUD_DEG (default 62) is the panel's
 * width; RT_FULLVIEW_EYE=game sets game_eye. Read when it is turned on.
 */
#define METAL_D3D9_FULLVIEW_MARGIN 0.06f
#define METAL_D3D9_FULLVIEW_HUD_DEG 62.0f
/* A discipline's aura held to the camera: across its own panel, wider
   than the HUD's so its ring shows, narrower than the eye so the ring is
   not lost at the lens's edge and its layers cost half the eye's pixels. */
#define METAL_D3D9_FULLVIEW_AURA_DEG 90.0f
/* This headset's views as a device log gave them, mirrored per eye. */
#define METAL_D3D9_FULLVIEW_TAN_OUT 1.732f
#define METAL_D3D9_FULLVIEW_TAN_NOSE 1.0f
#define METAL_D3D9_FULLVIEW_TAN_DOWN 1.19f
#define METAL_D3D9_FULLVIEW_TAN_UP 1.0f
void metal_d3d9_fullview_enable(int on);
int metal_d3d9_fullview_enabled(void);
void metal_d3d9_fullview_set_views(const float tan[8], const float eye_m[6]);
int metal_d3d9_fullview_game_tan(float *tan_x, float *tan_y);
/* Whether the game's frames draw a view model: -1 not known yet (none
 * seen this session), 0 none (the game's third person, which it switches
 * per weapon), 1 drawn (first person). Read by the runtime each frame. */
int metal_d3d9_fullview_view_model(void);
void metal_d3d9_immersive_view_play(int play);
void metal_d3d9_fullview_game_eye(int on);

/*
 * The space's opening and closing order (space.h), for a host that shows
 * the game in a window until the space has shown it. Each call applies
 * where Present sends frames: the ring, the window's layer, or both.
 *
 *   request         before the runtime starts: frames go to the ring and
 *                   still to the window from the first Present.
 *   should_open     a driver's poll: 1 claims the open (call
 *                   openImmersiveSpace, then open_result), 0 polls again,
 *                   -1 nothing to open. The claim waits for a frame in the
 *                   ring, so SDL's window has taken the window scene.
 *   open_result     one of the results below; ABANDONED hands the claim
 *                   back for another driver.
 *   compositor_started, first_frame_shown   from the compositor; the
 *                   second once its GPU finished a frame with the game.
 *   take_dismiss    1 once the space is open and has shown a frame: the
 *                   window's layer is off, dismiss the window now.
 *   closed          the space closed; 1 if the window must come back
 *                   first, then window_back once it holds the game.
 *   reopen          after window_back (Return to Immersive): the space
 *                   is wanted again and the same order runs again, as
 *                   often as asked; only frames from then on count.
 *   watchdog        logs the state of each step once if the space is not
 *                   open 5 s after the request, and once 5 s after the
 *                   first frame reached the ring. The backend also checks
 *                   it from its render thread.
 *
 * Any thread. None waits on the compositor or the game.
 */
enum {
    METAL_D3D9_SPACE_OPENED = 0,
    METAL_D3D9_SPACE_DECLINED = 1,
    METAL_D3D9_SPACE_FAILED = 2,
    METAL_D3D9_SPACE_ABANDONED = 3,
};
void metal_d3d9_space_request(void);
int metal_d3d9_space_should_open(void);
void metal_d3d9_space_open_result(int result);
void metal_d3d9_space_compositor_started(void);
void metal_d3d9_space_first_frame_shown(void);
int metal_d3d9_space_take_dismiss(void);
int metal_d3d9_space_closed(void);
void metal_d3d9_space_window_back(void);
/* Return to Immersive: after the window took the game back, the space is
   wanted again, in the same order as the first open. 1 when taken. */
int metal_d3d9_space_reopen(void);
void metal_d3d9_space_watchdog(void);

/*
 * The virtual screen. Matrices are column-major float[16] (m[col * 4 +
 * row]), the layout of simd_float4x4, in ARKit's world space: metres, +Y
 * up, -Z forward from the head.
 *
 * mtl_screen_place pins the screen from the head pose at the moment the
 * space opens: its centre `distance` metres ahead along the head's
 * forward direction flattened onto the floor, at the head's height,
 * upright and turned to face the head. Only yaw is taken from the head,
 * so a lowered head does not tilt the screen into the floor. The model
 * matrix maps the unit quad (x and y in -0.5..0.5, z = 0) to a screen
 * `width` metres wide and width / aspect tall, with +Y the top and +Z
 * towards the viewer.
 */
static inline void mtl_screen_identity(float m[16])
{
    for (int i = 0; i < 16; i++) m[i] = (i % 5) == 0 ? 1.0f : 0.0f;
}

/* out = a * b. out may not alias a or b. */
static inline void mtl_screen_mul(const float a[16], const float b[16], float out[16])
{
    for (int c = 0; c < 4; c++)
        for (int r = 0; r < 4; r++) {
            float s = 0;
            for (int k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
            out[c * 4 + r] = s;
        }
}

/* The inverse of a rotation plus translation, which every pose here is. */
static inline void mtl_screen_rigid_inverse(const float m[16], float out[16])
{
    for (int c = 0; c < 3; c++)
        for (int r = 0; r < 3; r++) out[c * 4 + r] = m[r * 4 + c];
    for (int r = 0; r < 3; r++) {
        out[12 + r] = -(out[0 * 4 + r] * m[12] + out[1 * 4 + r] * m[13] + out[2 * 4 + r] * m[14]);
        out[3 + r * 4] = 0;
    }
    out[15] = 1;
}

static inline void mtl_screen_transform(const float m[16], const float v[4], float out[4])
{
    for (int r = 0; r < 4; r++) out[r] = m[r] * v[0] + m[4 + r] * v[1] + m[8 + r] * v[2] + m[12 + r] * v[3];
}

static inline void mtl_screen_place(const float head[16], float distance, float width, float aspect, float model[16])
{
    /* Forward is the head's -Z, flattened. Looking straight up or down
       leaves no heading to flatten, so the head's up vector, which then
       lies nearly flat, gives it instead (looking down, +Z points up
       and up points forward; looking up, both flip, hence the sign). */
    float fx = -head[8], fz = -head[10];
    float len = sqrtf(fx * fx + fz * fz);
    if (len < 1e-3f) {
        float s = head[9] > 0 ? 1.0f : -1.0f;
        fx = s * head[4];
        fz = s * head[6];
        len = sqrtf(fx * fx + fz * fz);
        if (len < 1e-6f) {
            fx = 0;
            fz = -1;
            len = 1;
        }
    }
    fx /= len;
    fz /= len;
    float height = width / (aspect > 0 ? aspect : 1.0f);
    /* Local +Z faces back towards the head; +X is right as seen from it. */
    float zx = -fx, zz = -fz;
    float xx = zz, xz = -zx;
    mtl_screen_identity(model);
    model[0] = xx * width;
    model[1] = 0;
    model[2] = xz * width;
    model[4] = 0;
    model[5] = height;
    model[6] = 0;
    model[8] = zx;
    model[9] = 0;
    model[10] = zz;
    model[12] = head[12] + fx * distance;
    model[13] = head[13];
    model[14] = head[14] + fz * distance;
}

/* projection * inverse(origin_from_view) * model: what the vertex stage
   multiplies the unit quad's corners by, for one view of one drawable.
   origin_from_view is the device anchor's transform times the view's. */
static inline void mtl_screen_mvp(const float projection[16], const float origin_from_view[16], const float model[16],
                                  float mvp[16])
{
    float view[16], vm[16];
    mtl_screen_rigid_inverse(origin_from_view, view);
    mtl_screen_mul(view, model, vm);
    mtl_screen_mul(projection, vm, mvp);
}

/*
 * Stereo's eye plane. A stereo frame's eye image is a symmetric frustum
 * about that eye's forward axis (stereo.h), while the headset's views are
 * off-axis and differ per eye, so the image cannot be stretched over a
 * view. It is put instead on a plane `distance` metres ahead of the eye,
 * perpendicular to the head's forward direction and centred on the eye's
 * own axis, 2 tan_x distance wide and 2 tan_y distance tall: a pixel at
 * NDC (x, y) of the image is then in the direction (x tan_x, y tan_y)
 * from that eye, which is the direction the game rendered it for. The
 * eye's own projection does the rest.
 *
 * device_from_view is the view's transform (views[i].transform, the eye
 * in the device's space); the plane keeps the device's axes, which the
 * game's eyes share (they are offset along its right only). The plane is
 * head locked, as stage A's camera is. The unit quad is the screen's,
 * x and y in -0.5..0.5 with +Y the image's top.
 */
/*
 * Late rotation (stage C). Once the game's camera follows the head, a
 * frame is drawn for the head pose the runtime read before it, and is
 * shown a frame or two later at another. The eye plane is then turned by
 * the rotation between the two: from the device's axes now to the axes
 * the frame was drawn with, so every pixel stays in the world direction
 * it was drawn for and the world holds still between game frames. turn
 * is that rotation in the device's space (mtl_late_rotation); the plane
 * stays centred on the eye's position now, since only rotation can be
 * corrected on a flat image. The identity gives the head-locked plane.
 */
static inline void mtl_stereo_plane_model_turned(const float device_from_view[16], const float turn[16], float tan_x,
                                                 float tan_y, float distance, float model[16])
{
    float w = 2.0f * tan_x * distance, h = 2.0f * tan_y * distance;
    mtl_screen_identity(model);
    for (int r = 0; r < 3; r++) {
        model[0 + r] = turn[0 + r] * w;
        model[4 + r] = turn[4 + r] * h;
        model[8 + r] = turn[8 + r];
        model[12 + r] = device_from_view[12 + r] - turn[8 + r] * distance;
    }
}

static inline void mtl_stereo_plane_model(const float device_from_view[16], float tan_x, float tan_y, float distance,
                                          float model[16])
{
    float id[16];
    mtl_screen_identity(id);
    mtl_stereo_plane_model_turned(device_from_view, id, tan_x, tan_y, distance, model);
}

static inline void mtl_stereo_plane_mvp_turned(const float projection[16], const float device_from_view[16],
                                               const float turn[16], float tan_x, float tan_y, float distance,
                                               float mvp[16])
{
    float model[16];
    mtl_stereo_plane_model_turned(device_from_view, turn, tan_x, tan_y, distance, model);
    mtl_screen_mvp(projection, device_from_view, model, mvp);
}

static inline void mtl_stereo_plane_mvp(const float projection[16], const float device_from_view[16], float tan_x,
                                        float tan_y, float distance, float mvp[16])
{
    float id[16];
    mtl_screen_identity(id);
    mtl_stereo_plane_mvp_turned(projection, device_from_view, id, tan_x, tan_y, distance, mvp);
}

/*
 * The fixed screen (theater). Without head look the game's camera is its
 * own, so a plane that follows the head turns the whole picture with
 * every glance. The eye planes are instead pinned where the head was when
 * the screen was placed: the anchor is that pose levelled (its heading
 * and position, no pitch or roll), and each eye's plane is where it would
 * have been for that head, centred on that eye's axis then, so the pair
 * is a stereo screen standing in the room. Turning the head looks across
 * it; leaning moves the eyes off the axes the images were drawn for, as
 * at any 3D screen, and the depth bends a little with it rather than the
 * screen swimming.
 */
static inline void mtl_stereo_anchor(const float origin_from_device[16], float anchor[16])
{
    float fx = -origin_from_device[8], fz = -origin_from_device[10];
    float len = sqrtf(fx * fx + fz * fz);
    if (len < 1e-3f) {
        float s = origin_from_device[9] > 0 ? 1.0f : -1.0f;
        fx = s * origin_from_device[4];
        fz = s * origin_from_device[6];
        len = sqrtf(fx * fx + fz * fz);
        if (len < 1e-6f) {
            fx = 0;
            fz = -1;
            len = 1;
        }
    }
    fx /= len;
    fz /= len;
    mtl_screen_identity(anchor);
    /* +X right, +Y up, +Z back: forward is -Z. */
    anchor[0] = -fz;
    anchor[2] = fx;
    anchor[8] = -fx;
    anchor[10] = -fz;
    for (int r = 0; r < 3; r++) anchor[12 + r] = origin_from_device[12 + r];
}

/* The eye plane in the device's space now: device_from_anchor is the
   rigid move from the anchor to the device now,
   inverse(origin_from_device) * anchor. The identity is the head-locked
   plane of mtl_stereo_plane_model. */
static inline void mtl_stereo_plane_model_fixed(const float device_from_view[16], const float device_from_anchor[16],
                                                float tan_x, float tan_y, float distance, float model[16])
{
    float w = 2.0f * tan_x * distance, h = 2.0f * tan_y * distance;
    float c[4] = {device_from_view[12], device_from_view[13], device_from_view[14] - distance, 1.0f}, at[4];
    mtl_screen_transform(device_from_anchor, c, at);
    mtl_screen_identity(model);
    for (int r = 0; r < 3; r++) {
        model[0 + r] = device_from_anchor[0 + r] * w;
        model[4 + r] = device_from_anchor[4 + r] * h;
        model[8 + r] = device_from_anchor[8 + r];
        model[12 + r] = at[r];
    }
}

static inline void mtl_stereo_plane_mvp_fixed(const float projection[16], const float device_from_view[16],
                                              const float origin_from_device[16], const float anchor[16], float tan_x,
                                              float tan_y, float distance, float mvp[16])
{
    float device_from_origin[16], device_from_anchor[16], model[16];
    mtl_screen_rigid_inverse(origin_from_device, device_from_origin);
    mtl_screen_mul(device_from_origin, anchor, device_from_anchor);
    mtl_stereo_plane_model_fixed(device_from_view, device_from_anchor, tan_x, tan_y, distance, model);
    mtl_screen_mvp(projection, device_from_view, model, mvp);
}

/*
 * Full view's eye plane. A full view eye image spans its eye's tangents
 * (left, right, down, up, the margin included) about the head's axes as
 * the frame was drawn, so it goes on an off-axis plane `distance` ahead
 * of the eye covering exactly that: (l + r) distance wide, (d + u)
 * distance tall, its centre ((r - l) / 2, (u - d) / 2) distance off the
 * eye's forward axis. turn is the late rotation (mtl_late_rotation), the
 * identity for the head-locked plane. The plane keeps the device's axes,
 * which the eye images were drawn in, not the view's own: each view is
 * turned a fraction of a degree from the device, and the image's
 * directions are the head's. With the identity turn, a view with the
 * device's axes and tan that view's own tangents, the corners land on
 * NDC (-1, -1) and (1, 1); with the margin they land outside, and the
 * margin is what the late rotation turns into view.
 */
static inline void mtl_fullview_plane_model(const float device_from_view[16], const float turn[16],
                                            const float tan[4], float distance, float model[16])
{
    float w = (tan[0] + tan[1]) * distance, h = (tan[2] + tan[3]) * distance;
    float c[3] = {(tan[1] - tan[0]) * 0.5f * distance, (tan[3] - tan[2]) * 0.5f * distance, -distance};
    mtl_screen_identity(model);
    for (int r = 0; r < 3; r++) {
        model[0 + r] = turn[0 + r] * w;
        model[4 + r] = turn[4 + r] * h;
        model[8 + r] = turn[8 + r];
        model[12 + r] = device_from_view[12 + r] + turn[0 + r] * c[0] + turn[4 + r] * c[1] + turn[8 + r] * c[2];
    }
}

static inline void mtl_fullview_plane_mvp(const float projection[16], const float device_from_view[16],
                                          const float turn[16], const float tan[4], float distance, float mvp[16])
{
    float model[16];
    mtl_fullview_plane_model(device_from_view, turn, tan, distance, model);
    mtl_screen_mvp(projection, device_from_view, model, mvp);
}

/* A pose's heading and elevation in degrees, in the convention of
   metal_d3d9_shown_frame's pose: yaw positive turning left from -Z, pitch
   positive looking up. Straight up or down leaves no heading in the
   forward axis, so it comes from the up axis, as in mtl_screen_place. */
static inline void mtl_head_yaw_pitch(const float m[16], float *yaw_deg, float *pitch_deg)
{
    const float deg = 57.29577951f;
    float fx = -m[8], fy = -m[9], fz = -m[10];
    if (fy > 1.0f) fy = 1.0f;
    if (fy < -1.0f) fy = -1.0f;
    float hx = fx, hz = fz;
    if (hx * hx + hz * hz < 1e-6f) {
        float s = m[9] > 0 ? 1.0f : -1.0f;
        hx = s * m[4];
        hz = s * m[6];
    }
    *yaw_deg = hx * hx + hz * hz > 0 ? atan2f(-hx, -hz) * deg : 0.0f;
    *pitch_deg = asinf(fy) * deg;
}

/* The rotation of a head at yaw and pitch with no roll: yaw about +Y,
   then pitch about the turned +X, column major with no translation. */
static inline void mtl_view_rotation(float yaw_deg, float pitch_deg, float r[16])
{
    const float rad = 0.01745329252f;
    float cy = cosf(yaw_deg * rad), sy = sinf(yaw_deg * rad), cp = cosf(pitch_deg * rad), sp = sinf(pitch_deg * rad);
    mtl_screen_identity(r);
    r[0] = cy;
    r[1] = 0;
    r[2] = -sy;
    r[4] = sy * sp;
    r[5] = cp;
    r[6] = cy * sp;
    r[8] = sy * cp;
    r[9] = -sp;
    r[10] = cy * cp;
}

/* The turn for mtl_stereo_plane_model_turned: the drawn orientation (yaw
   and pitch, no roll) seen from the device's orientation now,
   transpose(R_now) * R_drawn. The head's roll is in R_now and not in
   the drawn frame, so the plane rolls against the head and the world
   stays level: the game cannot roll, the compositor does it here. */
static inline void mtl_late_rotation(const float origin_from_device[16], float yaw_deg, float pitch_deg,
                                     float turn[16])
{
    float drawn[16], now_t[16];
    mtl_view_rotation(yaw_deg, pitch_deg, drawn);
    mtl_screen_identity(now_t);
    for (int c = 0; c < 3; c++)
        for (int r = 0; r < 3; r++) now_t[c * 4 + r] = origin_from_device[r * 4 + c];
    mtl_screen_mul(now_t, drawn, turn);
    turn[12] = turn[13] = turn[14] = 0;
    turn[3] = turn[7] = turn[11] = 0;
    turn[15] = 1;
}

/* The angle of a rotation in degrees, for the late rotation's log: from
   both its cosine (the trace) and its sine (the skew part), since the
   cosine alone rounds away the tenths of a degree the log is about. */
static inline float mtl_rotation_angle_deg(const float r[16])
{
    float c = (r[0] + r[5] + r[10] - 1.0f) * 0.5f;
    float x = r[6] - r[9], y = r[8] - r[2], z = r[1] - r[4];
    return atan2f(0.5f * sqrtf(x * x + y * y + z * z), c) * 57.29577951f;
}

/* A view's half-angle tangents from its projection (computeProjection's
   right-up-back convention, column major): left, right, down, up, each
   positive outwards. NDC x = (P00 t - P20) at a direction of tangent t
   with z forward negative, so the edges are t = (1 + P20) / P00 and
   (1 - P20) / P00, and the same for y. */
static inline void mtl_stereo_tangents(const float p[16], float out[4])
{
    out[0] = (1.0f - p[8]) / p[0];
    out[1] = (1.0f + p[8]) / p[0];
    out[2] = (1.0f - p[9]) / p[5];
    out[3] = (1.0f + p[9]) / p[5];
}

#endif
