/*
 * Immersive mode's Present: the back buffer goes into the handoff ring
 * (handoff.h) for a compositor to sample, instead of into a window's
 * layer. See immersive.h for the host's side.
 *
 * The copy is encoded at the end of the frame's own command buffer, on
 * whichever thread replays lists, and the slot is published from that
 * command buffer's completion. Present still only closes the list. What
 * the game loses is the pacing a layer gave it (nextDrawable waits when
 * three drawables are queued), so a cheap scene such as a menu would
 * otherwise run as fast as the GPU allows and heat the headset for frames
 * nobody sees. RT_IMMERSIVE_MAX_HZ (default 90; 0 for none) holds the
 * game thread back after each Present, once the frame is queued.
 *
 * That wait used to sit on the replaying thread, between a frame's replay
 * and its commit. It made the replay the bottleneck, so the game filled
 * both in-flight frames behind it and every frame waited about two pacing
 * periods before the GPU saw it: 30 of the 38 ms median age the first
 * device run measured. On the game thread it falls between frames, and
 * the next frame starts later and samples input later, which is where a
 * frame cap's wait belongs. The in-flight depth is also one frame while
 * immersive (frame.m): the compositor paces itself, so the game's frames
 * need no queue in front of the GPU.
 *
 * The slot textures are made on first use and remade only while their
 * slot is the writer's, so a texture the compositor holds is never
 * released under it.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "handoff.h"
#include "immersive.h"
#include "mtl_objc.h"
#include "space.h"

#include <mach/mach_time.h>
#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Seconds between the counters' log lines. */
#define LOG_PERIOD_NS 5000000000ull

static atomic_int s_enabled;
/* While immersive, whether Present also shows the frame in the window's
   layer: on from the space's request until the window is dismissed, so
   the window shows the game while the space opens (space.h). */
static atomic_int s_window_too;
static mtl_space s_space;
/* Set once on, so the exit report still runs after the space closed and
   Present went back to the window. */
static atomic_int s_ever;
static mtl_handoff s_ring;
static pthread_once_t s_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t s_log_mu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t s_period_start;
static uint64_t s_pace_ns, s_last_present; /* s_last_present: game thread only */
/* The slot the list being replayed writes, from its present to its
   commit: touched only by the one thread that replays lists. */
static int s_write_slot = -1;
static _Atomic uint64_t s_pace_waits, s_pace_wait_ns;
static FILE *s_trace;
static int s_trace_tried;
static _Atomic uint64_t s_copies, s_copy_gpu_ns, s_aborts;
/* What each slot's stereo frame was drawn with (immersive.h): written by
   the replaying thread between a slot's begin and publish, read by the
   compositor while it holds the slot, so the ring's own lock orders them. */
static float s_slot_view[MTL_HANDOFF_SLOTS][4];
/* And the head pose each slot's frame was drawn for (valid, yaw, pitch),
   under the same rule. */
static float s_slot_pose[MTL_HANDOFF_SLOTS][3];
/* The head pose the runtime gave for the next Present: the two angles'
   float bits in one word, so a Present never pairs one pose's yaw with
   another's pitch, and whether it is valid. */
static _Atomic uint64_t s_pose_bits;
static atomic_int s_pose_valid;
/* Stereo: asked for by the host, and the half IPD in game units, kept as
   float bits so the replaying thread reads it without a lock. */
static atomic_int s_stereo;
static _Atomic uint32_t s_half_ipd_bits;
static float s_units_per_m;
/* The eye planes' size for the session (stereo.h, mtl_stereo_screen_a):
   RT_STEREO_SCREEN_DEG, degrees across a 4:3 view as the game's own FOV
   is given, 0 for the nominal 75. s_screen_logged holds the aspect's
   float bits last logged, so the size is logged once, and again only if
   a Reset changes the back buffer's shape. */
static float s_screen_deg;
static _Atomic uint32_t s_screen_logged;
/* Full view (immersive.h): asked for, in play (from the runtime, once a
   game frame), and the check that keeps the game's projection per eye.
   The views and the settings read when it is turned on are under
   s_views_mu: set from the compositor, read once a frame where the frame
   is latched. The back buffer's aspect at the last Present is the shape
   the game's field is fitted to. */
static atomic_int s_fullview, s_play, s_game_eye;
static pthread_mutex_t s_views_mu = PTHREAD_MUTEX_INITIALIZER;
static float s_view_tan[8], s_eye_m[6];
static int s_views_set;
static float s_fv_margin = METAL_D3D9_FULLVIEW_MARGIN, s_fv_hud_deg = METAL_D3D9_FULLVIEW_HUD_DEG;
static float s_fv_sky_scale = 1.0f / 16.0f, s_fv_aura_deg = METAL_D3D9_FULLVIEW_AURA_DEG;
/* Whether the frames draw a view model (draw.m, mtl_fv_scan): -1 not
   known yet, 0 none (third person), 1 drawn. Read by the runtime. */
static _Atomic int s_view_model = -1;
static _Atomic uint32_t s_back_aspect_bits;
static atomic_int s_fv_logged;
/* And each slot's full view frame, under the rule s_slot_view keeps. */
static int s_slot_fv[MTL_HANDOFF_SLOTS];
static float s_slot_tan[MTL_HANDOFF_SLOTS][2][4];
/* Its HUD panel's half tangents and the game's crosshair on it (draw.m,
   crosshair), latched with the frame as its pose is. */
static float s_slot_panel[MTL_HANDOFF_SLOTS][2];
static mtl_crosshair_frame s_slot_xhair[MTL_HANDOFF_SLOTS];
/* The third person aim: the runtime's reader, and each slot's samples,
   the GPU's pixels and reads at them (a shared buffer of
   MTL_AIM_BUF_FLOATS, made once), and the search's answer with the projection that turns it
   into tangents, written by the frame's completion before it is
   published. */
static _Atomic(int (*)(float *, float *)) s_aim_fn;
static _Atomic(int (*)(float *, int *)) s_aim_trace_fn;
static _Atomic(int (*)(float *)) s_hands_fn;
static _Atomic(int (*)(float *)) s_hands_bind_fn;
static mtl_aim_ray s_slot_ray[MTL_HANDOFF_SLOTS];
static mtl_aim_probes s_slot_probes[MTL_HANDOFF_SLOTS];
static void *s_slot_aim_buf[MTL_HANDOFF_SLOTS];
static int s_slot_aim_on[MTL_HANDOFF_SLOTS];
static mtl_aim_hit s_slot_hit[MTL_HANDOFF_SLOTS];
/* The view model's hands each slot's frame drew, and its projection. */
static mtl_vm_hands s_slot_hands[MTL_HANDOFF_SLOTS];
static mtl_stereo_proj_full s_slot_hands_p[MTL_HANDOFF_SLOTS];
static int s_slot_hit_ok[MTL_HANDOFF_SLOTS];
static mtl_stereo_proj_full s_slot_aim_p[MTL_HANDOFF_SLOTS];
/* The arm HUD (draw.m, arm_side): on unless RT_ARM_HUD=0, and each side
   as the compositor last said it can show that side's card. Each slot's
   copy of the cards, its own texture remade only while the slot is the
   writer's as the frame's is, which sides it holds and where. */
static int s_arm_env = 1;
static atomic_int s_arm_side[2];
static void *s_slot_arm[MTL_HANDOFF_SLOTS];
static int s_slot_arm_on[MTL_HANDOFF_SLOTS][2];
static float s_slot_arm_uv[MTL_HANDOFF_SLOTS][2][4];
/* With the orbs: each slot's pictures, the game's own textures (0 the
   weapon's, 1 the discipline's), retained while the slot holds them
   (draw.m, orb_note), and each side's label on the slot's copy of the
   labels' target: on, and its uv there. */
static void *s_slot_icon[MTL_HANDOFF_SLOTS][2];
/* The scene's depth (metal_d3d9_fullview_scene_depth): asked for, and each
   slot's target, made at a quarter of the frame's pixels the first time
   and again when the frame's size changes, and whether it holds this
   slot's frame. */
static atomic_int s_scene_depth;
static void *s_slot_scene[MTL_HANDOFF_SLOTS];
static int s_slot_scene_on[MTL_HANDOFF_SLOTS];
static int s_slot_scene_vm[MTL_HANDOFF_SLOTS];
static int s_slot_label_on[MTL_HANDOFF_SLOTS][2];
static float s_slot_label_uv[MTL_HANDOFF_SLOTS][2][4];

static uint64_t now_ns(void)
{
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    return mach_absolute_time() * tb.numer / tb.denom;
}

static void init_once(void)
{
    mtl_handoff_init(&s_ring);
    mtl_space_init(&s_space);
    const char *v = getenv("RT_IMMERSIVE_MAX_HZ");
    double hz = v && *v ? atof(v) : 90.0;
    s_pace_ns = hz > 0 ? (uint64_t)(1e9 / hz) : 0;
    /* The engine's unit is an inch (a 72 unit tall player); a mod or a
       test may say otherwise. */
    const char *u = getenv("RT_STEREO_UNITS_PER_M");
    double upm = u && *u ? atof(u) : 1.0 / 0.0254;
    s_units_per_m = upm > 0 ? (float)upm : (float)(1.0 / 0.0254);
    /* The gauges leave the panel for the orbs (the default) or for the
       old cards (RT_ARM_HUD=1); with neither they stay on it. */
    const char *arm = getenv("RT_ARM_HUD"), *orb = getenv("RT_ORB_HUD");
    s_arm_env = (arm && !strcmp(arm, "1")) || !(orb && !strcmp(orb, "0"));
    if (!s_arm_env) mtl_log("full view: no orbs nor cards (RT_ORB_HUD=0), the gauges stay on the panel");
    const char *sd = getenv("RT_STEREO_SCREEN_DEG");
    s_screen_deg = sd && *sd ? (float)atof(sd) : 0.0f;
    float h = 0.032f * s_units_per_m;
    uint32_t bits;
    memcpy(&bits, &h, 4);
    atomic_store(&s_half_ipd_bits, bits);
}

static void ensure_init(void) { pthread_once(&s_once, init_once); }

D9_EXPORT void metal_d3d9_immersive_enable(int on)
{
    ensure_init();
    atomic_store(&s_enabled, on != 0);
    if (on) atomic_store(&s_ever, 1);
    if (on && s_pace_ns)
        mtl_log("immersive: Present hands frames to the compositor (%d slots, one frame in flight, the game "
                "held to one frame every %.2f ms after its Present)",
                MTL_HANDOFF_SLOTS, s_pace_ns / 1e6);
    else if (on)
        mtl_log("immersive: Present hands frames to the compositor (%d slots, one frame in flight, no frame cap)",
                MTL_HANDOFF_SLOTS);
}

D9_EXPORT int metal_d3d9_immersive_enabled(void) { return atomic_load(&s_enabled); }

D9_EXPORT uint64_t metal_d3d9_immersive_frames(void)
{
    ensure_init();
    mtl_handoff_stats s;
    mtl_handoff_totals(&s_ring, &s);
    return s.published;
}

int mtl_immersive_on(void) { return atomic_load_explicit(&s_enabled, memory_order_relaxed); }

int mtl_immersive_window_too(void) { return atomic_load_explicit(&s_window_too, memory_order_relaxed); }

/* The space's outputs, applied after each of its steps. The ring is
   logged only as it turns on, as metal_d3d9_immersive_enable does. */
static void apply_space(void)
{
    int ring, window;
    mtl_space_outputs(&s_space, &ring, &window);
    atomic_store(&s_window_too, window);
    if (ring != atomic_load(&s_enabled)) metal_d3d9_immersive_enable(ring);
}

D9_EXPORT void metal_d3d9_space_request(void)
{
    ensure_init();
    mtl_space_request(&s_space, now_ns());
    apply_space();
}

D9_EXPORT int metal_d3d9_space_should_open(void)
{
    ensure_init();
    int r = mtl_space_should_open(&s_space, atomic_load(&s_copies), now_ns());
    if (r == 1) mtl_log("immersive: a frame is in the ring; opening the space");
    return r;
}

D9_EXPORT void metal_d3d9_space_open_result(int result)
{
    ensure_init();
    mtl_space_open_result(&s_space, result, now_ns());
    apply_space();
}

D9_EXPORT void metal_d3d9_space_compositor_started(void)
{
    ensure_init();
    mtl_space_compositor_started(&s_space, now_ns());
}

D9_EXPORT void metal_d3d9_space_first_frame_shown(void)
{
    ensure_init();
    mtl_space_first_shown(&s_space, now_ns());
}

D9_EXPORT int metal_d3d9_space_take_dismiss(void)
{
    ensure_init();
    int r = mtl_space_take_dismiss(&s_space, now_ns());
    apply_space();
    return r;
}

D9_EXPORT int metal_d3d9_space_closed(void)
{
    ensure_init();
    int r = mtl_space_closed(&s_space, now_ns());
    apply_space();
    return r;
}

D9_EXPORT void metal_d3d9_space_window_back(void)
{
    ensure_init();
    mtl_space_window_back(&s_space, now_ns());
    apply_space();
}

D9_EXPORT int metal_d3d9_space_reopen(void)
{
    ensure_init();
    int r = mtl_space_reopen(&s_space, atomic_load(&s_copies), now_ns());
    if (r) mtl_log("immersive: the space is wanted again (Return to Immersive); frames to the ring");
    apply_space();
    return r;
}

D9_EXPORT void metal_d3d9_space_watchdog(void)
{
    ensure_init();
    char line[1024];
    if (mtl_space_watchdog(&s_space, atomic_load(&s_copies), now_ns(), line, sizeof line))
        mtl_log("immersive %s", line);
}

D9_EXPORT void metal_d3d9_immersive_log(const char *msg) { mtl_log("immersive: %s", msg ? msg : ""); }

D9_EXPORT void metal_d3d9_immersive_view_pose(int valid, float yaw_deg, float pitch_deg)
{
    uint32_t y, p;
    memcpy(&y, &yaw_deg, 4);
    memcpy(&p, &pitch_deg, 4);
    atomic_store_explicit(&s_pose_bits, (uint64_t)y << 32 | p, memory_order_relaxed);
    atomic_store_explicit(&s_pose_valid, valid != 0, memory_order_release);
}

/* The latest hand poses, one slot per side, behind a sequence word that
   is odd while a write is under way, so a later reader takes a whole
   pose. Nothing reads them yet. */
static struct {
    _Atomic uint32_t seq;
    int state;
    float grip[16], aim[16];
    double time;
} s_hands[2];

D9_EXPORT void metal_d3d9_immersive_hand_pose(int side, int state, const float grip[16], const float aim[16], double time)
{
    if (side < 0 || side > 1) return;
    atomic_fetch_add_explicit(&s_hands[side].seq, 1, memory_order_acq_rel);
    s_hands[side].state = state;
    if (grip) memcpy(s_hands[side].grip, grip, sizeof s_hands[side].grip);
    if (aim) memcpy(s_hands[side].aim, aim, sizeof s_hands[side].aim);
    s_hands[side].time = time;
    atomic_fetch_add_explicit(&s_hands[side].seq, 1, memory_order_acq_rel);
}

D9_EXPORT void metal_d3d9_fullview_scene_depth(int on)
{
    if (atomic_exchange_explicit(&s_scene_depth, on ? 1 : 0, memory_order_relaxed) != (on ? 1 : 0))
        mtl_log("full view: the scene's depth %s", on ? "handed to the compositor" : "no longer handed on");
}

static id<MTLTexture> slot_scene_texture(id<MTLDevice> dev, int slot, uint32_t w, uint32_t h)
{
    id<MTLTexture> t = (__bridge id<MTLTexture>)s_slot_scene[slot];
    uint32_t sw = w / 2 > 0 ? w / 2 : 1, sh = h / 2 > 0 ? h / 2 : 1;
    if (t && t.width == sw && t.height == sh) return t;
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float
                                                                                  width:sw
                                                                                 height:sh
                                                                              mipmapped:NO];
    td.textureType = MTLTextureType2DArray;
    td.arrayLength = 2;
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    td.storageMode = MTLStorageModePrivate;
    id<MTLTexture> n = [dev newTextureWithDescriptor:td];
    if (!n) return nil;
    n.label = [NSString stringWithFormat:@"immersive slot %d scene depth", slot];
    if (s_slot_scene[slot]) CFRelease(s_slot_scene[slot]);
    s_slot_scene[slot] = (void *)CFBridgingRetain(n);
    return n;
}

D9_EXPORT void metal_d3d9_fullview_arm_hud(int side, int on)
{
    if (side < 0 || side > 1) return;
    on = on < 0 ? 0 : on > 2 ? 2 : on;
    if (atomic_exchange_explicit(&s_arm_side[side], on, memory_order_relaxed) != on)
        mtl_log("full view: the %s gauge %s", side ? "right" : "left",
                on == 2   ? "left to its orb at the controller"
                : on == 1 ? "on its card at the controller"
                          : "on the panel (no controller for it)");
}

void mtl_immersive_pose_now(float pose[3])
{
    int valid = atomic_load_explicit(&s_pose_valid, memory_order_acquire);
    uint64_t b = atomic_load_explicit(&s_pose_bits, memory_order_relaxed);
    uint32_t y = (uint32_t)(b >> 32), p = (uint32_t)b;
    pose[0] = valid ? 1.0f : 0.0f;
    memcpy(&pose[1], &y, 4);
    memcpy(&pose[2], &p, 4);
}

D9_EXPORT void metal_d3d9_stereo_enable(int on)
{
    ensure_init();
    if (atomic_exchange(&s_stereo, on != 0) == (on != 0)) return;
    float h;
    uint32_t bits = atomic_load(&s_half_ipd_bits);
    memcpy(&h, &bits, 4);
    mtl_log("stereo: %s (half IPD %.2f game units, %.1f units a metre)",
            on ? "each frame replayed once per eye from its next frame" : "off, one replay a frame", h, s_units_per_m);
}

D9_EXPORT int metal_d3d9_stereo_enabled(void) { return atomic_load(&s_stereo); }

D9_EXPORT void metal_d3d9_stereo_set_half_ipd(float metres)
{
    ensure_init();
    if (!(metres >= 0.0f) || metres > 0.1f) return;
    float h = metres * s_units_per_m;
    uint32_t bits;
    memcpy(&bits, &h, 4);
    if (atomic_exchange(&s_half_ipd_bits, bits) != bits)
        mtl_log("stereo: half IPD %.1f mm, %.3f game units", metres * 1000.0f, h);
}

int mtl_stereo_on(void)
{
    return atomic_load_explicit(&s_stereo, memory_order_relaxed) && atomic_load_explicit(&s_enabled, memory_order_relaxed);
}

float mtl_stereo_hud_z(void)
{
    ensure_init();
    return METAL_D3D9_STEREO_HUD_M * s_units_per_m;
}

float mtl_stereo_screen_a_for(float aspect)
{
    ensure_init();
    float a = mtl_stereo_screen_a(aspect, s_screen_deg);
    uint32_t bits;
    memcpy(&bits, &aspect, 4);
    if (atomic_exchange_explicit(&s_screen_logged, bits, memory_order_relaxed) != bits) {
        int set = s_screen_deg >= 10.0f && s_screen_deg <= 170.0f;
        mtl_log("stereo: eye planes fixed for the session at %.1f by %.1f degrees (a %.4f, %.1f across 4:3, hor+, "
                "aspect %.4f; %s), at %.2f m",
                2.0 * atan(1.0 / a) * 180.0 / M_PI, 2.0 * atan(1.0 / (a * aspect)) * 180.0 / M_PI, a,
                set ? s_screen_deg : MTL_STEREO_NOMINAL_DEG, aspect,
                set ? "RT_STEREO_SCREEN_DEG" : "the game's nominal FOV, RT_STEREO_SCREEN_DEG to change",
                METAL_D3D9_STEREO_HUD_M);
    }
    return a;
}

/* The headset's views until the compositor gives its own. */
static void default_views(float tan[8], float eye_m[6])
{
    const float l[8] = {METAL_D3D9_FULLVIEW_TAN_OUT, METAL_D3D9_FULLVIEW_TAN_NOSE, METAL_D3D9_FULLVIEW_TAN_DOWN,
                        METAL_D3D9_FULLVIEW_TAN_UP,  METAL_D3D9_FULLVIEW_TAN_NOSE, METAL_D3D9_FULLVIEW_TAN_OUT,
                        METAL_D3D9_FULLVIEW_TAN_DOWN, METAL_D3D9_FULLVIEW_TAN_UP};
    const float e[6] = {-0.032f, 0, 0, 0.032f, 0, 0};
    memcpy(tan, l, sizeof l);
    memcpy(eye_m, e, sizeof e);
}

static float f32(uint32_t bits)
{
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static float back_aspect(void)
{
    float a = f32(atomic_load_explicit(&s_back_aspect_bits, memory_order_relaxed));
    return a > 0.0f ? a : 2048.0f / 1408.0f;
}

/* The game's symmetric field from the views: the union of both eyes'
   tangents with the margin, across and up, widened so that at the back
   buffer's shape (square pixels, so tan_y = tan_x / aspect) neither is
   cut. Under s_views_mu. */
static void game_tan_locked(float aspect, float *tx, float *ty)
{
    float lx = 0, ly = 0;
    for (int e = 0; e < 2; e++) {
        const float *t = s_view_tan + 4 * e;
        lx = fmaxf(lx, fmaxf(t[0], t[1]));
        ly = fmaxf(ly, fmaxf(t[2], t[3]));
    }
    lx *= 1.0f + s_fv_margin;
    ly *= 1.0f + s_fv_margin;
    *tx = fmaxf(lx, ly * aspect);
    *ty = *tx / aspect;
}

D9_EXPORT void metal_d3d9_fullview_enable(int on)
{
    ensure_init();
    if (on) {
        pthread_mutex_lock(&s_views_mu);
        if (!s_views_set) default_views(s_view_tan, s_eye_m);
        const char *m = getenv("RT_FULLVIEW_MARGIN"), *h = getenv("RT_FULLVIEW_HUD_DEG"), *g = getenv("RT_FULLVIEW_EYE"),
                   *k = getenv("RT_FULLVIEW_SKY"), *u = getenv("RT_FULLVIEW_AURA_DEG");
        /* The 3D sky's scale: the source engine's sky camera is usually 16
           to 1. 0 puts the sky at infinity. */
        float kv = k && *k ? (float)atof(k) : 16.0f;
        s_fv_sky_scale = kv >= 1.0f && kv <= 1024.0f ? 1.0f / kv : 0.0f;
        float mv = m && *m ? (float)atof(m) : METAL_D3D9_FULLVIEW_MARGIN, hv = h && *h ? (float)atof(h) : 0.0f;
        s_fv_margin = mv >= 0.0f && mv <= 0.5f ? mv : METAL_D3D9_FULLVIEW_MARGIN;
        s_fv_hud_deg = hv >= 10.0f && hv <= 150.0f ? hv : METAL_D3D9_FULLVIEW_HUD_DEG;
        float uv = u && *u ? (float)atof(u) : 0.0f;
        s_fv_aura_deg = uv >= 10.0f && uv <= 150.0f ? uv : METAL_D3D9_FULLVIEW_AURA_DEG;
        pthread_mutex_unlock(&s_views_mu);
        if (g && !strcmp(g, "game")) atomic_store(&s_game_eye, 1);
    }
    if (atomic_exchange(&s_fullview, on != 0) == (on != 0)) return;
    mtl_log("full view: %s (margin %.2f, HUD panel %.0f degrees across%s)",
            on ? "play frames fill each eye's view, other frames on the plane" : "off", s_fv_margin, s_fv_hud_deg,
            atomic_load(&s_game_eye) ? "; check: each eye keeps the game's projection" : "");
}

D9_EXPORT int metal_d3d9_fullview_enabled(void) { return atomic_load(&s_fullview); }

D9_EXPORT void metal_d3d9_fullview_game_eye(int on)
{
    if (atomic_exchange(&s_game_eye, on != 0) != (on != 0))
        mtl_log("full view: each eye %s", on ? "keeps the game's projection (check)" : "its own view");
}

D9_EXPORT void metal_d3d9_fullview_set_views(const float tan[8], const float eye_m[6])
{
    ensure_init();
    for (int i = 0; i < 8; i++)
        if (!(tan[i] > 0.05f && tan[i] < 20.0f)) return;
    pthread_mutex_lock(&s_views_mu);
    /* The compositor sends its views every frame and the eye positions
       wander by fractions of a millimetre as tracking settles, so only a
       real change (a thousandth of a tangent, half a millimetre) is
       logged; an exact compare logged one line per frame. */
    static float logged_tan[8], logged_eye[6];
    static int logged;
    int same = logged;
    for (int i = 0; i < 8 && same; i++) same = fabsf(logged_tan[i] - tan[i]) < 1e-3f;
    for (int i = 0; i < 6 && same; i++) same = fabsf(logged_eye[i] - eye_m[i]) < 5e-4f;
    if (!same) {
        memcpy(logged_tan, tan, sizeof logged_tan);
        memcpy(logged_eye, eye_m, sizeof logged_eye);
        logged = 1;
    }
    memcpy(s_view_tan, tan, sizeof s_view_tan);
    memcpy(s_eye_m, eye_m, sizeof s_eye_m);
    s_views_set = 1;
    float tx, ty;
    game_tan_locked(back_aspect(), &tx, &ty);
    pthread_mutex_unlock(&s_views_mu);
    if (!same)
        mtl_log("full view: views left %.3f %.3f %.3f %.3f, right %.3f %.3f %.3f %.3f (left, right, down, up), eyes "
                "%.1f mm apart; the game drawn %.1f by %.1f degrees",
                tan[0], tan[1], tan[2], tan[3], tan[4], tan[5], tan[6], tan[7],
                1000.0f * sqrtf((eye_m[3] - eye_m[0]) * (eye_m[3] - eye_m[0]) +
                                (eye_m[4] - eye_m[1]) * (eye_m[4] - eye_m[1]) +
                                (eye_m[5] - eye_m[2]) * (eye_m[5] - eye_m[2])),
                2.0 * atan(tx) * 180.0 / M_PI, 2.0 * atan(ty) * 180.0 / M_PI);
}

void mtl_immersive_view_model(int drawn) { atomic_store_explicit(&s_view_model, drawn ? 1 : 0, memory_order_relaxed); }

D9_EXPORT void metal_d3d9_fullview_aim_source(int (*fn)(float eye[3], float dir[3]))
{
    atomic_store_explicit(&s_aim_fn, fn, memory_order_release);
}

D9_EXPORT void metal_d3d9_fullview_aim_trace_source(int (*fn)(float *t, int *hit))
{
    atomic_store_explicit(&s_aim_trace_fn, fn, memory_order_release);
}

int mtl_immersive_aim_trace(float *t, int *hit)
{
    int (*fn)(float *, int *) = atomic_load_explicit(&s_aim_trace_fn, memory_order_acquire);
    return fn ? fn(t, hit) != 0 : 0;
}

D9_EXPORT void metal_d3d9_fullview_hands_source(int (*fn)(float pts[18]))
{
    atomic_store_explicit(&s_hands_fn, fn, memory_order_release);
}

D9_EXPORT void metal_d3d9_fullview_hands_bind_source(int (*fn)(float out[32]))
{
    atomic_store_explicit(&s_hands_bind_fn, fn, memory_order_release);
}

int mtl_immersive_hands_bind(float out[32])
{
    int (*fn)(float *) = atomic_load_explicit(&s_hands_bind_fn, memory_order_acquire);
    return fn ? fn(out) & 3 : 0;
}

int mtl_immersive_hands(float pts[18])
{
    int (*fn)(float *) = atomic_load_explicit(&s_hands_fn, memory_order_acquire);
    return fn ? fn(pts) & 3 : 0;
}

int mtl_immersive_aim(float eye[3], float dir[3])
{
    int (*fn)(float *, float *) = atomic_load_explicit(&s_aim_fn, memory_order_acquire);
    return fn ? fn(eye, dir) != 0 : 0;
}

/* The search's answer, the first time and then on a change between hit
   and no hit or every two seconds, the first 60 times: the ray, the hit
   along it and in the frame, and the frame's camera from its own rows
   beside the eye, which a run compares with the runtime's aim line. */
static void aim_log(const mtl_aim_ray *r, const mtl_aim_probes *p, const mtl_aim_hit *h, int ok)
{
    static int logged, last = -2;
    static uint64_t last_ns;
    uint64_t now = now_ns();
    int state = ok ? h->hit : -1;
    if (logged >= 60 || (state == last && now - last_ns < 2000000000ull)) return;
    logged++;
    last = state;
    last_ns = now;
    float cam[3] = {0, 0, 0};
    int cam_ok = mtl_stereo_camera_from_rows(r->rows[0], r->rows[1], r->rows[3], cam);
    float dx = cam[0] - r->eye[0], dy = cam[1] - r->eye[1], dz = cam[2] - r->eye[2];
    char how[64];
    if (p) snprintf(how, sizeof how, "sample %d, %d hidden, %d of %d on the image", h->k, h->hidden, p->on_image, p->n);
    else snprintf(how, sizeof how, "the engine's trace");
    mtl_log("full view: aim %s t %.1f (%s) at %.1f %.1f %.1f, NDC %.4f %.4f "
            "w %.1f; ray from %.1f %.1f %.1f along %.3f %.3f %.3f, the frame's camera %s%.1f %.1f %.1f, %.1f from the "
            "eye%s",
            !ok ? "gives no point:" : h->hit ? "hits at" : "finds no surface, the far end at", h->t, how, h->p[0], h->p[1], h->p[2], h->ndc[0], h->ndc[1], h->w, r->eye[0], r->eye[1],
            r->eye[2], r->dir[0], r->dir[1], r->dir[2], cam_ok ? "" : "(none) ", cam[0], cam[1], cam[2],
            sqrtf(dx * dx + dy * dy + dz * dz), logged == 60 ? "; no more of these" : "");
}

D9_EXPORT int metal_d3d9_fullview_view_model(void) { return atomic_load_explicit(&s_view_model, memory_order_relaxed); }

D9_EXPORT void metal_d3d9_immersive_view_play(int play) { atomic_store_explicit(&s_play, play != 0, memory_order_relaxed); }

D9_EXPORT int metal_d3d9_fullview_game_tan(float *tan_x, float *tan_y)
{
    ensure_init();
    pthread_mutex_lock(&s_views_mu);
    if (!s_views_set) default_views(s_view_tan, s_eye_m);
    game_tan_locked(back_aspect(), tan_x, tan_y);
    pthread_mutex_unlock(&s_views_mu);
    return mtl_fullview_wanted() && mtl_stereo_on();
}

int mtl_fullview_wanted(void)
{
    return atomic_load_explicit(&s_fullview, memory_order_relaxed) && atomic_load_explicit(&s_play, memory_order_relaxed);
}

void mtl_fullview_note_aspect(float aspect)
{
    if (!(aspect > 0.0f)) return;
    uint32_t bits;
    memcpy(&bits, &aspect, 4);
    atomic_store_explicit(&s_back_aspect_bits, bits, memory_order_relaxed);
}

void mtl_fullview_latch(float aspect, mtl_fullview_frame *out)
{
    ensure_init();
    memset(out, 0, sizeof *out);
    if (!(aspect > 0.0f)) aspect = back_aspect();
    float tan[8], eye[6], tx, ty, margin, hud_deg, aura_deg;
    pthread_mutex_lock(&s_views_mu);
    if (!s_views_set) default_views(s_view_tan, s_eye_m);
    memcpy(tan, s_view_tan, sizeof tan);
    memcpy(eye, s_eye_m, sizeof eye);
    game_tan_locked(aspect, &tx, &ty);
    margin = s_fv_margin;
    hud_deg = s_fv_hud_deg;
    aura_deg = s_fv_aura_deg;
    pthread_mutex_unlock(&s_views_mu);
    out->game_eye = atomic_load_explicit(&s_game_eye, memory_order_relaxed);
    /* The game's camera is the eyes' midpoint; each eye's offset from it
       goes from the device's axes (+Z back) to the camera's (+Z forward). */
    float mid[3] = {(eye[0] + eye[3]) * 0.5f, (eye[1] + eye[4]) * 0.5f, (eye[2] + eye[5]) * 0.5f};
    for (int e = 0; e < 2; e++) {
        if (out->game_eye) {
            const float g[4] = {tx, tx, ty, ty};
            memcpy(out->tan[e], g, sizeof g);
        } else {
            for (int k = 0; k < 4; k++) out->tan[e][k] = tan[4 * e + k] * (1.0f + margin);
            out->eye_t[e][0] = (eye[3 * e + 0] - mid[0]) * s_units_per_m;
            out->eye_t[e][1] = (eye[3 * e + 1] - mid[1]) * s_units_per_m;
            out->eye_t[e][2] = -(eye[3 * e + 2] - mid[2]) * s_units_per_m;
        }
        mtl_stereo_proj_from_tangents(out->tan[e], &out->eye_p[e]);
    }
    out->panel[0] = tanf(hud_deg * 0.5f * (float)M_PI / 180.0f);
    out->panel[1] = out->panel[0] / aspect;
    out->aura[0] = tanf(aura_deg * 0.5f * (float)M_PI / 180.0f);
    out->aura[1] = out->aura[0] / aspect;
    out->sky_scale = s_fv_sky_scale;
    for (int k = 0; k < 2; k++)
        out->arm[k] = s_arm_env && !out->game_eye ? atomic_load_explicit(&s_arm_side[k], memory_order_relaxed) : 0;
    out->on = 1;
    if (!atomic_exchange(&s_fv_logged, 1))
        mtl_log("full view: first frame, the game at %.1f by %.1f degrees, left eye image %.3f %.3f %.3f %.3f, right "
                "%.3f %.3f %.3f %.3f, eyes at %+.2f and %+.2f units, HUD panel %.0f degrees",
                2.0 * atan(tx) * 180.0 / M_PI, 2.0 * atan(ty) * 180.0 / M_PI, out->tan[0][0], out->tan[0][1],
                out->tan[0][2], out->tan[0][3], out->tan[1][0], out->tan[1][1], out->tan[1][2], out->tan[1][3],
                out->eye_t[0][0], out->eye_t[1][0], hud_deg);
}

float mtl_stereo_eye_x(int eye)
{
    ensure_init();
    float h;
    uint32_t bits = atomic_load_explicit(&s_half_ipd_bits, memory_order_relaxed);
    memcpy(&h, &bits, 4);
    return eye ? h : -h;
}

/* The game thread waits out what is left of the frame period, after its
   frame is queued. The period runs from one Present's return to the next,
   so a slow frame is not followed by a short one to catch up. */
void mtl_immersive_pace(void)
{
    ensure_init();
    if (!s_pace_ns) return;
    uint64_t t = now_ns();
    if (s_last_present && t < s_last_present + s_pace_ns) {
        uint64_t left = s_last_present + s_pace_ns - t;
        struct timespec ts = {(time_t)(left / 1000000000u), (long)(left % 1000000000u)};
        nanosleep(&ts, NULL);
        uint64_t woke = now_ns();
        atomic_fetch_add_explicit(&s_pace_waits, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&s_pace_wait_ns, woke - t, memory_order_relaxed);
        t = woke;
    }
    s_last_present = t;
}

void mtl_immersive_committed(uint64_t committed_ns)
{
    if (s_write_slot < 0) return;
    mtl_handoff_set_committed(&s_ring, s_write_slot, committed_ns);
    s_write_slot = -1;
}

/* layers: 1 for a mono frame (a 2D texture), 2 for a stereo one (a 2D
   array, slice 0 the left eye), so the compositor tells them apart by the
   texture alone and a switch can never pair a mode with the other's frame. */
static id<MTLTexture> slot_texture(id<MTLDevice> dev, int slot, id<MTLTexture> like, uint32_t layers)
{
    void *cur;
    uint32_t w, h;
    mtl_handoff_get_texture(&s_ring, slot, &cur, &w, &h);
    id<MTLTexture> t = (__bridge id<MTLTexture>)cur;
    if (t && w == like.width && h == like.height && t.pixelFormat == like.pixelFormat && t.arrayLength == layers &&
        (layers > 1) == (t.textureType == MTLTextureType2DArray))
        return t;
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:like.pixelFormat
                                                                                  width:like.width
                                                                                 height:like.height
                                                                              mipmapped:NO];
    if (layers > 1) {
        td.textureType = MTLTextureType2DArray;
        td.arrayLength = layers;
    }
    /* Read by the compositor, through an sRGB view: the back buffer holds
       gamma-encoded values that the compositor's sRGB target re-encodes. */
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
    td.storageMode = MTLStorageModePrivate;
    id<MTLTexture> n = [dev newTextureWithDescriptor:td];
    if (!n) return nil;
    n.label = [NSString stringWithFormat:@"immersive slot %d", slot];
    void *old = NULL;
    mtl_handoff_set_texture(&s_ring, slot, (void *)CFBridgingRetain(n), (uint32_t)n.width, (uint32_t)n.height, &old);
    if (old) CFRelease(old);
    return n;
}

/* A slot's copy of the arm HUD's cards, made like the frame's. Read raw
   by the compositor, which decodes its premultiplied colour itself. With
   mipmaps: a card at arm's length shows the gauge's 1584 rows on a few
   hundred pixels, where the portrait and figures would shimmer. */
static id<MTLTexture> slot_arm_texture(id<MTLDevice> dev, int slot, id<MTLTexture> like)
{
    id<MTLTexture> t = (__bridge id<MTLTexture>)s_slot_arm[slot];
    if (t && t.width == like.width && t.height == like.height && t.pixelFormat == like.pixelFormat) return t;
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:like.pixelFormat
                                                                                  width:like.width
                                                                                 height:like.height
                                                                              mipmapped:YES];
    td.usage = MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModePrivate;
    id<MTLTexture> n = [dev newTextureWithDescriptor:td];
    if (!n) return nil;
    n.label = [NSString stringWithFormat:@"immersive slot %d arm HUD", slot];
    if (s_slot_arm[slot]) CFRelease(s_slot_arm[slot]);
    s_slot_arm[slot] = (void *)CFBridgingRetain(n);
    return n;
}

/* Called where the list is replayed, in place of the layer present. */
void mtl_immersive_present(mtl_device *d, mtl_emit *e, void *back, void *right, uint64_t presented_ns)
{
    ensure_init();
    /* Checked here as well as by the host: this thread keeps running when
       the host's driver has gone quiet, which is the stall it exists to
       explain. A lock and a compare, once a frame. */
    metal_d3d9_space_watchdog();
    /* Taken whatever happens below, so a frame's cards are never handed
       on with the next frame's, nor its crosshair. */
    mtl_arm_frame arm;
    mtl_arm_take(d, &arm);
    mtl_crosshair_frame xh;
    mtl_crosshair_take(d, &xh);
    int slot = back ? mtl_handoff_begin(&s_ring) : -1;
    if (slot < 0) {
        for (int k = 0; k < 2; k++)
            if (arm.icon[k]) CFRelease(arm.icon[k]);
        return; /* counted as skipped when there was a frame */
    }
    /* The slot is the writer's now: its pictures are this frame's, or none. */
    for (int k = 0; k < 2; k++) {
        if (s_slot_icon[slot][k]) CFRelease(s_slot_icon[slot][k]);
        s_slot_icon[slot][k] = e->fv.on ? arm.icon[k] : NULL;
        if (arm.icon[k] && !e->fv.on) CFRelease(arm.icon[k]);
    }
    id<MTLTexture> bb = (__bridge id<MTLTexture>)back, rb = (__bridge id<MTLTexture>)right;
    if (rb && (rb.width != bb.width || rb.height != bb.height || rb.pixelFormat != bb.pixelFormat)) rb = nil;
    id<MTLTexture> t = slot_texture((__bridge id<MTLDevice>)d->mtl, slot, bb, rb ? 2 : 1);
    if (!t) {
        mtl_handoff_abort(&s_ring, slot);
        atomic_fetch_add(&s_aborts, 1);
        return;
    }
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)e->cb;
    id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
    [bl copyFromTexture:bb
             sourceSlice:0
             sourceLevel:0
            sourceOrigin:MTLOriginMake(0, 0, 0)
              sourceSize:MTLSizeMake(bb.width, bb.height, 1)
               toTexture:t
        destinationSlice:0
        destinationLevel:0
       destinationOrigin:MTLOriginMake(0, 0, 0)];
    if (rb)
        [bl copyFromTexture:rb
                 sourceSlice:0
                 sourceLevel:0
                sourceOrigin:MTLOriginMake(0, 0, 0)
                  sourceSize:MTLSizeMake(rb.width, rb.height, 1)
                   toTexture:t
            destinationSlice:1
            destinationLevel:0
           destinationOrigin:MTLOriginMake(0, 0, 0)];
    /* The cards only with a full view frame that drew one: anything else
       leaves both sides off, so a gauge never lingers on a card. */
    id<MTLTexture> at = rb && e->fv.on && arm.tex && (arm.drawn[0] || arm.drawn[1]) ? (__bridge id<MTLTexture>)arm.tex : nil;
    id<MTLTexture> st = at ? slot_arm_texture((__bridge id<MTLDevice>)d->mtl, slot, at) : nil;
    if (st)
        [bl copyFromTexture:at
                 sourceSlice:0
                 sourceLevel:0
                sourceOrigin:MTLOriginMake(0, 0, 0)
                  sourceSize:MTLSizeMake(at.width, at.height, 1)
                   toTexture:st
            destinationSlice:0
            destinationLevel:0
           destinationOrigin:MTLOriginMake(0, 0, 0)];
    if (st) [bl generateMipmapsForTexture:st];
    for (int k = 0; k < 2; k++) {
        s_slot_arm_on[slot][k] = st && arm.drawn[k] && arm.w[k] && arm.h;
        float *uv = s_slot_arm_uv[slot][k];
        if (s_slot_arm_on[slot][k]) {
            uv[0] = (float)arm.x[k] / (float)at.width;
            uv[1] = 0.0f;
            uv[2] = (float)(arm.x[k] + arm.w[k]) / (float)at.width;
            uv[3] = 1.0f;
        } else {
            memset(uv, 0, 4 * sizeof *uv);
        }
    }
    for (int k = 0; k < 2; k++) {
        s_slot_label_on[slot][k] = st && arm.label_on[k];
        if (s_slot_label_on[slot][k]) {
            s_slot_label_uv[slot][k][0] = arm.label[k][0] / (float)at.width;
            s_slot_label_uv[slot][k][1] = arm.label[k][1] / (float)at.height;
            s_slot_label_uv[slot][k][2] = arm.label[k][2] / (float)at.width;
            s_slot_label_uv[slot][k][3] = arm.label[k][3] / (float)at.height;
        }
    }
    [bl endEncoding];
    /* The third person aim: the shot's ray sampled through the frame's
       view projection into the right eye's image (its replay is the last
       to write its depth), and the depth there read on the GPU in this
       command buffer; the search runs at its completion below. The check
       (game_eye) draws each eye with the game's projection at the camera,
       so its correction is the identity. */
    s_slot_aim_on[slot] = 0;
    s_slot_hit_ok[slot] = 0;
    memset(&s_slot_hands[slot], 0, sizeof s_slot_hands[slot]);
    if (rb && e->fv.on && e->hands.mask &&
        mtl_stereo_proj_from_rows(e->hands.rows[0], e->hands.rows[1], e->hands.rows[2], e->hands.rows[3],
                                  &s_slot_hands_p[slot]))
        s_slot_hands[slot] = e->hands;
    if (rb && e->fv.on && e->aim.ok) {
        const mtl_aim_ray *r = &e->aim;
        mtl_stereo_proj_full gp;
        mtl_stereo_fix fix;
        if (mtl_stereo_proj_from_rows(r->rows[0], r->rows[1], r->rows[2], r->rows[3], &gp)) {
            if (e->fv.game_eye) mtl_stereo_identity(&fix);
            else mtl_stereo_correction3(&gp, &e->fv.eye_p[1], e->fv.eye_t[1], &fix);
            if (!s_slot_aim_buf[slot]) {
                id<MTLBuffer> b = [(__bridge id<MTLDevice>)d->mtl newBufferWithLength:MTL_AIM_BUF_FLOATS * sizeof(float)
                                                                               options:MTLResourceStorageModeShared];
                b.label = [NSString stringWithFormat:@"immersive slot %d aim depths", slot];
                s_slot_aim_buf[slot] = b ? (void *)CFBridgingRetain(b) : NULL;
            }
            s_slot_ray[slot] = *r;
            s_slot_aim_p[slot] = gp;
            /* The engine traced the shot itself: its point is the answer,
               and no depth is read. */
            if (r->traced) s_slot_aim_on[slot] = 1;
            else if (mtl_aim_probe(r, &fix, &s_slot_probes[slot]) &&
                     mtl_aim_gather(d, e->cb, mtl_frames_right_depth(d), &s_slot_probes[slot], s_slot_aim_buf[slot]))
                s_slot_aim_on[slot] = 1;
        }
    }
    /* The scene's depth: each eye's depth buffer as metres along its
       forward, from the frame's view projection and each eye's correction
       as the aim's are made. The left eye's depth is the game's own
       surface, the right's its twin (frame.m, right_eye_ready). */
    s_slot_scene_on[slot] = 0;
    if (rb && e->fv.on && e->aim.ok && atomic_load_explicit(&s_scene_depth, memory_order_relaxed) && d->auto_ds &&
        d->auto_ds->tex && mtl_frames_right_depth(d)) {
        const mtl_aim_ray *r = &e->aim;
        mtl_stereo_proj_full gp;
        float terms[16] = {0};
        int ok = mtl_stereo_proj_from_rows(r->rows[0], r->rows[1], r->rows[2], r->rows[3], &gp);
        for (int k = 0; ok && k < 2; k++) {
            mtl_stereo_fix fix;
            if (e->fv.game_eye) mtl_stereo_identity(&fix);
            else mtl_stereo_correction3(&gp, &e->fv.eye_p[k], e->fv.eye_t[k], &fix);
            ok = mtl_stereo_depth_terms(r->rows, &fix, &terms[2 * k], &terms[2 * k + 1]);
        }
        terms[4] = r->vp[4];
        terms[5] = r->vp[5] - r->vp[4];
        terms[6] = 1.0f / s_units_per_m;
        terms[7] = 0.0f;
        /* The view model's pixels by its own terms: Source draws it into a
           depth range of its own (the front of the buffer), with its own
           field of view, so the world's terms put the gun a few
           centimetres from the eye and the gloves took all of it as what
           is held (A.'s run, 2026-11-25: fingers through the grip). Told
           apart by depth when its range is its own. */
        const mtl_vm_depth *vz = &e->vmz;
        mtl_stereo_proj_full vp;
        int vm_own = vz->ok && vz->vp[1] > vz->vp[0] &&
                     (vz->vp[1] < r->vp[5] - 1e-4f || vz->vp[0] > r->vp[4] + 1e-4f) &&
                     mtl_stereo_proj_from_rows(vz->rows[0], vz->rows[1], vz->rows[2], vz->rows[3], &vp);
        for (int k = 0; vm_own && k < 2; k++) {
            mtl_stereo_fix fix;
            if (e->fv.game_eye) mtl_stereo_identity(&fix);
            else mtl_stereo_correction3(&vp, &e->fv.eye_p[k], e->fv.eye_t[k], &fix);
            vm_own = mtl_stereo_depth_terms(vz->rows, &fix, &terms[8 + 2 * k], &terms[9 + 2 * k]);
        }
        if (vm_own) {
            terms[12] = vz->vp[0];
            terms[13] = vz->vp[1] - vz->vp[0];
            terms[14] = 1.0f;
        }
        s_slot_scene_vm[slot] = vm_own;
        static int told = -1;
        if (told != vm_own) {
            told = vm_own;
            mtl_log("full view: the view model's depth %s (its range %.3f to %.3f, the world's %.3f to %.3f)",
                    vm_own ? "by its own projection" : "by the world's", vz->ok ? vz->vp[0] : 0.0, vz->ok ? vz->vp[1] : 0.0,
                    r->vp[4], r->vp[5]);
        }
        id<MTLTexture> zt = (__bridge id<MTLTexture>)d->auto_ds->tex;
        id<MTLTexture> st = ok ? slot_scene_texture((__bridge id<MTLDevice>)d->mtl, slot, (uint32_t)zt.width,
                                                    (uint32_t)zt.height)
                               : nil;
        if (st && mtl_scene_depth(d, e->cb, d->auto_ds->tex, mtl_frames_right_depth(d), terms, (__bridge void *)st))
            s_slot_scene_on[slot] = 1;
    }
    s_write_slot = slot;
    s_slot_fv[slot] = rb && e->fv.on;
    if (s_slot_fv[slot]) memcpy(s_slot_tan[slot], e->fv.tan, sizeof s_slot_tan[slot]);
    else memset(s_slot_tan[slot], 0, sizeof s_slot_tan[slot]);
    if (s_slot_fv[slot]) memcpy(s_slot_panel[slot], e->fv.panel, sizeof s_slot_panel[slot]);
    else memset(s_slot_panel[slot], 0, sizeof s_slot_panel[slot]);
    if (s_slot_fv[slot] && xh.shown) s_slot_xhair[slot] = xh;
    else memset(&s_slot_xhair[slot], 0, sizeof s_slot_xhair[slot]);
    if (rb) {
        float a = e->screen_a > 0.0f ? e->screen_a : MTL_STEREO_DEFAULT_A;
        s_slot_view[slot][0] = 1.0f / a;
        s_slot_view[slot][1] = (float)bb.height / ((float)bb.width * a);
        s_slot_view[slot][2] = fabsf(e->eye_x) / s_units_per_m;
        s_slot_view[slot][3] = METAL_D3D9_STEREO_HUD_M;
    } else {
        memset(s_slot_view[slot], 0, sizeof s_slot_view[slot]);
    }
    memcpy(s_slot_pose[slot], e->pose, sizeof s_slot_pose[slot]);
    uint64_t dequeued = e->dequeued_ns;
    /* Published only once the GPU has finished the copy, so an acquire
       never returns a slot still being written. */
    [cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
        if (b.status == MTLCommandBufferStatusCompleted) {
            if (s_slot_aim_on[slot] && s_slot_ray[slot].traced) {
                int ok = mtl_aim_traced(&s_slot_ray[slot], &s_slot_hit[slot]);
                s_slot_hit_ok[slot] = ok;
                aim_log(&s_slot_ray[slot], NULL, &s_slot_hit[slot], ok);
            } else if (s_slot_aim_on[slot]) {
                const float *depth = (const float *)[(__bridge id<MTLBuffer>)s_slot_aim_buf[slot] contents] + MTL_AIM_DEPTH_AT;
                int ok = mtl_aim_search(&s_slot_ray[slot], &s_slot_probes[slot], depth, &s_slot_hit[slot]);
                s_slot_hit_ok[slot] = ok;
                aim_log(&s_slot_ray[slot], &s_slot_probes[slot], &s_slot_hit[slot], ok);
            }
            /* GPUStartTime and GPUEndTime are host time in seconds, the
               mach_absolute_time clock the other stamps use. */
            mtl_handoff_times tm = {
                .presented_ns = presented_ns,
                .dequeued_ns = dequeued,
                .gpu_start_ns = b.GPUStartTime > 0 ? (uint64_t)(b.GPUStartTime * 1e9) : 0,
                .gpu_end_ns = b.GPUEndTime > 0 ? (uint64_t)(b.GPUEndTime * 1e9) : 0,
                .ready_ns = now_ns(),
            };
            mtl_handoff_publish(&s_ring, slot, &tm);
            atomic_fetch_add_explicit(&s_copies, 1, memory_order_relaxed);
            double g = b.GPUEndTime - b.GPUStartTime;
            if (g > 0) atomic_fetch_add_explicit(&s_copy_gpu_ns, (uint64_t)(g * 1e9), memory_order_relaxed);
        } else {
            mtl_handoff_abort(&s_ring, slot);
            atomic_fetch_add_explicit(&s_aborts, 1, memory_order_relaxed);
        }
    }];
}

static void log_stats(const char *what, const mtl_handoff_stats *s, double secs)
{
    double sec = secs > 0 ? secs : 1;
    mtl_log("immersive %s %.1f s: compositor frames %llu (%.1f Hz, %llu before the first game frame), game frames "
            "handed over %llu (%.1f Hz), skipped %llu, never shown %llu, shown more than once %llu (compositor "
            "frames repeating one %llu), shown frame age p50 %.2f p95 %.2f max %.2f mean %.2f ms; first shown %llu, "
            "mean (max) ms: queue %.2f (%.2f), replay %.2f (%.2f), GPU wait %.2f (%.2f), GPU %.2f (%.2f), publish "
            "%.2f (%.2f), compositor wait %.2f (%.2f)",
            what, secs, (unsigned long long)s->compositor_frames, s->compositor_frames / sec,
            (unsigned long long)s->empty, (unsigned long long)s->published, s->published / sec,
            (unsigned long long)s->skipped, (unsigned long long)s->never_shown, (unsigned long long)s->shown_again,
            (unsigned long long)s->repeats, s->age_p50_ms, s->age_p95_ms, s->age_max_ms, s->age_mean_ms,
            (unsigned long long)s->first_n, s->stage_mean_ms[MTL_AGE_QUEUE], s->stage_max_ms[MTL_AGE_QUEUE],
            s->stage_mean_ms[MTL_AGE_REPLAY], s->stage_max_ms[MTL_AGE_REPLAY], s->stage_mean_ms[MTL_AGE_GPU_WAIT],
            s->stage_max_ms[MTL_AGE_GPU_WAIT], s->stage_mean_ms[MTL_AGE_GPU], s->stage_max_ms[MTL_AGE_GPU],
            s->stage_mean_ms[MTL_AGE_PUBLISH], s->stage_max_ms[MTL_AGE_PUBLISH],
            s->stage_mean_ms[MTL_AGE_COMPOSITOR], s->stage_max_ms[MTL_AGE_COMPOSITOR]);
}

static void trace_line(const mtl_handoff_view *v, uint64_t comp_frame, uint64_t t)
{
    if (!s_trace_tried) {
        s_trace_tried = 1;
        const char *p = getenv("RT_COMPOSITOR_TRACE");
        if (p && *p) {
            s_trace = fopen(p, "w");
            if (s_trace) fprintf(s_trace, "compositor_frame,t_ms,game_frame,slot,age_ms,repeat\n");
            else mtl_log("immersive: could not open RT_COMPOSITOR_TRACE %s", p);
        }
    }
    if (!s_trace) return;
    fprintf(s_trace, "%llu,%.3f,%llu,%d,%.3f,%d\n", (unsigned long long)comp_frame, t / 1e6,
            (unsigned long long)v->frame, v->slot, v->age_ms, v->repeat);
    if (comp_frame % 90 == 0) fflush(s_trace);
}

D9_EXPORT int metal_d3d9_immersive_acquire(metal_d3d9_shown_frame *out)
{
    ensure_init();
    uint64_t t = now_ns();
    mtl_handoff_view v;
    int ok = mtl_handoff_acquire(&s_ring, t, &v);
    out->slot = v.slot;
    out->texture = v.tex;
    out->width = v.w;
    out->height = v.h;
    out->frame = v.frame;
    out->age_ms = v.age_ms;
    out->repeat = v.repeat;
    out->layers = ok && v.tex ? (uint32_t)((__bridge id<MTLTexture>)v.tex).arrayLength : 0;
    const float *sv = ok && v.slot >= 0 && v.slot < MTL_HANDOFF_SLOTS && out->layers == 2 ? s_slot_view[v.slot] : NULL;
    out->tan_x = sv ? sv[0] : 0.0f;
    out->tan_y = sv ? sv[1] : 0.0f;
    out->half_ipd_m = sv ? sv[2] : 0.0f;
    out->hud_m = sv ? sv[3] : 0.0f;
    const float *sp = ok && v.slot >= 0 && v.slot < MTL_HANDOFF_SLOTS ? s_slot_pose[v.slot] : NULL;
    out->pose_valid = sp && sp[0] > 0.5f;
    out->pose_yaw_deg = out->pose_valid ? sp[1] : 0.0f;
    out->pose_pitch_deg = out->pose_valid ? sp[2] : 0.0f;
    out->fullview = sv && s_slot_fv[v.slot];
    if (out->fullview) memcpy(out->eye_tan, s_slot_tan[v.slot], sizeof out->eye_tan);
    else memset(out->eye_tan, 0, sizeof out->eye_tan);
    const mtl_crosshair_frame *xh = out->fullview ? &s_slot_xhair[v.slot] : NULL;
    out->panel[0] = out->fullview ? s_slot_panel[v.slot][0] : 0.0f;
    out->panel[1] = out->fullview ? s_slot_panel[v.slot][1] : 0.0f;
    out->crosshair_shown = xh && xh->shown;
    out->crosshair_ndc[0] = out->crosshair_shown ? xh->ndc[0] : 0.0f;
    out->crosshair_ndc[1] = out->crosshair_shown ? xh->ndc[1] : 0.0f;
    out->crosshair_half = out->crosshair_shown ? xh->half : 0.0f;
    /* The aim's point as the compositor places it: tangents off the
       drawn forward and metres along it (mtl_stereo_world_point, no
       extent). A point the projection gives nothing for is not shown. */
    float wp[4] = {0, 0, 0, 0};
    const mtl_aim_hit *h = out->fullview && s_slot_hit_ok[v.slot] ? &s_slot_hit[v.slot] : NULL;
    out->aim_shown = h && mtl_stereo_world_point(&s_slot_aim_p[v.slot], h->ndc[0], h->ndc[1], 0.0f, h->w,
                                                 s_units_per_m, wp);
    out->aim_hit = out->aim_shown && h->hit;
    out->aim_tan[0] = out->aim_shown ? wp[0] : 0.0f;
    out->aim_tan[1] = out->aim_shown ? wp[1] : 0.0f;
    out->aim_m = out->aim_shown ? wp[3] : 0.0f;
    out->aim_t = out->aim_shown ? h->t : 0.0f;
    /* The view model's hands the same way: each point through the frame's
       rows to its NDC and w, then tangents and metres. */
    const mtl_vm_hands *vh = out->fullview ? &s_slot_hands[v.slot] : NULL;
    for (int k = 0; k < 2; k++) {
        int ok = vh && (vh->mask >> k & 1);
        for (int j = 0; ok && j < 3; j++) {
            float c[4], hp[4];
            mtl_aim_clip(vh->rows, vh->pts + 9 * k + 3 * j, c);
            ok = c[3] > 1e-3f && mtl_stereo_world_point(&s_slot_hands_p[v.slot], c[0] / c[3], c[1] / c[3], 0.0f, c[3],
                                                         s_units_per_m, hp);
            out->vm_hand_tan[3 * k + j][0] = ok ? hp[0] : 0.0f;
            out->vm_hand_tan[3 * k + j][1] = ok ? hp[1] : 0.0f;
            out->vm_hand_m[3 * k + j] = ok ? hp[3] : 0.0f;
        }
        out->vm_hand_shown[k] = ok;
        float pc[4], pp[4];
        int pok = ok;
        if (pok) {
            mtl_aim_clip(vh->rows, vh->palm + 3 * k, pc);
            pok = pc[3] > 1e-3f && mtl_stereo_world_point(&s_slot_hands_p[v.slot], pc[0] / pc[3], pc[1] / pc[3], 0.0f, pc[3],
                                                          s_units_per_m, pp);
        }
        out->vm_palm_tan[k][0] = pok ? pp[0] : 0.0f;
        out->vm_palm_tan[k][1] = pok ? pp[1] : 0.0f;
        out->vm_palm_m[k] = pok ? pp[3] : 0.0f;
        if (!ok)
            for (int j = 0; j < 3; j++) out->vm_hand_tan[3 * k + j][0] = out->vm_hand_tan[3 * k + j][1] = out->vm_hand_m[3 * k + j] = 0.0f;
    }
    out->scene_depth = out->fullview && s_slot_scene_on[v.slot] ? s_slot_scene[v.slot] : NULL;
    out->scene_depth_vm = out->scene_depth && s_slot_scene_vm[v.slot];
    out->weapon_texture = out->fullview ? s_slot_icon[v.slot][0] : NULL;
    out->icon_texture = out->fullview ? s_slot_icon[v.slot][1] : NULL;
    int arm = out->fullview && (s_slot_arm_on[v.slot][0] || s_slot_arm_on[v.slot][1]);
    for (int k = 0; k < 2; k++) {
        out->label_shown[k] = arm && s_slot_label_on[v.slot][k];
        if (out->label_shown[k]) memcpy(out->label_uv[k], s_slot_label_uv[v.slot][k], sizeof out->label_uv[k]);
        else memset(out->label_uv[k], 0, sizeof out->label_uv[k]);
    }
    out->arm_texture = arm ? s_slot_arm[v.slot] : NULL;
    for (int k = 0; k < 2; k++) {
        out->arm_shown[k] = arm && s_slot_arm_on[v.slot][k];
        if (out->arm_shown[k]) memcpy(out->arm_uv[k], s_slot_arm_uv[v.slot][k], sizeof out->arm_uv[k]);
        else memset(out->arm_uv[k], 0, sizeof out->arm_uv[k]);
    }
    /* The compositor calls from one thread; the lock only keeps the log
       and the trace whole against the exit report. */
    pthread_mutex_lock(&s_log_mu);
    mtl_handoff_stats tot;
    mtl_handoff_totals(&s_ring, &tot);
    trace_line(&v, tot.compositor_frames, t);
    if (!s_period_start) s_period_start = t;
    if (t - s_period_start >= LOG_PERIOD_NS) {
        mtl_handoff_stats p;
        mtl_handoff_take_period(&s_ring, &p);
        log_stats("last", &p, (t - s_period_start) / 1e9);
        s_period_start = t;
    }
    pthread_mutex_unlock(&s_log_mu);
    return ok;
}

D9_EXPORT void metal_d3d9_immersive_release(int slot)
{
    ensure_init();
    mtl_handoff_release(&s_ring, slot);
}

/* The whole run's counters, from metal_d3d9_shutdown. */
void mtl_immersive_report(void)
{
    if (!atomic_load(&s_ever)) return;
    pthread_mutex_lock(&s_log_mu);
    mtl_handoff_stats tot;
    mtl_handoff_totals(&s_ring, &tot);
    uint64_t copies = atomic_load(&s_copies);
    log_stats("run", &tot, 0);
    mtl_log("immersive copies %llu, %llu aborted, GPU %.3f ms a frame including its copy; game thread held by the "
            "frame cap %llu times, %.2f ms each", (unsigned long long)copies,
            (unsigned long long)atomic_load(&s_aborts), copies ? atomic_load(&s_copy_gpu_ns) / 1e6 / copies : 0.0,
            (unsigned long long)atomic_load(&s_pace_waits),
            atomic_load(&s_pace_waits) ? atomic_load(&s_pace_wait_ns) / 1e6 / atomic_load(&s_pace_waits) : 0.0);
    if (s_trace) fflush(s_trace);
    pthread_mutex_unlock(&s_log_mu);
}

/* Test hooks, exported for the headless test only. */

/* out: compositor frames, empty, published, skipped, never shown, shown,
   shown again, repeats. */
D9_EXPORT void metal_d3d9_immersive_stats(uint64_t out[8])
{
    ensure_init();
    mtl_handoff_stats s;
    mtl_handoff_totals(&s_ring, &s);
    out[0] = s.compositor_frames;
    out[1] = s.empty;
    out[2] = s.published;
    out[3] = s.skipped;
    out[4] = s.never_shown;
    out[5] = s.shown;
    out[6] = s.shown_again;
    out[7] = s.repeats;
}

/* A held slot's pixels, read on a queue of its own as a compositor would:
   one layer of it, the left eye or a mono frame being layer 0. */
D9_EXPORT int metal_d3d9_immersive_read_layer(int slot, uint32_t layer, uint32_t *out, uint32_t w, uint32_t h)
{
    void *p;
    uint32_t tw, th;
    mtl_handoff_get_texture(&s_ring, slot, &p, &tw, &th);
    if (!p || tw != w || th != h) return 0;
    @autoreleasepool {
        id<MTLTexture> t = (__bridge id<MTLTexture>)p;
        if (layer >= t.arrayLength) return 0;
        id<MTLBuffer> b = [t.device newBufferWithLength:(NSUInteger)w * h * 4 options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = [[t.device newCommandQueue] commandBuffer];
        id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
        [bl copyFromTexture:t
                         sourceSlice:layer
                         sourceLevel:0
                        sourceOrigin:MTLOriginMake(0, 0, 0)
                          sourceSize:MTLSizeMake(w, h, 1)
                            toBuffer:b
                   destinationOffset:0
              destinationBytesPerRow:w * 4
            destinationBytesPerImage:(NSUInteger)w * h * 4];
        [bl endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        memcpy(out, b.contents, (size_t)w * h * 4);
    }
    return 1;
}

/* A held slot's arm HUD cards, the whole texture, read as read_layer
   reads a frame. 0 when the slot holds none of that size. */
D9_EXPORT int metal_d3d9_fullview_read_arm(int slot, uint32_t *out, uint32_t w, uint32_t h)
{
    if (slot < 0 || slot >= MTL_HANDOFF_SLOTS || !s_slot_arm[slot]) return 0;
    @autoreleasepool {
        id<MTLTexture> t = (__bridge id<MTLTexture>)s_slot_arm[slot];
        if (t.width != w || t.height != h) return 0;
        id<MTLBuffer> b = [t.device newBufferWithLength:(NSUInteger)w * h * 4 options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = [[t.device newCommandQueue] commandBuffer];
        id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
        [bl copyFromTexture:t
                         sourceSlice:0
                         sourceLevel:0
                        sourceOrigin:MTLOriginMake(0, 0, 0)
                          sourceSize:MTLSizeMake(w, h, 1)
                            toBuffer:b
                   destinationOffset:0
              destinationBytesPerRow:w * 4
            destinationBytesPerImage:(NSUInteger)w * h * 4];
        [bl endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        memcpy(out, b.contents, (size_t)w * h * 4);
    }
    return 1;
}

D9_EXPORT int metal_d3d9_immersive_read(int slot, uint32_t *out, uint32_t w, uint32_t h)
{
    return metal_d3d9_immersive_read_layer(slot, 0, out, w, h);
}
