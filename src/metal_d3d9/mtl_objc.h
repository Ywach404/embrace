/*
 * Declarations shared by the Objective-C files only, since they carry
 * Metal protocol types the C side cannot name.
 */
#ifndef MTL_OBJC_H
#define MTL_OBJC_H

#import <Metal/Metal.h>

#include "mtl_d3d9.h"

/* Replaying a recorded list into a command buffer (draw.m), on whichever
   thread owns the replay: the render thread, or the game thread in inline
   mode. Objective-C objects are held as retained or borrowed void
   pointers, as elsewhere here. A second replay of the same list (the
   test's twin target) swaps the back buffer and its depth surface for
   other textures and skips what must happen once: readbacks and present. */
/* Full view (immersive.h, stereo.h): what a frame is replayed with, latched
   once where the frame's stereo choice is made and held to its Present. */
typedef struct {
    int on;                /* the eyes' own views, else stereo's plane */
    int game_eye;          /* the check: each eye the game's projection, at the camera */
    mtl_stereo_proj eye_p[2];
    float eye_t[2][3];     /* each eye's offset from the game's camera, game units */
    float tan[2][4];       /* each eye image's tangents as rendered, the margin included */
    float panel[2];        /* the HUD panel's half tangents */
    float aura[2];         /* a camera-held effect's panel, the same (RT_FULLVIEW_AURA_DEG) */
    float sky_scale;       /* the 3D sky's eye offset per the world's (RT_FULLVIEW_SKY, 1/16; 0 at infinity) */
    int arm[2];            /* the arm HUD: each side's gauge on its card (1) or left to the orbs (2), not the panel (draw.m, arm_side) */
} mtl_fullview_frame;

typedef struct mtl_emit {
    void *cb;               /* id<MTLCommandBuffer>, owned by the caller */
    void *enc;              /* the open MTLRenderCommandEncoder, retained */
    void *swap_from[2 + MTL_EYE_COPIES], *swap_to[2 + MTL_EYE_COPIES]; /* back buffer, its depth, the eye copies */
    int second;
    uint64_t dequeued_ns;   /* when the replaying thread took the list, for the immersive ages */
    /* Stereo (stereo.h): this replay is one eye, at eye_x game units along
       the camera's right. The first eye's Present is held here until the
       second eye has been replayed. */
    int stereo;
    float eye_x;
    /* The a the eye images are shown with and the HUD plane's depth in
       game units, latched with eye_x for the whole frame. */
    float screen_a, hud_z;
    int pend_present;
    void *pend_back;
    uint64_t pend_ns;
    /* The head pose the Present being emitted was drawn for (valid,
       yaw, pitch; immersive.h), from its recorded command. */
    float pose[3];
    /* Full view: both eyes' terms, and which eye this replay is. */
    mtl_fullview_frame fv;
    /* The third person aim of the Present being emitted (draw.m,
       mtl_fv_aim), from its recorded command. */
    mtl_aim_ray aim;
    mtl_vm_hands hands;
    mtl_vm_depth vmz;
    int eye;
    /* The pass open now, to reopen it after a camera-held effect's own
       small pass (draw.m, emit_aura), and whether that pass is open. */
    const void *pass;
    int in_aura;
    /* Whether the open encoder is the arm HUD's cards' (draw.m, arm_enter)
       rather than the eye's pass, which is reopened at the next command
       that is not a card's draw. */
    int in_arm;
    /* The full view shot (frame.m): a FILE * each replayed command of
       this eye is described into, else NULL, and the running counts. */
    void *dump;
    uint32_t dump_cmd, dump_draw;
} mtl_emit;

void mtl_emit_run(mtl_device *d, mtl_emit *e, const uint8_t *cmds, size_t from, size_t to);
void mtl_emit_close(mtl_emit *e); /* ends an encoder left open */
/* device.m. presented_ns is when the game called Present, for the
   immersive handoff's frame ages. */
void mtl_emit_present(mtl_device *d, mtl_emit *e, void *back, void *right, uint64_t presented_ns);

/* immersive.m: Present into the compositor's ring instead of a layer
   (immersive.h, handoff.h). */
int mtl_immersive_on(void);
/* While immersive: whether Present still shows the frame in the window's
   layer too, until the window is dismissed (space.h). */
int mtl_immersive_window_too(void);
/* right: the second eye's image when the frame was replayed per eye
   (back is then the first eye's), else NULL. */
void mtl_immersive_present(mtl_device *d, mtl_emit *e, void *back, void *right, uint64_t presented_ns);
/* The head pose the runtime last gave (metal_d3d9_immersive_view_pose),
   as valid, yaw, pitch: read where Present is recorded. */
void mtl_immersive_pose_now(float pose[3]);
/* The shot's ray from the runtime (metal_d3d9_fullview_aim_source), read
   now, on the game thread at Present: 1 in third person with a player. */
int mtl_immersive_aim(float eye[3], float dir[3]);
/* The view model's hands from the runtime (metal_d3d9_fullview_hands_source),
   read at the Present's submit: the mask, 0 for none. */
int mtl_immersive_hands(float pts[18]);
/* The arms' bind pose hands (metal_d3d9_fullview_hands_bind_source), read
   on the game thread when the first arm draw of a frame is recorded. */
int mtl_immersive_hands_bind(float out[32]);
/* The engine's trace of that ray (metal_d3d9_fullview_aim_trace_source),
   asked right after it: 1 with its distance and whether it met anything. */
int mtl_immersive_aim_trace(float *t, int *hit);
/* draw.m: reads the depth texture at each probe's pixel into out (one
   float each, the raw stored depth; 1 for a probe off the image), in a
   small compute pass on cb. 0 when it could not be encoded. */
int mtl_aim_gather(mtl_device *d, void *cb, void *depth, const mtl_aim_probes *p, void *out);
/* Each eye's depth as metres along its forward into out, a two slice
   R16Float array (draw.m): terms are the left eye's A, B, the right's,
   then the viewport's depth min and range and metres per unit. */
int mtl_scene_depth(mtl_device *d, void *cb, void *left, void *right, const float terms[16], void *out);
/* frame.m: the right eye's depth texture, as its replay left it. */
void *mtl_frames_right_depth(const mtl_device *d);
/* Whether frames are replayed once per eye: stereo asked for, and the
   ring in use. Latched by the replaying thread at each frame's start. */
int mtl_stereo_on(void);
/* An eye's offset along the camera's right in game units: eye 0 is the
   left, at minus half the IPD. */
float mtl_stereo_eye_x(int eye);
/* The HUD plane's distance in game units (immersive.h,
   METAL_D3D9_STEREO_HUD_M, changed with the units a metre). */
float mtl_stereo_hud_z(void);
/* The a the eye planes are sized by, fixed for the session, for a back
   buffer of this aspect (stereo.h, mtl_stereo_screen_a; logged once). */
float mtl_stereo_screen_a_for(float aspect);
void mtl_immersive_report(void);
/* Full view. wanted: on the game thread as a list is submitted, whether
   the frame is meant for the eyes' own views (on and in play); the
   replaying thread still needs stereo for it. latch: both eyes' terms
   for a frame on a back buffer of this aspect. note_aspect: at each
   Present, the back buffer's shape the game's field is fitted to. */
int mtl_fullview_wanted(void);
void mtl_fullview_latch(float aspect, mtl_fullview_frame *out);
/* draw.m's verdict on the view model, for metal_d3d9_fullview_view_model. */
void mtl_immersive_view_model(int drawn);
void mtl_fullview_note_aspect(float aspect);
/* After the replaying thread commits a list: when, for the ages, if that
   list wrote a slot. */
void mtl_immersive_committed(uint64_t committed_ns);
/* On the game thread after each presented list is queued: holds the game
   to RT_IMMERSIVE_MAX_HZ while immersive, between frames rather than
   inside one. */
void mtl_immersive_pace(void);

/* draw.m: the arm HUD's cards, one target holding the left gauge's
   columns of the back buffer then the right's, at its pixels. x and w
   are each side's part of it; drawn says which sides a frame drew. take,
   at a frame's Present: that frame's, and the next frame's first card
   draw clears the target. last: what the last take gave, for the shot.
   tex NULL when there is no target yet. */
typedef struct {
    void *tex;
    uint32_t x[2], w[2], h;
    int drawn[2];
    /* take only, with the orbs (an arm side in mode 2): each side's
       picture, the game's own texture retained for the taker to release
       (0 the weapon's, 1 the discipline's), or NULL; and when their
       labels were drawn, tex is the labels' target, whole, and label[k]
       each side's glyphs on it (u0, v0, u1, v1 in its pixels, top left
       first), label_on[k] 0 for none. Never set by last. */
    void *icon[2];
    float label[2][4];
    int label_on[2];
} mtl_arm_frame;
void mtl_arm_take(mtl_device *d, mtl_arm_frame *out);
void mtl_arm_last(mtl_device *d, mtl_arm_frame *out);

/* draw.m: the game's crosshair as a full view frame drew it, or would
   have (immersive.h, crosshair_shown): whether any of its draws came, the
   centre of their box and its x half extent in the game's NDC. take, at
   a frame's Present: that frame's, and the next frame starts empty. */
typedef struct {
    int shown;
    float ndc[2], half;
} mtl_crosshair_frame;
void mtl_crosshair_take(mtl_device *d, mtl_crosshair_frame *out);

/* frame.m: what the replay counts, read when the device is released. */
void mtl_frames_note_dropped(mtl_device *d);
void mtl_frames_note_no_drawable(mtl_device *d);
uint64_t mtl_frames_no_drawable(const mtl_device *d);

/* draw.m: the commands device.m records. */
void mtl_rec_readback(mtl_device *d, void *src, uint32_t level, uint32_t slice, uint32_t w, uint32_t h, void *dst,
                      uint32_t pitch, uint32_t rows);
void mtl_rec_present(mtl_device *d, void *back);

/* gpu.m: the device's capabilities, logged the first time per process. */
void mtl_caps_query(id<MTLDevice> dev, mtl_gpu_caps *out);

/* gpu.m: a 2D texture holding one surface's CPU memory as it is now, in
   the format sampling it needs (DXT decoded and 16-bit formats widened
   where the GPU lacks them). nil for a format with no Metal format. */
id<MTLTexture> mtl_texture_from_mem(mtl_device *d, uint32_t fmt, uint32_t w, uint32_t h, const void *mem,
                                    uint32_t pitch, uint32_t rows);

/* gpu.m: a texture's MTLTexture with pending uploads applied, or nil. */
id<MTLTexture> mtl_texture_gpu(mtl_texture *t);

/* draw.m: the texture a surface renders into or is read from, with the
   mip level and cube face it names. nil for a surface with CPU memory
   only. */
id<MTLTexture> mtl_surface_texture(mtl_surface *s, uint32_t *level, uint32_t *slice);

/* draw.m: gives a render target or depth surface its private texture. */
int mtl_surface_make_target(mtl_device *d, mtl_surface *s);

#endif
