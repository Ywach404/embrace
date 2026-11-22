/*
 * The handoff between the game's frames and an immersive compositor.
 *
 * In immersive mode (visionOS, CompositorServices) the game and the
 * compositor run at their own rates: the game presents about every 15 ms,
 * the compositor asks for a frame at 90 Hz, and neither may wait for the
 * other. So Present no longer shows the back buffer itself. The replay of
 * a frame's list ends by copying the back buffer into one slot of a small
 * ring, the slot is published when that command buffer completes, and the
 * compositor samples whichever slot was published last, as many times as
 * it needs to.
 *
 * This file is the ring's bookkeeping only, plain C with no Metal, so a
 * test can drive every state with a fake clock. The textures are opaque
 * pointers the Metal side (immersive.m) creates and swaps.
 *
 * The two GPU rules live in how the states are used, not in this file:
 *
 *   - A slot is published from its command buffer's completion, so the
 *     compositor never samples a copy the GPU has not finished.
 *   - A slot the compositor acquired is held until the compositor's own
 *     command buffer completes, so the game never overwrites a slot the
 *     compositor's GPU work may still read.
 *
 * The writer takes a slot that is neither held nor the latest published.
 * With three slots that is always one when the game has a single copy in
 * flight. When there is none (two copies in flight and the compositor
 * holding the third) the frame is skipped, not waited for, and counted.
 */
#ifndef MTL_HANDOFF_H
#define MTL_HANDOFF_H

#include <pthread.h>
#include <stdint.h>

#define MTL_HANDOFF_SLOTS 3

/* Age histogram: 0.25 ms buckets up to 250 ms, then one overflow bucket. */
#define MTL_HANDOFF_AGE_BUCKETS 1001
#define MTL_HANDOFF_AGE_STEP_US 250

enum { MTL_SLOT_FREE, MTL_SLOT_WRITING, MTL_SLOT_READY };

/* Where a frame's time went between the game's Present and the
   compositor frame that first showed it, as stages that add up to that
   age. The timestamps are all on one clock (mach_absolute_time in ns). */
enum {
    MTL_AGE_QUEUE,      /* Present to the render thread taking the list */
    MTL_AGE_REPLAY,     /* replaying the list, to its commit */
    MTL_AGE_GPU_WAIT,   /* commit to the GPU starting it, behind earlier work */
    MTL_AGE_GPU,        /* the GPU running it, copy included */
    MTL_AGE_PUBLISH,    /* GPU end to the ring hearing of it */
    MTL_AGE_COMPOSITOR, /* published to the first compositor acquire */
    MTL_AGE_STAGES
};

/* A frame's timestamps, given at publish. Any later than the next is taken
   as equal to it, so a missing one (0) puts its stage's time into the
   stage before. */
typedef struct {
    uint64_t presented_ns, dequeued_ns, committed_ns, gpu_start_ns, gpu_end_ns, ready_ns;
} mtl_handoff_times;

typedef struct {
    int state;
    uint32_t holds;        /* compositor frames still reading it */
    uint64_t frame;        /* game frame number, 1 up, set at publish */
    uint64_t presented_ns; /* when the game called Present for it */
    uint64_t ready_ns;     /* when its copy completed on the GPU */
    uint64_t committed_ns; /* when its list was committed, 0 until then */
    uint64_t stage_us[MTL_AGE_STAGES - 1]; /* the stages up to publish, set there */
    uint32_t shown;        /* compositor frames that have shown it */
    void *tex;             /* the Metal side's texture, opaque here */
    uint32_t w, h;
} mtl_handoff_slot;

typedef struct {
    uint64_t compositor_frames; /* acquires, with or without a frame to show */
    uint64_t empty;             /* acquires before any frame was published */
    uint64_t published;         /* game frames handed over */
    uint64_t skipped;           /* game frames with no free slot, not handed over */
    uint64_t never_shown;       /* published, then replaced before any compositor frame showed it */
    uint64_t shown;             /* distinct game frames shown at least once */
    uint64_t shown_again;       /* distinct game frames shown more than once */
    uint64_t repeats;           /* compositor frames that showed a frame already shown */
    uint64_t age_n;
    double age_p50_ms, age_p95_ms, age_max_ms, age_mean_ms;
    /* The stages of each frame's age at its first showing: how many
       frames, and each stage's mean and max in ms. */
    uint64_t first_n;
    double stage_mean_ms[MTL_AGE_STAGES], stage_max_ms[MTL_AGE_STAGES];
} mtl_handoff_stats;

typedef struct {
    uint64_t n;
    uint64_t sum_us[MTL_AGE_STAGES], max_us[MTL_AGE_STAGES];
} mtl_handoff_stages;

typedef struct {
    pthread_mutex_t mu;
    mtl_handoff_slot slot[MTL_HANDOFF_SLOTS];
    int latest; /* the slot published last, -1 before the first */
    uint64_t next_frame;
    mtl_handoff_stats total;
    /* Ages since the last mtl_handoff_take_period, and for the whole run. */
    uint32_t period_hist[MTL_HANDOFF_AGE_BUCKETS], total_hist[MTL_HANDOFF_AGE_BUCKETS];
    uint64_t period_max_us, total_max_us;
    double period_sum_ms, total_sum_ms;
    mtl_handoff_stats period_base; /* the counters at the start of this period */
    mtl_handoff_stages period_stages, total_stages;
} mtl_handoff;

/* What the compositor gets for one of its frames. */
typedef struct {
    int slot;
    void *tex;
    uint32_t w, h;
    uint64_t frame;
    double age_ms;  /* from the game's Present to this acquire */
    int repeat;     /* this game frame was already shown by an earlier compositor frame */
} mtl_handoff_view;

void mtl_handoff_init(mtl_handoff *h);
void mtl_handoff_destroy(mtl_handoff *h);

/* Writer side, on the thread that replays lists. begin returns the slot
   to copy into (now WRITING, its texture and size readable until publish
   or abort), or -1 when every slot is held, latest or being written. */
int mtl_handoff_begin(mtl_handoff *h);
void mtl_handoff_set_texture(mtl_handoff *h, int slot, void *tex, uint32_t w, uint32_t h_px, void **old);
void mtl_handoff_get_texture(mtl_handoff *h, int slot, void **tex, uint32_t *w, uint32_t *h_px);
/* After the list that writes the slot is committed, for the age's stages.
   Ignored once the slot is no longer being written: the GPU can finish
   and publish before the committing thread gets here. */
void mtl_handoff_set_committed(mtl_handoff *h, int slot, uint64_t committed_ns);
/* From the copy's completion: the slot becomes the latest. A committed
   time given here wins over none; otherwise the one set above is used. */
uint64_t mtl_handoff_publish(mtl_handoff *h, int slot, const mtl_handoff_times *t);
void mtl_handoff_abort(mtl_handoff *h, int slot);

/* Compositor side. acquire returns 1 with the latest slot held, or 0 when
   nothing has been published yet; every call counts as a compositor
   frame. release is called once per successful acquire, from the
   compositor command buffer's completion. */
int mtl_handoff_acquire(mtl_handoff *h, uint64_t now_ns, mtl_handoff_view *out);
void mtl_handoff_release(mtl_handoff *h, int slot);

/* Counters for the whole run, and for the period since the last take
   (which then starts a new period). */
void mtl_handoff_totals(mtl_handoff *h, mtl_handoff_stats *out);
void mtl_handoff_take_period(mtl_handoff *h, mtl_handoff_stats *out);

#endif
