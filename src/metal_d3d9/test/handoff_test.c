/*
 * Unit test of the immersive handoff's bookkeeping (handoff.c) and the
 * virtual screen's placement math (immersive.h), with no Metal and a
 * fake clock: the slot rules, the counters a device run's log reports,
 * a two-thread run that checks the writer never takes a slot the reader
 * holds, and the four corners of the screen projected for sample views
 * against clip coordinates worked out by hand. Then the immersive space's
 * opening order (space.c) with a fake game, host and compositor: request,
 * open, first frame, dismiss, close, the watchdog, and the cancelled host
 * driver that once left the space unopened. And stereo's pure part
 * (stereo.c): the clip-space correction against a per-eye projection
 * worked out from view space, its identity case bit for bit, the
 * disparity's sign and size, recovering the game's a from a view
 * projection's rows and from a D3D matrix, the classing of draws, and the
 * scan of a vertex shader's position constants on hand-made bytecode.
 *
 * Usage: handoff_test
 */
#include "handoff.h"
#include "immersive.h"
#include "space.h"
#include "stereo.h"

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int s_fail, s_pass;

#define CHECK(cond, ...)                                                                                               \
    do {                                                                                                               \
        if (cond) {                                                                                                    \
            s_pass++;                                                                                                  \
            printf("ok   ");                                                                                           \
        } else {                                                                                                       \
            s_fail++;                                                                                                  \
            printf("FAIL ");                                                                                           \
        }                                                                                                              \
        printf(__VA_ARGS__);                                                                                           \
        printf("\n");                                                                                                  \
    } while (0)

#define MS(x) ((uint64_t)((x) * 1000000.0))

/* A publish with only Present and ready known, as the older tests need. */
static uint64_t publish2(mtl_handoff *h, int slot, uint64_t presented_ns, uint64_t ready_ns)
{
    mtl_handoff_times t = {.presented_ns = presented_ns, .ready_ns = ready_ns};
    return mtl_handoff_publish(h, slot, &t);
}

static void ring_rules(void)
{
    mtl_handoff h;
    mtl_handoff_init(&h);
    mtl_handoff_view v;
    mtl_handoff_stats s;

    int got = mtl_handoff_acquire(&h, MS(1), &v);
    mtl_handoff_totals(&h, &s);
    CHECK(!got && v.slot == -1 && s.compositor_frames == 1 && s.empty == 1,
          "acquire before any frame: nothing held, counted as a compositor frame with no game frame");

    int a = mtl_handoff_begin(&h);
    CHECK(a >= 0, "first begin gets a slot (%d)", a);
    got = mtl_handoff_acquire(&h, MS(2), &v);
    CHECK(!got, "a slot being written is not visible to the compositor");
    uint64_t f1 = publish2(&h, a, MS(10), MS(14));
    got = mtl_handoff_acquire(&h, MS(20), &v);
    CHECK(got && v.slot == a && v.frame == f1 && f1 == 1 && fabs(v.age_ms - 10.0) < 1e-9 && !v.repeat,
          "published slot is acquired: frame %llu, age %.3f ms from Present", (unsigned long long)v.frame, v.age_ms);
    mtl_handoff_view v2;
    got = mtl_handoff_acquire(&h, MS(31), &v2);
    CHECK(got && v2.slot == a && v2.repeat && fabs(v2.age_ms - 21.0) < 1e-9,
          "the same game frame acquired again is a repeat, age now %.3f ms", v2.age_ms);

    /* a is held twice and is the latest: the writer must take another. */
    int b = mtl_handoff_begin(&h);
    CHECK(b >= 0 && b != a, "begin skips the held latest slot (got %d, held %d)", b, a);
    publish2(&h, b, MS(40), MS(44));
    int c = mtl_handoff_begin(&h);
    CHECK(c >= 0 && c != a && c != b, "begin skips the held slot and the new latest (got %d)", c);
    int none = mtl_handoff_begin(&h);
    mtl_handoff_totals(&h, &s);
    CHECK(none == -1 && s.skipped == 1, "no slot free (one held, one latest, one being written): skipped, not waited");
    publish2(&h, c, MS(50), MS(55));
    mtl_handoff_totals(&h, &s);
    CHECK(s.never_shown == 1, "frame %d replaced before any compositor frame showed it counts as never shown", b);
    int d = mtl_handoff_begin(&h);
    CHECK(d == b, "a replaced, unheld frame's slot is reused (got %d, want %d)", d, b);
    mtl_handoff_abort(&h, d);
    mtl_handoff_release(&h, a);
    mtl_handoff_release(&h, a);
    int e = mtl_handoff_begin(&h);
    CHECK(e >= 0 && e != c, "after both releases the old slot is free again (got %d)", e);
    mtl_handoff_abort(&h, e);

    got = mtl_handoff_acquire(&h, MS(60), &v);
    mtl_handoff_release(&h, v.slot);
    got = mtl_handoff_acquire(&h, MS(71), &v);
    mtl_handoff_release(&h, v.slot);
    got = mtl_handoff_acquire(&h, MS(82), &v);
    mtl_handoff_release(&h, v.slot);
    mtl_handoff_totals(&h, &s);
    CHECK(s.compositor_frames == 7 && s.empty == 2 && s.published == 3 && s.shown == 2 && s.shown_again == 2 &&
              s.repeats == 3,
          "counters: compositor %llu, empty %llu, published %llu, shown %llu, shown again %llu, repeats %llu",
          (unsigned long long)s.compositor_frames, (unsigned long long)s.empty, (unsigned long long)s.published,
          (unsigned long long)s.shown, (unsigned long long)s.shown_again, (unsigned long long)s.repeats);
    mtl_handoff_destroy(&h);
}

static void ring_ages(void)
{
    mtl_handoff h;
    mtl_handoff_init(&h);
    int a = mtl_handoff_begin(&h);
    publish2(&h, a, MS(1000), MS(1001));
    mtl_handoff_view v;
    for (int i = 1; i <= 100; i++) {
        mtl_handoff_acquire(&h, MS(1000 + i), &v);
        mtl_handoff_release(&h, v.slot);
    }
    mtl_handoff_stats p;
    mtl_handoff_take_period(&h, &p);
    CHECK(p.age_n == 100 && fabs(p.age_p50_ms - 50.0) <= 0.25 && fabs(p.age_p95_ms - 95.0) <= 0.25 &&
              fabs(p.age_max_ms - 100.0) < 1e-6 && fabs(p.age_mean_ms - 50.5) < 1e-6 && p.repeats == 99,
          "ages 1..100 ms: p50 %.2f p95 %.2f max %.2f mean %.2f, %llu repeats", p.age_p50_ms, p.age_p95_ms, p.age_max_ms,
          p.age_mean_ms, (unsigned long long)p.repeats);
    mtl_handoff_acquire(&h, MS(1300), &v);
    mtl_handoff_release(&h, v.slot);
    mtl_handoff_take_period(&h, &p);
    mtl_handoff_stats t;
    mtl_handoff_totals(&h, &t);
    CHECK(p.age_n == 1 && p.compositor_frames == 1 && fabs(p.age_max_ms - 300.0) < 1e-6 &&
              fabs(p.age_p50_ms - 250.25) < 1e-6 && t.compositor_frames == 101 && t.age_n == 101,
          "a new period starts empty (n %llu, max %.1f, p50 in the overflow bucket %.2f) while the totals keep counting",
          (unsigned long long)p.age_n, p.age_max_ms, p.age_p50_ms);
    mtl_handoff_destroy(&h);
}

static int near(double a, double b) { return fabs(a - b) < 1e-6; }

/* Where a frame's age went: the stages of its first showing, which add up
   to that age, with the committed time from either side of the race
   between the committing thread and the GPU's completion. */
static void ring_stages(void)
{
    mtl_handoff h;
    mtl_handoff_init(&h);
    mtl_handoff_view v;
    mtl_handoff_stats p;

    /* Present 100, dequeued 101, committed 103 (set after commit), GPU
       104..108, ready 108.5, first acquire 112: 1 + 2 + 1 + 4 + 0.5 + 3.5. */
    int a = mtl_handoff_begin(&h);
    mtl_handoff_set_committed(&h, a, MS(103));
    mtl_handoff_times t = {MS(100), MS(101), 0, MS(104), MS(108), MS(108.5)};
    mtl_handoff_publish(&h, a, &t);
    mtl_handoff_acquire(&h, MS(112), &v);
    mtl_handoff_release(&h, v.slot);
    /* A repeat adds nothing to the stages: only first showings count. */
    mtl_handoff_acquire(&h, MS(120), &v);
    mtl_handoff_release(&h, v.slot);
    mtl_handoff_take_period(&h, &p);
    const double *m = p.stage_mean_ms;
    double sum = 0;
    for (int i = 0; i < MTL_AGE_STAGES; i++) sum += m[i];
    CHECK(p.first_n == 1 && near(m[MTL_AGE_QUEUE], 1) && near(m[MTL_AGE_REPLAY], 2) && near(m[MTL_AGE_GPU_WAIT], 1) &&
              near(m[MTL_AGE_GPU], 4) && near(m[MTL_AGE_PUBLISH], 0.5) && near(m[MTL_AGE_COMPOSITOR], 3.5) &&
              near(sum, 12),
          "stages of a first showing: queue %.2f replay %.2f GPU wait %.2f GPU %.2f publish %.2f compositor %.2f, "
          "sum %.2f ms = its age",
          m[0], m[1], m[2], m[3], m[4], m[5], sum);

    /* The GPU finished first: the slot is published before the committing
       thread says when it committed, which then must not touch it, and
       the commit's time folds into replay up to the GPU's start. */
    int b = mtl_handoff_begin(&h);
    mtl_handoff_times t2 = {MS(200), MS(201), 0, MS(203), MS(205), MS(206)};
    mtl_handoff_publish(&h, b, &t2);
    mtl_handoff_set_committed(&h, b, MS(202));
    CHECK(h.slot[b].state == MTL_SLOT_READY && h.slot[b].committed_ns == 0,
          "a commit time after publish leaves the published slot alone");
    mtl_handoff_acquire(&h, MS(207), &v);
    mtl_handoff_release(&h, v.slot);
    mtl_handoff_take_period(&h, &p);
    CHECK(p.first_n == 1 && near(m[MTL_AGE_REPLAY], 2) && near(m[MTL_AGE_GPU_WAIT], 0) && near(m[MTL_AGE_GPU], 2) &&
              near(m[MTL_AGE_COMPOSITOR], 1),
          "no commit time: replay runs to the GPU's start (%.2f), GPU wait %.2f", m[MTL_AGE_REPLAY],
          m[MTL_AGE_GPU_WAIT]);

    /* A slot begun again starts with no committed time from its last use. */
    int c = mtl_handoff_begin(&h);
    mtl_handoff_set_committed(&h, c, MS(303));
    mtl_handoff_abort(&h, c);
    int c2 = mtl_handoff_begin(&h);
    CHECK(c2 == c && h.slot[c2].committed_ns == 0, "a slot begun again forgets the aborted write's commit time");
    mtl_handoff_abort(&h, c2);

    /* Stamps out of order (clocks read on different threads) are clamped,
       never negative, and the stages still add up to the age. */
    int d = mtl_handoff_begin(&h);
    mtl_handoff_times t3 = {MS(400), MS(402), MS(401), MS(403), MS(409), MS(408)};
    mtl_handoff_publish(&h, d, &t3);
    mtl_handoff_acquire(&h, MS(410), &v);
    mtl_handoff_release(&h, v.slot);
    mtl_handoff_take_period(&h, &p);
    sum = 0;
    int neg = 0;
    for (int i = 0; i < MTL_AGE_STAGES; i++) {
        sum += m[i];
        neg |= m[i] < 0;
    }
    CHECK(!neg && near(sum, 10) && near(m[MTL_AGE_GPU_WAIT], 2) && near(m[MTL_AGE_PUBLISH], 0),
          "out of order stamps: stages clamped (GPU wait %.2f, publish %.2f), sum %.2f ms = the age",
          m[MTL_AGE_GPU_WAIT], m[MTL_AGE_PUBLISH], sum);

    mtl_handoff_stats tot;
    mtl_handoff_totals(&h, &tot);
    CHECK(tot.first_n == 3 && near(tot.stage_max_ms[MTL_AGE_GPU], 5) && near(tot.stage_max_ms[MTL_AGE_COMPOSITOR], 3.5),
          "totals keep every period's first showings (%llu, GPU max %.2f, compositor wait max %.2f ms)",
          (unsigned long long)tot.first_n, tot.stage_max_ms[MTL_AGE_GPU], tot.stage_max_ms[MTL_AGE_COMPOSITOR]);
    mtl_handoff_destroy(&h);
}

/* Two threads, as on the device: the writer stamps a slot with the frame
   it is about to publish, the reader checks a held slot keeps its stamp
   for as long as it holds it. Any overlap of a write with a hold shows
   as a changed stamp or a write into a slot marked as being read. */
#define STRESS_FRAMES 200000

static mtl_handoff s_h;
static _Atomic uint64_t s_stamp[MTL_HANDOFF_SLOTS];
static atomic_int s_reading[MTL_HANDOFF_SLOTS];
static atomic_int s_done;
static _Atomic uint64_t s_overlaps, s_torn, s_reads;

static void *writer(void *arg)
{
    (void)arg;
    uint64_t t = 0;
    for (uint64_t i = 1; i <= STRESS_FRAMES; i++) {
        int s = mtl_handoff_begin(&s_h);
        if (s < 0) continue;
        if (atomic_load(&s_reading[s])) atomic_fetch_add(&s_overlaps, 1);
        atomic_store(&s_stamp[s], i);
        t += 1000;
        publish2(&s_h, s, t, t);
    }
    atomic_store(&s_done, 1);
    return NULL;
}

static void *reader(void *arg)
{
    (void)arg;
    uint64_t t = 0, last = 0;
    while (!atomic_load(&s_done)) {
        mtl_handoff_view v;
        t += 700;
        if (!mtl_handoff_acquire(&s_h, t, &v)) continue;
        atomic_fetch_add(&s_reading[v.slot], 1);
        uint64_t before = atomic_load(&s_stamp[v.slot]);
        for (volatile int k = 0; k < 50; k++) {
        }
        uint64_t after = atomic_load(&s_stamp[v.slot]);
        if (before != after) atomic_fetch_add(&s_torn, 1);
        if (v.frame < last) atomic_fetch_add(&s_torn, 1); /* frames only move forward */
        last = v.frame;
        atomic_fetch_sub(&s_reading[v.slot], 1);
        mtl_handoff_release(&s_h, v.slot);
        atomic_fetch_add(&s_reads, 1);
    }
    return NULL;
}

static void ring_threads(void)
{
    mtl_handoff_init(&s_h);
    pthread_t w, r;
    pthread_create(&r, NULL, reader, NULL);
    pthread_create(&w, NULL, writer, NULL);
    pthread_join(w, NULL);
    pthread_join(r, NULL);
    mtl_handoff_stats s;
    mtl_handoff_totals(&s_h, &s);
    int held = 0;
    for (int i = 0; i < MTL_HANDOFF_SLOTS; i++) held += s_h.slot[i].holds;
    CHECK(!atomic_load(&s_overlaps) && !atomic_load(&s_torn) && held == 0 && s.published + s.skipped == STRESS_FRAMES &&
              s.published > 0,
          "two threads, %d frames: %llu published, %llu skipped, %llu reads, %llu writes into a held slot, %llu "
          "changed under a hold, %d holds left",
          STRESS_FRAMES, (unsigned long long)s.published, (unsigned long long)s.skipped,
          (unsigned long long)atomic_load(&s_reads), (unsigned long long)atomic_load(&s_overlaps),
          (unsigned long long)atomic_load(&s_torn), held);
    mtl_handoff_destroy(&s_h);
}

/* Screen math */

static void translate(float m[16], float x, float y, float z)
{
    mtl_screen_identity(m);
    m[12] = x;
    m[13] = y;
    m[14] = z;
}

/* Yaw about +Y, then pitch about the yawed +X: a head turned and nodded. */
static void head_pose(float m[16], float yaw_deg, float pitch_deg, float x, float y, float z)
{
    float a = yaw_deg * (float)M_PI / 180.0f, p = pitch_deg * (float)M_PI / 180.0f;
    float yaw[16], pitch[16], r[16];
    mtl_screen_identity(yaw);
    yaw[0] = cosf(a);
    yaw[2] = -sinf(a);
    yaw[8] = sinf(a);
    yaw[10] = cosf(a);
    mtl_screen_identity(pitch);
    pitch[5] = cosf(p);
    pitch[6] = sinf(p);
    pitch[9] = -sinf(p);
    pitch[10] = cosf(p);
    mtl_screen_mul(yaw, pitch, r);
    memcpy(m, r, sizeof r);
    m[12] = x;
    m[13] = y;
    m[14] = z;
}

/* A reverse-Z projection with an infinite far plane, as CompositorServices
   gives (rightUpBack: the view looks down -Z), from the view's tangents. */
static void projection(float m[16], float left, float right, float up, float down, float near)
{
    memset(m, 0, 16 * sizeof(float));
    m[0] = 2.0f / (left + right);
    m[5] = 2.0f / (up + down);
    m[8] = (right - left) / (left + right);
    m[9] = (up - down) / (up + down);
    m[11] = -1.0f;
    m[14] = near;
}

static const float k_corner[4][4] = {{-0.5f, 0.5f, 0, 1}, {0.5f, 0.5f, 0, 1}, {-0.5f, -0.5f, 0, 1}, {0.5f, -0.5f, 0, 1}};
static const char *const k_name[4] = {"top left", "top right", "bottom left", "bottom right"};

static int near4(const float a[4], const float b[4], float eps)
{
    for (int i = 0; i < 4; i++)
        if (fabsf(a[i] - b[i]) > eps) return 0;
    return 1;
}

static int corners_are(const char *what, const float mvp[16], const float want[4][4])
{
    int ok = 1;
    for (int i = 0; i < 4; i++) {
        float c[4];
        mtl_screen_transform(mvp, k_corner[i], c);
        if (!near4(c, want[i], 1e-4f)) {
            printf("     %s %s: clip (%.5f, %.5f, %.5f, %.5f), want (%.5f, %.5f, %.5f, %.5f)\n", what, k_name[i], c[0],
                   c[1], c[2], c[3], want[i][0], want[i][1], want[i][2], want[i][3]);
            ok = 0;
        }
    }
    return ok;
}

static void screen_math(void)
{
    const float dist = 2.5f, width = 2.5f, aspect = 4.0f / 3.0f, near = 0.1f;
    float head[16], model[16], proj[16], mvp[16];

    /* 1. Head at the origin looking down -Z, a symmetric 90 degree view.
       The corners sit at x = +-1.25, y = +-0.9375, z = -2.5; with tangent
       1 the clip x and y are the eye-space x and y, z is the near plane
       (reverse-Z) and w is the distance, 2.5: NDC (+-0.5, +-0.375, 0.04). */
    mtl_screen_identity(head);
    mtl_screen_place(head, dist, width, aspect, model);
    projection(proj, 1, 1, 1, 1, near);
    mtl_screen_mvp(proj, head, model, mvp);
    const float want1[4][4] = {
        {-1.25f, 0.9375f, near, 2.5f}, {1.25f, 0.9375f, near, 2.5f}, {-1.25f, -0.9375f, near, 2.5f}, {1.25f, -0.9375f, near, 2.5f}};
    CHECK(corners_are("ahead", mvp, want1), "screen straight ahead: corners at clip (+-1.25, +-0.9375, 0.1, 2.5), NDC "
                                            "(+-0.5, +-0.375, 0.04)");

    /* 2. The same view from one eye 32 mm to the right: every corner moves
       0.032 left in clip x, nothing else. */
    float eye[16], off[16];
    translate(off, 0.032f, 0, 0);
    mtl_screen_mul(head, off, eye);
    mtl_screen_mvp(proj, eye, model, mvp);
    float want2[4][4];
    memcpy(want2, want1, sizeof want2);
    for (int i = 0; i < 4; i++) want2[i][0] -= 0.032f;
    CHECK(corners_are("right eye", mvp, want2), "right eye 32 mm over: the screen shifts 0.032 left in clip x, same depth");

    /* 3. Opened with the head turned 90 degrees left (facing -X), nodded
       20 degrees down, standing at (1, 1.6, 2). The screen is 2.5 m along
       -X at head height, upright: centre (-1.5, 1.6, 2). Seen from the
       same place looking level the same way, it projects as in case 1. */
    head_pose(head, 90, -20, 1, 1.6f, 2);
    mtl_screen_place(head, dist, width, aspect, model);
    float centre[4], o[4] = {0, 0, 0, 1};
    mtl_screen_transform(model, o, centre);
    float top[4], top_local[4] = {0, 0.5f, 0, 1};
    mtl_screen_transform(model, top_local, top);
    CHECK(fabsf(centre[0] + 1.5f) < 1e-5f && fabsf(centre[1] - 1.6f) < 1e-5f && fabsf(centre[2] - 2) < 1e-5f &&
              fabsf(top[0] - centre[0]) < 1e-5f && fabsf(top[2] - centre[2]) < 1e-5f,
          "turned and nodded: centre (%.3f, %.3f, %.3f), upright (top edge straight above it)", centre[0], centre[1],
          centre[2]);
    float level[16];
    head_pose(level, 90, 0, 1, 1.6f, 2);
    mtl_screen_mvp(proj, level, model, mvp);
    CHECK(corners_are("turned", mvp, want1), "turned: seen level from where it was placed, the corners match case 1");

    /* 4. Nodded 20 degrees down at the same spot, the view sees the screen
       higher in the frame: the corners' eye-space y and z rotate by +20
       degrees about x. Worked: top edge y' = 0.9375 cos20 + 2.5 sin20,
       z' = 0.9375 sin20 - 2.5 cos20, w = -z'. */
    mtl_screen_mvp(proj, head, model, mvp);
    float c20 = cosf(20 * (float)M_PI / 180), s20 = sinf(20 * (float)M_PI / 180);
    float want4[4][4];
    for (int i = 0; i < 4; i++) {
        float y = i < 2 ? 0.9375f : -0.9375f, z = -2.5f;
        float y2 = y * c20 - z * s20, z2 = y * s20 + z * c20;
        want4[i][0] = i % 2 ? 1.25f : -1.25f;
        want4[i][1] = y2;
        want4[i][2] = near;
        want4[i][3] = -z2;
    }
    CHECK(corners_are("nodded", mvp, want4), "nodded 20 degrees down: corners rise in clip y as worked by hand (top %.4f)",
          want4[0][1]);

    /* 5. An asymmetric view, as a headset's eye has: tangents left 1.0,
       right 0.8, up 0.9, down 1.1. x / -z in [-1.0, 0.8] maps to NDC
       [-1, 1], so x_c = 2x / 1.8 + (0.8 - 1.0) / 1.8 * z, and likewise
       y_c = 2y / 2.0 + (0.9 - 1.1) / 2.0 * z. */
    mtl_screen_identity(head);
    mtl_screen_place(head, dist, width, aspect, model);
    projection(proj, 1.0f, 0.8f, 0.9f, 1.1f, near);
    mtl_screen_mvp(proj, head, model, mvp);
    float want5[4][4];
    for (int i = 0; i < 4; i++) {
        float x = i % 2 ? 1.25f : -1.25f, y = i < 2 ? 0.9375f : -0.9375f, z = -2.5f;
        want5[i][0] = 2 * x / 1.8f + (-0.2f / 1.8f) * z;
        want5[i][1] = 2 * y / 2.0f + (-0.2f / 2.0f) * z;
        want5[i][2] = near;
        want5[i][3] = -z;
    }
    CHECK(corners_are("asymmetric", mvp, want5), "asymmetric eye tangents: corners as worked (top left %.4f, %.4f)",
          want5[0][0], want5[0][1]);

    /* 6. Opened looking straight down: no heading to flatten, so the head's
       up vector (which points where the face was turned) gives it. */
    head_pose(head, 0, -90, 0, 1.6f, 0);
    mtl_screen_place(head, dist, width, aspect, model);
    mtl_screen_transform(model, o, centre);
    CHECK(fabsf(centre[0]) < 1e-4f && fabsf(centre[1] - 1.6f) < 1e-5f && fabsf(centre[2] + 2.5f) < 1e-4f,
          "opened looking straight down: the screen goes ahead of the feet, centre (%.3f, %.3f, %.3f)", centre[0],
          centre[1], centre[2]);
    head_pose(head, 0, 90, 0, 1.6f, 0);
    mtl_screen_place(head, dist, width, aspect, model);
    mtl_screen_transform(model, o, centre);
    CHECK(fabsf(centre[0]) < 1e-4f && fabsf(centre[2] + 2.5f) < 1e-4f,
          "opened looking straight up: still ahead, centre (%.3f, %.3f, %.3f)", centre[0], centre[1], centre[2]);
}

/* The immersive space's order, with a fake game publishing into a real
   ring, a fake host driving the open as the visionOS host does, and a
   fake compositor reading the ring. */
typedef struct {
    mtl_handoff ring;
    mtl_space sp;
    uint64_t t;
    uint64_t game_frames, game_stalls;
    int dismissed, dismiss_bad;
} fake;

static void fake_init(fake *f)
{
    mtl_handoff_init(&f->ring);
    mtl_space_init(&f->sp);
    f->t = MS(1000);
    f->game_frames = f->game_stalls = 0;
    f->dismissed = f->dismiss_bad = 0;
}

static void fake_destroy(fake *f)
{
    mtl_space_destroy(&f->sp);
    mtl_handoff_destroy(&f->ring);
}

static uint64_t ring_frames(fake *f)
{
    mtl_handoff_stats st;
    mtl_handoff_totals(&f->ring, &st);
    return st.published;
}

/* The game presents while the ring is on, as Present does: it takes a
   slot or skips, never waits. A missing slot would be the game held up
   by the compositor. */
static void game_present(fake *f)
{
    f->t += MS(11);
    int ring, window;
    mtl_space_outputs(&f->sp, &ring, &window);
    if (!ring) return;
    int slot = mtl_handoff_begin(&f->ring);
    if (slot < 0) {
        f->game_stalls++;
        return;
    }
    publish2(&f->ring, slot, f->t, f->t + MS(4));
    f->game_frames++;
}

/* The host's dismiss, as WindowCoordinator does it: only on the claim,
   and the window's layer must already be off when the window goes. */
static void host_dismiss_if_ready(fake *f)
{
    if (!mtl_space_take_dismiss(&f->sp, f->t)) return;
    int ring, window;
    mtl_space_outputs(&f->sp, &ring, &window);
    if (window || !ring || !f->sp.opened_ns || !f->sp.shown_ns) f->dismiss_bad++;
    f->dismissed++;
}

/* One compositor frame: the newest game frame, shown and released when
   its GPU work completes. Returns whether a game frame was on screen. */
static int compositor_frame(fake *f)
{
    f->t += MS(8);
    mtl_handoff_view v;
    int got = mtl_handoff_acquire(&f->ring, f->t, &v);
    if (!got) return 0;
    mtl_handoff_release(&f->ring, v.slot);
    mtl_space_first_shown(&f->sp, f->t);
    return 1;
}

static void space_order(void)
{
    fake f;
    int ring, window, fired;
    char line[1024];

    /* 1. Request, open, first frame, dismiss, in that order. */
    fake_init(&f);
    mtl_space_outputs(&f.sp, &ring, &window);
    CHECK(!ring && window, "space: before the request, frames go to the window only");
    mtl_space_request(&f.sp, f.t);
    mtl_space_outputs(&f.sp, &ring, &window);
    CHECK(ring && window, "space: from the request, frames go to the ring and still to the window");
    int polled = mtl_space_should_open(&f.sp, ring_frames(&f), f.t);
    CHECK(polled == 0, "space: no open before a frame is in the ring (poll %d)", polled);
    for (int i = 0; i < 10; i++) game_present(&f);
    CHECK(f.game_frames == 10 && f.game_stalls == 0 && ring_frames(&f) == 10,
          "space: with no compositor yet the game hands over every frame and never waits (%llu handed, %llu stalled)",
          (unsigned long long)f.game_frames, (unsigned long long)f.game_stalls);
    polled = mtl_space_should_open(&f.sp, ring_frames(&f), f.t);
    int second = mtl_space_should_open(&f.sp, ring_frames(&f), f.t);
    CHECK(polled == 1 && second == 0, "space: a frame in the ring claims the open once (%d), a second driver waits (%d)",
          polled, second);
    for (int i = 0; i < 5; i++) game_present(&f);
    mtl_space_outputs(&f.sp, &ring, &window);
    CHECK(ring && window, "space: while the system opens the space the window keeps getting frames");
    mtl_space_open_result(&f.sp, METAL_D3D9_SPACE_OPENED, f.t);
    host_dismiss_if_ready(&f);
    mtl_space_outputs(&f.sp, &ring, &window);
    CHECK(!f.dismissed && window, "space: opened but nothing shown yet, the window stays and keeps its frames");
    mtl_space_compositor_started(&f.sp, f.t);
    int shown = compositor_frame(&f);
    host_dismiss_if_ready(&f);
    host_dismiss_if_ready(&f);
    mtl_space_outputs(&f.sp, &ring, &window);
    CHECK(shown && f.dismissed == 1 && !f.dismiss_bad && ring && !window,
          "space: the compositor's first frame dismisses the window once, its layer already off (%d dismissed, %d "
          "early)",
          f.dismissed, f.dismiss_bad);
    for (int i = 0; i < 20; i++) {
        game_present(&f);
        compositor_frame(&f);
    }
    CHECK(f.game_stalls == 0, "space: with the compositor reading, the game still never waits (%llu stalled)",
          (unsigned long long)f.game_stalls);
    CHECK(!mtl_space_watchdog(&f.sp, ring_frames(&f), f.t + MS(60000), line, sizeof line),
          "space: no watchdog line once the space is open");

    /* 2. The space closes after the window went: the window comes back
       before the frames do. */
    int reopen = mtl_space_closed(&f.sp, f.t);
    mtl_space_outputs(&f.sp, &ring, &window);
    CHECK(reopen == 1 && ring && !window,
          "space: closed with the window gone, a window is opened and frames wait in the ring until it is back");
    mtl_space_window_back(&f.sp, f.t);
    mtl_space_outputs(&f.sp, &ring, &window);
    CHECK(!ring && window, "space: the window is back and frames go to it alone");
    CHECK(mtl_space_should_open(&f.sp, ring_frames(&f), f.t) == -1, "space: a closed space is not opened again");

    /* 2b. Return to Immersive, three times over: the same order as the
       first open each time, only frames handed over after the request
       claim it, and each close brings the window back before the frames. */
    for (int round = 1; round <= 3; round++) {
        int was_dismissed = f.dismissed;
        CHECK(mtl_space_reopen(&f.sp, ring_frames(&f), f.t) == 1 && mtl_space_reopen(&f.sp, ring_frames(&f), f.t) == 0,
              "space reopen %d: taken once from the window, a second tap while it is wanted does nothing", round);
        mtl_space_outputs(&f.sp, &ring, &window);
        polled = mtl_space_should_open(&f.sp, ring_frames(&f), f.t);
        CHECK(ring && window && polled == 0 && ring_frames(&f) > 0,
              "space reopen %d: frames to the ring and still to the window; the %llu old frames claim nothing (poll %d)",
              round, (unsigned long long)ring_frames(&f), polled);
        game_present(&f);
        polled = mtl_space_should_open(&f.sp, ring_frames(&f), f.t);
        second = mtl_space_should_open(&f.sp, ring_frames(&f), f.t);
        CHECK(polled == 1 && second == 0, "space reopen %d: a new frame claims the open once (%d, then %d)", round, polled,
              second);
        mtl_space_open_result(&f.sp, METAL_D3D9_SPACE_OPENED, f.t);
        host_dismiss_if_ready(&f);
        CHECK(f.dismissed == was_dismissed, "space reopen %d: opened, nothing shown yet, the window stays", round);
        mtl_space_compositor_started(&f.sp, f.t);
        game_present(&f);
        shown = compositor_frame(&f);
        host_dismiss_if_ready(&f);
        host_dismiss_if_ready(&f);
        mtl_space_outputs(&f.sp, &ring, &window);
        CHECK(shown && f.dismissed == was_dismissed + 1 && !f.dismiss_bad && ring && !window,
              "space reopen %d: the new compositor's first frame dismisses the window once, its layer already off", round);
        CHECK(!mtl_space_watchdog(&f.sp, ring_frames(&f), f.t + MS(60000), line, sizeof line),
              "space reopen %d: no watchdog line once it is open", round);
        reopen = mtl_space_closed(&f.sp, f.t);
        mtl_space_outputs(&f.sp, &ring, &window);
        CHECK(reopen == 1 && ring && !window, "space reopen %d: closed, the window comes back first", round);
        mtl_space_window_back(&f.sp, f.t);
        mtl_space_outputs(&f.sp, &ring, &window);
        CHECK(!ring && window && mtl_space_should_open(&f.sp, ring_frames(&f), f.t) == -1,
              "space reopen %d: back in the window, frames to it alone, nothing to open (%llu spaces opened)", round,
              (unsigned long long)f.sp.opens);
    }
    fake_destroy(&f);

    /* 2c. A reopen is refused while a space is wanted or open, and it
       watches the new request as it did the first. */
    fake_init(&f);
    mtl_space_request(&f.sp, f.t);
    CHECK(mtl_space_reopen(&f.sp, 0, f.t) == 0, "space reopen: refused while the first open is pending");
    game_present(&f);
    mtl_space_should_open(&f.sp, ring_frames(&f), f.t);
    mtl_space_open_result(&f.sp, METAL_D3D9_SPACE_OPENED, f.t);
    CHECK(mtl_space_reopen(&f.sp, ring_frames(&f), f.t) == 0, "space reopen: refused while the space is open");
    compositor_frame(&f);
    host_dismiss_if_ready(&f);
    mtl_space_closed(&f.sp, f.t);
    CHECK(mtl_space_reopen(&f.sp, ring_frames(&f), f.t) == 0, "space reopen: refused until the window holds the game");
    mtl_space_window_back(&f.sp, f.t);
    uint64_t rq = f.t + MS(1000);
    CHECK(mtl_space_reopen(&f.sp, ring_frames(&f), rq) == 1, "space reopen: taken once the window is back");
    fired = 0;
    for (int i = 0; i < 2; i++) fired += mtl_space_watchdog(&f.sp, ring_frames(&f), rq + MS(5000 + i), line, sizeof line);
    CHECK(fired == 1 && strstr(line, "frames in the ring 0") && strstr(line, "open called no"),
          "space reopen watchdog: one line 5 s after the new request, counting only its frames: %s", line);
    fake_destroy(&f);

    /* 3. The compositor's first frame completes before openImmersiveSpace
       returns to the host: the open's result does the dismiss. */
    fake_init(&f);
    mtl_space_request(&f.sp, f.t);
    game_present(&f);
    mtl_space_should_open(&f.sp, ring_frames(&f), f.t);
    mtl_space_compositor_started(&f.sp, f.t);
    compositor_frame(&f);
    host_dismiss_if_ready(&f);
    CHECK(!f.dismissed, "space: a frame shown before the open's result does not dismiss yet");
    mtl_space_open_result(&f.sp, METAL_D3D9_SPACE_OPENED, f.t);
    host_dismiss_if_ready(&f);
    CHECK(f.dismissed == 1 && !f.dismiss_bad, "space: the open's result then dismisses the window once");
    fake_destroy(&f);

    /* 4. The regression. SwiftUI cancels and restarts a view's task when
       its id changes, which the launcher's state does in the same view's
       onAppear. The first driver polled, was cancelled, and the old
       one-shot flag it had taken left the restarted driver nothing to do.
       Here a driver cancelled before a frame holds no claim, and one
       cancelled after its claim hands it back. */
    fake_init(&f);
    mtl_space_request(&f.sp, f.t);
    CHECK(mtl_space_should_open(&f.sp, ring_frames(&f), f.t) == 0, "space: a driver that polls before a frame claims nothing");
    /* ...and is cancelled; its replacement polls. */
    game_present(&f);
    CHECK(mtl_space_should_open(&f.sp, ring_frames(&f), f.t) == 1,
          "space: the restarted driver still gets the open after the first was cancelled");
    mtl_space_open_result(&f.sp, METAL_D3D9_SPACE_ABANDONED, f.t);
    mtl_space_outputs(&f.sp, &ring, &window);
    CHECK(ring && window, "space: an abandoned open keeps the ring and the window fed");
    CHECK(mtl_space_should_open(&f.sp, ring_frames(&f), f.t) == 1, "space: an abandoned claim goes to the next driver");
    mtl_space_open_result(&f.sp, METAL_D3D9_SPACE_OPENED, f.t);
    compositor_frame(&f);
    host_dismiss_if_ready(&f);
    CHECK(f.dismissed == 1 && f.sp.attempts == 2, "space: the next driver opens it and the window goes (%llu attempts)",
          (unsigned long long)f.sp.attempts);
    fake_destroy(&f);

    /* 5. Declined: the game never left the window. */
    fake_init(&f);
    mtl_space_request(&f.sp, f.t);
    game_present(&f);
    mtl_space_should_open(&f.sp, ring_frames(&f), f.t);
    mtl_space_open_result(&f.sp, METAL_D3D9_SPACE_DECLINED, f.t);
    mtl_space_outputs(&f.sp, &ring, &window);
    CHECK(!ring && window && mtl_space_should_open(&f.sp, ring_frames(&f), f.t) == -1,
          "space: a declined space leaves frames in the window alone and is not retried");
    fake_destroy(&f);

    /* 6. Closed before the window went: frames are back on it at once. */
    fake_init(&f);
    mtl_space_request(&f.sp, f.t);
    game_present(&f);
    mtl_space_should_open(&f.sp, ring_frames(&f), f.t);
    mtl_space_open_result(&f.sp, METAL_D3D9_SPACE_OPENED, f.t);
    reopen = mtl_space_closed(&f.sp, f.t);
    mtl_space_outputs(&f.sp, &ring, &window);
    CHECK(reopen == 0 && !ring && window, "space: closed with the window still up, the window has the frames at once");
    fake_destroy(&f);

    /* 7. The watchdog: the stalled run, where no driver ever opened the
       space while the game kept handing over frames. */
    fake_init(&f);
    uint64_t req = f.t;
    mtl_space_request(&f.sp, req);
    mtl_space_should_open(&f.sp, 0, req + MS(100));
    CHECK(!mtl_space_watchdog(&f.sp, 0, req + MS(4999), line, sizeof line), "space watchdog: quiet before 5 s");
    fired = mtl_space_watchdog(&f.sp, 0, req + MS(5000), line, sizeof line);
    CHECK(fired && strstr(line, "frames in the ring 0") && strstr(line, "host polls 1") &&
              strstr(line, "open called no"),
          "space watchdog: at 5 s it logs each step: %s", fired ? line : "(nothing)");
    CHECK(!mtl_space_watchdog(&f.sp, 0, req + MS(6000), line, sizeof line), "space watchdog: once per cause");
    /* Frames arrive at 8 s, seen only by the render thread's check. */
    mtl_space_watchdog(&f.sp, 40, req + MS(8000), line, sizeof line);
    CHECK(!mtl_space_watchdog(&f.sp, 400, req + MS(12999), line, sizeof line),
          "space watchdog: quiet until 5 s after the first frame");
    fired = mtl_space_watchdog(&f.sp, 400, req + MS(13000), line, sizeof line);
    CHECK(fired && strstr(line, "frames in the ring 400, first at 8.00 s") && strstr(line, "last 12.90 s ago") &&
              strstr(line, "waiting for a driver") && strstr(line, "to the window on"),
          "space watchdog: 5 s after the first frame it logs again, showing the driver gone quiet: %s",
          fired ? line : "(nothing)");
    CHECK(!mtl_space_watchdog(&f.sp, 900, req + MS(60000), line, sizeof line), "space watchdog: never a third line");
    fake_destroy(&f);
}

/* Stereo */

static void apply_fix(const mtl_stereo_fix *f, const float c[4], float o[4])
{
    o[0] = f->x[0] * c[0] + f->x[1] * c[3] + f->x[2];
    o[1] = f->y[0] * c[1] + f->y[1] * c[3] + f->y[2];
    o[2] = c[2];
    o[3] = c[3];
}

/* A D3D left-handed perspective with an off-centre: clip from view. */
static void persp_clip(const mtl_stereo_proj *p, float q, float r, const float v[3], float c[4])
{
    c[0] = p->a * v[0] + p->cx * v[2];
    c[1] = p->b * v[1] + p->cy * v[2];
    c[2] = q * v[2] + r;
    c[3] = v[2];
}

static uint32_t dst_tok(int type, int num, uint32_t mask)
{
    return 0x80000000u | (uint32_t)num | ((uint32_t)(type & 7) << 28) | ((uint32_t)(type & 0x18) << 8) | mask << 16;
}
static uint32_t src_tok(int type, int num)
{
    return 0x80000000u | (uint32_t)num | ((uint32_t)(type & 7) << 28) | ((uint32_t)(type & 0x18) << 8) | 0xE4u << 16;
}

static void stereo_math(void)
{
    const float q = 4096.0f / (4096.0f - 4.0f), r = -4.0f * q;
    mtl_stereo_proj game = {1.3032f, 0.0f, 1.7377f, 0.0f};
    mtl_stereo_fix fix;

    /* Identity: the eye's projection is the game's and the eye is at the
       camera, so every vertex keeps its bits. */
    mtl_stereo_correction(&game, &game, 0.0f, &fix);
    int same = 1;
    for (int i = 0; i < 200; i++) {
        float v[3] = {(float)(i % 17) * 7.3f - 60.0f, (float)(i % 11) * 5.1f - 30.0f, 4.0f + (float)i * 9.7f};
        float c[4], o[4];
        persp_clip(&game, q, r, v, c);
        apply_fix(&fix, c, o);
        same &= !memcmp(c, o, sizeof c);
    }
    CHECK(same && fix.x[0] == 1.0f && fix.x[1] == 0.0f && fix.y[0] == 1.0f && fix.y[1] == 0.0f,
          "stereo identity: P_eye = P_game, eye at the camera, 200 vertices bit for bit");

    /* The general case against the eye's view worked out directly. */
    mtl_stereo_proj g2 = {1.21f, 0.031f, 1.62f, -0.017f}, eye = {0.93f, -0.087f, 0.88f, 0.052f};
    float e = 1.26f, worst = 0;
    mtl_stereo_correction(&g2, &eye, e, &fix);
    for (int i = 0; i < 200; i++) {
        float v[3] = {(float)(i % 13) * 9.1f - 55.0f, (float)(i % 7) * 6.3f - 20.0f, 6.0f + (float)i * 13.1f};
        float c[4], o[4], want[4], ve[3] = {v[0] - e, v[1], v[2]};
        persp_clip(&g2, q, r, v, c);
        apply_fix(&fix, c, o);
        persp_clip(&eye, q, r, ve, want);
        for (int k = 0; k < 4; k++) {
            float d = fabsf(o[k] - want[k]) / fmaxf(1.0f, fabsf(want[k]));
            if (d > worst) worst = d;
        }
    }
    CHECK(worst < 1e-5f, "stereo correction to another projection and eye matches the eye's own view (worst %.2g)",
          (double)worst);

    /* Stage A's disparity: left eye at -h shifts a point right, the right
       eye left, by a h / z each in NDC, so nearer means more. */
    float h = 0.032f / 0.0254f, d_near = 0, d_far = 0;
    for (int k = 0; k < 2; k++) {
        float z = k ? 400.0f : 50.0f, v[3] = {3.0f, 1.0f, z}, c[4], l[4], rr[4];
        persp_clip(&game, q, r, v, c);
        mtl_stereo_fix fl, fr;
        mtl_stereo_correction(&game, &game, -h, &fl);
        mtl_stereo_correction(&game, &game, h, &fr);
        apply_fix(&fl, c, l);
        apply_fix(&fr, c, rr);
        float disp = l[0] / l[3] - rr[0] / rr[3];
        if (k) d_far = disp;
        else d_near = disp;
        CHECK(fabsf(disp - 2.0f * game.a * h / z) < 1e-5f && l[2] == c[2] && rr[3] == c[3],
              "stereo disparity at z %.0f: %.5f NDC (want 2 a h / z = %.5f), depth and w unchanged", (double)z,
              (double)disp, (double)(2.0f * game.a * h / z));
    }
    CHECK(d_near > d_far && d_far > 0, "stereo disparity is crossed (left image right) and shrinks with depth");

    /* a from a view projection's rows, for a camera turned and tilted. */
    float yaw = 0.7f, pitch = -0.3f;
    float right[3] = {cosf(yaw), 0, -sinf(yaw)};
    float fwd[3] = {sinf(yaw) * cosf(pitch), sinf(pitch), cosf(yaw) * cosf(pitch)};
    float pos[3] = {120, 40, -300};
    float xr[4], wr[4];
    for (int k = 0; k < 3; k++) {
        xr[k] = g2.a * right[k] + g2.cx * fwd[k];
        wr[k] = fwd[k];
    }
    xr[3] = -(xr[0] * pos[0] + xr[1] * pos[1] + xr[2] * pos[2]);
    wr[3] = -(wr[0] * pos[0] + wr[1] * pos[1] + wr[2] * pos[2]);
    float a = mtl_stereo_a_from_rows(xr, wr);
    float ortho_w[4] = {0, 0, 0, 1};
    CHECK(fabsf(a - g2.a) < 1e-5f && mtl_stereo_a_from_rows(xr, ortho_w) == 0.0f,
          "stereo a from a turned camera's view projection rows: %.5f (want %.5f); orthographic gives 0", (double)a,
          (double)g2.a);

    float pd[16] = {0}, od[16] = {0};
    pd[0] = 1.3032f, pd[5] = 1.7377f, pd[10] = q, pd[11] = 1, pd[14] = r;
    od[0] = 2.0f / 1402, od[5] = -2.0f / 1052, od[10] = 1, od[12] = -1, od[13] = 1, od[15] = 1;
    CHECK(mtl_stereo_a_from_d3d(pd) == 1.3032f && mtl_stereo_a_from_d3d(od) == 0.0f,
          "stereo a from a D3D perspective matrix's _11; an orthographic one gives 0");

    /* Classing. */
    float vsf[256 * 4] = {0};
    memcpy(vsf + 4 * 8, xr, 16);
    memcpy(vsf + 4 * 11, wr, 16);
    memcpy(vsf + 4 * 7, wr, 16);
    float ca, seen;
    mtl_stereo_draw_in in = {.to_back_buffer = 1, .pos_regs = MTL_POS_VIEWPROJ, .vsf = vsf};
    int c1 = mtl_stereo_classify(&in, &ca, &seen);
    CHECK(c1 == MTL_STEREO_WORLD && fabsf(ca - g2.a) < 1e-5f && seen == ca, "stereo class: view projection draw is world");
    in.to_back_buffer = 0;
    CHECK(mtl_stereo_classify(&in, &ca, &seen) == MTL_STEREO_OFFSCREEN && ca == 0, "stereo class: offscreen left alone");
    in.to_back_buffer = 1;
    memcpy(vsf + 4 * 11, ortho_w, 16);
    CHECK(mtl_stereo_classify(&in, &ca, &seen) == MTL_STEREO_FLAT && ca == 0, "stereo class: orthographic c8-c11 is flat");
    in.pos_regs = MTL_POS_MVP;
    CHECK(mtl_stereo_classify(&in, &ca, &seen) == MTL_STEREO_NO_A, "stereo class: model view projection before any a");
    in.last_a = 1.25f;
    CHECK(mtl_stereo_classify(&in, &ca, &seen) == MTL_STEREO_WORLD && ca == 1.25f && seen == 0,
          "stereo class: model view projection borrows the frame's a");
    in.pos_regs = 0;
    CHECK(mtl_stereo_classify(&in, &ca, &seen) == MTL_STEREO_UNKNOWN && ca == 0, "stereo class: unrecognised shader");
    mtl_stereo_draw_in ff = {.to_back_buffer = 1, .ff_vs = 1, .ff_proj = pd};
    CHECK(mtl_stereo_classify(&ff, &ca, &seen) == MTL_STEREO_WORLD && ca == pd[0], "stereo class: FF perspective is world");
    ff.ff_proj = od;
    CHECK(mtl_stereo_classify(&ff, &ca, &seen) == MTL_STEREO_FLAT, "stereo class: FF orthographic is flat");
    ff.ff_proj = pd;
    ff.positiont = 1;
    CHECK(mtl_stereo_classify(&ff, &ca, &seen) == MTL_STEREO_PRETRANSFORMED, "stereo class: POSITIONT left alone");

    /* The scan, on hand-made vs_1_1 bytecode. */
    uint32_t b1[64], n = 0;
    b1[n++] = 0xFFFE0101u;
    b1[n++] = 0xFFFEu | 2u << 16, b1[n++] = 0x12345678u, b1[n++] = 0x9u; /* a comment holding a dp4 lookalike */
    for (int k = 0; k < 4; k++) b1[n++] = 9, b1[n++] = dst_tok(4, 0, 1u << k), b1[n++] = src_tok(1, 0), b1[n++] = src_tok(2, 8 + k);
    b1[n++] = 0xFFFFu;
    uint32_t b2[16], n2 = 0;
    b2[n2++] = 0xFFFE0101u;
    b2[n2++] = 20, b2[n2++] = dst_tok(4, 0, 0xF), b2[n2++] = src_tok(1, 0), b2[n2++] = src_tok(2, 4);
    b2[n2++] = 0xFFFFu;
    uint32_t b3[64], n3 = 0;
    b3[n3++] = 0xFFFE0200u;
    b3[n3++] = 0x51u, b3[n3++] = dst_tok(2, 90, 0xF), b3[n3++] = 0x3F800000u, b3[n3++] = 0, b3[n3++] = 0, b3[n3++] = 0;
    for (int k = 0; k < 4; k++) b3[n3++] = 9, b3[n3++] = dst_tok(0, 2, 1u << k), b3[n3++] = src_tok(0, 0), b3[n3++] = src_tok(2, 8 + k);
    b3[n3++] = 1, b3[n3++] = dst_tok(4, 0, 0xF), b3[n3++] = src_tok(0, 2);
    b3[n3++] = 0xFFFFu;
    uint32_t b4[8], n4 = 0;
    b4[n4++] = 0xFFFE0101u;
    b4[n4++] = 1, b4[n4++] = dst_tok(4, 0, 0xF), b4[n4++] = src_tok(1, 0);
    b4[n4++] = 0xFFFFu;
    int s1 = mtl_stereo_scan_pos(b1, n), s2 = mtl_stereo_scan_pos(b2, n2), s3 = mtl_stereo_scan_pos(b3, n3),
        s4 = mtl_stereo_scan_pos(b4, n4);
    CHECK(s1 == 8 && s2 == 4 && s3 == 8 && s4 == 0,
          "stereo scan: dp4 oPos with c8-c11 %d, m4x4 oPos c4 %d, dp4 into r2 then mov oPos %d, passthrough %d (want "
          "8 4 8 0)",
          s1, s2, s3, s4);
}

/* A headset view's projection as computeProjection gives it: right-up-
   back, reverse Z with an infinite far plane, from its four tangents. */
static void eye_projection(float l, float r, float d, float u, float near_m, float p[16])
{
    memset(p, 0, 16 * sizeof(float));
    p[0] = 2.0f / (l + r);
    p[8] = (r - l) / (l + r);
    p[5] = 2.0f / (u + d);
    p[9] = (u - d) / (u + d);
    p[11] = -1.0f;
    p[14] = near_m;
}

/* The tangent of the direction a view shows a clip position in. */
static void view_dir(const float p[16], const float c[4], float t[2])
{
    t[0] = (c[0] / c[3] + p[8]) / p[0];
    t[1] = (c[1] / c[3] + p[9]) / p[5];
}

/* A point of an eye image, at its NDC, through the eye plane and the
   eye's own projection, to the direction that eye sees it in. */
static void plane_dir(const float mvp[16], const float p[16], float nx, float ny, float t[2])
{
    float corner[4] = {0.5f * nx, 0.5f * ny, 0, 1}, c[4];
    mtl_screen_transform(mvp, corner, c);
    view_dir(p, c, t);
}

/* The eye-identity test against real asymmetric projections. The views'
   tangents are the ones measured on this headset in another project's
   log (1.7320 outward, 1.0000 towards the nose, mirrored between eyes);
   the vertical ones are a sample, as no device log of this one has them
   yet. The chain is the one that runs: a game clip position, the eye's
   correction (stereo.c), the eye plane (immersive.h), the eye's
   projection; then the direction each eye sees, and where the two rays
   meet. */
static void stereo_plane_math(void)
{
    const float half_ipd = 0.0314f, upm = 1.0f / 0.0254f, h = half_ipd * upm;
    const float a = 1.30322540f, b = a * 1402.0f / 1051.0f, tx = 1.0f / a, ty = 1.0f / b;
    float pl[16], pr[16], vl[16], vr[16], ml[16], mr[16];
    eye_projection(1.7320f, 1.0000f, 1.2500f, 1.0000f, 0.1f, pl);
    eye_projection(1.0000f, 1.7320f, 1.2500f, 1.0000f, 0.1f, pr);
    mtl_screen_identity(vl);
    mtl_screen_identity(vr);
    vl[12] = -half_ipd;
    vr[12] = half_ipd;
    float tg[4];
    mtl_stereo_tangents(pl, tg);
    CHECK(fabsf(tg[0] - 1.732f) < 1e-5f && fabsf(tg[1] - 1.0f) < 1e-5f && fabsf(tg[2] - 1.25f) < 1e-5f &&
              fabsf(tg[3] - 1.0f) < 1e-5f,
          "eye tangents back from a projection: left %.4f right %.4f down %.4f up %.4f", (double)tg[0], (double)tg[1],
          (double)tg[2], (double)tg[3]);

    /* What stage A did: the image over the whole view. Straight ahead in
       the game's image, NDC 0, is a different direction in each eye. */
    float c0[4] = {0, 0, 0.5f, 1}, sl[2], sr[2];
    view_dir(pl, c0, sl);
    view_dir(pr, c0, sr);
    printf("note: stage A's full-view image puts the game's centre at %.3f and %.3f rad across: %.1f degrees apart, "
           "diverging, which no eye fuses\n",
           (double)atanf(sl[0]), (double)atanf(sr[0]), (double)((atanf(sl[0]) - atanf(sr[0])) * 57.29578f));

    const float dist = METAL_D3D9_STEREO_HUD_M;
    mtl_stereo_plane_mvp(pl, vl, tx, ty, dist, ml);
    mtl_stereo_plane_mvp(pr, vr, tx, ty, dist, mr);
    float worst = 0;
    for (int i = 0; i <= 20; i++)
        for (int j = 0; j <= 20; j++) {
            float nx = -1.0f + 0.1f * (float)i, ny = -1.0f + 0.1f * (float)j, t[2];
            for (int e = 0; e < 2; e++) {
                plane_dir(e ? mr : ml, e ? pr : pl, nx, ny, t);
                worst = fmaxf(worst, fmaxf(fabsf(t[0] - nx * tx), fabsf(t[1] - ny * ty)));
            }
        }
    CHECK(worst < 1e-5f,
          "eye plane: every image pixel (21 x 21 per eye) lies in the direction the game drew it for, through each "
          "eye's own off-axis projection (worst tangent error %.2g)",
          (double)worst);

    /* World points through the whole chain. */
    const float q = 4096.0f / (4096.0f - 4.0f), r = -4.0f * q;
    mtl_stereo_proj game = {a, 0, b, 0};
    mtl_stereo_fix fl, fr;
    mtl_stereo_correction(&game, &game, -h, &fl);
    mtl_stereo_correction(&game, &game, h, &fr);
    const float zs_m[] = {0.5f, 1.0f, 2.0f, 5.0f, 20.0f, 100.0f, 1e5f};
    float prev = 1e9f;
    int shrinks = 1, meets = 1;
    for (size_t k = 0; k < sizeof zs_m / sizeof zs_m[0]; k++) {
        float z = zs_m[k] * upm, v[3] = {0.3f * z, -0.1f * z, z}, c[4], cl[4], cr[4], il[2], ir[2], tl[2], trr[2];
        persp_clip(&game, q, r, v, c);
        apply_fix(&fl, c, cl);
        apply_fix(&fr, c, cr);
        il[0] = cl[0] / cl[3], il[1] = cl[1] / cl[3], ir[0] = cr[0] / cr[3], ir[1] = cr[1] / cr[3];
        plane_dir(ml, pl, il[0], il[1], tl);
        plane_dir(mr, pr, ir[0], ir[1], trr);
        float disp = tl[0] - trr[0], want = 2.0f * half_ipd / zs_m[k];
        /* The rays from x = -h and +h with these tangents meet at depth
           2 h / (tl - tr), in metres. */
        float met = disp > 0 ? 2.0f * half_ipd / disp : INFINITY;
        printf("note: world point %.1f m ahead: angular disparity %.6f rad (want 2 h / z = %.6f), rays meet at %.3f m, "
               "vertical %.2g\n",
               (double)zs_m[k], (double)disp, (double)want, (double)met, (double)(tl[1] - trr[1]));
        shrinks &= disp < prev && disp > 0;
        meets &= fabsf(disp - want) < 2e-5f && fabsf(tl[1] - trr[1]) < 1e-5f;
        prev = disp;
    }
    CHECK(shrinks && meets && prev < 1e-6f,
          "eye plane: world disparity is 2 h / z at 0.5 m to 100 km, shrinking with depth to %.2g rad at the far end, "
          "with no vertical disparity",
          (double)prev);
    {
        /* At infinity: the correction's shift, a e / z in NDC, has gone
           to zero, so both eye images hold the point at the same NDC. */
        float c[4] = {0.37f, -0.21f, 1.0f, 1.0f}, cl[4], cr[4], tl[2], trr[2];
        mtl_stereo_fix il, ir;
        mtl_stereo_identity(&il);
        mtl_stereo_identity(&ir);
        apply_fix(&il, c, cl);
        apply_fix(&ir, c, cr);
        plane_dir(ml, pl, cl[0], cl[1], tl);
        plane_dir(mr, pr, cr[0], cr[1], trr);
        CHECK(fabsf(tl[0] - trr[0]) < 1e-6f && fabsf(tl[1] - trr[1]) < 1e-6f,
              "eye plane: a point at infinity (same NDC in both images) is the same direction in both eyes (%.3g, "
              "%.3g apart)",
              (double)(tl[0] - trr[0]), (double)(tl[1] - trr[1]));
    }

    /* The HUD: a flat draw at NDC (-0.7, 0.8), w 1, through the HUD
       correction; the rays must meet at the HUD plane. */
    float hz = METAL_D3D9_STEREO_HUD_M * upm;
    mtl_stereo_fix hl, hr;
    mtl_stereo_hud_correction(a, -h, hz, &hl);
    mtl_stereo_hud_correction(a, h, hz, &hr);
    float worst_m = 0;
    for (int k = 0; k < 2; k++) {
        float w = k ? 0.25f : 1.0f, c[4] = {-0.7f * w, 0.8f * w, 0.5f * w, w}, cl[4], cr[4], tl[2], trr[2];
        apply_fix(&hl, c, cl);
        apply_fix(&hr, c, cr);
        plane_dir(ml, pl, cl[0] / cl[3], cl[1] / cl[3], tl);
        plane_dir(mr, pr, cr[0] / cr[3], cr[1] / cr[3], trr);
        float met = 2.0f * half_ipd / (tl[0] - trr[0]);
        worst_m = fmaxf(worst_m, fabsf(met - METAL_D3D9_STEREO_HUD_M));
        if (!k)
            printf("note: HUD point: angular disparity %.6f rad (want %.6f), rays meet at %.4f m\n",
                   (double)(tl[0] - trr[0]), (double)(2.0f * half_ipd / METAL_D3D9_STEREO_HUD_M), (double)met);
    }
    CHECK(worst_m < 1e-3f, "eye plane: the HUD's rays meet at the %.2f m HUD plane for w 1 and w 0.25 (worst %.2g m)",
          (double)METAL_D3D9_STEREO_HUD_M, (double)worst_m);

    /* The frame's main a: a view model's a does not win over the world. */
    mtl_stereo_tally t;
    memset(&t, 0, sizeof t);
    for (int i = 0; i < 40; i++) mtl_stereo_tally_add(&t, a);
    for (int i = 0; i < 9; i++) mtl_stereo_tally_add(&t, 2.1435f);
    mtl_stereo_tally_add(&t, 0.0f);
    float got = mtl_stereo_tally_take(&t, 9.0f), empty = mtl_stereo_tally_take(&t, 9.0f);
    CHECK(got == a && empty == 9.0f, "stereo tally: the world's a %.4f over a view model's 2.1435; an empty frame keeps "
                                     "the last (%.1f)",
          (double)got, (double)empty);
    /* The session's plane: the game's hor+ 75 at 4:3 is the default a; at
       the device's 1402x1051 it is the a play logged there (tan 0.76769);
       a set size is degrees across 4:3 too, and nonsense falls back. */
    float s43 = mtl_stereo_screen_a(4.0f / 3.0f, 0.0f), sdev = mtl_stereo_screen_a(1402.0f / 1051.0f, 0.0f);
    float s90 = mtl_stereo_screen_a(4.0f / 3.0f, 90.0f), sbad = mtl_stereo_screen_a(4.0f / 3.0f, 400.0f);
    CHECK(fabsf(s43 - MTL_STEREO_DEFAULT_A) < 1e-5f && fabsf(1.0f / sdev - 0.76769197f) < 1e-5f &&
              fabsf(s90 - 1.0f) < 1e-5f && sbad == s43,
          "stereo screen a: 4:3 %.5f (want %.5f), 1402x1051 tan %.5f (device play 0.76769), 90 degrees %.5f, out of "
          "range %.5f",
          (double)s43, (double)MTL_STEREO_DEFAULT_A, (double)(1.0f / sdev), (double)s90, (double)sbad);
    mtl_stereo_fix id;
    mtl_stereo_hud_correction(a, 0.0f, hz, &id);
    CHECK(id.x[0] == 1.0f && id.x[1] == 0.0f && id.x[2] == 0.0f, "stereo HUD correction with no eye offset is identity");
    mtl_stereo_draw_in in = {.to_back_buffer = 1, .positiont = 1, .samples_rt = 1};
    float ca, seen;
    int c1 = mtl_stereo_classify(&in, &ca, &seen);
    in.samples_rt = 0;
    int c2 = mtl_stereo_classify(&in, &ca, &seen);
    CHECK(c1 == MTL_STEREO_SCREEN_PASS && !mtl_stereo_class_is_hud(c1) && c2 == MTL_STEREO_PRETRANSFORMED &&
              mtl_stereo_class_is_hud(c2),
          "stereo class: a POSITIONT pass over a render target stays with the eye's image, one over a texture is HUD");
}

/* Stage C: the eye plane turned by the late rotation. A head pose with
   roll as well, which the game never draws. */
static void head_pose_rolled(float m[16], float yaw_deg, float pitch_deg, float roll_deg)
{
    float yp[16], roll[16], r = roll_deg * (float)M_PI / 180.0f;
    head_pose(yp, yaw_deg, pitch_deg, 0, 0, 0);
    mtl_screen_identity(roll);
    roll[0] = cosf(r);
    roll[1] = sinf(r);
    roll[4] = -sinf(r);
    roll[5] = cosf(r);
    mtl_screen_mul(yp, roll, m);
    m[12] = 0.3f;
    m[13] = 1.6f;
    m[14] = -0.2f;
}

static void normalise(float v[3])
{
    float n = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    for (int i = 0; i < 3; i++) v[i] /= n;
}

/* The worst angle, in degrees, between where an image pixel lies in the
   world through the turned plane and where the frame drew it, over a
   21 x 21 grid of the image: the eye at device position eye, the head now
   at `now`, the frame drawn at yaw and pitch. */
static float late_worst(const float now[16], float yaw, float pitch, const float eye[3], float tx, float ty, float dist)
{
    float turn[16], view[16], model[16], drawn[16];
    mtl_late_rotation(now, yaw, pitch, turn);
    translate(view, eye[0], eye[1], eye[2]);
    mtl_stereo_plane_model_turned(view, turn, tx, ty, dist, model);
    mtl_view_rotation(yaw, pitch, drawn);
    float worst = 0;
    for (int i = 0; i <= 20; i++)
        for (int j = 0; j <= 20; j++) {
            float nx = -1.0f + 0.1f * (float)i, ny = -1.0f + 0.1f * (float)j;
            float corner[4] = {0.5f * nx, 0.5f * ny, 0, 1}, pd[4];
            mtl_screen_transform(model, corner, pd);
            float d[4] = {pd[0] - eye[0], pd[1] - eye[1], pd[2] - eye[2], 0}, world[4];
            mtl_screen_transform(now, d, world);
            float want_local[4] = {nx * tx, ny * ty, -1, 0}, want[4];
            mtl_screen_transform(drawn, want_local, want);
            normalise(world);
            normalise(want);
            /* atan2 of the cross and dot products: acos near 1 loses
               the small angles this checks to float rounding. */
            float c = world[0] * want[0] + world[1] * want[1] + world[2] * want[2];
            float x0 = world[1] * want[2] - world[2] * want[1], x1 = world[2] * want[0] - world[0] * want[2],
                  x2 = world[0] * want[1] - world[1] * want[0];
            float a = atan2f(sqrtf(x0 * x0 + x1 * x1 + x2 * x2), c) * 57.29578f;
            if (a > worst) worst = a;
        }
    return worst;
}

static void late_rotation_math(void)
{
    const float a = 1.30322540f, b = a * 1402.0f / 1051.0f, tx = 1.0f / a, ty = 1.0f / b;
    const float dist = METAL_D3D9_STEREO_HUD_M, eye[3] = {-0.0314f, 0.0f, 0.0f};

    /* The head's angles back from a pose, the host's half of stage B. */
    static const float k_yp[][2] = {{0, 0}, {30, 10}, {-120, -45}, {179, 80}, {-90, 89.9f}, {45, -89.9f}};
    float worst_yp = 0;
    for (size_t i = 0; i < sizeof k_yp / sizeof *k_yp; i++) {
        float m[16], y, p;
        head_pose(m, k_yp[i][0], k_yp[i][1], 0, 1.6f, 0);
        mtl_head_yaw_pitch(m, &y, &p);
        float dy = fabsf(remainderf(y - k_yp[i][0], 360.0f)), dp = fabsf(p - k_yp[i][1]);
        worst_yp = fmaxf(worst_yp, fmaxf(dy * cosf(k_yp[i][1] * (float)M_PI / 180.0f), dp));
    }
    CHECK(worst_yp < 0.01f, "late rotation: a head's yaw and pitch back from its pose, to 89.9 up and down (worst %.3g deg)",
          (double)worst_yp);
    float up[16], y, p;
    head_pose(up, 60, 90, 0, 0, 0);
    mtl_head_yaw_pitch(up, &y, &p);
    CHECK(fabsf(y - 60) < 0.01f && fabsf(p - 90) < 0.01f,
          "late rotation: straight up keeps its heading from the up axis (yaw %.3f, pitch %.3f)", (double)y, (double)p);

    /* Drawn and shown at the same pose: no turn, and the plane is the
       head-locked one bit for bit. */
    float now[16], turn[16], view[16], m0[16], m1[16];
    head_pose(now, 30, 10, 0.3f, 1.6f, -0.2f);
    mtl_late_rotation(now, 30, 10, turn);
    float ang = mtl_rotation_angle_deg(turn);
    translate(view, eye[0], eye[1], eye[2]);
    mtl_stereo_plane_model(view, tx, ty, dist, m0);
    float id[16];
    mtl_screen_identity(id);
    mtl_stereo_plane_model_turned(view, id, tx, ty, dist, m1);
    CHECK(ang < 0.05f && !memcmp(m0, m1, sizeof m0),
          "late rotation: none when the head has not moved (%.3g deg), and the identity turn is the head-locked plane",
          (double)ang);

    /* The head turned on since the frame was drawn: every pixel stays in
       the world direction it was drawn for. */
    static const float k_case[][5] = {
        /* now yaw, pitch, roll; drawn yaw, pitch */
        {10, 0, 0, 0, 0},       {-25, 5, 0, -20, 3}, {91, -30, 0, 85, -28}, {0, 0, 15, 0, 0},
        {40, 20, -10, 36, 18},  {170, 0, 0, -175, 0},
    };
    float worst = 0, worst_deg_err = 0;
    for (size_t i = 0; i < sizeof k_case / sizeof *k_case; i++) {
        head_pose_rolled(now, k_case[i][0], k_case[i][1], k_case[i][2]);
        worst = fmaxf(worst, late_worst(now, k_case[i][3], k_case[i][4], eye, tx, ty, dist));
    }
    head_pose_rolled(now, 10, 0, 0);
    mtl_late_rotation(now, 0, 0, turn);
    worst_deg_err = fabsf(mtl_rotation_angle_deg(turn) - 10.0f);
    CHECK(worst < 0.01f,
          "late rotation: under a known head turn, roll included, every image pixel (21 x 21) stays in the world "
          "direction the frame was drawn for (worst %.3g deg)",
          (double)worst);
    CHECK(worst_deg_err < 0.01f, "late rotation: a 10 degree head turn is logged as 10 degrees (off by %.3g)",
          (double)worst_deg_err);

    /* What stage A did, for scale: the head-locked plane under the same
       10 degree turn puts the world 10 degrees off. */
    head_pose_rolled(now, 10, 0, 0);
    float locked[16], world_centre[4], c4[4] = {0, 0, 0, 1};
    mtl_stereo_plane_model(view, tx, ty, dist, locked);
    float pd[4];
    mtl_screen_transform(locked, c4, pd);
    float d[4] = {pd[0] - eye[0], pd[1] - eye[1], pd[2] - eye[2], 0};
    mtl_screen_transform(now, d, world_centre);
    float drag = atan2f(-world_centre[0], -world_centre[2]) * 57.29578f;
    CHECK(fabsf(drag - 10.0f) < 0.01f, "late rotation: without it the head-locked plane drags the world %.2f deg with a "
          "10 degree turn", (double)drag);
}

/* The fixed screen: the eye planes stand in the room. At the head that
   placed them they are the head-locked planes; under any head after, a
   corner of each is the same point of the room. */
static void fixed_screen_math(void)
{
    const float a = 1.30322540f, b = a * 1402.0f / 1051.0f, tx = 1.0f / a, ty = 1.0f / b;
    const float dist = METAL_D3D9_STEREO_HUD_M;
    static const float k_eye[2] = {-0.0314f, 0.0314f};
    float placed[16], anchor[16];
    head_pose(placed, 25, 0, 0.3f, 1.6f, -0.2f);
    mtl_stereo_anchor(placed, anchor);
    float worst_at = 0, worst_move = 0;
    for (int e = 0; e < 2; e++) {
        float view[16], locked[16], fixed[16], dfo[16], dfa[16];
        translate(view, k_eye[e], 0, 0);
        mtl_stereo_plane_model(view, tx, ty, dist, locked);
        mtl_screen_rigid_inverse(placed, dfo);
        mtl_screen_mul(dfo, anchor, dfa);
        mtl_stereo_plane_model_fixed(view, dfa, tx, ty, dist, fixed);
        for (int i = 0; i < 16; i++) worst_at = fmaxf(worst_at, fabsf(fixed[i] - locked[i]));
        static const float k_corner[4][2] = {{-0.5f, -0.5f}, {0.5f, -0.5f}, {0.5f, 0.5f}, {-0.5f, 0.5f}};
        static const float k_head[][6] = {{25, 0, 0, 0.3f, 1.6f, -0.2f}, {70, 20, 10, 0.3f, 1.6f, -0.2f},
                                          {-40, -35, 0, 0.5f, 1.4f, 0.1f}, {150, 60, -20, -0.2f, 1.8f, -0.6f}};
        float ref[4][3];
        for (size_t h = 0; h < sizeof k_head / sizeof *k_head; h++) {
            float now[16], rolled[16], model[16], dfo2[16], dfa2[16];
            head_pose_rolled(rolled, k_head[h][0], k_head[h][1], k_head[h][2]);
            memcpy(now, rolled, sizeof now);
            now[12] = k_head[h][3];
            now[13] = k_head[h][4];
            now[14] = k_head[h][5];
            mtl_screen_rigid_inverse(now, dfo2);
            mtl_screen_mul(dfo2, anchor, dfa2);
            mtl_stereo_plane_model_fixed(view, dfa2, tx, ty, dist, model);
            for (int c = 0; c < 4; c++) {
                float q[4] = {k_corner[c][0], k_corner[c][1], 0, 1}, d[4], w[4];
                mtl_screen_transform(model, q, d);
                mtl_screen_transform(now, d, w);
                if (!h) memcpy(ref[c], w, sizeof ref[c]);
                else
                    for (int r = 0; r < 3; r++) worst_move = fmaxf(worst_move, fabsf(w[r] - ref[c][r]));
            }
        }
    }
    CHECK(worst_at < 1e-5f, "fixed screen: at the head that placed it each eye plane is the head-locked one (worst %.3g)",
          (double)worst_at);
    CHECK(worst_move < 1e-4f,
          "fixed screen: turned, rolled and moved heads leave every corner of both eye planes where it stood (worst %.3g m)",
          (double)worst_move);
    float tilted[16], level[16];
    head_pose(tilted, -60, 40, 1, 1.5f, 2);
    mtl_stereo_anchor(tilted, level);
    float yaw = atan2f(level[8], level[10]) * 57.29578f;
    CHECK(fabsf(level[9]) < 1e-6f && fabsf(level[5] - 1) < 1e-6f && fabsf(yaw + 60) < 0.01f && level[12] == tilted[12] &&
              level[13] == tilted[13] && level[14] == tilted[14],
          "fixed screen: the anchor keeps the head's heading (%.2f) and position, upright", (double)yaw);
}

/* Full view (stereo.h's mtl_stereo_correction3, immersive.h's
   mtl_fullview_plane_model). The fix with all four rows, its identity
   bit for bit, a game clip position re-projected to an off-axis eye
   moved in x, y and z against that eye's own view of the point, the
   projection's terms recovered from rows and from a D3D matrix, the HUD
   panel's directions, and the whole chain: a world point through the
   game's wide projection, an eye's fix, the eye's plane and the view's
   own projection, to the direction that eye really sees it in. */
static void apply_fix4(const mtl_stereo_fix *f, const float c[4], float o[4])
{
    o[0] = f->x[0] * c[0] + f->x[1] * c[3] + f->x[2];
    o[1] = f->y[0] * c[1] + f->y[1] * c[3] + f->y[2];
    o[2] = f->z[0] * c[2] + f->z[1] * c[3] + f->z[2];
    o[3] = f->w[0] * c[3] + f->w[1] * c[3] + f->w[2];
}

static void full_clip(const mtl_stereo_proj_full *p, float r, const float v[3], float c[4])
{
    c[0] = p->a * v[0] + p->cx * v[2];
    c[1] = p->b * v[1] + p->cy * v[2];
    c[2] = p->A * v[2] + r;
    c[3] = v[2];
}

static void fullview_math(void)
{
    const float q = 4096.0f / (4096.0f - 4.0f), r = -4.0f * q;
    mtl_stereo_proj_full game = {1.3032f, 0.0f, 1.7377f, 0.0f, q};
    mtl_stereo_proj game_xy = {game.a, game.cx, game.b, game.cy};
    const float zero[3] = {0, 0, 0};
    mtl_stereo_fix fix;

    mtl_stereo_correction3(&game, &game_xy, zero, &fix);
    int same = 1;
    for (int i = 0; i < 200; i++) {
        float v[3] = {(float)(i % 17) * 7.3f - 60.0f, (float)(i % 11) * 5.1f - 30.0f, 4.0f + (float)i * 9.7f};
        float c[4], o[4];
        full_clip(&game, r, v, c);
        apply_fix4(&fix, c, o);
        same &= !memcmp(c, o, sizeof c);
    }
    CHECK(same && fix.z[0] == 1.0f && fix.w[0] == 1.0f,
          "full view identity: P_eye = P_game, t = 0, 200 vertices bit for bit in x, y, z and w");

    /* An off-axis eye, moved in x, y and z, against its own view. */
    mtl_stereo_proj_full g2 = {1.21f, 0.031f, 1.62f, -0.017f, q};
    const float tan_eye[4] = {1.836f, 1.06f, 1.261f, 1.06f};
    mtl_stereo_proj eye;
    mtl_stereo_proj_from_tangents(tan_eye, &eye);
    const float t[3] = {1.26f, -0.4f, 0.9f};
    mtl_stereo_correction3(&g2, &eye, t, &fix);
    float worst = 0;
    for (int i = 0; i < 200; i++) {
        float v[3] = {(float)(i % 13) * 9.1f - 55.0f, (float)(i % 7) * 6.3f - 20.0f, 6.0f + (float)i * 13.1f};
        float c[4], o[4], ve[3] = {v[0] - t[0], v[1] - t[1], v[2] - t[2]}, want[4];
        mtl_stereo_proj_full ef = {eye.a, eye.cx, eye.b, eye.cy, g2.A};
        full_clip(&g2, r, v, c);
        apply_fix4(&fix, c, o);
        full_clip(&ef, r, ve, want);
        for (int k = 0; k < 4; k++) worst = fmaxf(worst, fabsf(o[k] - want[k]) / fmaxf(1.0f, fabsf(want[k])));
    }
    CHECK(worst < 1e-5f,
          "full view correction to an off-axis eye moved in x, y and z matches that eye's own view, depth included "
          "(worst %.2g)",
          (double)worst);

    /* The eye's terms put its view's edges on NDC -1 and 1. */
    float el = eye.a * -tan_eye[0] + eye.cx, er = eye.a * tan_eye[1] + eye.cx, ed = eye.b * -tan_eye[2] + eye.cy,
          eu = eye.b * tan_eye[3] + eye.cy;
    CHECK(fabsf(el + 1) < 1e-6f && fabsf(er - 1) < 1e-6f && fabsf(ed + 1) < 1e-6f && fabsf(eu - 1) < 1e-6f,
          "full view: an eye's terms from its tangents put its view's edges at NDC -1 and 1 (%.6f %.6f %.6f %.6f)",
          (double)el, (double)er, (double)ed, (double)eu);

    /* The terms from a turned camera's view projection rows, scaled. */
    float yaw = -0.4f, pitch = 0.25f, sc = 2.5f;
    float right[3] = {cosf(yaw), 0, -sinf(yaw)};
    float fwd[3] = {sinf(yaw) * cosf(pitch), sinf(pitch), cosf(yaw) * cosf(pitch)};
    float up[3] = {fwd[1] * right[2] - fwd[2] * right[1], fwd[2] * right[0] - fwd[0] * right[2],
                   fwd[0] * right[1] - fwd[1] * right[0]};
    float pos[3] = {-80, 25, 140};
    float rows[4][4];
    for (int k = 0; k < 3; k++) {
        rows[0][k] = sc * (g2.a * right[k] + g2.cx * fwd[k]);
        rows[1][k] = sc * (g2.b * up[k] + g2.cy * fwd[k]);
        rows[2][k] = sc * g2.A * fwd[k];
        rows[3][k] = sc * fwd[k];
    }
    for (int j = 0; j < 4; j++) rows[j][3] = -(rows[j][0] * pos[0] + rows[j][1] * pos[1] + rows[j][2] * pos[2]);
    rows[2][3] += sc * r;
    mtl_stereo_proj_full got;
    int ok = mtl_stereo_proj_from_rows(rows[0], rows[1], rows[2], rows[3], &got);
    CHECK(ok && fabsf(got.a - g2.a) < 1e-5f && fabsf(got.cx - g2.cx) < 1e-5f && fabsf(got.b - g2.b) < 1e-5f &&
              fabsf(got.cy - g2.cy) < 1e-5f && fabsf(got.A - g2.A) < 1e-5f,
          "full view terms from a turned camera's view projection rows scaled by 2.5: a %.5f cx %.5f b %.5f cy %.5f "
          "A %.5f (want %.5f %.5f %.5f %.5f %.5f)",
          (double)got.a, (double)got.cx, (double)got.b, (double)got.cy, (double)got.A, (double)g2.a, (double)g2.cx,
          (double)g2.b, (double)g2.cy, (double)g2.A);
    float pd[16] = {0};
    pd[0] = g2.a, pd[5] = g2.b, pd[8] = g2.cx, pd[9] = g2.cy, pd[10] = q, pd[11] = 1, pd[14] = r;
    mtl_stereo_proj_full gd;
    ok = mtl_stereo_proj_from_d3d(pd, &gd);
    float od[16] = {0};
    od[0] = 2.0f / 1402, od[5] = -2.0f / 1052, od[10] = 1, od[12] = -1, od[13] = 1, od[15] = 1;
    mtl_stereo_proj_full go;
    CHECK(ok && gd.a == g2.a && gd.cx == g2.cx && gd.b == g2.b && gd.cy == g2.cy && gd.A == q &&
              !mtl_stereo_proj_from_d3d(od, &go) && go.a == 0,
          "full view terms from a D3D perspective matrix; an orthographic one gives none");

    /* The classing with terms agrees with stage A's, and a model view
       projection borrows the frame's whole projection. */
    float vsf[256 * 4] = {0};
    for (int j = 0; j < 4; j++) memcpy(vsf + 4 * (8 + j), rows[j], 16);
    memcpy(vsf + 4 * 7, rows[3], 16);
    mtl_stereo_draw_in in = {.to_back_buffer = 1, .pos_regs = MTL_POS_VIEWPROJ, .vsf = vsf};
    mtl_stereo_proj_full pp, seen;
    float ca, sa, pa, psa;
    int c1 = mtl_stereo_classify_proj(&in, &pa, &psa, &pp, &seen), c2 = mtl_stereo_classify(&in, &ca, &sa);
    in.pos_regs = MTL_POS_MVP;
    in.last_a = ca;
    in.last_proj = &pp;
    mtl_stereo_proj_full pm, sm;
    float ma, msa;
    int c3 = mtl_stereo_classify_proj(&in, &ma, &msa, &pm, &sm);
    CHECK(c1 == MTL_STEREO_WORLD && c2 == c1 && !memcmp(&pa, &ca, 4) && !memcmp(&psa, &sa, 4) &&
              fabsf(pp.a - g2.a) < 1e-5f && !memcmp(&seen, &pp, sizeof pp) && c3 == MTL_STEREO_WORLD && ma == ca &&
              msa == 0 && !memcmp(&pm, &pp, sizeof pm) && sm.a == 0,
          "full view class: stage A's a to the bit beside the whole terms; a model view projection borrows both");

    /* The HUD panel: game NDC (x, y) at (px x, py y) hud_z from the head,
       seen from each eye where that eye really sees that point. */
    const float hud_z = METAL_D3D9_STEREO_HUD_M / 0.0254f, px = 0.6f, py = 0.6f / 1.4545f;
    float worst_hud = 0;
    for (int e = 0; e < 2; e++) {
        float te[3] = {e ? 1.22f : -1.22f, 0.05f, 0.0f};
        const float tl[4] = {e ? 1.0f : 1.732f, e ? 1.732f : 1.0f, 1.19f, 1.0f};
        mtl_stereo_proj ep;
        mtl_stereo_proj_from_tangents(tl, &ep);
        mtl_stereo_panel_correction(&ep, te, hud_z, px, py, &fix);
        for (int i = 0; i < 25; i++) {
            float nx = -1.0f + 0.5f * (float)(i % 5), ny = -1.0f + 0.5f * (float)(i / 5), w = 1.0f + (float)i;
            float c[4] = {nx * w, ny * w, 0.5f * w, w}, o[4];
            apply_fix4(&fix, c, o);
            float want_x = (px * nx * hud_z - te[0]) / hud_z, want_y = (py * ny * hud_z - te[1]) / hud_z;
            float got_x = (o[0] / o[3] - ep.cx) / ep.a, got_y = (o[1] / o[3] - ep.cy) / ep.b;
            worst_hud = fmaxf(worst_hud, fmaxf(fabsf(got_x - want_x), fabsf(got_y - want_y)));
        }
    }
    CHECK(worst_hud < 1e-5f, "full view HUD: the panel's points where each eye sees them, any w (worst %.2g tangent)",
          (double)worst_hud);

    /* The eye plane: with no turn, a view with the device's axes and its
       own tangents, the corners are NDC -1 and 1. */
    const float near_m = 0.1f, dist = METAL_D3D9_STEREO_HUD_M;
    float worst_corner = 0;
    for (int e = 0; e < 2; e++) {
        const float tl[4] = {e ? 1.0f : 1.732f, e ? 1.732f : 1.0f, 1.19f, 1.0f};
        float p[16], view[16], id[16], mvp[16];
        eye_projection(tl[0], tl[1], tl[2], tl[3], near_m, p);
        translate(view, e ? 0.031f : -0.031f, -0.025f, -0.033f);
        mtl_screen_identity(id);
        mtl_fullview_plane_mvp(p, view, id, tl, dist, mvp);
        static const float k_c[4][2] = {{-0.5f, -0.5f}, {0.5f, -0.5f}, {0.5f, 0.5f}, {-0.5f, 0.5f}};
        for (int c = 0; c < 4; c++) {
            float qd[4] = {k_c[c][0], k_c[c][1], 0, 1}, o[4];
            mtl_screen_transform(mvp, qd, o);
            worst_corner = fmaxf(worst_corner, fmaxf(fabsf(o[0] / o[3] - 2 * k_c[c][0]), fabsf(o[1] / o[3] - 2 * k_c[c][1])));
        }
    }
    CHECK(worst_corner < 1e-5f, "full view plane: each eye's image fills its view, corners at NDC -1 and 1 (worst %.2g)",
          (double)worst_corner);

    /* The chain. The game draws the union of both eyes' views with the
       margin, widened to the back buffer's shape; each eye replays it
       into its own view with the margin; a point of the world lands in
       the direction its eye really sees it, through that eye's plane and
       projection. Then a late rotation: the plane turned with the head
       keeps every pixel in the world direction it was drawn for. */
    const float m = METAL_D3D9_FULLVIEW_MARGIN, aspect = 2048.0f / 1408.0f, upm = 1.0f / 0.0254f;
    float lx = 1.732f * (1 + m), ly = 1.19f * (1 + m), gtx = fmaxf(lx, ly * aspect), gty = gtx / aspect;
    mtl_stereo_proj_full gw = {1.0f / gtx, 0, 1.0f / gty, 0, q};
    float worst_dir = 0, worst_turn = 0;
    for (int e = 0; e < 2; e++) {
        const float tl[4] = {e ? 1.0f : 1.732f, e ? 1.732f : 1.0f, 1.19f, 1.0f};
        float tm[4], p[16], view[16], id[16], mvp[16];
        for (int k = 0; k < 4; k++) tm[k] = tl[k] * (1 + m);
        mtl_stereo_proj ep;
        mtl_stereo_proj_from_tangents(tm, &ep);
        float ex = e ? 0.031f : -0.031f, te[3] = {ex * upm, 0, 0};
        mtl_stereo_correction3(&gw, &ep, te, &fix);
        eye_projection(tl[0], tl[1], tl[2], tl[3], near_m, p);
        translate(view, ex, 0, 0);
        mtl_screen_identity(id);
        mtl_fullview_plane_mvp(p, view, id, tm, dist, mvp);
        float turn[16], rot[16], model[16];
        mtl_view_rotation(12.0f, -5.0f, rot);
        memcpy(turn, rot, sizeof turn);
        mtl_fullview_plane_model(view, turn, tm, dist, model);
        for (int i = 0; i < 60; i++) {
            /* Points inside this eye's view, near and far. */
            float zv = 30.0f + (float)i * 37.0f;
            float txw = -0.9f * tl[0] + 1.8f * (tl[0] + tl[1]) * 0.5f * (float)(i % 6) / 5.0f;
            float tyw = -0.9f * tl[2] + 1.8f * (tl[2] + tl[3]) * 0.5f * (float)((i / 6) % 5) / 4.0f;
            float v[3] = {txw * zv + te[0], tyw * zv, zv}, c[4], o[4];
            full_clip(&gw, r, v, c);
            apply_fix4(&fix, c, o);
            float nx = o[0] / o[3], ny = o[1] / o[3], dir[2];
            plane_dir(mvp, p, nx, ny, dir);
            float want_x = (v[0] - te[0]) / v[2], want_y = (v[1] - te[1]) / v[2];
            worst_dir = fmaxf(worst_dir, fmaxf(fabsf(dir[0] - want_x), fabsf(dir[1] - want_y)));
            /* Turned: the plane's point, from the eye, is the drawn
               direction turned. */
            float corner[4] = {0.5f * nx, 0.5f * ny, 0, 1}, at[4];
            float want_v[4] = {want_x * dist, want_y * dist, -dist, 0}, tv[4];
            mtl_screen_transform(model, corner, at);
            mtl_screen_transform(turn, want_v, tv);
            for (int k = 0; k < 3; k++) worst_turn = fmaxf(worst_turn, fabsf(at[k] - view[12 + k] - tv[k]));
        }
    }
    CHECK(worst_dir < 2e-5f,
          "full view chain: 120 world points through the game's %.1f by %.1f degree field, each eye's fix, plane and "
          "own projection land where that eye sees them (worst %.2g tangent)",
          2.0 * atan(gtx) * 57.29578, 2.0 * atan(gty) * 57.29578, (double)worst_dir);
    CHECK(worst_turn < 1e-4f, "full view late rotation: the turned plane keeps each pixel in the drawn direction, turned "
          "(worst %.2g m)", (double)worst_turn);
}

/* Which camera a draw is seen from. The la_hub_1 case: the 3D sky drawn
   through c8 from its own camera, then the world's first draws through the
   model view projection in c4 with the engine's c8 already the player's
   view. Those must read the player's camera from c8, not keep the sky's. */
/* mtl_stereo_covers_view: a HUD draw over the whole view fills each eye
   in full view; anything smaller, or behind the camera, stays on the
   panel. */
static void overlay_cover_math(void)
{
    const float full[4][3] = {{-1, 1, 1}, {1, 1, 1}, {1, -1, 1}, {-1, -1, 1}};
    const float scaled[4][3] = {{-2.5f, 2.5f, 2.5f}, {2.5f, 2.5f, 2.5f}, {2.5f, -2.5f, 2.5f}, {-2.5f, -2.5f, 2.5f}};
    const float bar[4][3] = {{-1, -0.8f, 1}, {1, -0.8f, 1}, {1, -1, 1}, {-1, -1, 1}};
    const float wide[4][3] = {{-1.2f, 1.1f, 1}, {1.3f, 1.1f, 1}, {1.3f, -1.2f, 1}, {-1.2f, -1.2f, 1}};
    const float behind[4][3] = {{-1, 1, 1}, {1, 1, -1}, {1, -1, 1}, {-1, -1, 1}};
    CHECK(mtl_stereo_covers_view(full, 4) && mtl_stereo_covers_view(scaled, 4) && mtl_stereo_covers_view(wide, 4),
          "overlay cover: a quad at the view's edges, the same with w 2.5, and one past them all cover the view");
    CHECK(!mtl_stereo_covers_view(bar, 4) && !mtl_stereo_covers_view(behind, 4) && !mtl_stereo_covers_view(full, 2),
          "overlay cover: a bar along the bottom, a quad with a vertex behind the camera, and two vertices do not");
}

/* mtl_stereo_arm_side and its columns: the la_hub_1 gauges' boxes go to
   their cards, the banner text, the transient group, a box across a line
   and an unreadable box stay on the panel. */
static void arm_side_math(void)
{
    const float health[4] = {-0.97f, -0.80f, -0.95f, -0.35f}, portrait[4] = {-0.99f, -0.74f, -0.30f, 0.05f};
    const float blood[4] = {0.80f, 0.93f, -0.95f, -0.35f}, disc[4] = {0.77f, 0.99f, -0.30f, 0.10f};
    const float banner[4] = {-0.30f, 0.30f, 0.86f, 0.94f}, transient[4] = {0.53f, 0.82f, 0.60f, 1.0f};
    const float across_l[4] = {-0.95f, -0.69f, 0.0f, 0.2f}, across_r[4] = {0.75f, 0.95f, 0.0f, 0.2f};
    const float nan_box[4] = {NAN, -0.9f, 0.0f, 0.1f}, empty[4] = {INFINITY, -INFINITY, INFINITY, -INFINITY};
    CHECK(mtl_stereo_arm_side(health) == 1 && mtl_stereo_arm_side(portrait) == 1 && mtl_stereo_arm_side(blood) == 2 &&
              mtl_stereo_arm_side(disc) == 2,
          "arm side: health and the portrait go left, blood and the disc right");
    CHECK(!mtl_stereo_arm_side(banner) && !mtl_stereo_arm_side(transient) && !mtl_stereo_arm_side(across_l) &&
              !mtl_stereo_arm_side(across_r) && !mtl_stereo_arm_side(nan_box) && !mtl_stereo_arm_side(empty),
          "arm side: the banner, the transient group, boxes across a line and no box stay on the panel");
    uint32_t l0, l1, r0, r1, n0, n1;
    mtl_stereo_arm_columns(1, 2304, &l0, &l1);
    mtl_stereo_arm_columns(2, 2304, &r0, &r1);
    mtl_stereo_arm_columns(0, 2304, &n0, &n1);
    CHECK(l0 == 0 && l1 == 346 && r0 == 2027 && r1 == 2304 && n0 == n1,
          "arm columns at 2304: left 0..%u (346), right %u..%u (2027..2304), none empty", l1, r0, r1);
}

static void draw_camera_math(void)
{
    /* A view projection for a camera at p looking down +z: x, y and w rows
       of perspective (a, b) times a translation by -p. */
    float sky[3] = {1998.1f, 191.4f, 704.6f}, world[3] = {229.1f, -1076.9f, -7367.9f};
    float vsf[4 * 64];
    memset(vsf, 0, sizeof vsf);
#define SET_VP(p)                                                                                                      \
    do {                                                                                                               \
        float a_ = 0.5442f, b_ = 0.7916f;                                                                              \
        float *x_ = vsf + 4 * 8, *y_ = vsf + 4 * 9, *z_ = vsf + 4 * 10, *w_ = vsf + 4 * 11;                            \
        x_[0] = a_, x_[1] = 0, x_[2] = 0, x_[3] = -a_ * (p)[0];                                                        \
        y_[0] = 0, y_[1] = b_, y_[2] = 0, y_[3] = -b_ * (p)[1];                                                        \
        z_[0] = 0, z_[1] = 0, z_[2] = 1, z_[3] = -(p)[2] - 4;                                                          \
        w_[0] = 0, w_[1] = 0, w_[2] = 1, w_[3] = -(p)[2];                                                              \
    } while (0)
    float cam[3];
    SET_VP(sky);
    int ok_sky = mtl_stereo_draw_camera(0, vsf, NULL, cam);
    float e_sky = fmaxf(fabsf(cam[0] - sky[0]), fmaxf(fabsf(cam[1] - sky[1]), fabsf(cam[2] - sky[2])));
    /* The world's view set in c8, the draw itself placed through c4 (a
       prop's model view projection, a different matrix entirely). */
    SET_VP(world);
    for (int i = 0; i < 16; i++) vsf[4 * 4 + i] = (float)(i * 7 % 5) + 0.25f;
    int ok_mvp = mtl_stereo_draw_camera(0, vsf, NULL, cam);
    float e_mvp = fmaxf(fabsf(cam[0] - world[0]), fmaxf(fabsf(cam[1] - world[1]), fabsf(cam[2] - world[2])));
    /* No view projection in c8: no camera, so the scan keeps the last. */
    memset(vsf + 4 * 8, 0, 16 * sizeof(float));
    int ok_none = mtl_stereo_draw_camera(0, vsf, NULL, cam);
    /* Fixed function: from the view matrix (row vectors, camera at world). */
    float view[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -world[0], -world[1], -world[2], 1}, cff[3];
    int ok_ff = mtl_stereo_draw_camera(1, NULL, view, cff);
    float e_ff = fmaxf(fabsf(cff[0] - world[0]), fmaxf(fabsf(cff[1] - world[1]), fabsf(cff[2] - world[2])));
#undef SET_VP
    CHECK(ok_sky && e_sky < 0.05f, "draw camera: a c8 draw is seen from its own view's camera (off %.3g)", (double)e_sky);
    CHECK(ok_mvp && e_mvp < 0.05f,
          "draw camera: a draw placed through c4 after the sky is seen from the view in c8, the player's (off %.3g)",
          (double)e_mvp);
    CHECK(!ok_none, "draw camera: an empty c8 gives no camera, so the last one stands (%d)", ok_none);
    CHECK(ok_ff && e_ff < 0.05f, "draw camera: a fixed-function draw from its view matrix (off %.3g)", (double)e_ff);
}

int main(void)
{
    ring_rules();
    ring_ages();
    ring_stages();
    ring_threads();
    screen_math();
    space_order();
    stereo_math();
    stereo_plane_math();
    late_rotation_math();
    fixed_screen_math();
    fullview_math();
    draw_camera_math();
    overlay_cover_math();
    arm_side_math();
    printf("%s: %d passed, %d failed\n", s_fail ? "FAILED" : "ALL PASS", s_pass, s_fail);
    return s_fail ? 1 : 0;
}
