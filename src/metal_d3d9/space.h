/*
 * The order in which immersive mode takes over from the window: the space
 * is requested, opened, shows its first game frame, and only then is the
 * window dismissed; when the space closes, the window comes back first and
 * the frames follow it.
 *
 * Two outputs say where Present sends each frame, and the backend applies
 * them (immersive.m): `ring`, the compositor's handoff ring (handoff.h),
 * and `window`, the window's layer. Both are on from the moment immersive
 * is requested, so the game is visible in the window while the space is
 * still opening and the compositor has frames waiting the moment it starts.
 * The window's layer is turned off before the window is dismissed and on
 * again only once a window scene holds it, since an off-screen layer's
 * nextDrawable blocks the render thread for up to a second.
 *
 * The open is claimed by whichever host driver asks once frames are in the
 * ring, and a driver that gives up releases it. The host's first driver
 * lived in a SwiftUI view task that SwiftUI cancels and restarts as the
 * launcher's state changes; a one-shot flag taken by the cancelled task
 * left the restarted one nothing to do, and the space never opened. So no
 * claim here outlives the driver that took it without an answer.
 *
 * Plain C with no Metal and a caller's clock, so a test can drive every
 * order of events (test/handoff_test.c). Every call may come from any
 * thread; none waits for anything but the struct's own lock.
 */
#ifndef MTL_SPACE_H
#define MTL_SPACE_H

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>

#include "immersive.h"

/* The watchdog's wait: a space not open this long after it was requested,
   or after the first frame reached the ring, is reported. */
#define MTL_SPACE_WATCHDOG_NS 5000000000ull

enum {
    MTL_SPACE_OFF,       /* the window shows the game; no space wanted */
    MTL_SPACE_REQUESTED, /* wanted; waiting for a frame in the ring and a driver */
    MTL_SPACE_OPENING,   /* a driver asked the system to open it */
    MTL_SPACE_OPEN,      /* the space reported opened */
    MTL_SPACE_RETURNING, /* the space closed after the window went; a window scene is coming */
};

typedef struct {
    pthread_mutex_t mu;
    int phase;
    int ring, window;  /* where Present sends frames */
    int window_up;     /* the host's window scene is up */
    uint64_t frames;   /* frames in the ring since this request, last seen */
    uint64_t frames_base; /* the ring's running total when this request was made */
    uint64_t opens;    /* spaces opened in the run, the first included */
    /* When each step happened, 0 until it has. */
    uint64_t requested_ns, first_frame_ns, open_called_ns, opened_ns, compositor_ns, shown_ns, dismissed_ns;
    uint64_t polls, last_poll_ns, attempts;
    int result;        /* the last open's METAL_D3D9_SPACE_* result, -1 for none */
    int watchdog_fired; /* bit 0: after the request, bit 1: after the first frame */
} mtl_space;

void mtl_space_init(mtl_space *s);
void mtl_space_destroy(mtl_space *s);

/* Immersive is wanted: frames to the ring and still to the window. */
void mtl_space_request(mtl_space *s, uint64_t now_ns);
/* Immersive is wanted again after a space closed and the window took the
   game back (the window's Return to Immersive). The same order as the
   first open follows: frames to the ring, a driver's claim, open, first
   frame, dismiss. frames_now is the ring's running total, so only a frame
   handed over after this request can claim the open: one left from the
   last space would let the driver open before frames flow again. 1 when
   taken; 0 when a space is still wanted, open, or on its way back. */
int mtl_space_reopen(mtl_space *s, uint64_t frames_now, uint64_t now_ns);
/* A host driver's poll, with the ring's frame count: 1 claims the open
   (the driver must open the space and report the result), 0 asks it to
   poll again, -1 says no open is wanted. */
int mtl_space_should_open(mtl_space *s, uint64_t frames, uint64_t now_ns);
void mtl_space_open_result(mtl_space *s, int result, uint64_t now_ns);
void mtl_space_compositor_started(mtl_space *s, uint64_t now_ns);
/* The compositor's GPU finished a frame with the game on the screen. */
void mtl_space_first_shown(mtl_space *s, uint64_t now_ns);
/* 1 once, when the space is open, has shown a frame and the window is up:
   the window's layer is already off and the host dismisses the window. */
int mtl_space_take_dismiss(mtl_space *s, uint64_t now_ns);
/* The space closed. 1: the window was dismissed, so the host opens a new
   window scene and calls window_back once it holds the game's window;
   until then frames stay off the window. 0: the window never went, and
   the frames are back on it already. */
int mtl_space_closed(mtl_space *s, uint64_t now_ns);
void mtl_space_window_back(mtl_space *s, uint64_t now_ns);
/* 1 and a line in buf when the space is overdue: once 5 s after the
   request, once 5 s after the first frame reached the ring, never once
   the space opened or was given up. */
int mtl_space_watchdog(mtl_space *s, uint64_t frames, uint64_t now_ns, char *buf, size_t cap);

/* What Present does now. */
void mtl_space_outputs(mtl_space *s, int *ring, int *window);

#endif
