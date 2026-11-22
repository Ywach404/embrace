/* The immersive space's opening and closing order: see space.h. */
#include "space.h"

#include <stdio.h>
#include <string.h>

void mtl_space_init(mtl_space *s)
{
    memset(s, 0, sizeof *s);
    pthread_mutex_init(&s->mu, NULL);
    s->window = 1;
    s->window_up = 1;
    s->result = -1;
}

void mtl_space_destroy(mtl_space *s) { pthread_mutex_destroy(&s->mu); }

/* Under the lock. The first frame is noted by whoever sees it first: the
   host's poll, or the render thread's watchdog when no poll comes. */
static void note_frames(mtl_space *s, uint64_t frames, uint64_t now_ns)
{
    uint64_t since = frames > s->frames_base ? frames - s->frames_base : 0;
    if (since > s->frames) s->frames = since;
    if (s->frames && !s->first_frame_ns && s->requested_ns) s->first_frame_ns = now_ns;
}

static void back_to_window(mtl_space *s)
{
    s->phase = MTL_SPACE_OFF;
    s->ring = 0;
    s->window = 1;
    s->window_up = 1;
}

void mtl_space_request(mtl_space *s, uint64_t now_ns)
{
    pthread_mutex_lock(&s->mu);
    if (s->phase == MTL_SPACE_OFF && !s->requested_ns) {
        s->phase = MTL_SPACE_REQUESTED;
        s->ring = 1;
        s->window = 1;
        s->requested_ns = now_ns;
    }
    pthread_mutex_unlock(&s->mu);
}

int mtl_space_reopen(mtl_space *s, uint64_t frames_now, uint64_t now_ns)
{
    pthread_mutex_lock(&s->mu);
    int r = s->phase == MTL_SPACE_OFF && s->requested_ns;
    if (r) {
        /* Every step's time and the watchdog's lines are per open, so the
           second space is watched as closely as the first. */
        s->frames = 0;
        s->frames_base = frames_now;
        s->first_frame_ns = s->open_called_ns = s->opened_ns = s->compositor_ns = s->shown_ns = s->dismissed_ns = 0;
        s->attempts = 0;
        s->result = -1;
        s->watchdog_fired = 0;
        s->phase = MTL_SPACE_REQUESTED;
        s->ring = 1;
        s->window = 1;
        s->requested_ns = now_ns;
    }
    pthread_mutex_unlock(&s->mu);
    return r;
}

int mtl_space_should_open(mtl_space *s, uint64_t frames, uint64_t now_ns)
{
    pthread_mutex_lock(&s->mu);
    s->polls++;
    s->last_poll_ns = now_ns;
    note_frames(s, frames, now_ns);
    int r;
    if (s->phase == MTL_SPACE_REQUESTED && s->frames) {
        /* Claimed here and released only by a result, so a second driver
           cannot open the space twice, and one that gives up hands the
           open back rather than keeping it. */
        s->phase = MTL_SPACE_OPENING;
        s->attempts++;
        s->open_called_ns = now_ns;
        r = 1;
    } else {
        r = s->phase == MTL_SPACE_REQUESTED || s->phase == MTL_SPACE_OPENING ? 0 : -1;
    }
    pthread_mutex_unlock(&s->mu);
    return r;
}

void mtl_space_open_result(mtl_space *s, int result, uint64_t now_ns)
{
    pthread_mutex_lock(&s->mu);
    if (s->phase == MTL_SPACE_OPENING) {
        s->result = result;
        switch (result) {
        case METAL_D3D9_SPACE_OPENED:
            s->phase = MTL_SPACE_OPEN;
            s->opened_ns = now_ns;
            s->opens++;
            break;
        case METAL_D3D9_SPACE_ABANDONED:
            s->phase = MTL_SPACE_REQUESTED;
            break;
        default:
            /* Declined or failed: the game stays in the window, which
               never stopped showing it. */
            back_to_window(s);
            break;
        }
    }
    pthread_mutex_unlock(&s->mu);
}

void mtl_space_compositor_started(mtl_space *s, uint64_t now_ns)
{
    pthread_mutex_lock(&s->mu);
    if (!s->compositor_ns) s->compositor_ns = now_ns;
    pthread_mutex_unlock(&s->mu);
}

void mtl_space_first_shown(mtl_space *s, uint64_t now_ns)
{
    pthread_mutex_lock(&s->mu);
    if (!s->shown_ns) s->shown_ns = now_ns;
    pthread_mutex_unlock(&s->mu);
}

int mtl_space_take_dismiss(mtl_space *s, uint64_t now_ns)
{
    pthread_mutex_lock(&s->mu);
    int r = s->phase == MTL_SPACE_OPEN && s->shown_ns && s->window_up;
    if (r) {
        /* Off before the host dismisses the window, so the render thread
           never asks a layer with no scene for a drawable. */
        s->window = 0;
        s->window_up = 0;
        s->dismissed_ns = now_ns;
    }
    pthread_mutex_unlock(&s->mu);
    return r;
}

int mtl_space_closed(mtl_space *s, uint64_t now_ns)
{
    (void)now_ns;
    pthread_mutex_lock(&s->mu);
    int r = 0;
    if (s->phase == MTL_SPACE_OPEN || s->phase == MTL_SPACE_OPENING) {
        if (s->window_up) {
            back_to_window(s);
        } else {
            /* The ring keeps the frames until a window can take them. */
            s->phase = MTL_SPACE_RETURNING;
            r = 1;
        }
    }
    pthread_mutex_unlock(&s->mu);
    return r;
}

void mtl_space_window_back(mtl_space *s, uint64_t now_ns)
{
    (void)now_ns;
    pthread_mutex_lock(&s->mu);
    if (s->phase == MTL_SPACE_RETURNING) back_to_window(s);
    pthread_mutex_unlock(&s->mu);
}

static double secs(uint64_t from, uint64_t to) { return to > from ? (double)(to - from) / 1e9 : 0.0; }

/* A step's time since the request, or "no". */
static const char *at(char *b, size_t cap, uint64_t req, uint64_t t)
{
    if (!t) return "no";
    snprintf(b, cap, "at %.2f s", secs(req, t));
    return b;
}

static const char *result_name(int r)
{
    switch (r) {
    case -1: return "none";
    case METAL_D3D9_SPACE_OPENED: return "opened";
    case METAL_D3D9_SPACE_DECLINED: return "declined";
    case METAL_D3D9_SPACE_FAILED: return "failed";
    case METAL_D3D9_SPACE_ABANDONED: return "abandoned";
    default: return "unknown";
    }
}

int mtl_space_watchdog(mtl_space *s, uint64_t frames, uint64_t now_ns, char *buf, size_t cap)
{
    pthread_mutex_lock(&s->mu);
    note_frames(s, frames, now_ns);
    int bit = 0;
    if (s->phase == MTL_SPACE_REQUESTED || s->phase == MTL_SPACE_OPENING) {
        if (!(s->watchdog_fired & 1) && now_ns >= s->requested_ns + MTL_SPACE_WATCHDOG_NS) bit = 1;
        else if (!(s->watchdog_fired & 2) && s->first_frame_ns && now_ns >= s->first_frame_ns + MTL_SPACE_WATCHDOG_NS)
            bit = 2;
    }
    if (bit) {
        /* Firing for the request also covers the first frame's turn when
           that is already overdue, so one stall gives one line. */
        s->watchdog_fired |= bit;
        if (bit == 1 && s->first_frame_ns && now_ns >= s->first_frame_ns + MTL_SPACE_WATCHDOG_NS)
            s->watchdog_fired |= 2;
        uint64_t q = s->requested_ns;
        char b[6][32];
        snprintf(buf, cap,
                 "watchdog: the space is not open %.2f s after it was requested (%s); requested yes; frames in the "
                 "ring %llu, first %s; host polls %llu, last %s; open called %s, %llu attempts, phase %s; open "
                 "result %s; compositor loop %s; first frame shown %s; window dismissed %s; Present sends frames to "
                 "the ring %s, to the window %s",
                 secs(q, now_ns), bit == 1 ? "after the request" : "after the first frame reached the ring",
                 (unsigned long long)s->frames, at(b[0], sizeof b[0], q, s->first_frame_ns),
                 (unsigned long long)s->polls,
                 s->last_poll_ns ? (snprintf(b[1], sizeof b[1], "%.2f s ago", secs(s->last_poll_ns, now_ns)), b[1])
                                 : "never",
                 at(b[2], sizeof b[2], q, s->open_called_ns), (unsigned long long)s->attempts,
                 s->phase == MTL_SPACE_OPENING ? "waiting for the system's answer" : "waiting for a driver",
                 result_name(s->result), at(b[3], sizeof b[3], q, s->compositor_ns),
                 at(b[4], sizeof b[4], q, s->shown_ns), at(b[5], sizeof b[5], q, s->dismissed_ns),
                 s->ring ? "on" : "off", s->window ? "on" : "off");
    }
    pthread_mutex_unlock(&s->mu);
    return bit != 0;
}

void mtl_space_outputs(mtl_space *s, int *ring, int *window)
{
    pthread_mutex_lock(&s->mu);
    *ring = s->ring;
    *window = s->window;
    pthread_mutex_unlock(&s->mu);
}
