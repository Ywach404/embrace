/* The handoff ring's bookkeeping: see handoff.h. */
#include "handoff.h"

#include <string.h>

void mtl_handoff_init(mtl_handoff *h)
{
    memset(h, 0, sizeof *h);
    pthread_mutex_init(&h->mu, NULL);
    h->latest = -1;
}

void mtl_handoff_destroy(mtl_handoff *h) { pthread_mutex_destroy(&h->mu); }

static int valid(int slot) { return slot >= 0 && slot < MTL_HANDOFF_SLOTS; }

int mtl_handoff_begin(mtl_handoff *h)
{
    pthread_mutex_lock(&h->mu);
    int pick = -1;
    /* A free slot first, then a published one that a newer frame has
       replaced and nobody holds. Either way never the latest, which the
       compositor may acquire at any moment, and never a held one. */
    for (int i = 0; i < MTL_HANDOFF_SLOTS && pick < 0; i++)
        if (h->slot[i].state == MTL_SLOT_FREE && !h->slot[i].holds && i != h->latest) pick = i;
    for (int i = 0; i < MTL_HANDOFF_SLOTS && pick < 0; i++)
        if (h->slot[i].state == MTL_SLOT_READY && !h->slot[i].holds && i != h->latest) pick = i;
    if (pick >= 0) {
        h->slot[pick].state = MTL_SLOT_WRITING;
        h->slot[pick].committed_ns = 0;
    } else {
        h->total.skipped++;
    }
    pthread_mutex_unlock(&h->mu);
    return pick;
}

void mtl_handoff_set_texture(mtl_handoff *h, int slot, void *tex, uint32_t w, uint32_t h_px, void **old)
{
    *old = NULL;
    if (!valid(slot)) return;
    pthread_mutex_lock(&h->mu);
    mtl_handoff_slot *s = &h->slot[slot];
    if (s->state == MTL_SLOT_WRITING) {
        *old = s->tex;
        s->tex = tex;
        s->w = w;
        s->h = h_px;
    } else {
        *old = tex; /* not the writer's slot: hand the new one back to be dropped */
    }
    pthread_mutex_unlock(&h->mu);
}

void mtl_handoff_get_texture(mtl_handoff *h, int slot, void **tex, uint32_t *w, uint32_t *h_px)
{
    *tex = NULL;
    *w = *h_px = 0;
    if (!valid(slot)) return;
    pthread_mutex_lock(&h->mu);
    *tex = h->slot[slot].tex;
    *w = h->slot[slot].w;
    *h_px = h->slot[slot].h;
    pthread_mutex_unlock(&h->mu);
}

void mtl_handoff_set_committed(mtl_handoff *h, int slot, uint64_t committed_ns)
{
    if (!valid(slot)) return;
    pthread_mutex_lock(&h->mu);
    if (h->slot[slot].state == MTL_SLOT_WRITING) h->slot[slot].committed_ns = committed_ns;
    pthread_mutex_unlock(&h->mu);
}

/* The stages from Present to publish, each from one timestamp to the next.
   A missing stamp takes the next one's value, so its stage's time goes to
   the stage before (no commit time: replay runs to the GPU's start). Stamps
   out of order, read on different threads, are clamped between their
   neighbours, so no stage is negative and they still add up to
   ready - presented. */
static void split_stages(const mtl_handoff_times *t, uint64_t committed_ns, uint64_t out_us[MTL_AGE_STAGES - 1])
{
    uint64_t c = t->committed_ns ? t->committed_ns : committed_ns;
    uint64_t at[MTL_AGE_STAGES] = {t->presented_ns, t->dequeued_ns, c, t->gpu_start_ns, t->gpu_end_ns, t->ready_ns};
    for (int i = MTL_AGE_STAGES - 2; i > 0; i--)
        if (!at[i] || at[i] > at[i + 1]) at[i] = at[i + 1];
    if (at[0] > at[MTL_AGE_STAGES - 1]) at[0] = at[MTL_AGE_STAGES - 1];
    for (int i = 1; i < MTL_AGE_STAGES; i++)
        if (at[i] < at[i - 1]) at[i] = at[i - 1];
    for (int i = 0; i < MTL_AGE_STAGES - 1; i++) out_us[i] = (at[i + 1] - at[i] + 500) / 1000;
}

uint64_t mtl_handoff_publish(mtl_handoff *h, int slot, const mtl_handoff_times *t)
{
    if (!valid(slot)) return 0;
    pthread_mutex_lock(&h->mu);
    mtl_handoff_slot *s = &h->slot[slot];
    uint64_t frame = 0;
    if (s->state == MTL_SLOT_WRITING) {
        split_stages(t, s->committed_ns, s->stage_us);
        if (h->latest >= 0 && !h->slot[h->latest].shown) h->total.never_shown++;
        frame = ++h->next_frame;
        s->state = MTL_SLOT_READY;
        s->frame = frame;
        s->presented_ns = t->presented_ns;
        s->ready_ns = t->ready_ns;
        s->shown = 0;
        h->latest = slot;
        h->total.published++;
    }
    pthread_mutex_unlock(&h->mu);
    return frame;
}

void mtl_handoff_abort(mtl_handoff *h, int slot)
{
    if (!valid(slot)) return;
    pthread_mutex_lock(&h->mu);
    if (h->slot[slot].state == MTL_SLOT_WRITING) h->slot[slot].state = MTL_SLOT_FREE;
    pthread_mutex_unlock(&h->mu);
}

static void note_age(mtl_handoff *h, double ms)
{
    uint64_t us = ms > 0 ? (uint64_t)(ms * 1000.0 + 0.5) : 0;
    uint64_t b = us / MTL_HANDOFF_AGE_STEP_US;
    if (b >= MTL_HANDOFF_AGE_BUCKETS) b = MTL_HANDOFF_AGE_BUCKETS - 1;
    h->period_hist[b]++;
    h->total_hist[b]++;
    if (us > h->period_max_us) h->period_max_us = us;
    if (us > h->total_max_us) h->total_max_us = us;
    h->period_sum_ms += ms;
    h->total_sum_ms += ms;
    h->total.age_n++;
}

int mtl_handoff_acquire(mtl_handoff *h, uint64_t now_ns, mtl_handoff_view *out)
{
    memset(out, 0, sizeof *out);
    out->slot = -1;
    pthread_mutex_lock(&h->mu);
    h->total.compositor_frames++;
    if (h->latest < 0) {
        h->total.empty++;
        pthread_mutex_unlock(&h->mu);
        return 0;
    }
    mtl_handoff_slot *s = &h->slot[h->latest];
    s->holds++;
    s->shown++;
    if (s->shown == 1) {
        h->total.shown++;
        /* The first showing is the frame's age as the eye first gets it;
           repeats only add the compositor's own interval. */
        uint64_t st[MTL_AGE_STAGES];
        memcpy(st, s->stage_us, sizeof s->stage_us);
        st[MTL_AGE_COMPOSITOR] = now_ns > s->ready_ns ? (now_ns - s->ready_ns + 500) / 1000 : 0;
        mtl_handoff_stages *g[2] = {&h->period_stages, &h->total_stages};
        for (int k = 0; k < 2; k++) {
            g[k]->n++;
            for (int i = 0; i < MTL_AGE_STAGES; i++) {
                g[k]->sum_us[i] += st[i];
                if (st[i] > g[k]->max_us[i]) g[k]->max_us[i] = st[i];
            }
        }
    } else {
        h->total.repeats++;
        if (s->shown == 2) h->total.shown_again++;
    }
    double age = now_ns > s->presented_ns ? (double)(now_ns - s->presented_ns) / 1e6 : 0.0;
    note_age(h, age);
    out->slot = h->latest;
    out->tex = s->tex;
    out->w = s->w;
    out->h = s->h;
    out->frame = s->frame;
    out->age_ms = age;
    out->repeat = s->shown > 1;
    pthread_mutex_unlock(&h->mu);
    return 1;
}

void mtl_handoff_release(mtl_handoff *h, int slot)
{
    if (!valid(slot)) return;
    pthread_mutex_lock(&h->mu);
    if (h->slot[slot].holds) h->slot[slot].holds--;
    pthread_mutex_unlock(&h->mu);
}

/* The bucket holding the q-th fraction of n samples, as its upper edge
   in ms: an over-estimate by at most one bucket (0.25 ms). */
static double quantile(const uint32_t *hist, uint64_t n, double q)
{
    if (!n) return 0;
    uint64_t want = (uint64_t)(q * (double)n + 0.999999), seen = 0;
    if (!want) want = 1;
    for (uint32_t b = 0; b < MTL_HANDOFF_AGE_BUCKETS; b++) {
        seen += hist[b];
        if (seen >= want) return (double)(b + 1) * MTL_HANDOFF_AGE_STEP_US / 1000.0;
    }
    return (double)MTL_HANDOFF_AGE_BUCKETS * MTL_HANDOFF_AGE_STEP_US / 1000.0;
}

static void fill_stages(mtl_handoff_stats *o, const mtl_handoff_stages *g)
{
    o->first_n = g->n;
    for (int i = 0; i < MTL_AGE_STAGES; i++) {
        o->stage_mean_ms[i] = g->n ? (double)g->sum_us[i] / 1000.0 / (double)g->n : 0;
        o->stage_max_ms[i] = (double)g->max_us[i] / 1000.0;
    }
}

static void fill_ages(mtl_handoff_stats *o, const uint32_t *hist, uint64_t n, uint64_t max_us, double sum)
{
    o->age_n = n;
    o->age_p50_ms = quantile(hist, n, 0.50);
    o->age_p95_ms = quantile(hist, n, 0.95);
    o->age_max_ms = (double)max_us / 1000.0;
    o->age_mean_ms = n ? sum / (double)n : 0;
}

void mtl_handoff_totals(mtl_handoff *h, mtl_handoff_stats *out)
{
    pthread_mutex_lock(&h->mu);
    *out = h->total;
    fill_ages(out, h->total_hist, h->total.age_n, h->total_max_us, h->total_sum_ms);
    fill_stages(out, &h->total_stages);
    pthread_mutex_unlock(&h->mu);
}

void mtl_handoff_take_period(mtl_handoff *h, mtl_handoff_stats *out)
{
    pthread_mutex_lock(&h->mu);
    const mtl_handoff_stats *b = &h->period_base, *t = &h->total;
    out->compositor_frames = t->compositor_frames - b->compositor_frames;
    out->empty = t->empty - b->empty;
    out->published = t->published - b->published;
    out->skipped = t->skipped - b->skipped;
    out->never_shown = t->never_shown - b->never_shown;
    out->shown = t->shown - b->shown;
    out->shown_again = t->shown_again - b->shown_again;
    out->repeats = t->repeats - b->repeats;
    fill_ages(out, h->period_hist, t->age_n - b->age_n, h->period_max_us, h->period_sum_ms);
    fill_stages(out, &h->period_stages);
    memset(&h->period_stages, 0, sizeof h->period_stages);
    h->period_base = *t;
    memset(h->period_hist, 0, sizeof h->period_hist);
    h->period_max_us = 0;
    h->period_sum_ms = 0;
    pthread_mutex_unlock(&h->mu);
}
