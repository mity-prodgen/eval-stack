1. **Before generation, the conflict check works.** On the 101 stale/current items, the six hosted decision models
   catch 99–100% of the conflicts (n=202 per model) and raise a false alarm on 0–4% of the agreeing controls (n=80).
   Strands, running locally, catches 96% with 8% false alarms. On real WikiContradict pairs, catch falls to 52–72% for
   the hosted models and 15% for Strands (n=40). It's cheap and quick: Perplexity's flash model costs $0.005 per 1,000
   decisions at a 120 ms median, and Jev is fastest at 83 ms. Asking the same request twice almost never changed a
   verdict (0–1%).

2. **After generation, nothing catches the stale answer.** At P(yes) ≥ 0.5, the seven decision models catch 0–8% of
   the 95 stale or blended answers. The Sonnet 5.5 faithfulness judge from post 1 passes 94% of them, including 26 of the
   27 the human labeled stale. Nearly all of these answers come from Gemma 4 31B (94 of 95). Claude Haiku 5.5 gave
   the newer value in 193 of its 202 stale/current answers and flagged the conflict in 7 more.

3. **The good-looking Job 2 ROC is mostly a shortcut, and the "future change" answer fools most models.** Across all
   answers, Job 2 AUC is 0.95–1.00 (Luna 0.58), but stale answers mention the old value and current ones mostly don't.
   Within the 107 human-labeled timeline answers, where every answer mentions both values, AUC falls to 0.57–0.92 (only
   8 of them are current). Between stale answers (the newer value presented as a future change) and blended ones, four
   models — Jev, Perplexity Flash, Clef, Clef-flash — score the stale answers as *fresher* (AUC 0.26–0.39). The LLM
   labelers fell for the same shape: on the 107 timeline answers they agreed with the human on 54 (Sonnet) and 58
   (Sol), and called 0 and 4 of them stale. The human called 27 of them stale.

**Harness bugs and run notes**

- **The ledger recorded batch spend only when a step closed.** The first full-run process stopped while waiting on
  OpenAI's batch, and the cost of Sonnet's finished label batch ($0.52) was never recorded. I priced it from
  Anthropic's usage numbers and added it to the ledger by hand. Fix: record each batch's spend as soon as it ends.
- **The polling process belonged to the Claude session.** It ended with that session. The rerun resumed both batches
  by their saved ids, and nothing was paid twice.
- **Cloudflare's free-plan limit wasn't checked.** The estimate showed list prices but never compared them against the
  account's 10,000-neuron daily free allocation. Job 1 and Job 2 used about 12,300 neurons; Cloudflare let the overrun
  through, then answered every later call with 429 (code 4006). The client treated that as a retryable rate limit,
  and the noise and latency calls waited a day and a plan upgrade. Fix: treat 4006 as fatal and check the plan's limit
  at estimate time.
- **Perplexity rate limits:** 63 (v1.2) and 233 (flash) decisions needed a retry after a 429. They are excluded from
  latency, which came from a separate sequential sample with no retries.
- **The pilot and the full run shared a latency cache tag.** One sampled call per model was already cached, so each
  model has 99 timed calls instead of 100.
- **Analysis changes made after the first full-run report:** latency had counted that cached call, H3 named a
  "lowest" date format when all three were tied, and two within-timeline AUC rows were added to Job 2 once the
  shortcut showed up. No headline metric or threshold changed.
