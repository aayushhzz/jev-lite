# Jev Lite

A local decision API for agents that follows the architecture in the JEV-as-a-Judge paper: typed decisions with probabilities, a two-order confidence gate, and a cascade that escalates uncertain items to a stronger judge. It runs on your machine through Ollama.

It is not TypeSafe AI's Jev. The paper's hosted model is proprietary, so this build uses local open-weight models in both tiers. The architecture and the evaluation protocol follow the paper. Results will not match the paper's.

## Architecture

```
          question + two candidate responses (a, b)
                          |
        +-----------------v------------------+
        |  first stage: Jev Lite (4B, local) |   judge (a,b) and (b,a)
        |  p(a) = 1/2 [p1(a,b) + 1 - p1(b,a)] |   average aligned probabilities (paper eq. 1)
        |  q    = max(p(a), 1 - p(a))        |   confidence, temperature-scaled
        +-----------------+------------------+
                          |
                q >= tau ?  ----yes---->  accept first-stage verdict
                          |
                          no
                          v
        +------------------------------------+
        |  fallback: stronger local model    |   judges (a,b) in base order, alone
        |  (8B, thinking)                               |   its verdict replaces the first stage's
        +------------------------------------+
```

- **Primitives** (`POST /api/system-one`): choice, score, noul, and pairwise. Each answer is a probability distribution over options you supply, from one model step.
- **Two-order gate**: each pair is judged in both presentation orders, and the probabilities are averaged, which cancels position bias. Confidence is q, the largest probability.
- **Cascade** (`POST /api/cascade`): accepts the first stage when q is at least tau. Otherwise the fallback decides, and it judges only escalated items.
- **Threshold**: chosen on a selection split with the paper's lower-bound rule (one-sided 95%, margin -0.02, grid 0.5 to 0.99). Ties go to the stricter threshold. If nothing qualifies, everything escalates.
- **Temperature scaling**: fitted on the selection split by minimizing NLL, as in paper section 7.2. Checked on held-out pairs.
- **Frozen policy**: `policy.json` holds the temperature and the threshold, and the server loads it at startup.

Jev Lite's other endpoints: `POST /api/task-complexity` and `POST /api/coding-route` (used by [codex-langgraph](https://github.com/aayushhzz/codex-langgraph)). Their response shapes did not change.

## Run it

Requires Python 3.9 or newer, Ollama, and both models:

```bash
ollama pull qwen3:4b-instruct
ollama pull qwen3:8b
ollama serve            # or: brew services start ollama
python3 server.py       # listens on http://127.0.0.1:8876
```

Environment variables:

- `JEV_PORT`: port, default `8876`
- `JEV_MODEL`: first-stage model, default `qwen3:4b-instruct`
- `JEV_FALLBACK_MODEL`: fallback model, default `qwen3:8b`
- `OLLAMA_URL`: loopback Ollama URL, default `http://127.0.0.1:11434`

Examples:

```bash
# Cascade: needs policy.json (run the evaluation below first)
curl -s localhost:8876/api/cascade -H 'Content-Type: application/json' -d '{
  "state": "Which city is the capital of Australia?",
  "instructions": "Which response correctly and fully answers the question?",
  "a": "Canberra", "b": "Sydney"
}'

# Primitives: choice, score, noul, pairwise
curl -s localhost:8876/api/system-one -H 'Content-Type: application/json' -d '{
  "state": "Rewrites the payment retry logic and changes the database schema.",
  "questions": {"risk": {"type": "score", "instructions": "How risky is this change?",
                         "criteria": ["trivial", "minor", "moderate", "high", "dangerous"]}}
}'
```

Tests (no Ollama needed):

```bash
python3 -m unittest discover -s tests
```

Evaluation (needs both models; about 30 minutes, because the thinking fallback takes about 5 seconds per item):

```bash
python3 bench/cascade_eval.py     # writes policy.json and bench/results/
```

## Evaluation protocol

The harness follows the paper's order of operations:

1. Judge the selection pairs with both tiers. Apply the temperature and threshold rules to them.
2. Freeze the temperature and threshold in `policy.json`. Nothing is tuned after this.
3. Judge the held-out pairs offline, and run the frozen cascade live through the server's own method.
4. Report accuracy, escalation rate, paired difference against the fallback with a 2,000-resample bootstrap, ECE over 10 bins, Brier, NLL, error AUROC of 1 - q, and selective accuracy at 80% and 50% coverage.

The labels are 150 selection pairs, 146 fresh held-out pairs, and 77 legacy pairs. They are hand-written in [`bench/pairs.py`](bench/pairs.py), [`bench/pairs_extra.py`](bench/pairs_extra.py), and [`bench/pairs_hard.py`](bench/pairs_hard.py). The hard pairs need multi-step counting, probability, Python and SQL semantics, graph theory, or obscure facts. Each correct answer was fixed when its pair was written, and the position of the correct answer is randomized. The sets are small and written by one person, so the results are a sanity check, not a benchmark.

## Results

Hardware: Apple M2 Pro, 16 GB, Ollama 0.40.16. First stage `qwen3:4b-instruct`, two orders, read from logprobs. Fallback `qwen3:8b` with thinking enabled, one order, read as a verdict letter. Raw judgments and full metrics are in [`bench/results/`](bench/results/).

Three sets: 150 selection pairs (100 easy, 50 hard; fit T and tau), 146 fresh held-out pairs (98 easy, 48 hard; scored once after freezing), and 77 legacy pairs that I had already inspected (sanity only).

**Selection (150 pairs)**

| | First stage | Fallback (thinking) |
| --- | --- | --- |
| All 150 | 90.0% | 100% |
| Easy (100) | 96.0% | 100% |
| Hard (50) | 78.0% | 100% |

The temperature fit picked T = 6.0, and selection NLL fell from 0.512 to 0.207. The lower-bound rule picked tau = 0.95. With no temperature (T = 1) no threshold qualified, so every item would escalate.

**Fresh held-out (146 pairs)**

| Metric | Value |
| --- | --- |
| First stage alone | 92.5% (135/146) |
| Fallback alone, thinking | 100% (146/146) |
| Frozen cascade, live | 100%, escalated 32.2% (47/146) |
| Cascade minus fallback | 0.0 points |
| First-stage errors that were escalated | all of them |
| Escalated, easy / hard | 24.5% / 47.9% |
| First stage, easy / hard | 94.9% / 87.5% |
| Live decisions agreeing with offline simulation | 100% |
| ECE, 10 bins (T = 1 to T = 6, calibrated subsets above) | 0.051 to 0.018 |
| Latency, p50: accepted path / escalated path / fallback alone | 215 ms / 10,789 ms / 7,199 ms |

**Legacy set (77 pairs, previously inspected)**: cascade 100%, escalated 14.3%. The first stage gets 93.5% and the fallback 100%.

**Post hoc threshold sweep (fresh held-out, not used for the policy)**

These numbers score thresholds on the held-out set, so they are for reading only. Choosing tau from them would fit the test set.

| tau | Escalated | Cascade minus fallback |
| --- | --- | --- |
| 0.5 | 0% | -7.5 |
| 0.6 | 7.5% | -4.1 |
| 0.7 | 12.3% | -2.7 |
| 0.8 | 15.1% | -2.1 |
| 0.9 | 24.0% | -0.7 |
| 0.95 (frozen) | 32.2% | 0.0 |
| 0.99 | 100% | 0.0 |

**What this shows.** The harder set does what it was meant to do for the first stage: it is clearly worse on hard items, and the gate sends those items to the fallback. The gate caught every first-stage error and matched the fallback's accuracy while escalating about a third of items. But the thinking fallback still got all 373 labeled pairs right, including the hard ones. So this set still can't show what the cascade does when the fallback itself is wrong. The next round needs items that defeat an 8B reasoning model.

## Limits

- **Not Jev.** The paper's hosted model was trained with a method that is not published. This build uses generic local models, so the gap to the paper's accuracy and calibration is expected.
- **Small, hand-labeled set.** The paper evaluates on 1,850 preference pairs and more. The threshold fit uses 100 pairs and the fresh held-out set has 98, so differences of a few points are within noise. The bootstrap interval is about plus or minus 5 points.
- **Temperature transfers unevenly.** The paper reports this too (section 7.2). The fit here may help or hurt on held-out pairs.
- **Serial cascade only.** The paper's fusion and debate variants are not built.
- **Ceiling.** The thinking fallback is correct on all 373 labeled pairs, including 98 hard ones. These sets can't show what happens when the fallback is wrong. Items that defeat an 8B reasoning model are the next step.
- **Temperature also changes the routing confidences.** `/api/coding-route` and `/api/task-complexity` use the same first-stage model, so with `policy.json` they report temperature-scaled confidence. Temperature is monotone, so the chosen role does not change.
- **Escalation rule.** `JEV_POLICY` lets you point at another policy file for tests. The policy is only as good as its selection split.
- **No auth.** It binds to loopback. Do not expose it.
- **Stateless.** Nothing is stored between requests.

## Sources

- Li, Y., Miao, Y., Krishnan, R., and Padman, R. *JEV-as-a-Judge: Accept When Confident, Escalate When Unsure.* arXiv:2609.26550 (v6, 29 Oct 2026). https://arxiv.org/abs/2609.26550
- Deußer, T., Ramamurthy, R., Sparrenberg, L., and Sifa, R. *Evaluating and Benchmarking the System One Model Jev.* arXiv:2609.37647 (v2, 29 Oct 2026). https://arxiv.org/abs/2609.37647
- TypeSafe AI, Jev documentation. https://docs.typesafe.ai/introduction
- TypeSafe AI, *Introducing System One models and Jev.* https://typesafe.ai/blog/introducing-system-one-models-and-jev
- Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. *On Calibration of Modern Neural Networks.* ICML 2017. Cited by both papers for temperature scaling and the confidence definition; not read directly here.
