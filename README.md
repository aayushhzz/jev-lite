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
        |  (8B)                               |   its verdict replaces the first stage's
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

Evaluation (needs both models, roughly 10 minutes):

```bash
python3 bench/cascade_eval.py     # writes policy.json and bench/results/
```

## Evaluation protocol

The harness follows the paper's order of operations:

1. Judge the selection pairs with both tiers. Apply the temperature and threshold rules to them.
2. Freeze the temperature and threshold in `policy.json`. Nothing is tuned after this.
3. Judge the held-out pairs offline, and run the frozen cascade live through the server's own method.
4. Report accuracy, escalation rate, paired difference against the fallback with a 2,000-resample bootstrap, ECE over 10 bins, Brier, NLL, error AUROC of 1 - q, and selective accuracy at 80% and 50% coverage.

The labels are 100 selection pairs, 98 fresh held-out pairs, and 77 legacy pairs. They are hand-written in [`bench/pairs.py`](bench/pairs.py) and [`bench/pairs_extra.py`](bench/pairs_extra.py). Each correct answer was fixed when its pair was written, and the position of the correct answer is randomized. The sets are small and written by one person, so the results are a sanity check, not a benchmark.

## Results

Hardware: Apple M2 Pro, 16 GB, Ollama 0.40.16. First stage `qwen3:4b-instruct`, two orders. Fallback `qwen3:8b`, one order, thinking disabled (see Limits). Raw judgments and full metrics are in [`bench/results/`](bench/results/).

Three sets: 100 selection pairs (fit T and tau), 98 fresh held-out pairs (scored once, after the policy was frozen), and 77 legacy pairs from the first set. I had already inspected the legacy pairs, so they are a sanity check only.

**Selection (100 pairs)**

| | First stage | Fallback |
| --- | --- | --- |
| Accuracy (ties at half credit) | 96% | 92% |

The temperature fit picked T = 4.0, and selection NLL fell from 0.211 to 0.105. The lower-bound rule picked tau = 0.5. Every confidence is at least 0.5, so the frozen gate escalates nothing.

**Fresh held-out (98 pairs)**

| Metric | Value |
| --- | --- |
| First stage alone | 94.9% (93/98) |
| Fallback alone | 92.9% (91/98) |
| Frozen cascade, live | 94.9%, escalated 0% |
| Cascade minus fallback | +2.0 points, 95% bootstrap interval [-3.1, +7.1] |
| Live decisions agreeing with offline simulation | 100% |
| ECE, 10 bins (T = 1 to T = 4) | 0.056 to 0.027 |
| Brier (T = 1 to T = 4) | 0.101 to 0.089 |
| NLL (T = 1 to T = 4) | 0.340 to 0.148 |
| Error AUROC of 1 - q | 0.91 |
| Latency, p50: first-stage path / fallback alone | 215 ms / 236 ms |

Temperature scaling improved calibration on this set, which the paper saw on only one of its three datasets.

**Legacy set (77 pairs, previously inspected)**

The same frozen policy gives 93.5% against the fallback's 96.1%, a difference of -2.6 points with a 95% interval of [-9.1, +3.9].

**Post hoc threshold sweep (fresh held-out, not used for the policy)**

These numbers score thresholds on the held-out set, so they are for reading only. Choosing tau from them would fit the test set.

| tau | Escalated | Cascade minus fallback |
| --- | --- | --- |
| 0.5 (frozen) | 0% | +2.0 |
| 0.6 | 4.1% | +2.0 |
| 0.7 | 7.1% | +2.0 |
| 0.8 | 8.2% | +2.0 |
| 0.9 | 12.2% | +2.0 |
| 0.95 | 19.4% | 0.0 |
| 0.99 | 29.6% | 0.0 |

Escalating more items lowers accuracy against the first stage alone. On those items the fallback is often wrong where the first stage is right.

**What this shows.** The larger set confirms the earlier result. The threshold rule escalates nothing, so the cascade equals the first stage. The limit now is the fallback. Without thinking, the 8B model is less accurate than the two-order 4B first stage on both selection and held-out pairs, so escalating adds no accuracy. The paper's cascade needs a stronger fallback, and this build doesn't have one. A live escalation check on an earlier run (tau = 0.99) confirmed that the escalated path returns the fallback's decision.

## Limits

- **Not Jev.** The paper's hosted model was trained with a method that is not published. This build uses generic local models, so the gap to the paper's accuracy and calibration is expected.
- **Small, hand-labeled set.** The paper evaluates on 1,850 preference pairs and more. The threshold fit uses 100 pairs and the fresh held-out set has 98, so differences of a few points are within noise. The bootstrap interval is about plus or minus 5 points.
- **Temperature transfers unevenly.** The paper reports this too (section 7.2). The fit here may help or hurt on held-out pairs.
- **Serial cascade only.** The paper's fusion and debate variants are not built.
- **The fallback is not the stronger judge here.** Qwen3's thinking mode would make the 8B model stronger, but a one-token logprob read cannot use it, so `reasoning_effort: "none"` turns thinking off. On the selection set the 4B first stage beats it (96% against 92%). The paper's fallback is a reasoning model. The next step is a fallback with thinking enabled, which needs its own read path, since a logprob read takes one token.
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
