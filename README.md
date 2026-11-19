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

The labels are 100 selection pairs, 98 fresh held-out pairs, and 77 legacy pairs. They are hand-written in [`bench/pairs.py`](bench/pairs.py) and [`bench/pairs_extra.py`](bench/pairs_extra.py). Each correct answer was fixed when its pair was written, and the position of the correct answer is randomized. The sets are small and written by one person, so the results are a sanity check, not a benchmark.

## Results

Hardware: Apple M2 Pro, 16 GB, Ollama 0.40.16. First stage `qwen3:4b-instruct`, two orders, read from logprobs. Fallback `qwen3:8b` with thinking enabled, one order, read as a verdict letter from the final answer. Raw judgments and full metrics are in [`bench/results/`](bench/results/).

Three sets: 100 selection pairs (fit T and tau), 98 fresh held-out pairs (scored once, after the policy was frozen), and 77 legacy pairs that I had already inspected (sanity only).

**Selection (100 pairs)**

| | First stage | Fallback (thinking) |
| --- | --- | --- |
| Accuracy (ties at half credit) | 96% | 100% |

The temperature fit picked T = 4.0, and selection NLL fell from 0.211 to 0.105. The lower-bound rule picked tau = 0.99. With no temperature (T = 1) no threshold qualified, so every item would escalate.

**Fresh held-out (98 pairs)**

| Metric | Value |
| --- | --- |
| First stage alone | 94.9% (93/98) |
| Fallback alone, thinking | 100% (98/98) |
| Frozen cascade, live | 100%, escalated 29.6% |
| Cascade minus fallback | 0.0 points (both 100%) |
| First-stage errors that were escalated | 5 of 5 |
| Live decisions agreeing with offline simulation | 100% |
| ECE, 10 bins (T = 1 to T = 4) | 0.056 to 0.027 |
| Error AUROC of 1 - q | 0.91 |
| Latency, p50: accepted path / escalated path / fallback alone | 218 ms / 9,221 ms / 6,425 ms |

The cascade matches the fallback's accuracy and sends 70% of items to the cheap first stage. The first stage misses on five items, and the gate escalates all five.

**Legacy set (77 pairs, previously inspected)**: cascade 100%, escalated 14.3%. The first stage gets 93.5% and the fallback 100%.

**Post hoc threshold sweep (fresh held-out, not used for the policy)**

These numbers score thresholds on the held-out set, so they are for reading only. Choosing tau from them would fit the test set.

| tau | Escalated | Cascade minus fallback |
| --- | --- | --- |
| 0.5 | 0% | -5.1 |
| 0.6 | 4.1% | -4.1 |
| 0.7 | 7.1% | -3.1 |
| 0.8 | 8.2% | -3.1 |
| 0.9 | 12.2% | -2.0 |
| 0.95 | 19.4% | -1.0 |
| 0.99 (frozen) | 29.6% | 0.0 |

Lower thresholds let first-stage errors through, and the sweep shows the cost.

**What changed from the run without thinking.** Before this change (commit `03d8833`), the fallback was weaker than the first stage, so the gate escalated nothing. With thinking on, the fallback is correct on every labeled pair, so the rule now escalates the items the first stage isn't sure about.

**Caveats**

- **Ceiling.** The fallback gets all 275 labeled pairs right. These sets can't show what happens when both stages are unsure, and the labels are mostly short factual and arithmetic questions.
- **Cost.** Each escalation is a thinking call of 6 to 9 seconds. That is the paper's trade-off: the stronger judge is paid for only on escalated items.
- **Noise.** The held-out set has five first-stage errors, so the AUROC and selective-accuracy figures are noisy.

## Limits

- **Not Jev.** The paper's hosted model was trained with a method that is not published. This build uses generic local models, so the gap to the paper's accuracy and calibration is expected.
- **Small, hand-labeled set.** The paper evaluates on 1,850 preference pairs and more. The threshold fit uses 100 pairs and the fresh held-out set has 98, so differences of a few points are within noise. The bootstrap interval is about plus or minus 5 points.
- **Temperature transfers unevenly.** The paper reports this too (section 7.2). The fit here may help or hurt on held-out pairs.
- **Serial cascade only.** The paper's fusion and debate variants are not built.
- **Easy labels.** The fallback is correct on all 275 labeled pairs, so these sets can't show how the cascade behaves when both stages are unsure. Harder, verified labels are the next step.
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
