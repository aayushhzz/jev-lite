# Jev Lite

A small local decision API for agents, modeled on the shape of TypeSafe AI's Jev. It answers typed questions about a piece of state with probabilities over options you supply, never free text. It runs on your machine through Ollama.

It is not affiliated with TypeSafe AI and it is not Jev. It uses a generic 4B instruct model with no special training, so its probabilities are much less calibrated than Jev's.

## How it works

Each question takes one model step:

1. Build a prompt: your instructions, the state wrapped in tags, and the options as letters (`A. ...`, `B. ...`), then "Answer with one letter."
2. Ask Ollama for one output token (`max_tokens: 1`) and the top 20 log-probabilities for that token.
3. Keep only the option letters, exponentiate their log-probabilities, and renormalize. That gives a probability for every option.
4. Derive the answer:
   - **choice**: the most probable option, plus `probabilities` and `confidence = (p_max - 1/n) / (1 - 1/n)`.
   - **score**: the probability-weighted level number, plus `probabilities`, `legend`, and a confidence from the spread.
   - **noul**: P(yes), the probability of option A in a yes/no question.

Questions in one request run in parallel threads. The model never writes JSON or prose, so there is nothing to parse or retry.

The confidence formulas follow the ones documented for Jev. This project implements them from that description and has not checked them against Jev's code.

## Endpoints

| Endpoint | Purpose |
| --- | --- |
| `POST /api/system-one` | Jev-shaped: `{"state": ..., "questions": {"id": {"type": "choice" \| "score" \| "noul", "instructions": ..., "criteria": ...}}}`. Returns `{"answers": {...}, "latency_ms": ...}`. |
| `POST /api/task-complexity` | `{"task", "context"?}` returns `{"complexity": "simple"\|"complex", "route": "codex"\|"langgraph"}`. |
| `POST /api/coding-route` | `{"task", "context"?, "agents"?}` returns `{"agent", "assignment"}` from the five coding roles. |

Criteria:

- choice: an object of option names to descriptions, 2 to 20 options.
- score: an array of level descriptions, lowest first, 2 to 10 levels.
- noul: optional `{"true": "...", "false": "..."}`.

Errors: `400` for a bad request, `502` if the model gives no option letter, `503` if Ollama is unreachable or lacks logprobs.

The `assignment` text matches `AGENT_GUIDANCE` in [codex-langgraph](https://github.com/aayushhzz/codex-langgraph), which rejects any other text. Change both together.

Example:

```bash
curl -s localhost:8876/api/system-one -H 'Content-Type: application/json' -d '{
  "state": "Rewrites the payment retry logic and changes the database schema.",
  "questions": {
    "risk": {"type": "score", "instructions": "How risky is this change?",
             "criteria": ["trivial", "minor", "moderate", "high", "dangerous"]},
    "money": {"type": "noul", "instructions": "Does this touch money movement?"}
  }
}'
```

## Run it

Requires Python 3.9 or newer and Ollama. Ollama must support token logprobs, as the 0.40 build here does.

```bash
ollama pull qwen3:4b-instruct
ollama serve            # or: brew services start ollama
python3 server.py
```

The server listens on `http://127.0.0.1:8876`. Open that page for a small form.

Environment variables:

- `JEV_PORT`: port, default `8876`
- `JEV_MODEL`: Ollama model, default `qwen3:4b-instruct`
- `OLLAMA_URL`: loopback Ollama URL, default `http://127.0.0.1:11434`

Tests (no Ollama needed):

```bash
python3 -m unittest discover -s tests
```

Benchmark (needs the running server and Ollama):

```bash
python3 bench/run_bench.py
```

## Measured results

Hardware: Apple M2 Pro, 16 GB, Ollama 0.40.16, `qwen3:4b-instruct`.

| Check | Result |
| --- | --- |
| Offline tests | 25 pass (stub model) |
| Support-ticket benchmark, 40 hand-labeled tickets, 4 categories | accuracy 92.5% (37/40), Brier 0.150, ECE 0.073 (10 bins, top probability) |
| Latency, one choice question, warm | p50 182 ms, p95 186 ms |
| Latency, four questions in one request | p50 244 ms, p95 249 ms |
| Prompt injection in state (one sample, choice) | resisted: returned `no` |

The 40 labels were written before any model run. The sample is small, so treat these numbers as a sanity check, not a benchmark.

## Limits

- **Calibration.** An ECE of 0.073 means the top probability is off from observed accuracy by about 7 points on average in this set. The bins were not split by direction, so this does not show whether it is over- or under-confident. A calibration step, such as fitting a temperature on held-out labels, is not implemented yet.
- **Injection.** Option letters block the old JSON-based failure, but one sample proves little. Treat state as untrusted.
- **Not Jev.** Jev is a hosted model trained with a method the company has not published. This build uses a generic model and makes no claim to Jev's accuracy or latency at scale.
- **No auth.** It binds to loopback. Do not expose it.
- **Stateless.** Each request is independent. Nothing is stored.
