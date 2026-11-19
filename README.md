# Jev Lite

A small local decision API for agents. Jev Lite answers typed questions about a piece of text state and makes routing decisions for coding agents. It runs on your machine, uses a local model through Ollama, and checks every answer before returning it.

It is not affiliated with TypeSafe AI's Jev. It is a conventional LLM wrapper that borrows the idea of typed decisions.

## What it answers

| Endpoint | Input | Output |
| --- | --- | --- |
| `POST /api/decide` | `kind` (`choice`, `score`, or `unit`), `question`, `state`, and `options` for choice | `{"kind":"choice","choice":"yes"}`, `{"kind":"score","score":7}` (0 to 10), or `{"kind":"unit","value":0.25}` (0 to 1) |
| `POST /api/task-complexity` | `task`, optional `context` | `{"complexity":"complex","route":"langgraph"}`. `simple` maps to `codex`. |
| `POST /api/coding-route` | `task`, optional `context`, optional `agents` | `{"agent":"debugger","assignment":"..."}` from `implementer`, `debugger`, `test_writer`, `refactorer`, `researcher` |

Errors: `400` for a bad request, `502` if the model gives no valid answer after one retry, `503` if Ollama is not reachable.

The coding-route assignment text matches `AGENT_GUIDANCE` in [codex-langgraph](https://github.com/aayushhzz/codex-langgraph), which rejects any other text. Change both together.

## Run it

Requires Python 3.9 or newer and [Ollama](https://ollama.com) with a model pulled:

```bash
ollama pull qwen3:4b-instruct
python3 server.py
```

The server listens on `http://127.0.0.1:8876`. Open that page for a small form to try the decision endpoint.

Environment variables:

- `JEV_PORT`: port, default `8876`
- `JEV_MODEL`: Ollama model, default `qwen3:4b-instruct`
- `OLLAMA_URL`: loopback Ollama URL, default `http://127.0.0.1:11434`

Run the offline tests (no Ollama needed):

```bash
python3 -m unittest discover -s tests
```

## Design

- **Standard library only.** No dependencies to install.
- **Loopback only, no authentication.** Do not expose the port on a network.
- **Untrusted input.** Question, state, and task text are wrapped in tags so they cannot close their own block early. Model replies are parsed and checked in code. The model's JSON schema constrains the reply shape, but the server still validates it.
- **Bounded outputs.** Choices must be one of the options you supply. Scores are integers from 0 to 10. Estimates are numbers from 0 to 1.

## Status

- Verified: 21 offline tests pass, using a stub model in place of Ollama. They cover validation, retry, prompt fencing, the HTTP contract, and client-compatible routing text.
- Verified: a live server starts, serves its page and health check, and returns `503` when Ollama is absent.
- Not verified: answers from a live Ollama model. Quality and calibration are unmeasured.

## Limits

- Scores and estimates are subjective model outputs. They are not calibrated probabilities, and a valid shape does not mean a correct answer.
- Routing is a suggestion. The calling agent's sandbox, not Jev, decides what a worker can change.
- Each request is independent. Jev keeps no state between requests.
