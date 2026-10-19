# rolloutscope

**A linter for RL rollouts.**

[简体中文](README.md) · English

Existing tools answer *"what did the model generate?"* — trace viewers, span explorers, dashboards full of completions.
`rolloutscope` answers the other question: **"how much of this training run was worth computing, and what exactly is broken?"**

Point it at the rollout files you already have. It tells you:

- how many **GRPO groups produced zero advantage** (all-correct or all-wrong → no gradient),
- at which step **entropy** and **output length** started to collapse,
- which rollouts look like **reward hacking** (top-decile reward with a fraction of the usual work),
- how many rollouts are **structurally invalid** (truncated, malformed tool call, tool loop),
- and how many GPU-hours were burned on all of the above.

No GPU. No re-run. No SDK to integrate.

![rolloutscope report](docs/report.png)

```
pip install rolloutscope
rolloutscope demo
```

---

## Why

Three things are true at the same time:

1. **Everyone logs rollouts, nobody analyses them.** verl ships `rollout_trace` backends (weave / mlflow / trackio), TRL logs completions, rLLM has its own UI. Each one is a *viewer* for one framework.
2. **The metric that matters is already known but collapsed into one scalar.** TRL logs `frac_reward_zero_std`; verl logs entropy and response length. None of them tell you *which* prompts wasted the step, or whether "zero std" meant too easy or too hard.
3. **Uploading every token to a SaaS is not viable.** verl's own docs note that a training run produces tens of GB of trace data per day against a 1 GB/month free tier.

`rolloutscope` is the analysis layer that sits on top of whatever you already dump: local files in, one HTML report out.

---

## Quickstart

```bash
pip install rolloutscope

# 1) synthetic run with a story baked in (no GPU, offline, reproducible)
rolloutscope demo
open demo/out/report.html

# 2) your own dump — format is auto-detected
rolloutscope report ./checkpoints/exp1/rollout_dump --out report.html

# 3) CI: fail the job when rollout health drops
rolloutscope scan ./rollout_dump --fail-under 60 --json findings.json

# 4) export to OpenTelemetry and keep using your existing observability stack
rolloutscope convert ./rollout_dump --to otel --out spans.json
```

The demo writes four artefacts, which double as a tour of the supported inputs:

| path | what it is |
| --- | --- |
| `demo/out/traces.jsonl` | native `AgentTrace` schema (with text) — used for the main report |
| `demo/out/verl_dump/` | verl-shaped parquet (`uid`, `responses`, `response_mask`, `rollout_log_probs`, `rm_scores`) |
| `demo/out/trl_run/` | TRL `trainer_state.json` + `completions.jsonl` |
| `demo/out/spans.json` | the same run as OTLP/JSON spans |

```bash
rolloutscope report demo/out/verl_dump    # token ids only, exactly like verl
rolloutscope report demo/out/trl_run      # metrics-only fallback still works
rolloutscope report demo/out/spans.json --adapter otel
```

### What each source can see

Different dumps carry different information, and the report is explicit about it rather than guessing:

| input | reward | length | tool calls | logprobs | text |
| --- | --- | --- | --- | --- | --- |
| native `AgentTrace` jsonl | yes | yes | yes | optional | yes |
| verl parquet dump | yes (`rm_scores`) | yes (`response_mask`) | no | yes (`rollout_log_probs`) | only with `--tokenizer` |
| TRL completions table | yes | yes (`completion_tokens` or estimate) | no | no | yes |
| TRL `trainer_state.json` | aggregate only | yes | no | entropy scalar | no |
| OTel spans | yes (`rolloutscope.reward`) | yes | yes | aggregate only | yes |

A rule that cannot see its input is skipped with a reason, never faked. That is why `invalid-rollout` reads ~9% on a native dump but ~6% on the verl parquet: the parquet has no tool-call column, so malformed tool calls are invisible there.

---

## What it finds

| rule | question | how it is computed |
| --- | --- | --- |
| `group-difficulty` | How much of this step can move the policy? | buckets groups into saturated (`pass=1`), informative (`0<pass<1`), collapsed (`pass=0`); **ERR** = informative / all; gradient density = `Σ|A|` per 1k tokens |
| `zero-advantage` | How many groups produced no gradient? | `σ(reward)=0` ⇒ every `A_i=0`; split into all-correct vs all-wrong, then convert wasted tokens to GPU-hours |
| `entropy-collapse` | Did the policy stop exploring? | per-step median entropy (or `-log p(token)` proxy) vs. an early baseline; flags a sustained drop below 60% |
| `length-collapse` | Did the model learn to stop talking? | median output length vs. baseline, plus within-step length CV; separates an intended length penalty (reward up) from collapse (reward flat) |
| `reward-hacking` | Is the reward gameable? | per step: top-decile reward **and** (<35% of median length **or** ≤20% of median tool calls), plus the `lift` of "short ⇒ high reward" |
| `invalid-rollout` | How many rollouts were dead on arrival? | truncation, malformed tool-call JSON, tool-call syntax leaking into text, empty output, repeated identical tool calls |

Every finding carries a number, the affected trace ids, and a concrete next action. Nothing is reported as "low quality" without evidence you can open.

### The number that is usually missing

GRPO computes `A_i = (r_i − mean(r)) / (std(r) + ε)` inside each group. If all `n` rollouts of a prompt get the same reward, `std = 0`, so **every advantage is exactly zero** — the group contributes nothing to the gradient, while its forward passes, reward calls and tool executions have already been paid for.

That failure has two completely different causes with opposite fixes:

- **all correct** → the prompt is too easy; down-sample or retire it;
- **all wrong** → the prompt is too hard; add hints, or use a dynamic baseline so the group still produces signal.

A single `frac_reward_zero_std` scalar cannot tell you which one you are in. `rolloutscope` splits it, attributes it per step, and prices it.

---

## The schema

`AgentTrace` is **not a new format** — it is a superset of the [OpenTelemetry GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai). A rollout is emitted as a normal OTel trace that any GenAI-aware backend already understands:

```
invoke_agent {agent}                gen_ai.operation.name = invoke_agent
├── chat {model}                    gen_ai.usage.*, gen_ai.input.messages, gen_ai.response.finish_reasons
│   └── execute_tool {tool}         gen_ai.tool.name / call.id / call.arguments / call.result
└── chat {model} ...
```

The RL dimensions OTel does not define yet ride along as `rolloutscope.*` attributes, so the `gen_ai.*` namespace stays clean:

| attribute | meaning |
| --- | --- |
| `rolloutscope.train.step` | training step |
| `rolloutscope.group.id` | GRPO group (same prompt) |
| `rolloutscope.rollout.index` | n-th sample in the group |
| `rolloutscope.reward` / `.advantage` | scalar reward / normalised advantage |
| `rolloutscope.status` / `.invalid.reason` | `ok` / `invalid` + reason |
| `rolloutscope.schema.version` | schema version |

```python
from rolloutscope import load, analyze
from rolloutscope.otel import to_otlp

batch = load("./rollout_dump")          # auto-detects verl / TRL / generic / OTel
report = analyze(batch)
print(report.score, report.summary)

spans = to_otlp(batch)                  # feed any OTel backend
```

Because the mapping is bidirectional, agents already instrumented with OpenInference, Langfuse or Traceloop can be diagnosed without touching their code: export spans, run `rolloutscope report spans.json --adapter otel`.

---

## Configuration

All thresholds live in one dataclass and can be overridden with a JSON file:

```bash
rolloutscope scan ./dump --config rolloutscope.json --pass-threshold 1.0
```

```json
{
  "zero_adv_std": 1e-6,
  "pass_threshold": 1.0,
  "entropy_drop_ratio": 0.6,
  "length_drop_ratio": 0.6,
  "hack_reward_quantile": 0.9,
  "hack_length_ratio": 0.35,
  "weight_zero_adv": 30,
  "weight_invalid": 25,
  "tokens_per_second": 2500,
  "cost_per_gpu_hour": 2.0
}
```

The 0–100 **rollout health score** is a documented weighted sum (`30·zero-adv + 25·invalid + 20·reward-hack + 15·entropy + 10·length`), clipped at 0. It is a heuristic for tracking a run over time, not a benchmark.

---

## Design principles

1. **Zero instrumentation.** Read what the trainer already wrote. Requiring a re-run kills adoption.
2. **Standard, not another standard.** OTel GenAI underneath, `rolloutscope.*` on top.
3. **Every finding is actionable.** Phenomenon → number → evidence → cause → fix → cost.
4. **Fail soft.** A missing column skips a rule; it never breaks the report.

---

## Status

v0.1 — the diagnostics engine, four adapters and the offline report work end-to-end on synthetic and verl/TRL-shaped data.
Next: calibrate thresholds on real runs, add `rolloutscope diff` for comparing two experiments, publish the `rolloutscope.*` attributes as an OTel proposal.

Licence: MIT.
