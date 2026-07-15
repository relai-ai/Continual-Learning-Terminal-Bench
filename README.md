# Continual-Learning Evaluation on Terminal-Bench 2.0

This repository holds the agent-harness artifacts accompanying the preprint *"Do Agent
Optimizers Compound? A Continual-Learning Evaluation on Terminal-Bench 2.0."* It evaluates
three agent-harness optimizers — GEPA, Meta Harness, and RELAI's Verifiable Continual Learning
engine (RELAI-VCL) — under a two-phase protocol: each optimizer starts from a shared baseline
agent, is optimized on an initial set of tasks (Phase 1), and is then optimized again on an
expanded task set that includes newly introduced tasks (Phase 2). These artifacts are the actual
agent snapshots produced at each stage of that protocol.

## Paper

📄 [**Do Agent Optimizers Compound? A Continual-Learning Evaluation on Terminal-Bench 2.0**](https://github.com/relai-ai/Continual-Learning-Terminal-Bench/blob/main/do_agent_optimizers_compound.pdf)

## Layout

```
baseline_agent/    Three copies of the shared, unoptimized starting agent
GEPA/              GEPA's optimized artifacts (prompt-only)
meta-harness/      Meta Harness's optimized artifacts (harness code)
RELAI/             RELAI-VCL's optimized artifacts (full harness package)
```

### `baseline_agent/`

The fixed starting agent ("TerminusKira," a Harbor-compatible agent built on Harbor's Terminus2
base class with native LLM tool calling) that every optimizer begins from. It appears here three
times — `gepa_tbench_agent/`, `metaharness_tbench_agent/`, `relai_tbench_agent/` — once per
optimizer's own experiment, since each optimizer's harness code imports the agent under its own
package name. The three copies are otherwise identical: same model (`openai/gpt-5.5`), same tool
definitions, same prompt template, same harness logic.

### `GEPA/`, `meta-harness/`, `RELAI/`

Each has an `initial/`, `phase1/`, and `phase2/` subfolder holding that optimizer's agent
snapshot at three points: before any optimization, after Phase 1, and after Phase 2.

The internal structure of these folders is *not* uniform across optimizers, and that's
intentional: it reflects what each optimizer is actually allowed to change.

- **`GEPA/`** — GEPA optimizes prompts only, so each stage is a single prompt file
  (`initial-prompt.txt`, `phase1.best-prompt.txt`, `phase2.best-prompt.txt`). A code-mutation
  variant of GEPA was also evaluated but failed to produce a valid candidate during Phase 1, so
  it has no corresponding artifacts here.
- **`meta-harness/`** — Meta Harness edits harness code directly, so each stage is a single
  Python file (`baseline_kira.py`, `io_boundary_hardening.py`, `output_noise_compaction.py`,
  named for what that stage's edit does) plus the prompt template in effect at that stage.
- **`RELAI/`** — RELAI-VCL searches over a broader space (prompts, tools, workflows, memory,
  skills, and code), so each stage is a full agent package
  (`relai_tbench_agent/`: `__init__.py`, `config.py`, `harbor_harness.py`, `kira_agent.py`,
  `anthropic_caching.py`, `prompt_templates/`).

