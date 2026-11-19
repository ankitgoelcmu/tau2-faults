# tau2-faults

[![PyPI](https://img.shields.io/pypi/v/tau2-faults)](https://pypi.org/project/tau2-faults/) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24356187.svg)](https://doi.org/10.5281/zenodo.24356187)

Tool-fault injection for [τ²-bench](https://github.com/sierra-research/tau2-bench), without changing τ²-bench's code.

Real tools fail. A write can succeed while its reply is lost, a service can return a transient error, and a read can come from a replica that is one write behind. `tau2-faults` adds these faults to τ²-bench's airline and retail domains, so you can measure how agents recover:

| Fault | What the agent sees | What happened |
|---|---|---|
| transient error | `503 Service Unavailable` | nothing ran |
| lost response | `504 Gateway Timeout` | the write **ran**, but the reply was lost |
| stale read | an older value | the read came from a replica one write behind |

Faults can be **uniform** or **bursty** (a Gilbert–Elliott process: healthy and degraded periods). An optional **idempotent** condition deduplicates repeated identical writes, like an API with idempotency keys.

## Why a plug-in is needed: grading stays exact

τ²-bench grades a run by replaying the conversation into a fresh database and comparing hashes. A tool that fails after or before committing breaks that replay ([issue #608](https://github.com/sierra-research/tau2-bench/issues/608)): replay may count a write that never happened, or miss one that ran before its reply was lost.

`tau2-faults` records what each faulted call did and replays it the same way: a call that failed before committing is skipped, a write that ran but lost its reply is applied with the recorded reply, and a deduplicated write changes nothing the second time. Tests check that, under every fault kind at a 40% fault rate, replay rebuilds exactly the live final database (25 airline and 25 retail tasks).

## Install

τ²-bench is not on PyPI, so install it first. Python 3.12 or later.

```bash
git clone https://github.com/sierra-research/tau2-bench
git -C tau2-bench checkout 033bf082f48e4a3c6a1256208e93cf2ce9c3bb6f   # the version tested
pip install -e tau2-bench
pip install tau2-faults
```

Other τ²-bench versions may work, but the plug-in wraps `Environment.get_response` and `Environment.set_state`, so changes to those can break it. Run the tests below after upgrading.

## Try it free (no API keys)

```bash
python examples/quickstart.py
```

A scripted agent replays each airline task's reference actions while half of its tool calls fail. It prints how many tasks still end correct with and without idempotency, and confirms that replay grading matched the live database.

## Run a real agent under faults

τ²-bench simulates the customer with a second model (default `gpt-4.1-2025-04-14`), so set `OPENAI_API_KEY` as well as your agent's key.

```bash
tau2-faults --domain airline --condition baseline   --fault-rate 0.2 \
    --agent-llm claude-sonnet-5-5 --num-trials 3 --out runs/airline-baseline-p20
tau2-faults --domain airline --condition idempotent --fault-rate 0.2 \
    --agent-llm claude-sonnet-5-5 --num-trials 3 --out runs/airline-idem-p20
```

Both conditions draw the same fault sequence for the same task and trial, so results can be compared pair by pair. Useful options: `--fault-model bursty`, `--burst-len 4`, `--lost-share 0.5`, `--stale-rate 0.1`, `--task-ids`, `--seed`. Local models through Ollama: `--agent-llm openai/<model> --agent-api-base http://localhost:11434/v1`.

Each run writes τ²-bench's `results.json` and a `faults.json` with every injected fault per simulation. Rerunning the same command resumes, and retries simulations that ended in an infrastructure error.

## Use it from Python

```python
import tau2_faults
tau2_faults.CONFIG = tau2_faults.FaultConfig(condition="idempotent", fault_rate=0.2, fault_model="bursty")
tau2_faults.install()   # wraps tau2's Environment; then run tau2 as usual
```

## Tests

```bash
python tests/test_replay.py    # 4 "ok" lines, no API calls
```

## Cite

If you use `tau2-faults`, please cite the paper it was built for, and the software (DOI [10.5281/zenodo.24356187](https://doi.org/10.5281/zenodo.24356187)):

```bibtex
@misc{goel2026toolcontracts,
  author = {Goel, Ankit},
  title  = {Tool Contracts and Output Defenses Under Agent Faults:
            A Pre-Registered Replication and $\tau^2$-bench Extension},
  year   = {2026},
  note   = {Preprint. Code: https://github.com/ankitgoelcmu/faultline-agent}
}
```

The paper's experiments, results on four models and redacted traces are in [faultline-agent](https://github.com/ankitgoelcmu/faultline-agent).

## License

MIT
