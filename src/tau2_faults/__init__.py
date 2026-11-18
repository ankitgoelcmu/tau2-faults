"""Tool-fault injection for tau2-bench (τ²), without modifying tau2.

Needs tau2-bench installed in the same Python 3.12+ environment:

  tau2-faults --domain airline --condition baseline \\
      --fault-rate 0.1 --agent-llm claude-sonnet-5-5 --num-trials 3 --out runs/airline-baseline-p10

or from Python:  import tau2_faults; tau2_faults.CONFIG = tau2_faults.FaultConfig(...); tau2_faults.install()

How it works. tau2 sends every tool call through Environment.get_response()
and grades a run by replaying the conversation's tool calls into a fresh
environment (Environment.set_state), then comparing database hashes. This
module wraps both methods:

  get_response  injects faults into the agent's tool calls:
                  transient      "503". Nothing ran. Marked execution_status
                                 "rejected", which tau2's replay skips.
                  lost_response  write tools only. The write ran, but the agent
                                 gets a timeout. Replay runs the write again
                                 and accepts the recorded timeout as its reply.
                  stale_read     read tools only. Served from the database as
                                 it was before the latest write.
                Under the "idempotent" condition, a repeated write (same tool,
                same arguments) returns the first result without running
                again, like an API with idempotency keys.
  set_state     replays with faults off and the same dedup rule, so grading
                sees exactly the writes that happened live. Everything replay
                needs is in the conversation itself, so saved runs can be
                re-graded later (set CONFIG from the run's .faults.json).

Conditions. tau2 replays each tool call once, so harness-level retries
(several executions behind one message) can't be graded faithfully. The
conditions therefore test the agent's own handling of errors:
  baseline    faults reach the agent; every write it sends runs.
  idempotent  faults reach the agent; repeated identical writes are deduplicated.

Pairing. Each episode's fault sequence is seeded by (task id, trial seed), and
the k-th tool call always sees the k-th fault draw, so both conditions face the
same dice. tau2 fixes trial seeds from --seed.
"""

import argparse
import contextvars
import copy
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass, field

FAULT_MODELS = ["uniform", "bursty"]
BURST_FAULT_RATE = 0.8   # share of attempts that fail while the service is degraded (bursty model)
__version__ = "0.1.0"

CONDITIONS = ["baseline", "idempotent"]
TRANSIENT_MSG = "Error: 503 Service Unavailable. Please retry."
LOST_MSG = ("Error: 504 Gateway Timeout: no response from service. "
            "The request may or may not have been applied.")
SCHEDULE_LEN = 1000   # far more tool calls than any tau2 episode makes


@dataclass
class FaultConfig:
    condition: str = "baseline"
    fault_rate: float = 0.0
    fault_model: str = "uniform"
    burst_len: float = 4.0
    lost_share: float = 0.5
    stale_rate: float = 0.0


@dataclass
class Episode:
    """Fault draws for one episode, fixed in advance so both conditions share them."""
    faulted: list
    lost_roll: list
    stale_roll: list
    calls: int = 0
    log: list = field(default_factory=list)


def make_schedule(cfg, task_id, seed):
    h = hashlib.sha256(f"tau2|{task_id}|{seed}|{cfg.fault_rate}".encode()).hexdigest()
    rng = random.Random(int(h[:12], 16))
    faulted = []
    if cfg.fault_model == "uniform":
        faulted = [rng.random() < cfg.fault_rate for _ in range(SCHEDULE_LEN)]
    else:   # Gilbert-Elliott, started in its long-run state mix
        share = cfg.fault_rate / BURST_FAULT_RATE
        p_recover = 1.0 / cfg.burst_len
        p_degrade = share * p_recover / (1 - share) if share < 1 else 1.0
        degraded = rng.random() < share
        for _ in range(SCHEDULE_LEN):
            degraded = (rng.random() >= p_recover) if degraded else (rng.random() < p_degrade)
            faulted.append(degraded and rng.random() < BURST_FAULT_RATE)
    return Episode(faulted=faulted,
                   lost_roll=[rng.random() for _ in range(SCHEDULE_LEN)],
                   stale_roll=[rng.random() for _ in range(SCHEDULE_LEN)])


CONFIG = FaultConfig()
_EPISODE = contextvars.ContextVar("tau2_fault_episode", default=None)
EPISODE_LOGS = {}   # (task id, seed) -> list of injected events, for the run summary


def _write_key(call):
    blob = json.dumps({"t": call.name, "a": call.arguments}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def install():
    """Wrap tau2's Environment and per-task runner. Safe to call twice."""
    from tau2.data_model.message import ToolMessage
    from tau2.environment.environment import Environment
    from tau2.runner import batch

    if getattr(Environment, "_faults_installed", False):
        return
    orig_get_response = Environment.get_response
    orig_set_state = Environment.set_state
    orig_run_single_task = batch.run_single_task

    def run_single_task(config, task, *args, seed=None, **kwargs):
        ep = make_schedule(CONFIG, task.id, seed)
        EPISODE_LOGS[(task.id, seed)] = ep.log
        token = _EPISODE.set(ep)
        try:
            return orig_run_single_task(config, task, *args, seed=seed, **kwargs)
        finally:
            _EPISODE.reset(token)

    def msg(call, content, error, status):
        return ToolMessage(id=call.id, role="tool", content=content, requestor=call.requestor,
                           error=error, execution_status=status)

    def get_response(self, call):
        replay = getattr(self, "_fault_replay", None)
        if replay is not None:
            # Grading replay (tau2 calls this only for writes that weren't
            # rejected). Apply the same dedup rule as the live run, and hand
            # back the recorded message for lost responses and dedups so
            # tau2's strict comparison matches.
            recorded = replay["msgs"].get(call.id)
            if CONFIG.condition == "idempotent":
                if _write_key(call) in replay["cache"]:
                    return recorded
                replay["cache"].add(_write_key(call))
            real = orig_get_response(self, call)
            return recorded if recorded is not None and recorded.content == LOST_MSG else real
        ep = getattr(self, "_fault_episode", None) or _EPISODE.get()
        if ep is None or call.requestor != "assistant":
            return orig_get_response(self, call)
        self._fault_episode = ep
        k = ep.calls
        ep.calls += 1
        known = self._has_tool(call.name)
        writes = known and self._is_mutating_tool(call.name)
        cache = self.__dict__.setdefault("_fault_cache", {})

        def event(kind):
            ep.log.append({"call": k, "tool": call.name, "kind": kind})

        if ep.faulted[k % SCHEDULE_LEN] and known:
            if writes and ep.lost_roll[k % SCHEDULE_LEN] < CONFIG.lost_share:
                real = self._run_write(call, cache)
                event("lost_response" if real is not None else "lost_response_dedup")
                return msg(call, LOST_MSG, True, "executed")
            event("transient")
            return msg(call, TRANSIENT_MSG, True, "rejected")
        if writes:
            if CONFIG.condition == "idempotent" and _write_key(call) in cache:
                event("dedup")
                done = cache[_write_key(call)]
                return msg(call, done.content, done.error, "executed")
            return self._run_write(call, cache)
        if (known and CONFIG.stale_rate and getattr(self, "_fault_replica", None) is not None
                and ep.stale_roll[k % SCHEDULE_LEN] < CONFIG.stale_rate):
            event("stale_read")
            live = self.tools.db
            self.tools.db = self._fault_replica
            try:
                return orig_get_response(self, call)
            finally:
                self.tools.db = live
        return orig_get_response(self, call)

    def _run_write(self, call, cache):
        """Run a write (unless deduplicated). Returns its ToolMessage, or None if deduplicated."""
        if CONFIG.condition == "idempotent" and _write_key(call) in cache:
            return None
        self._fault_replica = copy.deepcopy(self.tools.db)
        real = orig_get_response(self, call)
        if CONFIG.condition == "idempotent":
            cache[_write_key(call)] = real
        return real

    def set_state(self, initialization_data, initialization_actions, message_history, strict=True):
        self._fault_replay = {"cache": set(),
                              "msgs": {m.id: m for m in message_history
                                       if getattr(m, "role", None) == "tool"}}
        try:
            return orig_set_state(self, initialization_data, initialization_actions,
                                  message_history, strict=strict)
        finally:
            self._fault_replay = None

    Environment.get_response = get_response
    Environment._run_write = _run_write
    Environment.set_state = set_state
    Environment._faults_installed = True
    batch.run_single_task = run_single_task


def main():
    p = argparse.ArgumentParser(description="Run tau2-bench with tool-fault injection")
    p.add_argument("--domain", default="airline", choices=["airline", "retail"])
    p.add_argument("--condition", choices=CONDITIONS, default="baseline")
    p.add_argument("--fault-rate", type=float, default=0.0)
    p.add_argument("--fault-model", choices=FAULT_MODELS, default="uniform")
    p.add_argument("--burst-len", type=float, default=4.0)
    p.add_argument("--lost-share", type=float, default=0.5)
    p.add_argument("--stale-rate", type=float, default=0.0)
    p.add_argument("--agent-llm", required=True, help="litellm model name, e.g. claude-sonnet-5-5")
    p.add_argument("--agent-api-base", default=None,
                   help="OpenAI-compatible server for the agent, e.g. http://localhost:11434/v1 for "
                        "Ollama (use --agent-llm openai/<model>). litellm's ollama_chat route doesn't "
                        "return tool results to the model correctly.")
    p.add_argument("--user-llm", default="gpt-4.1-2025-04-14", help="tau2's default user simulator")
    p.add_argument("--num-trials", type=int, default=1)
    p.add_argument("--num-tasks", type=int, default=None)
    p.add_argument("--task-ids", default=None, help="comma-separated tau2 task ids")
    p.add_argument("--seed", type=int, default=300, help="tau2 trial seeds; keep fixed across conditions")
    p.add_argument("--max-errors", type=int, default=30,
                   help="tau2 ends a run after this many tool errors (its default is 10)")
    p.add_argument("--max-steps", type=int, default=200, help="tau2's step limit per simulation")
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--out", required=True,
                   help="results folder: tau2 writes results.json there (resumes if present), "
                        "this plug-in writes faults.json")
    a = p.parse_args()
    out = os.path.abspath(a.out)   # tau2 joins this onto its data dir; absolute keeps it here
    os.makedirs(out, exist_ok=True)
    # Current Claude models reject `temperature`; other agents run at temperature 0.
    agent_args = {} if "claude" in a.agent_llm else {"temperature": 0.0}
    if a.agent_api_base:
        # Local (Ollama) agents: cap each reply at 8,192 tokens. Without a cap, a
        # looping reply can run ~67k tokens per turn and starve the other episodes.
        agent_args.update(api_base=a.agent_api_base, api_key="local", max_tokens=8192)

    global CONFIG
    CONFIG = FaultConfig(a.condition, a.fault_rate, a.fault_model, a.burst_len,
                         a.lost_share, a.stale_rate)
    install()
    from tau2.data_model.simulation import TextRunConfig
    from tau2.runner.batch import run_domain

    config = TextRunConfig(
        domain=a.domain, llm_agent=a.agent_llm, llm_args_agent=agent_args,
        llm_user=a.user_llm, llm_args_user={"temperature": 0.0},
        num_trials=a.num_trials, num_tasks=a.num_tasks,
        task_ids=a.task_ids.split(",") if a.task_ids else None,
        seed=a.seed, max_errors=a.max_errors, max_steps=a.max_steps, max_concurrency=a.concurrency,
        hallucination_retries=0, auto_resume=True, save_to=out)
    meta_path = os.path.join(out, "faults.json")
    meta = {"fault_config": CONFIG.__dict__, "agent_llm": a.agent_llm, "user_llm": a.user_llm,
            "seed": a.seed, "max_errors": a.max_errors, "max_steps": a.max_steps, "faults": {}}
    if os.path.exists(meta_path):   # resuming: refuse different settings, keep earlier episodes
        prev = json.load(open(meta_path))
        if prev["fault_config"] != meta["fault_config"]:
            raise SystemExit(f"{out} was run with {prev['fault_config']}. Use a new --out folder.")
        meta["faults"] = prev["faults"]
    meta["infra_errors_rerun"] = prev.get("infra_errors_rerun", 0) if os.path.exists(meta_path) else 0
    meta["infra_errors_rerun"] += drop_infra_errors(os.path.join(out, "results.json"))
    results = run_domain(config)
    meta["faults"].update({f"{t}|{s}": log for (t, s), log in EPISODE_LOGS.items()})
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    errs = [s for s in results.simulations if str(s.termination_reason).endswith("infrastructure_error")]
    rewards = [s.reward_info.reward for s in results.simulations if s.reward_info]
    print(f"\n{len(rewards)} graded simulations, mean reward {sum(rewards) / max(len(rewards), 1):.3f}. "
          f"Results in {out}", file=sys.stderr)
    if errs:
        print(f"WARNING: {len(errs)} simulations ended in an infrastructure error (e.g. "
              f"{str((errs[0].info or {}).get('error', ''))[:150]}). They have no reward; rerun the "
              f"same command to retry them.", file=sys.stderr)


def drop_infra_errors(results_path):
    """Remove simulations that ended in an infrastructure error (API failures,
    bad requests) from a results file, so a resumed run retries them instead of
    keeping them as done. Returns how many were removed."""
    if not os.path.exists(results_path):
        return 0
    with open(results_path) as f:
        data = json.load(f)
    sims = data.get("simulations", [])
    keep = [s for s in sims if s.get("termination_reason") != "infrastructure_error"]
    if len(keep) == len(sims):
        return 0
    with open(results_path + ".bak", "w") as f:
        json.dump(data, f)
    data["simulations"] = keep
    with open(results_path, "w") as f:
        json.dump(data, f)
    print(f"Retrying {len(sims) - len(keep)} simulations that ended in an infrastructure error.",
          file=sys.stderr)
    return len(sims) - len(keep)


if __name__ == "__main__":
    main()
