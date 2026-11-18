"""Quick start for the tau2-bench fault plug-in. Free: no model or API calls.

  python examples/quickstart.py

A scripted agent replays each airline task's reference actions through the
plug-in. Half of its tool calls fail, and every failed write is a lost response:
the write ran, but the agent sees a timeout. The agent retries a failed call
up to twice, as a careful agent might. It prints, per condition:

  - how many tasks still end in the correct state (tau2's own grader),
  - whether tau2's replay grading rebuilt exactly the database the live run left.

Without idempotency, some retries repeat a write that already ran, and the task
fails. With idempotency, the repeat is deduplicated. Replay matches live in both.
"""
from loguru import logger                                         # noqa: E402
import tau2_faults as tf
from tau2.data_model.message import AssistantMessage, ToolCall    # noqa: E402
from tau2.evaluator.evaluator_env import EnvironmentEvaluator     # noqa: E402
from tau2.registry import registry                                # noqa: E402

DOMAIN, N_TASKS, SEED = "airline", 30, 7


def run(task, cfg, retries=2):
    """Reference actions through the fault layer. Returns (live env, trajectory, episode)."""
    tf.CONFIG = cfg
    env = registry.get_env_constructor(DOMAIN)()
    if task.initial_state is not None:
        env.set_state(task.initial_state.initialization_data,
                      task.initial_state.initialization_actions,
                      task.initial_state.message_history or [])
    ep = tf.make_schedule(cfg, task.id, SEED)
    token = tf._EPISODE.set(ep)
    traj, n = [], 0
    try:
        for action in task.evaluation_criteria.actions or []:
            for _ in range(retries + 1):
                n += 1
                call = ToolCall(id=f"call_{n}", name=action.name, arguments=action.arguments,
                                requestor="assistant")
                traj.append(AssistantMessage(role="assistant", tool_calls=[call]))
                out = env.get_response(call)
                traj.append(out)
                if not out.error:
                    break
    finally:
        tf._EPISODE.reset(token)
    return env, traj, ep


def replay_hash(task, traj):
    env = registry.get_env_constructor(DOMAIN)()
    init = task.initial_state
    env.set_state(init.initialization_data if init else None,
                  init.initialization_actions if init else None,
                  list((init.message_history or []) if init else []) + traj)
    return env.get_db_hash()


def main():
    logger.remove()
    tf.install()
    ctor = registry.get_env_constructor(DOMAIN)
    tasks = [t for t in registry.get_tasks_loader(DOMAIN)()
             if t.evaluation_criteria and t.evaluation_criteria.actions][:N_TASKS]
    print(f"tau2-bench {DOMAIN}, {len(tasks)} tasks, scripted agent, 50% faults, every write fault a lost response\n")
    for cond in tf.CONDITIONS:
        cfg = tf.FaultConfig(cond, fault_rate=0.5, fault_model="uniform", lost_share=1.0)
        solved = matched = lost = dedup = 0
        for task in tasks:
            env, traj, ep = run(task, cfg)
            solved += EnvironmentEvaluator.calculate_reward(ctor, task, traj).reward >= 1.0
            matched += replay_hash(task, traj) == env.get_db_hash()
            lost += sum(e["kind"].startswith("lost_response") for e in ep.log)
            dedup += sum(e["kind"] == "dedup" for e in ep.log)
        print(f"{cond:<11} tasks correct {solved}/{len(tasks)}   lost responses {lost:>3}   "
              f"deduplicated writes {dedup:>3}   replay matched live {matched}/{len(tasks)}")
    print("\nSame fault draws in both conditions; only deduplication differs.")


if __name__ == "__main__":
    main()
