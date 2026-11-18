"""Checks for the tau2-bench fault plug-in. No model calls.

Run with the tau2 environment:  python tests/test_replay.py

The key property: tau2 grades by replaying the conversation into a fresh
database, so that replay must rebuild exactly the database the live run left
behind, whatever faults were injected. Otherwise every reward is wrong.
"""

import tau2_faults as tf
from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.evaluator.evaluator_env import EnvironmentEvaluator
from tau2.registry import registry

tf.install()


def scripted_episode(domain, task, cfg, seed, retries=2):
    """Run a task's reference actions through the fault layer, like an agent that
    retries a failed call up to `retries` times. Returns (live env, trajectory, episode)."""
    tf.CONFIG = cfg
    env = registry.get_env_constructor(domain)()
    if task.initial_state is not None:
        env.set_state(task.initial_state.initialization_data,
                      task.initial_state.initialization_actions,
                      task.initial_state.message_history or [])
    ep = tf.make_schedule(cfg, task.id, seed)
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


def replayed_db_hash(domain, task, traj):
    """The database tau2's grader rebuilds from the trajectory."""
    env = registry.get_env_constructor(domain)()
    env.set_state(task.initial_state.initialization_data if task.initial_state else None,
                  task.initial_state.initialization_actions if task.initial_state else None,
                  list((task.initial_state.message_history or []) if task.initial_state else []) + traj)
    return env.get_db_hash()


def _tasks(domain, n):
    tasks = registry.get_tasks_loader(domain)()
    return [t for t in tasks if t.evaluation_criteria and t.evaluation_criteria.actions][:n]


def test_replay_matches_live_under_every_fault():
    kinds = set()
    for domain in ("airline", "retail"):
        for task in _tasks(domain, 25):
            for cond in tf.CONDITIONS:
                for model in ("uniform", "bursty"):
                    cfg = tf.FaultConfig(cond, 0.4, model, 4.0, 0.5, 0.3)
                    for seed in (1, 2, 3):
                        env, traj, ep = scripted_episode(domain, task, cfg, seed)
                        kinds |= {e["kind"] for e in ep.log}
                        assert replayed_db_hash(domain, task, traj) == env.get_db_hash(), \
                            (domain, task.id, cond, model, seed, ep.log)
    # The test must actually exercise every fault type.
    assert {"transient", "lost_response", "dedup"} <= kinds, kinds


def test_no_faults_gives_full_reward_on_reference_actions():
    for domain in ("airline", "retail"):
        ctor = registry.get_env_constructor(domain)
        for task in _tasks(domain, 15):
            for cond in tf.CONDITIONS:
                _, traj, ep = scripted_episode(domain, task, tf.FaultConfig(cond), 1)
                assert not ep.log
                r = EnvironmentEvaluator.calculate_reward(ctor, task, traj)
                assert r.reward == 1.0, (domain, task.id, cond)


def test_lost_response_duplicates_only_without_idempotency():
    """Find an airline task where a lost write plus a retry changes the outcome,
    and check the idempotent condition prevents it."""
    ctor = registry.get_env_constructor("airline")
    rewards = {c: [] for c in tf.CONDITIONS}
    for task in _tasks("airline", 50):
        for cond in tf.CONDITIONS:
            cfg = tf.FaultConfig(cond, 0.5, "uniform", 4.0, 1.0, 0.0)   # every write fault is lost
            _, traj, ep = scripted_episode("airline", task, cfg, 7)
            rewards[cond].append(EnvironmentEvaluator.calculate_reward(ctor, task, traj).reward)
    # Same fault dice; only the dedup differs. Idempotency must never do worse,
    # and must rescue at least one task the duplicate broke.
    pairs = list(zip(rewards["baseline"], rewards["idempotent"]))
    assert all(i >= b for b, i in pairs), pairs
    assert any(i > b for b, i in pairs), pairs


def test_same_dice_for_both_conditions():
    t = _tasks("airline", 1)[0]
    a = tf.make_schedule(tf.FaultConfig("baseline", 0.2), t.id, 5)
    b = tf.make_schedule(tf.FaultConfig("idempotent", 0.2), t.id, 5)
    assert a.faulted == b.faulted and a.lost_roll == b.lost_roll


if __name__ == "__main__":
    from loguru import logger
    logger.remove()
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
