"""DDPL potmax solver inside NePPO.

Needs the jaxtrl source tree with the DDPL package on the path, e.g.
    PYTHONPATH=../jaxtrl/src JAX_PLATFORMS=cpu python tests/test_ddpl_pm.py
"""

import copy
import os
import sys

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _root)

import jax
import numpy as np
import torch as th

from marl.algorithms.neppo.ddpl_pm import make_phi_fn, phi_params, tds_to_batch, _phi_kind
from marl.algorithms.neppo.mixer import NPConvexNN, NPLambda, NPMixer, NPMixerWithDelta, NPNoMix
from marl.algorithms.neppo.neppo import EnsemblePhiWrapper


def _randomize(m, seed):
    g = th.Generator().manual_seed(seed)
    with th.no_grad():
        for p in m.parameters():
            p.copy_(th.randn(p.shape, generator=g))
    return m


def _phis(d=7, n=3):
    return {
        "lambda": NPLambda(output_dim=n),
        "mixer": NPMixer(input_dim=d, output_dim=n),
        "mixer_delta": NPMixerWithDelta(input_dim=d, output_dim=n),
        "nonmix": NPNoMix(input_dim=d, hidden_size=8),
        "convex": NPConvexNN(input_dim=d, output_dim=n, hidden_size=5),
    }


def test_phi_jax_matches_torch():
    d, n, T = 7, 3, 11
    s = th.randn(T, d, generator=th.Generator().manual_seed(0))
    r = th.randn(n, T, generator=th.Generator().manual_seed(1))  # NePPO stacks rewards as [n, T]
    for k, (name, phi) in enumerate(_phis(d, n).items()):
        _randomize(phi, k)
        ref = phi(s, r).detach().numpy()
        got = make_phi_fn(_phi_kind(phi))(phi_params(phi), jax.numpy.asarray(s.numpy()), jax.numpy.asarray(r.T.numpy()))
        np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-5, atol=1e-5, err_msg=name)


def test_phi_ensemble_matches_torch():
    d, n, T = 7, 3, 9
    members = [_randomize(NPMixer(input_dim=d, output_dim=n), 10 + m) for m in range(4)]
    phi = EnsemblePhiWrapper(members, 0.3)
    s, r = th.randn(T, d), th.randn(n, T)
    ref = phi(s, r).detach().numpy()
    got = make_phi_fn(_phi_kind(phi))(phi_params(phi), jax.numpy.asarray(s.numpy()), jax.numpy.asarray(r.T.numpy()))
    np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-5, atol=1e-5)


def test_tds_to_batch_terminal_and_state():
    from tensordict import TensorDict

    T, L = 6, 3  # two episodes of length 3
    step = th.tensor([0, 1, 2, 0, 1, 2])
    trunc = th.tensor([False, False, True, False, False, True])
    tds = []
    for i, (do, da) in enumerate(((4, 2), (3, 1))):
        tds.append(TensorDict({
            "observation": th.full((T, do), float(i)), "action": th.zeros(T, da), "step_count": step,
            "state": th.ones(T, 5),
            "next": {"observation": th.full((T, do), 10.0 + i), "reward": th.arange(T, dtype=th.float32) * (i + 1),
                     "terminated": th.zeros(T, dtype=th.bool), "truncated": trunc},
        }, batch_size=T))
    bounds = [(-np.ones(2, np.float32), np.ones(2, np.float32)), (-np.ones(1, np.float32), np.ones(1, np.float32))]
    b = tds_to_batch(tds, bounds, episode_len=L)
    assert np.asarray(b.terminated).tolist() == [False, False, True, False, False, True]  # time limit is terminal
    assert b.state.shape == (T, 4 + 3 + 1) and b.obs[0].shape == (T, 4) and b.obs[1].shape == (T, 3)
    np.testing.assert_allclose(np.asarray(b.state[:, -1]), step.numpy() / L)
    np.testing.assert_allclose(np.asarray(b.next_state[:, -1]), (step.numpy() + 1) / L)
    np.testing.assert_allclose(np.asarray(b.reward_vec[:, 1]), 2 * np.arange(T))
    # a forced collector truncation that is NOT the time limit must bootstrap
    tds[0]["next", "truncated"] = th.tensor([False, True, False, False, False, True])
    b2 = tds_to_batch(tds, bounds, episode_len=L)
    assert not bool(b2.terminated[1])


def _run_neppo(pm_solver, cycles=2, ddpl_extra=None):
    import mpe_neppo

    logs = []

    class _Logger:
        def __init__(self, *a, **k):
            pass

        def log_scalars(self, d, timestep):
            logs.append(dict(d))

    mpe_neppo.ExtWandbLogger = _Logger
    cfg = copy.deepcopy(mpe_neppo.CONFIG)
    cfg["seed"] = 0
    cfg["trainer"]["algo_kwargs"]["pm_solver"] = pm_solver
    cfg["trainer"]["algo_kwargs"]["num_epochs"] = 1
    cfg["trainer"]["ddpl"]["ddpl_kwargs"].update(
        hidden_sizes=[32], batch_size=16, num_candidates=4, learning_starts=1, buffer_capacity=4096, diffusion_steps=5)
    cfg["trainer"]["ddpl"].update(ddpl_extra or {})
    n_agents, per_rollout = 6, 2 * mpe_neppo.TIMESTEPS_PER_ENV  # test mode: 2 envs
    cfg["num_env_steps"] = cycles * (2 + 2 * n_agents + 2 * (n_agents + 1)) * per_rollout
    cwd = os.getcwd()
    try:
        os.chdir(os.environ.get("NEPPO_TEST_TMP", "/tmp"))
        mpe_neppo.train(test=True, config=cfg, wandb_kwargs={})
    finally:
        os.chdir(cwd)
    return logs


def test_full_neppo_cycles_with_ddpl_pm():
    logs = _run_neppo("ddpl")
    branches = [d["debug/branch"] for d in logs]
    assert set(branches) == {0, 1, 2, 3, 4, 5} and branches.count(0) == 2
    pm_logs = [d for d in logs if d["debug/branch"] == 0]
    assert all(d["pm/train/epochs_this_step"] >= 1 and d["pm/train/all_finite"] == 1.0 for d in pm_logs)
    hat = [d for d in logs if d["debug/branch"] == 1]
    assert all("hat_pm/train/q_loss/0" in d for d in hat)
    assert any("pu/train/lse" in d for d in logs)  # the phi update ran


def test_full_neppo_cycle_with_ppo_pm_unchanged():
    logs = _run_neppo("ppo", cycles=1)
    assert {d["debug/branch"] for d in logs} == {0, 1, 2, 3, 4, 5}
    assert any("pm/train/0/loss_objective" in d for d in logs)


REF_CKPT = os.path.join(_root, "checkpoints/MPE_simple_world_comm_v3_ippo_11_11-21_48_00/actor_60000000")


def test_reference_mode_rollout_and_br_policies():
    """pi_ref collects pm/hat_pm rollouts; BR opponents and potdiff use the learned DDPL actor."""
    if not os.path.isdir(REF_CKPT):
        print("skip: no reference checkpoint")
        return
    from marl.algorithms.neppo.neppo import NePPO

    seen = []  # (branch, [policy objects]) per collect
    orig = NePPO.current_policies

    def spy(self):
        pols = orig.fget(self)
        seen.append((self.branch, self, list(pols)))
        return pols

    NePPO.current_policies = property(spy)
    try:
        logs = _run_neppo("ddpl", cycles=1, ddpl_extra={"reference_path": REF_CKPT, "reference_buffer_capacity": 2048})
    finally:
        NePPO.current_policies = orig
    algo = seen[0][1]
    refs = algo.pm.reference_policies
    assert refs is not None and len(refs) == 6
    learned = algo.pm.actors
    assert all(r is not l for r, l in zip(refs, learned))
    for branch, a, pols in seen:
        if branch in (0, 1):  # pm / hat_pm: the fixed pi_ref
            assert all(p is r for p, r in zip(pols, refs)), branch
        elif branch == 2:  # br: learned DDPL opponents + one PPO deviator
            assert sum(p is l for p, l in zip(pols, learned)) == 5
            assert not any(p is r for p in pols for r in refs)
    pm_logs = [d for d in logs if d["debug/branch"] in (0, 1)]
    assert pm_logs and all(d.get("pm/train/all_finite", d.get("hat_pm/train/all_finite")) == 1.0 for d in pm_logs)
    assert any("pu/train/lse" in d for d in logs)
    assert algo.pm.trainer.state.buffer.capacity == 2048  # reference_buffer_capacity applied
    # the learned actor is evaluated after each pm update (pm/* rollout metrics describe pi_ref)
    evals = [d for d in logs if "pm_eval/avg_phi" in d]
    assert len(evals) == 1 and evals[0]["debug/branch"] == 0
    assert "pm_eval/rollout/0/step_rew_mean" in evals[0] and np.isfinite(evals[0]["pm_eval/avg_phi"])
    assert not any("pm_eval/avg_phi" in d for d in _run_neppo("ddpl", cycles=1))  # paper mode: no extra eval


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}", flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}", flush=True)
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
