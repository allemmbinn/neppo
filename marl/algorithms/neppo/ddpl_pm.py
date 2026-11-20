"""DDPL as NePPO's potmax (pm) solver.

``DDPLPotMax`` is a drop-in for the ``StepTrain`` object NePPO keeps in ``self.pm`` / ``self.hat_pm``.
It implements only what the outer loop touches on a pm solver:

    actors                      list of jaxtrl Policies (DDPL diffusion actors, local observations)
    pm(tds, log_prefix)         one potmax update on a rollout -> (log_dict, True, num_transitions)
    copy_networks(phi=...)      swap the potential (pm gets check_phi, hat_pm gets hat_phi)
    rebuild_modules(), make_params_contiguous()
    copy.deepcopy(pm)           -> clone (hat_pm); replay arrays are shared, everything else copied
    save(path)

Mapping of NePPO's potmax step onto DDPL (Algorithm 2) — see jaxtrl/docs/ddpl_neppo_integration.md:
* common reward r~_t = phi_w(s_t, r_vec_t). The replay stores r_vec and phi's input; rewards are
  recomputed with the solver's CURRENT phi on every sampled minibatch, because phi changes every
  cycle and differs between pm (phi - delta u) and hat_pm (phi + delta u).
* actors condition on their own local observation (same policy class as the PPO BR deviations);
  the per-agent critics Q_i(s, a_i) see the concatenated observations plus a time feature t / T.
* the episode end (time limit) is a TERMINAL transition, matching NePPO's finite-horizon Phi.
* gamma is NePPO's gamma (importance weights scale with eta / (1 - gamma)).

Reference mode (``reference_policies`` given, e.g. a pretrained IPPO checkpoint via ``load_reference_policies``):
* pi_ref is FIXED. It collects every pm / hat_pm rollout (``rollout_policies``) and proposes every DDPL candidate,
  so the whole replay stays on-policy for pi_ref across NePPO cycles; only phi (relabeling) changes.
* the learned actor fits pi_ref * exp(eta/(1-gamma) Q^{pi_ref}_phi). ``actors`` is always that learned actor:
  it is what the BR opponents and the potdiff datasets use.
"""

from __future__ import annotations

import copy
import os

import jax
import numpy as np
import torch as th
from jax import numpy as jnp

try:
    from jaxtrl.marl.ddpl import DDPLConfig, DDPLTrainer, make_decentralized_batch
except ImportError as e:  # the paper pin (adkala/jaxtrl@03ca19b) predates DDPL
    import jaxtrl as _jt

    raise ImportError(
        f"the jaxtrl found at {list(getattr(_jt, '__path__', []))} has no DDPL (jaxtrl.marl.ddpl). Install the fork "
        "pinned in pyproject.toml (pip install -e .), or a local checkout: pip install -e ../jaxtrl"
    ) from e

from marl.algorithms.neppo.mixer import NPConvexNN, NPLambda, NPMixer, NPMixerWithDelta, NPNoMix

# ------------------------------------------------------------------ phi (Torch) -> pure JAX


def _np(t):
    return jnp.asarray(t.detach().cpu().numpy())


def _lin(p, x):
    return x @ p["W"].T + p["b"]


def _linp(m):
    return {"W": _np(m.weight), "b": _np(m.bias) if m.bias is not None else jnp.zeros(m.weight.shape[0])}


def _phi_kind(phi):
    if hasattr(phi, "phi_ensemble"):
        return ("ensemble", _phi_kind(phi.phi_ensemble[0]), len(phi.phi_ensemble), float(phi.alpha))
    for cls, name in ((NPMixerWithDelta, "mixer_delta"), (NPMixer, "mixer"), (NPLambda, "lambda"),
                      (NPNoMix, "nonmix"), (NPConvexNN, "convex")):
        if isinstance(phi, cls):
            return name
    raise NotImplementedError(f"no JAX port of potential {type(phi).__name__}")


def phi_params(phi):
    """Current parameters of a NePPO potential as a JAX PyTree (read each pm step)."""
    if hasattr(phi, "phi_ensemble"):
        return [phi_params(p) for p in phi.phi_ensemble]
    if isinstance(phi, NPMixerWithDelta):
        return {"lin": _linp(phi.linear), "delta": _linp(phi.linear_delta)}
    if isinstance(phi, NPMixer):
        return {"lin": _linp(phi.linear)}
    if isinstance(phi, NPLambda):
        return {"logits": _np(phi.logits)}
    if isinstance(phi, NPNoMix):
        return {"l0": _linp(phi.net[0]), "l1": _linp(phi.net[2])}
    if isinstance(phi, NPConvexNN):
        z = phi.output_layer_z
        return {"inp": _linp(phi.input_layer), "y": _linp(phi.output_layer_y),
                "z": {"W": _np(z.weight), "b": jnp.zeros(z.weight.shape[0])}}  # z.weight is the relu'd weight
    raise NotImplementedError(type(phi).__name__)


def make_phi_fn(kind):
    """Pure JAX phi(params, state [B, d], reward_vec [B, n]) -> [B], matching mixer.py's forward."""
    if isinstance(kind, tuple) and kind[0] == "ensemble":
        member, _, alpha = make_phi_fn(kind[1]), kind[2], kind[3]

        def ens(params, s, r):
            out = jnp.stack([member(p, s, r) for p in params])
            return out.mean(0) - alpha * out.var(0)

        return ens
    if kind == "lambda":
        return lambda p, s, r: r @ jax.nn.softmax(p["logits"])
    if kind == "mixer":
        return lambda p, s, r: jnp.sum(jax.nn.softmax(_lin(p["lin"], s), -1) * r, -1)
    if kind == "mixer_delta":
        return lambda p, s, r: (jnp.sum(jax.nn.softmax(_lin(p["lin"], s), -1) * r, -1)
                                + _lin(p["delta"], s)[..., 0])
    if kind == "nonmix":
        return lambda p, s, r: _lin(p["l1"], jax.nn.gelu(_lin(p["l0"], s), approximate=False))[..., 0]
    if kind == "convex":
        return lambda p, s, r: -(_lin(p["y"], s) + _lin(p["z"], jax.nn.softplus(_lin(p["inp"], s))))[..., 0]
    raise NotImplementedError(kind)


# ------------------------------------------------------------------ rollout (TensorDicts) -> DDPL batch


def _t(x):
    return x.detach().cpu().numpy()


def tds_to_batch(tds, action_bounds, episode_len: int):
    """NePPO's per-agent TensorDicts (one rollout) -> a DDPL TransitionBatch in decentralized mode.

    state  = [o_1, ..., o_n, t / T]          (critic input; t = step index within the episode)
    obs_i  = o_i                              (actor i input)
    terminal = terminated | (time limit reached)  -- episodes are finite-horizon games in NePPO
    reward_vec = per-agent env rewards; phi_input = td["state"] (whatever NePPO's phi consumes)
    """
    n = len(tds)
    T = tds[0].shape[0]
    obs = [_t(td["observation"]).reshape(T, -1) for td in tds]
    nobs = [_t(td["next", "observation"]).reshape(T, -1) for td in tds]
    t = _t(tds[0]["step_count"]).reshape(T, 1).astype(np.float32)
    state = np.concatenate([*obs, t / episode_len], -1)
    next_state = np.concatenate([*nobs, (t + 1) / episode_len], -1)
    term = _t(tds[0]["next", "terminated"]).reshape(T).astype(bool)
    trunc = _t(tds[0]["next", "truncated"]).reshape(T).astype(bool)
    term = term | (trunc & ((t[:, 0] + 1) >= episode_len))
    reward_vec = np.stack([_t(td["next", "reward"]).reshape(T) for td in tds], -1)
    phi_input = _t(tds[0]["state"]).reshape(T, -1)
    actions = [_t(td["action"]).reshape(T, -1) for td in tds]
    return make_decentralized_batch(state, next_state, obs, nobs, actions, term, trunc,
                                    action_bounds=action_bounds, reward_vec=reward_vec, phi_input=phi_input)


# ------------------------------------------------------------------ the pm solver


REFERENCE_ACTOR_KWARGS = {  # the jaxtrl IPPO checkpoints in checkpoints/*_ippo_*
    "net_arch": [64, 64],
    "activation_func": "relu",
    "distribution": "tanh",
    "distribution_kwargs": {"tanh_loc": True},
}


def load_reference_policies(agents, path, key, actor_kwargs=None):
    """One jaxtrl MLPActor per agent from ``{path}/{agent.name}.pth`` (a saved ``trl_actor`` state dict)."""
    from jaxtrl.rl.ppo.actor.mlp import MLPActor

    kw = {**REFERENCE_ACTOR_KWARGS, **(actor_kwargs or {}), "device": "cpu"}
    refs = []
    for a, k in zip(agents, jax.random.split(key, len(agents))):
        f = os.path.join(path, f"{a.name}.pth")
        if not os.path.isfile(f):
            raise FileNotFoundError(f"reference checkpoint has no {a.name}.pth: {f}")
        act = MLPActor(a.pi_obs_spec, a.pi_action_spec, k, **kw)
        act.trl_actor.load_state_dict(th.load(f, map_location="cpu"))
        refs.append(act)
    return refs


class DDPLPotMax:
    def __init__(self, hp, agents, phi, ddpl_kwargs, key, episode_len: int, phi_dim: int, device=None,
                 reference_policies=None, reference_contexts=None):
        """``reference_policies``/``reference_contexts`` (with ``ddpl_kwargs["proposal"] = "reference"``):
        a FIXED jaxtrl Policy per agent (e.g. a pretrained IPPO MLPActor) that both collects the pm rollouts
        and proposes the DDPL candidates. See jaxtrl DDPLTrainer."""
        self.hp = hp
        self.num_actors = len(agents)
        self.episode_len = int(episode_len)
        kw = dict(ddpl_kwargs or {})
        if "gamma" in kw and abs(kw["gamma"] - hp.gamma) > 1e-12:
            raise ValueError(f"DDPL gamma {kw['gamma']} != NePPO gamma {hp.gamma}; they must match")
        kw["gamma"] = hp.gamma
        preset = kw.pop("sampler_preset", "jaxddpm")
        cfg = DDPLConfig.jaxddpm_sampler(**kw) if preset == "jaxddpm" else DDPLConfig(**kw)
        self._kind = _phi_kind(phi)
        state_dim = sum(int(a.pi_obs_spec.shape[0]) for a in agents) + 1
        self.trainer = DDPLTrainer(agents, cfg, key, device=device, state_dim=state_dim,
                                   reward_fn=make_phi_fn(self._kind), phi_dim=int(phi_dim),
                                   reference_policies=reference_policies, reference_contexts=reference_contexts)
        self.action_bounds = list(zip(self.trainer.nets.action_low, self.trainer.nets.action_high))
        self.reference_policies = None if reference_policies is None else list(reference_policies)
        self.phi = None
        self.copy_networks(phi=phi)

    # ---- the StepTrain surface used by NePPO

    @property
    def actors(self):
        """The learned DDPL actors (BR opponents, potdiff datasets, checkpoints)."""
        return self.trainer.policies

    @property
    def rollout_policies(self):
        """Who collects this solver's pm rollouts: pi_ref in reference mode, else the learned actors."""
        return self.reference_policies if self.reference_policies is not None else self.actors

    def copy_networks(self, actors=None, pm_critic=None, br_critics=None, phi=None):
        if actors is not None or pm_critic is not None or br_critics is not None:
            raise ValueError("DDPLPotMax owns its actors and critics; only phi can be copied in")
        if phi is not None:
            if _phi_kind(phi) != self._kind:
                raise ValueError(f"potential type changed from {self._kind} to {_phi_kind(phi)}")
            self.phi = phi if hasattr(phi, "phi_ensemble") else copy.deepcopy(phi)  # as StepTrain

    def rebuild_modules(self):
        pass

    def make_params_contiguous(self):
        if self.phi is not None and not hasattr(self.phi, "phi_ensemble"):
            for p in self.phi.parameters():
                p.data = p.data.contiguous()

    def __deepcopy__(self, memo):
        other = object.__new__(type(self))
        other.hp, other.num_actors, other.episode_len = self.hp, self.num_actors, self.episode_len
        other._kind, other.action_bounds = self._kind, self.action_bounds
        other.reference_policies = self.reference_policies  # fixed, shared
        other.trainer = self.trainer.clone()
        other.phi = self.phi if hasattr(self.phi, "phi_ensemble") else copy.deepcopy(self.phi, memo)
        return other

    def pm(self, tds, log_prefix="pm"):
        batch = tds_to_batch(tds, self.action_bounds, self.episode_len)
        log = self.trainer.step_batch(batch, phi_params(self.phi))
        out = {f"{log_prefix}/train/{k.removeprefix('ddpl/')}": v for k, v in log.items()
               if k.startswith("ddpl/")}
        # phi-relabeled reward of THIS rollout (current phi). In reference mode this rollout is pi_ref's,
        # so these describe pi_ref; the learned actor is measured by evaluate_phi (see mpe_neppo pm_eval/*)
        out |= self._phi_stats(tds, batch, np.asarray(batch.reward), f"{log_prefix}/train")
        return out, True, batch.size

    def evaluate_phi(self, tds, log_prefix="pm_eval"):
        """phi-return stats of an arbitrary rollout under this solver's current phi; no training."""
        batch = tds_to_batch(tds, self.action_bounds, self.episode_len)
        r = np.asarray(self.trainer.reward_fn(phi_params(self.phi), batch.phi_input, batch.reward_vec))
        return self._phi_stats(tds, batch, r, log_prefix)

    def _phi_stats(self, tds, batch, r, prefix):
        out = {f"{prefix}/avg_phi": float(r.mean())}
        num_done = int(np.asarray(batch.terminated | batch.truncated).sum())
        if num_done > 0:
            steps = _t(tds[0]["step_count"]).reshape(-1)
            out[f"{prefix}/avg_sum_phi"] = float(r.sum() / num_done)
            out[f"{prefix}/avg_Phi"] = float((r * self.hp.gamma ** steps).sum() / num_done)
        return out

    def save(self, path, include_replay=False):
        os.makedirs(path, exist_ok=True)
        self.trainer.save(os.path.join(path, "ddpl"), include_replay=include_replay)
