"""DDPL potmax with a FIXED pretrained reference policy on MPE simple_world_comm (check, not a NePPO run).

pi_ref = the IPPO checkpoint's MLPActors (one per agent). Every pm rollout is collected with pi_ref, the replay
keeps all of them, and DDPL draws its candidates from pi_ref (jaxtrl proposal="reference", external). The
learned diffusion actors should then approximate pi_ref(a|o) * exp(eta/(1-gamma) Q^{pi_ref}), i.e. one KL-
regularized improvement step on pi_ref, so their phi-return should be >= pi_ref's.

phi = NPLambda with zero logits (the mean of the agents' rewards), NePPO's initial potential.

Usage:
  PYTHONPATH=../jaxtrl/src JAX_PLATFORMS=cpu python ddpl_reference_check.py \
      --ref checkpoints/MPE_simple_world_comm_v3_ippo_11_11-21_48_00/actor_7005000 --eta 0.1 --steps 10
"""

import argparse
import json
import os
import time

import jax
import numpy as np
import torch as th
from jax import numpy as jnp
from jaxtrl.rl.ppo.actor.mlp import MLPActor
from jaxtrl.sim.collect import collect_factory
from jaxtrl.sim.policy import PolicyAggregator

from marl.algorithms.neppo.ddpl_pm import DDPLPotMax
from marl.algorithms.neppo.mixer import NPLambda
from marl.algorithms.neppo.neppo import Hyperparameters
from marl.utils.collect import tdl, transitions_to_tensordicts
from mpe_env import JaxMARLSim, make_jaxmarl_env

ENV = "MPE_simple_world_comm_v3"
T, GAMMA = 25, 0.9851


def load_reference(agents, path, key):
    refs = []
    for a, k in zip(agents, jax.random.split(key, len(agents))):
        act = MLPActor(a.pi_obs_spec, a.pi_action_spec, k, net_arch=[64, 64], activation_func="relu",
                       distribution="tanh", distribution_kwargs={"tanh_loc": True})
        act.trl_actor.load_state_dict(th.load(os.path.join(path, f"{a.name}.pth"), map_location="cpu"))
        refs.append(act)
    return refs


def episode_stats(transitions, phi_w):
    """Undiscounted per-episode phi-return and per-agent env returns (each env runs exactly one 25-step episode)."""
    r = np.stack([np.asarray(x) for x in transitions.reward], -1).reshape(-1, T, len(phi_w))  # [E, T, n]
    phi = (r * phi_w).sum(-1)
    disc = GAMMA ** np.arange(T)
    return {"phi_return": float(phi.sum(1).mean()), "Phi_discounted": float((phi * disc).sum(1).mean()),
            "agent_returns": [float(x) for x in r.sum(1).mean(0)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=10, help="pm steps (one pi_ref rollout + epochs each)")
    ap.add_argument("--num-envs", type=int, default=128)
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--diffusion-steps", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--eval-envs", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    key = jax.random.key(args.seed)
    th.manual_seed(args.seed)
    env, agents, rewards, state_agent = make_jaxmarl_env(ENV, local_ratio=0.0)
    sim = JaxMARLSim(env, agents, rewards, T)
    n = len(agents)
    key, k_ref, k_ctx, k_pm = jax.random.split(key, 4)
    refs = load_reference(agents, args.ref, k_ref)
    ref_pol = PolicyAggregator(refs)
    ref_ctx = ref_pol.init_context(k_ctx)

    phi = NPLambda(output_dim=n)
    phi_w = np.full(n, 1.0 / n)
    hp = Hyperparameters(gamma=GAMMA)
    ddpl_kwargs = dict(weight_mode="global_max", eta=args.eta, hidden_sizes=tuple(args.hidden),
                       batch_size=args.batch, num_candidates=args.candidates, diffusion_steps=args.diffusion_steps,
                       rollout_batch_size=20, learning_starts=args.batch, buffer_capacity=1_000_000,
                       lr_decay_updates=args.steps * args.num_envs * T // 20, proposal="reference")
    pm = DDPLPotMax(hp, agents, phi, ddpl_kwargs, k_pm, episode_len=T, phi_dim=state_agent._state_dim,
                    device="cpu", reference_policies=refs, reference_contexts=ref_ctx)

    def make_collect(policies, num_envs):
        return jax.jit(collect_factory(sim.reset, sim.step, PolicyAggregator(list(policies)), timesteps_per_env=T,
                                       num_envs=num_envs, extended_transition=True))

    collect_ref = make_collect(refs, args.num_envs)
    eval_ref, eval_ddpl = make_collect(refs, args.eval_envs), make_collect(pm.actors, args.eval_envs)
    eval_key = jax.random.key(10_000 + args.seed)

    def evaluate():
        ref = episode_stats(eval_ref(eval_key, ref_ctx), phi_w)
        ddpl = episode_stats(eval_ddpl(eval_key, pm.trainer.get_policy_contexts()), phi_w)
        return {"ref": ref, "ddpl": ddpl}

    curve = [{"pm_step": 0, "epochs": 0, **evaluate()}]
    print(f"step 0: ref phi-return {curve[0]['ref']['phi_return']:.3f}  ddpl {curve[0]['ddpl']['phi_return']:.3f}",
          flush=True)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        key, sk = jax.random.split(key)
        tr = collect_ref(sk, ref_ctx)  # rollouts ALWAYS from pi_ref
        tds = transitions_to_tensordicts(tr)
        s, _ = jax.vmap(state_agent.obs_env2pi)(tr.raw_obs)
        s_next, _ = jax.vmap(state_agent.obs_env2pi)(tr.raw_next_obs)
        for td in tds:
            td["state"], td["next", "state"] = tdl(s), tdl(s_next)
        log, _, _ = pm.pm(tds)
        line = {k.split("/")[-2] + "/" + k.split("/")[-1]: v for k, v in log.items()
                if any(m in k for m in ("q_loss/0", "ess_mean/0", "w_underflow/0", "td_target_mean/0", "buffer_size"))}
        msg = f"step {step}: epochs {int(pm.trainer.state.epoch)} {json.dumps({k: round(v, 3) for k, v in line.items()})}"
        if step % args.eval_every == 0 or step == args.steps:
            curve.append({"pm_step": step, "epochs": int(pm.trainer.state.epoch), "train": line, **evaluate()})
            msg += (f" | phi-return ref {curve[-1]['ref']['phi_return']:.3f} ddpl {curve[-1]['ddpl']['phi_return']:.3f}"
                    f" ({time.time() - t0:.0f}s)")
        print(msg, flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "curve": curve}, f, indent=1)


if __name__ == "__main__":
    main()
