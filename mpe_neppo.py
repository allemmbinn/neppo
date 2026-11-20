import os
import time

# JAX (collector, DDPL nets + replay) and Torch (PPO/BR critics, phi) share one GPU. JAX's default grabs 75%
# of device memory up front, which starves Torch; allocate on demand instead. Set it in the shell to override.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np
import torch as th
from jax import numpy as jnp
from jaxtrl.sim.collect import collect_factory
from jaxtrl.sim.policy import PolicyAggregator
from jaxtrl.utils.logger import ExtWandbLogger

from mpe_env import JaxMARLSim, JaxMARLStateAgent, make_jaxmarl_env
from marl.algorithms.neppo.runner import NePPORunner
from marl.utils.collect import tdl, transitions_to_tensordicts
from marl.utils.torchrl import get_rollout_stats, save_actor

DEFAULT_NAME = "neppo"
JOB_TYPE = "neppo"
SEED = None

TIMESTEPS_PER_ENV = 25
NUM_ENVS = 128

CONFIG = {
    "trainer": {
        "algo": "neppo",
        "actor": "mlp",
        "critic": "mlp",
        "algo_kwargs": {
            "pm_zo_steps": 1,
            "br_zo_steps": 1,
            #
            "phi_lr": 0.01,
            "phi_epochs": 1,
            "zeroth_order_delta": 5e-4,
            "phi_max_grad_norm": 10,
            "phi_clip_grad_norm": True,
            #
            "entropy_eps": 0.02,
            "num_epochs": 10,
            "gamma": 0.9851,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2442,
            "critic_coef": 1,
            "lr": 0.0007,
            "loss_critic_type": "smooth_l1",
            # state-dependent potential softmax(W s + b) . r  (eq. 9 of the paper)
            "mixer": True,
            "mixer_input_dim": 20,
            "mixer_output_dim": 3,
            # potmax solver: "ppo" (default, unchanged) or "ddpl" (diffusion policies, see "ddpl" below)
            "pm_solver": "ppo",
        },
        # used only when pm_solver == "ddpl" (marl/algorithms/neppo/ddpl_pm.py); gamma comes from algo_kwargs
        "ddpl": {
            "episode_len": None,  # set from env max_steps in train()
            "jax_device": None,  # None -> JAX default device
            # fixed reference policy pi_ref (dir with {agent.name}.pth, e.g. an IPPO checkpoint). When set, pi_ref
            # collects every pm rollout and proposes every DDPL candidate (proposal="reference"); BR opponents
            # and potdiff still use the learned DDPL actor. None -> paper mode (proposals from theta^{k-1}).
            "reference_path": None,
            "reference_actor_kwargs": None,  # None -> ddpl_pm.REFERENCE_ACTOR_KWARGS (MLP 64x64 relu, tanh)
            # reference mode only: replaces ddpl_kwargs.buffer_capacity. pi_ref is fixed, so old data never goes
            # stale and 1M (paper) is not needed; ~4.8 GB per 1M transitions on world_comm, and hat_pm holds a
            # second copy. None -> keep ddpl_kwargs.buffer_capacity.
            "reference_buffer_capacity": 300_000,
            # reference mode only: after every N-th pm update, one extra rollout with the LEARNED DDPL actors,
            # logged as pm_eval/* (the pm/* rollout metrics describe pi_ref there). 0 disables.
            "eval_learned_every": 1,
            "ddpl_kwargs": {
                "sampler_preset": "jaxddpm",  # [DECISION] paper sampler does not learn (jaxtrl docs/ddpl_usage.md §9)
                "weight_mode": "global_max",  # [DECISION] literal exp(eta/(1-gamma) Q) overflows on MPE
                # weights exp(eta/(1-gamma) Q) ~ exp(67 eta Q): eta=1 is near-argmax over candidates. Tune with
                # pm/train/ess_mean (aim ~20-50% of num_candidates). See jaxtrl docs/ddpl_neppo_decisions.md
                "eta": 0.1,
                "hidden_sizes": [256, 256, 256],
                "batch_size": 256,
                "num_candidates": 16,
                "rollout_batch_size": 20,  # epochs per pm step = rollout size / 20
                "learning_starts": 256,
                "buffer_capacity": 1_000_000,
                "lr_decay_updates": 250_000,
            },
        },
        "actor_kwargs": {
            # one hidden layer, the architecture of the shipped Table 2 weights (pretrained/neppo_mixer_wc)
            "net_arch": [64],
            "activation_func": "relu",
            "distribution": "tanh",
            "distribution_kwargs": {"tanh_loc": True},
        },
        "critic_kwargs": {
            "net_arch": [64],
            "activation_func": "relu",
            "input_key": "state",
        },
    },
    "collect": {
        "num_envs": NUM_ENVS,
    },
    "save": {
        "freq": 1_000_000,
    },
    "env": {
        "name": "MPE_simple_world_comm_v3",
        "max_steps": 25,
    },
    "seed": None,
    "device": "cuda",
}

WANDB_KWARGS = {
    "job_type": JOB_TYPE,
    "project": "neppo",
    "sync_tensorboard": True,
}


# --- Training ---


def train(
    test=False,
    name=None,
    job_type=JOB_TYPE,
    seed=SEED,
    config=CONFIG,
    wandb_kwargs=WANDB_KWARGS,
    device=None,
    gpu_num=0,
    local_ratio=0.5,
    include_actions=False,
):
    if device is not None and not test:
        if device == "cuda":
            device = f"cuda:{gpu_num}"
            jax.config.update("jax_default_device", jax.devices("gpu")[gpu_num])
        config["device"] = device

    env_name = config["env"]["name"]
    name_tag = f"_{name}" if name else ""
    exp_name = f"{'test_' if test else ''}{env_name}_neppo{name_tag}_{time.strftime('%m_%d-%H_%M_%S')}"
    wandb_kwargs["job_type"] = job_type

    ENV_MAX_STEPS = config["env"]["max_steps"]

    if test:
        wandb_kwargs["offline"] = True
        wandb_kwargs["mode"] = "disabled"

        config["collect"]["num_envs"] = 2
        config["device"] = "cpu"

        ENV_MAX_STEPS = 10

    if config["seed"] is not None:
        seed = config["seed"]
    else:
        seed = np.random.randint(2**32 - 1)

    key = jax.random.key(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    log_dir = f".logs/{exp_name}"
    save_dir = f".checkpoints/{exp_name}"
    os.makedirs(log_dir, exist_ok=True)

    # Create JaxMARL environment
    env, agents, rewards, state_agent = make_jaxmarl_env(
        config["env"]["name"], local_ratio=local_ratio, include_actions=include_actions
    )

    config["trainer"]["critic_kwargs"]["input_size_override"] = state_agent._state_dim
    config["trainer"]["algo_kwargs"]["mixer_input_dim"] = state_agent._state_dim
    config["trainer"]["algo_kwargs"]["mixer_output_dim"] = len(agents)

    sim = JaxMARLSim(env, agents, rewards, ENV_MAX_STEPS)

    config["trainer"]["ddpl"]["episode_len"] = ENV_MAX_STEPS
    runner = NePPORunner(agents, config["trainer"], key, device=config["device"])

    # The jitted collector fixes each slot's policy CODE; weights flow through the contexts. With a
    # DDPL pm and PPO BR actors the slot types change between branches, so keep one compiled
    # collector per policy-type signature (N + 1 of them). For pm_solver="ppo" there is exactly one.
    collectors = {}

    # pi_ref is an MLPActor like the BR actors but with its own architecture, so it gets its own slot key
    ref_ids = {id(p) for p in (getattr(runner.algo.pm, "reference_policies", None) or [])}

    def get_collect(policies):
        sig = tuple(("ref", id(p)) if id(p) in ref_ids else type(p) for p in policies)
        if sig not in collectors:
            collectors[sig] = jax.jit(
                collect_factory(
                    sim.reset,
                    sim.step,
                    PolicyAggregator(list(policies)),
                    timesteps_per_env=TIMESTEPS_PER_ENV,
                    num_envs=config["collect"]["num_envs"],
                    extended_transition=True,
                )
            )
        return collectors[sig]

    def prepare_tds(transitions):
        """Per-agent tensordicts with the global state (critic / phi input) added."""
        tds = transitions_to_tensordicts(transitions)

        # Add state for critic
        state_obs, _ = jax.vmap(state_agent.obs_env2pi)(transitions.raw_obs)
        state_next_obs, _ = jax.vmap(state_agent.obs_env2pi)(transitions.raw_next_obs)
        if state_agent.include_actions:
            all_actions = jnp.concatenate(transitions.action, axis=-1)
            state_obs = jnp.concatenate([state_obs, all_actions], axis=-1)

            # Next-step actions for state_next_obs; zeros at episode boundaries
            n_envs = config["collect"]["num_envs"]
            actions_2d = all_actions.reshape(n_envs, TIMESTEPS_PER_ENV, -1)
            next_actions_2d = jnp.concatenate(
                [actions_2d[:, 1:], jnp.zeros_like(actions_2d[:, :1])], axis=1
            )
            done = transitions.terminated | transitions.truncated
            done_2d = done.reshape(n_envs, TIMESTEPS_PER_ENV)
            next_actions_2d = jnp.where(done_2d[..., None], 0.0, next_actions_2d)
            next_all_actions = next_actions_2d.reshape(-1, next_actions_2d.shape[-1])
            state_next_obs = jnp.concatenate(
                [state_next_obs, next_all_actions], axis=-1
            )
        state_obs = tdl(state_obs)
        state_next_obs = tdl(state_next_obs)
        for i in range(len(tds)):
            tds[i]["state"] = state_obs
            tds[i]["next", "state"] = state_next_obs

        return tds

    ddpl_cfg = config["trainer"]["ddpl"]
    ref_eval_every = (
        int(ddpl_cfg.get("eval_learned_every") or 0)
        if config["trainer"]["algo_kwargs"].get("pm_solver", "ppo") == "ddpl" and ddpl_cfg.get("reference_path")
        else 0
    )
    pm_updates = 0

    logger = ExtWandbLogger(
        exp_name=exp_name,
        **{
            **wandb_kwargs,
            "project": "neppo",
            "config": config,
        },
    )

    start_time = time.time()
    total_frames = 0
    next_save_frame = config["save"]["freq"]
    num_env_steps = config.get("num_env_steps", None)

    print("starting loop")
    while True:
        key, sk = jax.random.split(key)
        policy_context = runner.get_policy_contexts(sk)
        collect = get_collect(runner.policies)

        # collect
        key, sk = jax.random.split(key)
        collect_start_time = time.time()
        transitions = collect(sk, policy_context)
        collect_end_time = time.time()

        tds = prepare_tds(transitions)
        total_frames += transitions.step_count.shape[0]

        rewards_cache = {}
        for i in range(len(tds)):
            rewards_cache[i] = tds[i]["next", "reward"]

        # train
        train_start_time = time.time()
        log_dict = runner.step(tds)
        train_end_time = time.time()

        # reference mode: the pm rollout above was pi_ref's; measure the LEARNED potmax policy separately
        if ref_eval_every and log_dict["debug/branch"] == 0:
            pm_updates += 1
            if pm_updates % ref_eval_every == 0:
                eval_start_time = time.time()
                pm = runner.algo.pm
                key, ck, sk = jax.random.split(key, 3)
                eval_ctx = [p.init_context(k) for p, k in zip(pm.actors, jax.random.split(ck, len(pm.actors)))]
                eval_tr = get_collect(pm.actors)(sk, eval_ctx)
                eval_tds = prepare_tds(eval_tr)
                log_dict |= pm.evaluate_phi(eval_tds, "pm_eval")
                for j in range(len(eval_tds)):
                    log_dict |= {
                        f"pm_eval/rollout/{j}/{k}": v
                        for k, v in get_rollout_stats(eval_tds[j], include_len=j == 0).items()
                    }
                log_dict["time/eval_sec"] = time.time() - eval_start_time
                log_dict["time/eval_frames"] = int(eval_tr.step_count.shape[0])  # not in total_frames
                del eval_tr, eval_tds

        # logging
        pre = ""
        match log_dict["debug/branch"]:
            case 0:
                pre = "pm"
            case 1:
                pre = "hat_pm"
            case 2:
                pre = "br"
            case 3:
                pre = "hat_br"
            case 4:
                pre = "pd"
            case 5:
                pre = "hat_pd"

        branch = log_dict["debug/branch"]
        selective_branch = branch in {2, 3, 4, 5}
        if selective_branch:
            agent_idx = log_dict.get(f"{pre}/rollout/agent_idx")
            pre = f"{pre}/t{agent_idx}"
            if agent_idx >= runner.algo.num_actors:
                selective_branch = False

        for j in range(len(tds)):
            tds[j]["next", "reward"] = rewards_cache[j]
            log_dict |= {
                f"{pre}/rollout/{j}/{k}": v
                for k, v in get_rollout_stats(tds[j], include_len=j == 0).items()
            }

        log_dict.update(
            {
                "time/wall_min": (time.time() - start_time) / 60,
                "time/train_step_sec": train_end_time - train_start_time,
                "time/collect_step_sec": collect_end_time - collect_start_time,
                "time/sim_fps": transitions.step_count.shape[0]
                / (collect_end_time - collect_start_time),
                "time/total_frames": total_frames,
            }
        )

        logger.log_scalars(log_dict, timestep=total_frames)

        # save
        if total_frames >= next_save_frame:
            next_save_frame += config["save"]["freq"]
            if config["trainer"]["algo_kwargs"].get("pm_solver", "ppo") == "ddpl":
                runner.algo.pm.save(f"{save_dir}/actor_{total_frames}/pm_ddpl")
                save_actor(runner.algo.phi, f"{save_dir}/actor_{total_frames}/phi.pth")
            else:
                for i, agent in enumerate(agents):
                    save_actor(
                        runner.algo.pm.actors[i].trl_actor,
                        f"{save_dir}/actor_{total_frames}/{agent.name}.pth",
                    )
                    # phi in the zonp class
                    save_actor(
                        runner.algo.phi,
                        f"{save_dir}/actor_{total_frames}/phi.pth",
                    )

                    save_actor(
                        runner.algo.pm.actor_optims,
                        f"{save_dir}/actor_{total_frames}/actor_optims.pth",
                    )

                    save_actor(
                        runner.algo.pm.pm_critic,
                        f"{save_dir}/actor_{total_frames}/pm_critic.pth",
                    )

                    save_actor(
                        runner.algo.pm.pm_critic_optim,
                        f"{save_dir}/actor_{total_frames}/pm_critic_optim.pth",
                    )


        del transitions
        del tds

        if num_env_steps is not None and total_frames >= num_env_steps:
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--local_ratio", type=float, default=0.5)
    parser.add_argument("--include_actions", action="store_true")
    parser.add_argument("--phi_lr", type=float, default=None)
    parser.add_argument("--zeroth_order_delta", type=float, default=None)
    parser.add_argument("--phi_max_grad_norm", type=float, default=None)
    parser.add_argument("--pm_zo_steps", type=int, default=None)
    parser.add_argument("--br_zo_steps", type=int, default=None)
    parser.add_argument("--phi_epochs", type=int, default=None)
    parser.add_argument("--entropy_eps", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--num_env_steps", type=int, default=None)
    parser.add_argument("--pm_solver", type=str, default=None, choices=["ppo", "ddpl"])
    parser.add_argument("--ddpl_reference", type=str, default=None,
                        help="checkpoint dir for a fixed DDPL reference policy pi_ref (implies --pm_solver ddpl)")
    parser.add_argument("--ddpl_eta", type=float, default=None)
    parser.add_argument("--ddpl_buffer_capacity", type=int, default=None,
                        help="DDPL replay capacity (overrides the reference-mode default too)")
    args = parser.parse_args()
    if args.ddpl_reference is not None:
        if args.pm_solver == "ppo":
            parser.error("--ddpl_reference needs --pm_solver ddpl")
        args.pm_solver = "ddpl"
        CONFIG["trainer"]["ddpl"]["reference_path"] = args.ddpl_reference
    if args.ddpl_eta is not None:
        CONFIG["trainer"]["ddpl"]["ddpl_kwargs"]["eta"] = args.ddpl_eta
    if args.ddpl_buffer_capacity is not None:
        CONFIG["trainer"]["ddpl"]["ddpl_kwargs"]["buffer_capacity"] = args.ddpl_buffer_capacity
        CONFIG["trainer"]["ddpl"]["reference_buffer_capacity"] = None
    if args.pm_solver is not None:
        CONFIG["trainer"]["algo_kwargs"]["pm_solver"] = args.pm_solver
    if args.env is not None:
        CONFIG["env"]["name"] = args.env
    if args.num_envs is not None:
        CONFIG["collect"]["num_envs"] = args.num_envs
    if args.num_epochs is not None:
        CONFIG["trainer"]["algo_kwargs"]["num_epochs"] = args.num_epochs
    if args.lr is not None:
        CONFIG["trainer"]["algo_kwargs"]["lr"] = args.lr
    if args.phi_lr is not None:
        CONFIG["trainer"]["algo_kwargs"]["phi_lr"] = args.phi_lr
    if args.zeroth_order_delta is not None:
        CONFIG["trainer"]["algo_kwargs"]["zeroth_order_delta"] = args.zeroth_order_delta
    if args.phi_max_grad_norm is not None:
        CONFIG["trainer"]["algo_kwargs"]["phi_max_grad_norm"] = args.phi_max_grad_norm
    if args.pm_zo_steps is not None:
        CONFIG["trainer"]["algo_kwargs"]["pm_zo_steps"] = args.pm_zo_steps
    if args.br_zo_steps is not None:
        CONFIG["trainer"]["algo_kwargs"]["br_zo_steps"] = args.br_zo_steps
    if args.phi_epochs is not None:
        CONFIG["trainer"]["algo_kwargs"]["phi_epochs"] = args.phi_epochs
    if args.entropy_eps is not None:
        CONFIG["trainer"]["algo_kwargs"]["entropy_eps"] = args.entropy_eps
    if args.gamma is not None:
        CONFIG["trainer"]["algo_kwargs"]["gamma"] = args.gamma
    if args.num_env_steps is not None:
        CONFIG["num_env_steps"] = args.num_env_steps
    train(
        args.test,
        name=args.name,
        device=args.device,
        gpu_num=args.gpu_num,
        local_ratio=args.local_ratio,
        include_actions=args.include_actions,
    )
