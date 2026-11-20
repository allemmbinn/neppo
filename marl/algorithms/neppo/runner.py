import jax
import torch as th

#
from jaxtrl.rl.ppo.trainer import ACTOR_MAP as PPO_ACTOR_MAP
from jaxtrl.rl.ppo.trainer import CRITIC_FACTORY_MAP as PPO_CRITIC_FACTORY_MAP

from marl.algorithms.neppo.mixer import (
    NPConvexNN,
    NPLambda,
    NPMixer,
    NPMixerWithDelta,
    NPNoMix,
)
from marl.algorithms.neppo.neppo import Hyperparameters, NePPO


class NePPORunner:
    def __init__(self, agents, config, key, device=None):
        if device is not None:
            pass
        elif "device" in config:
            device = config["device"]
        else:
            device = "cuda" if th.cuda.is_available() else "cpu"

        algo_kwargs = config["algo_kwargs"]
        actor_kwargs = config["actor_kwargs"]
        critic_kwargs = config["critic_kwargs"]

        keys = jax.random.split(key, len(agents) + 1)
        actors = []
        for agent, key in zip(agents, keys[:-1]):
            actor = PPO_ACTOR_MAP[config["actor"]](
                agent.pi_obs_spec,
                agent.pi_action_spec,
                key,
                **{**actor_kwargs, "device": device},
            )
            actors.append(actor)

        PLACEHOLDER = agents[0].pi_obs_spec

        pm_critic = PPO_CRITIC_FACTORY_MAP[config["critic"]](
            PLACEHOLDER,
            None,
            **{
                **critic_kwargs,
                "device": device,
            },
        )

        br_critics = []
        for agent in agents:
            br_critic = PPO_CRITIC_FACTORY_MAP[config["critic"]](
                PLACEHOLDER,
                agent.pi_action_spec,
                **{
                    **critic_kwargs,
                    "device": device,
                },
            )
            br_critics.append(br_critic)

        if algo_kwargs.get("convex_nn", False):
            phi = NPConvexNN(
                input_dim=algo_kwargs["mixer_input_dim"],
                output_dim=algo_kwargs["mixer_output_dim"],
                hidden_size=algo_kwargs.get("convex_nn_hidden_size", 16),
            )
        elif algo_kwargs.get("nonmix", False):
            phi = NPNoMix(
                input_dim=algo_kwargs["mixer_input_dim"],
                hidden_size=algo_kwargs.get("nonmix_hidden_size", 64),
            )
        elif algo_kwargs["mixer"]:
            if algo_kwargs.get("mixer_with_delta", False):
                phi = NPMixerWithDelta(
                    input_dim=algo_kwargs["mixer_input_dim"],
                    output_dim=algo_kwargs["mixer_output_dim"],
                )
            else:
                phi = NPMixer(
                    input_dim=algo_kwargs["mixer_input_dim"],
                    output_dim=algo_kwargs["mixer_output_dim"],
                )
        else:
            phi = NPLambda(
                output_dim=algo_kwargs["mixer_output_dim"],
            )

        phi = phi.to(device)

        pm_step = None
        if algo_kwargs.get("pm_solver", "ppo") == "ddpl":
            from marl.algorithms.neppo.ddpl_pm import DDPLPotMax, load_reference_policies

            ddpl = config["ddpl"]
            ddpl_kwargs = dict(ddpl.get("ddpl_kwargs", {}))
            pm_key, ref_key = jax.random.split(keys[-1])
            refs = ref_ctx = None
            if ddpl.get("reference_path"):
                # fixed pi_ref (e.g. an IPPO checkpoint): collects the pm rollouts and proposes the candidates
                if ddpl_kwargs.setdefault("proposal", "reference") != "reference":
                    raise ValueError("ddpl.reference_path needs ddpl_kwargs.proposal='reference'")
                refs = load_reference_policies(
                    agents, ddpl["reference_path"], ref_key, ddpl.get("reference_actor_kwargs")
                )
                ref_ctx = [r.init_context(None) for r in refs]
                if ddpl.get("reference_buffer_capacity"):  # fixed pi_ref: a smaller replay suffices
                    ddpl_kwargs["buffer_capacity"] = int(ddpl["reference_buffer_capacity"])
            pm_step = DDPLPotMax(
                Hyperparameters(**algo_kwargs),
                agents,
                phi,
                ddpl_kwargs,
                pm_key,
                episode_len=ddpl["episode_len"],
                phi_dim=algo_kwargs["mixer_input_dim"],
                device=ddpl.get("jax_device"),
                reference_policies=refs,
                reference_contexts=ref_ctx,
            )

        self.algo = NePPO(
            actors, pm_critic, br_critics, phi, device, pm_step=pm_step, **algo_kwargs
        )

    def step(self, tds):
        return self.algo.step(tds)

    @property
    def policies(self):
        return self.algo.current_policies

    @property
    def num_frames(self):
        return self.algo.num_frames

    def get_policy_contexts(self, key):
        return [
            policy.init_context(key)
            for policy, key in zip(
                self.policies, jax.random.split(key, len(self.policies))
            )
        ]
        return [
            policy.init_context(key)
            for policy, key in zip(
                self.policies, jax.random.split(key, len(self.policies))
            )
        ]
