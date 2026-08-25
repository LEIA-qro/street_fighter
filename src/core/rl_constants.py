# rl_constants.py
#
# Single source of truth for constants that must stay bit-identical across
# modules developed independently of each other (env reward shaping vs.
# agent hyperparameters). No environment, no SB3, no side effects -- safe
# for both envs/ and agents/ to import without creating an envs -> agents
# (or agents -> envs) dependency.

# The RL discount factor. envs/reward.py's RewardConfig.gamma and
# agents/ppo/hyperparams.py's build_ppo_kwargs(...) both default to this.
#
# Why they must match: the distance-shaping term in reward.py is
# potential-based shaping, F(s, s') = gamma * Phi(s') - Phi(s), which is
# policy-invariant (Ng, Harada & Russell, "Policy invariance under reward
# transformations", ICML 1999) ONLY when gamma equals the discount of the
# agent that acts on the shaped reward. If the two literals drift apart, the
# shaping term stops telescoping to zero over a closed loop of states and
# becomes a real per-step reward -- see
# code_testing/pytest/test_cross_module_contracts.py for the regression
# test and src/envs/reward.py's module docstring for the shaping form.
AGENT_GAMMA = 0.995
