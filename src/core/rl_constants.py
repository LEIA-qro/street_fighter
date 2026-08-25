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

# --------------------------------------------------------------------------
# KNOWN CONFLICT -- documented, not fixed, by design (follow-up work):
#
# agents/ppo/optuna_study.py, agents/dqn/optuna_study.py and
# agents/sac/optuna_study.py all tune the discount via
# `trial.suggest_float("gamma", 0.95, 0.9999)` and construct their model
# with that trial value, while envs/reward.py's RewardConfig.gamma stays
# pinned at AGENT_GAMMA (0.995) for every trial. That is up to a +/-0.045
# mismatch between the acting agent's discount and the shaping potential's
# gamma.
#
# Per envs/reward.py's docstring, the distance-shaping term
# F(s, s') = gamma * Phi(s') - Phi(s) is policy-invariant (Ng, Harada &
# Russell, ICML 1999) ONLY when gamma equals the acting agent's discount.
# A 0.045 mismatch means a residual of up to 0.045 * Phi(s') per step; at
# spacing_weight=2.5 and peak_dist=70, Phi peaks at 2.5, so the residual
# reaches +/-0.1125/step, i.e. roughly +/-64 over a measured ~570-step
# round. That dominates every other reward term (win_bonus=65,
# loss_penalty=50, time_penalty*570=1.14) and makes trial scores
# INCOMPARABLE across the gamma search dimension -- a trial that happens to
# sample a high gamma gets a large, spurious stall-reward boost independent
# of policy quality.
#
# Net effect: gamma tuning results from all three optuna_study.py modules
# are currently untrustworthy and should not be used to pick a production
# gamma until this is fixed.
#
# The correct fix is to thread each trial's sampled gamma into
# RewardConfig(gamma=trial_gamma) at the same call site that builds the
# model, so the shaping potential and the acting agent always share one
# value even while gamma is being searched. That is a bigger change than a
# constant swap (RewardConfig is currently constructed from its default
# everywhere -- see SFv2_make_env) and is intentionally NOT done as part of
# this fix; it is being handed to the human as follow-up work.
# --------------------------------------------------------------------------
