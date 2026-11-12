This is a revision of an existing tutorial on reinforcement learning (the --doc
seed), with the GRPO video as an additional seed. The existing tutorial is the
backbone: keep all of its content and its overall arc from foundations to
reinforcement learning for language models.

SECTIONS 1-5 (Introduction; Foundations; Core Concepts: Policies, Value
Functions, and the Bellman Equations; Dynamic Programming; Model-Free Prediction
and Control) are in good shape. Keep their content, structure, and level of
detail; fact-check them and fix only what is wrong or unclear. Do not shorten
them.

SECTIONS 6-11 (Deep Value-Based RL; Policy Gradient Methods; Advanced Policy
Optimization; RL for Large Language Models; Offline RL; Deep Actor-Critic for
Continuous Control) are the focus of this revision. They currently read like a
collection of methods: rushed, with ideas introduced without enough motivation
or derivation. Rewrite them into a guided progression a learner can follow:

1. Storyline. Each method must appear as the answer to a concrete limitation of
   what the reader has just learned (e.g. why tabular Q-learning fails at scale
   -> function approximation -> why naive deep Q-learning is unstable -> replay
   and target networks; why value-based methods struggle with large or
   continuous action spaces -> policy gradients -> variance -> baselines and
   actor-critic -> unstable step sizes -> trust regions -> PPO -> applying PPO to
   language models and its costs -> GRPO and related methods). State the
   limitation precisely before introducing the fix. Split heavy topics into
   several sections rather than compressing them.

2. For every method, in this order:
   a. The problem it solves, set up precisely, and why the previous method fails.
   b. The intuition behind the solution.
   c. The full step-by-step derivation of its objective, gradient, or update
      rule, with every symbol defined and no skipped steps.
   d. The algorithm exactly as it is run in practice, as pseudocode in an
      algorithm environment: how data is collected (environment steps, rollouts,
      episodes, or prompts and the number of sampled completions per prompt),
      what is stored (replay buffer, trajectory batch), how batches and
      minibatches are formed, how advantages, returns, or targets are computed,
      every loss term, which networks or models are updated and how (policy,
      value/critic, target networks and their update rule, reference model,
      reward model), the number of epochs per batch, and typical hyperparameter
      values.
   e. A complexity analysis: compute per update (forward/backward passes, how
      many networks or models are run and kept in memory — e.g. policy,
      reference, reward, and value models for PPO-based RLHF versus GRPO), memory
      footprint, how cost scales with batch size, sequence length, and group
      size, and sample efficiency (on-policy versus off-policy data reuse).
   f. A small worked numerical example where it makes the mechanism tangible
      (e.g. computing advantages, the clipped objective, or group-normalized
      rewards for a toy batch).
   g. A comparison with the previous method: what is gained and what it costs.

3. Keep every method and result currently in sections 6-11 and everything the
   video teaches about GRPO; add the intermediate steps and connecting
   methods the progression needs.
