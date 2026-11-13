This is a revision of an existing tutorial on reinforcement learning (the --doc
seed), with the GRPO video as an additional seed. The existing tutorial is the
backbone: keep all of its content and its overall arc from foundations to
reinforcement learning for language models.

SECTIONS 1-5 (Introduction; Foundations; Core Concepts: Policies, Value
Functions, and the Bellman Equations; Dynamic Programming; Model-Free Prediction
and Control) are in good shape. Keep their content, structure, and level of
detail; fact-check them and fix only what is wrong or unclear. Do not shorten
them, and do not expand them much either.

SECTIONS 6-11 (Deep Value-Based RL; Policy Gradient Methods; Advanced Policy
Optimization; RL for Large Language Models; Offline RL; Deep Actor-Critic for
Continuous Control) are the focus of this revision. They currently read like a
collection of methods: rushed, with ideas introduced without enough motivation
or derivation. Rewrite them into a guided progression a learner can follow.

1. Storyline. Each method appears as the answer to a concrete limitation of
   what the reader has just learned (e.g. why tabular Q-learning fails at scale
   -> function approximation -> why naive deep Q-learning is unstable -> replay
   and target networks; why value-based methods struggle with large or
   continuous action spaces -> policy gradients -> variance -> baselines and
   actor-critic -> unstable step sizes -> trust regions -> PPO -> applying PPO to
   language models and its costs -> GRPO and related methods). State the
   limitation precisely before introducing the fix.

2. Depth by role. The main line above gets the full treatment: the problem,
   the intuition, the step-by-step derivation of its objective or update, the
   algorithm as it is run in practice, and — where it matters for the story —
   a cost analysis and a small worked example. The main line is: DQN; the
   policy gradient and REINFORCE; baselines, actor-critic, and GAE; TRPO; PPO;
   PPO-based RLHF; GRPO; DPO.
   Variants and refinements (e.g. Double/Dueling DQN, prioritized replay,
   distributional RL, Rainbow, A2C/A3C, Dr. GRPO, RLOO, DAPO, IPO/KTO, DDPG,
   TD3, SAC, the offline-RL methods) get a compact but complete treatment: the
   problem of the base method they address, the idea, the key equation and
   what changes, how it is used in practice, and the trade-off. Group related
   variants and compare them in a table where that helps.

3. Algorithms in practice. For main-line methods, give the algorithm as it is
   actually run: how data is collected (environment steps, rollouts, or prompts
   and the number of sampled completions per prompt), what is stored, how
   batches and minibatches are formed, how advantages or targets are computed,
   the loss terms, which networks or models are updated and how (policy,
   critic, target networks, reference and reward models), epochs per batch, and
   typical hyperparameters. For variants, show only what changes in the
   procedure.

4. Cost where it drives the story: e.g. the four models of PPO-based RLHF
   versus the critic-free GRPO, memory per parameter, how cost scales with
   batch size, sequence length, and group size, and on- versus off-policy data
   reuse.

5. Theorems (convergence, improvement bounds) are stated with their intuition
   and cited, not proved.
