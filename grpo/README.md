# GRPO

> [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)

GRPO (Group Relative Policy Optimization) abandons the critic model in PPO, estimates the baseline from group rollout to calculate the advantage, adds KL divergence regularization directly to the loss, significantly reduces the memory and computational overhead of training.

