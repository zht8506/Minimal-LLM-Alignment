# GRPO

Simple GRPO (Group Relative Policy Optimization) Training for LLM.

Key differences from PPO:
  - No critic / value model — advantages come from group-relative reward normalization.
  - Each prompt generates G responses (a "group"); rewards are normalized within the group.
  - Policy update uses PPO-clip objective + KL penalty with a frozen reference model.
