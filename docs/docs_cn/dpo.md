DPO（Direct Preference Optimization），直接偏好优化。

**论文**：https://arxiv.org/abs/2305.18290

**代码**：https://github.com/eric-mitchell/direct-preference-optimization

# 1 DPO原理

## 1.1 引言
PPO（Proximal Policy Optimization）涉及到Reward Model（RM）的训练和偏好训练，尤其在偏好训练时需要同时训练actor模型和critic模型，需要加载reward模型和reference模型进作为辅助，因此PPO训练流程长、需要资源多，实际上操作难度较大。
在训练奖励模型的过程中，我们就已经在考虑“什么回答是好的，什么回答是不好的”这个问题了。而对齐模型依然是在考虑这个问题 [1]。所以，我们能不能避开奖励模型的训练，直接一步到位训练对齐模型呢？


# 3 参考资料
[1] 人人都能看懂的DPO数学原理 \
[2] DPO及其衍生算法XX-O
[3] DPO和实现代码
[4] DPO: Direct Preference Optimization 论文解读及代码实践
