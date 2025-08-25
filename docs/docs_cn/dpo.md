DPO（Direct Preference Optimization），直接偏好优化。

**论文**：https://arxiv.org/abs/2305.18290

**代码**：https://github.com/eric-mitchell/direct-preference-optimization

# 1 DPO原理

## 1.1 引言
PPO（Proximal Policy Optimization）涉及到Reward Model（RM）的训练和偏好训练，尤其在偏好训练时需要同时训练actor模型和critic模型，需要加载reward模型和reference模型进作为辅助，因此PPO训练流程长、需要资源多，实际上操作难度较大。 

在训练奖励模型的过程中，我们就已经在考虑“什么回答是好的，什么回答是不好的”这个问题了。而对齐模型依然是在考虑这个问题 [1]。所以，我们能不能避开奖励模型的训练，直接一步到位训练对齐模型呢？

## 1.2 Policy model的形式最优解

（1）RM和PPO优化目标

首先介绍RM和PPO的优化目标，部分推导来自[2]。RLHF为两阶段训练，涉及Reward Model和PPO：

$\mathbb{E}_{(x,y_{\text{win}},y_{\text{lose}}) \sim \mathcal{D}}$

$$P(y_w≻y_l)=\frac{e^{λ_{y_w}}}{e^{λ_{y_w}}+e^{λ_{y_l}}}$$

```math
L_{rm} = \max_{r_\phi} \left\{ \mathbb{E}_{(x, y_{\text{win}}, y_{\text{lose}}) \sim \mathcal{D}} \left[ \log \sigma \left( r_\phi(x, y_{\text{win}}) - r_\phi(x, y_{\text{lose}}) \right) \right] \right\}
```

其中， $$r_{\phi}$$为Reward Model的打分。


# 3 参考资料
[1] [人人都能看懂的DPO数学原理](https://zhuanlan.zhihu.com/p/721073733) \
[2] [DPO及其衍生算法XX-O](https://zhuanlan.zhihu.com/p/30274484125) \
[3] [DPO和实现代码](https://zhuanlan.zhihu.com/p/715114620) \
[4] [DPO: Direct Preference Optimization 论文解读及代码实践](https://zhuanlan.zhihu.com/p/642569664)
