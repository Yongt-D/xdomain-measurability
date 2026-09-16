# GAPL-SegNet V5 三种子确认结果

## 验收结论

V5 confidence 已满足目标：原型分支具有直接、可验证的因果作用，并在 WHU-100、seeds 0/1/2 上取得稳定正增益。三个种子相对各自 A 基线和 V2 C 原型方案均为正增益，不是由 seed0 单个异常值驱动。

## 单种子结果

| Seed | Best epoch | Best Val IoU | Test IoU | Test F1 | Last Train IoU | Last Val IoU |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 109 | 0.864756 | 0.865368 | 0.927825 | 0.945445 | 0.851728 |
| 1 | 78 | 0.882617 | 0.860085 | 0.924780 | 0.949449 | 0.868677 |
| 2 | 93 | 0.879317 | 0.860791 | 0.925188 | 0.945975 | 0.866408 |

## 配对比较与统计

| Seed | A Test IoU | V2 C Test IoU | V5 Test IoU | V5−A | V5−C |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.859903 | 0.862963 | 0.865368 | +0.005464 | +0.002405 |
| 1 | 0.857702 | 0.855663 | 0.860085 | +0.002383 | +0.004421 |
| 2 | 0.851423 | 0.854249 | 0.860791 | +0.009368 | +0.006542 |

使用总体标准差 `numpy.std(ddof=0)`：

| 方法 | Test IoU mean ± std |
|---|---:|
| A：无原型基线 | 0.856343 ± 0.003593 |
| V2 C：普通单 support 原型 | 0.857625 ± 0.003818 |
| **V5 confidence** | **0.862081 ± 0.002342** |

V5 相对 A 平均提升 `+0.005738`，相对 V2 C 平均提升 `+0.004456`，且标准差更低。

## 核心公式

每个真实 support 的前景、背景原型由其 mask 池化。使用 query 与各 support 的全局语义相似度选择 support：

$$
a_k=\frac{\exp(\cos(\operatorname{GAP}(F_q),\operatorname{GAP}(F_s^{(k)}))/\tau_s)}{\sum_j\exp(\cos(\operatorname{GAP}(F_q),\operatorname{GAP}(F_s^{(j)}))/\tau_s)},
\qquad p_c=\sum_ka_kp_c^{(k)}.
$$

原型 logits 与可靠性门控为

$$
Z_p(i,c)=\frac{\cos(F_q(i),p_c)}{\tau_p},
$$

$$
G=0.25+0.75\tanh\left(\frac{|Z_p|}{2}\right),
\qquad Z=Z_{base}+\alpha_p(G\odot Z_p).
$$

query 只提供图像特征，不提供标签或 pseudo-mask。

## 因果证据

1. support mask 改变时 query logits 随之改变；
2. prototype temperature 和 support selector temperature 获得非零梯度；
3. support attention 权重和为 1；
4. 关闭 prototype 后最终输出发生可重复变化；
5. 最终 logits 显式包含 prototype logits，因此 $\partial Z/\partial p_c\ne0$；
6. 固定输入重复推理一致。

## 失败版本与最终机制

V2 的 query pseudo-prototype 闭环、V3 的稳定化闭环、V4 的 4-shot 等权聚合以及 V5 plain 均未取得稳定增益。真正有效的组合是：真实 support 监督、与 query 匹配的 support 选择、显式进入 logits 的原型通路，以及有界可靠性门控。

## 可复现入口

- 实现：`gaplsegnet_v5.py`
- seed0：`run_v5_seed0.sh confidence 1`
- seed1/2：`run_v5_confirm.sh <seed> <gpu>`
- 环境：`/CSTemp/fishfield/Anaconda3/envs/dinov3_old_dyt/bin/python`
- 权重：`weights/vgg16_bn-6c64b313.pth`
- SHA256：`6c64b3138f2f4fcb3bcc4cafde11619c4f440eb1631787e93a682fd88305888a`
- 结果：`outputs/v5_whu100_C_confidence_seed{0,1,2}`
