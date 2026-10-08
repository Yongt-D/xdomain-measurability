# GAPL-SegNet V5 最终状态

最终有效实现为 `gaplsegnet_v5.py` 的 `confidence` 模式：4-shot 真实 support、query-conditioned support selection、有界 prototype confidence gate。

WHU-100、seeds 0/1/2：

| 方法 | Test IoU mean ± std |
|---|---:|
| A 无原型 | 0.856343 ± 0.003593 |
| V2 C 普通原型 | 0.857625 ± 0.003818 |
| **V5 confidence** | **0.862081 ± 0.002342** |

三个种子中 V5 相对对应 A 与 C 的差值均为正。原型因果检查全部通过。完整公式、逐种子结果、失败版本和复现入口见 `V5_CONFIRM_RESULT.md`；论文实验章节的结论更新见项目根目录 `实验第二章_GAPL-SegNet_V5最终补充复盘.md`。

固定环境：`/CSTemp/fishfield/Anaconda3/envs/dinov3_old_dyt/bin/python`。

固定权重：`weights/vgg16_bn-6c64b313.pth`，SHA256 `6c64b3138f2f4fcb3bcc4cafde11619c4f440eb1631787e93a682fd88305888a`。
