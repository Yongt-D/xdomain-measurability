# GAPL-SegNet V5 远端实验备份清单

备份来源：`30482:/CSTemp/dyt/GAPLsegnetV2`

备份日期：2026-07-11

## 目录说明

- `checkpoints/`：V5 confidence seeds 0/1/2 的最佳模型；
- `logs/`：V3、V4、V5 plain 失败趋势以及 V5 confidence 三种子完整训练日志；
- `results/`：上述关键实验的 `results.json`；
- `source/`：远端实际运行的 V5 源码、启动脚本和最终报告。

项目预训练权重未在本目录重复保存，使用 `../../weights/vgg16_bn-6c64b313.pth`，SHA256 为 `6c64b3138f2f4fcb3bcc4cafde11619c4f440eb1631787e93a682fd88305888a`。

## 最终模型与结果哈希

| 文件 | SHA256 |
|---|---|
| `checkpoints/v5_confidence_seed0_best.pth` | `8f836b8e0058aa91f70d1b08d74ad08a5a8cf82918780d487add8e50434b3dcd` |
| `checkpoints/v5_confidence_seed1_best.pth` | `119a86e37f5890fbba7b2099e4ad577507dc3ae592230ef3ad408207427c0731` |
| `checkpoints/v5_confidence_seed2_best.pth` | `567697af8640b08efc2040e55d12f24867c021c1301c50ac4049e17653a2a1dc` |
| `results/v5_confidence_seed0.json` | `b88bd212d13dc1b3b6538a0e6e0a4fd432eb207af40cdd941d2c5c09979f11b8` |
| `results/v5_confidence_seed1.json` | `22fe0109dc1cb6d3635f7f17141c67d3b9e459f8904b9849906cbe4354c4a05d` |
| `results/v5_confidence_seed2.json` | `d129197b3098505f9663653a6b6368255c4817b23fe928f83edafc28d553f738` |

## 运行版本哈希

| 文件 | SHA256 |
|---|---|
| `source/gaplsegnet_v5.py` | `12560867f91051e17d361683b76de86d86ac77acd0443543d04943a9e26681a4` |
| `source/run_v5_seed0.sh` | `c13dccb962a5cc02326fc377d4a4d863672f864951db91a3d2e8cc24498faa7a` |
| `source/run_v5_confirm.sh` | `aff6d5db7e3a6ce13eeb671f2363ce5e36b069ebdc29864756081140ed7f7a2d` |
| `source/V5_CONFIRM_RESULT.md` | `c99fc465afd92139262adc8e14aa4b0cf0249d4867a3ae796c4cd3fa1ae3958e` |

## 最终结果摘要

| 方法 | WHU-100 Test IoU mean ± std |
|---|---:|
| A 无原型 | 0.856343 ± 0.003593 |
| V2 C 普通原型 | 0.857625 ± 0.003818 |
| V5 confidence | **0.862081 ± 0.002342** |

三个种子的 V5 结果相对对应 A 和 C 均为正增益。完整分析见 `source/V5_CONFIRM_RESULT.md`。
