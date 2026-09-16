# 3DA — 3D Adapter (几何空间增强连接器)

本目录存放 **3DA** 的模型定义、MS-Swift 注册文件与训练/推理脚本。

- **`my_qwen3_vggt_xattnv2/` 是主模型（main）**，对应论文与 Kaggle 上发布的 Xiaomi-SpatialEmbodied 权重；
- **`variants/` 下是实验性连接器变体**，用于对比不同 3D-2D 融合设计，不参与正式发布。

```
3DA/
├── README.md                        # 本文件
├── my_qwen3_vggt_xattnv2/           # ★ 主模型（Main）
│   ├── modeling_qwen3_vl_moe_vggt.py
│   ├── configuration_qwen3_vl_moe.py
│   ├── modular_qwen3_vl_moe.py
│   ├── plugin/
│   │   ├── qwen3_vl_moe_vggt_register.py    # MS-Swift 注册（--custom_register_path）
│   │   └── save_qwen3_vl_moe_vggt.py        # 合并 VGGT 权重到 fused checkpoint
│   ├── scripts/                     # 4 阶段 SFT + 推理脚本族
│   └── vggt/                        # VGGT backbone 代码（随主模型打包）
└── variants/                        # 实验性变体（仅模型定义 + 注册 + 单脚本）
    ├── my_qwen3_moe_vl_vggt_xattn_mlp/
    └── my_qwen3_moe_vl_vggt_qformer/
```

## 变体对照

| | **`my_qwen3_vggt_xattnv2`** ★主模型 | `variants/my_qwen3_moe_vl_vggt_xattn_mlp` | `variants/my_qwen3_moe_vl_vggt_qformer` |
|---|---|---|---|
| MS-Swift `model_type` | `qwen3_vl_moe_vggt` | `qwen3_vl_moe_vggt_xattn_mlp` | `qwen3_vl_moe_vggt_qformer` |
| 基座 | Qwen3-VL-30B-A3B (MoE) | 同左 | 同左 |
| **主连接器** | `VGGTEmbeddingMerger`：VGGT patch grid(14) 双线性插值到 Qwen3 grid(16) → token merge + RMSNorm + MLP 投影 → **cross-attention 残差融合**（Q=图像 token，K/V=VGGT 3D token） | 同族 merger，但 **`fusion_method` 可配**：`add` / `concat` / `gated` / `weighted` / `cross_attention`（默认 `cross_attention`） | `VGGTQFormerConnector`：language-initialized **Q-Former** 从 VGGT 空间特征中选出 **K=8** 个几何 token，**concat** 在原生视觉 token 之前 |
| **deepstack 多连接器** | ✗ 无（Qwen3 自身 deepstack 特征原样透传，VGGT 只注入 image token 流） | ✓ `deepstack_connectors`，每层一个 `VGGTEmbeddingMerger`，`deepstack_fusion_method` 默认 `add` | ✓ `deepstack_connectors`，每层一个轻量 MLP `VGGTDeepstackConnector`，融合默认 `add` |
| VGGT 中间层索引 | — | `[5, 11, 17]`（对应 Qwen3 `deepstack_visual_indexes=[8,16,24]`） | `[5, 11, 17]` |
| 训练脚本 | `scripts/` 共 20+ 个（`sft_ddp_s1~s4`、`CL1~3`、`planning` 等 4 阶段脚本族） | `scripts/sft_full.sh`（单文件 demo） | `scripts/sft_full.sh`（单文件 demo） |
| 随目录打包 `vggt/` | ✓ | ✗ | ✗ |
| 定位 | **正式发布 / 论文主结果** | 融合方式消融实验 | Q-Former 连接器消融实验 |

> 三个目录下的 `modeling_qwen3_vl_moe_vggt.py` 文件名相同、**内容不同**，且注册的 `model_type` 各不相同。同一次 `swift` 运行只能通过 `--custom_register_path` 挂载**一个**变体，不要在同一进程里混用。
>
> **为什么混用很危险**：三份 register 都靠「把自己目录插到 `sys.path[0]`」+ 裸名 `from modeling_qwen3_vl_moe_vggt import ...` 来定位模型类。Python 的 `sys.modules` 会缓存第一个导入的模块，**之后再调 `sys.path` 顺序也没用**。而 `--custom_register_path` 接受的是**列表**，逐个加载进同一进程——所以混挂是合法的 CLI 用法，不是笔误。
>
> 后果分两种：主模型 register 先加载、再加载 qformer 时，qformer 拿到的是主模型的类，其 `vggt_connector_config` 里的 `num_query_tokens` 等键被**静默忽略**，于是「model_type 叫 `qwen3_vl_moe_vggt_qformer`、实际构造的却是主模型架构」，无任何报错。反方向（xattn_mlp ↔ 主模型）因为双方配置类字段不同，会 `TypeError: unexpected keyword argument`，属于响亮失败。
>
> 结论：**一个进程只挂一个变体**。若确实需要在单进程内切换，必须先 `sys.modules.pop("modeling_qwen3_vl_moe_vggt", None)` 并清理 `sys.path_importer_cache`，仅调整 `sys.path` 顺序是无效的。

> ⚠️ 两个变体目录**只有** `modeling_qwen3_vl_moe_vggt.py`（配置类由它自带）；主模型目录另有 `configuration_qwen3_vl_moe.py` 与 `modular_qwen3_vl_moe.py`。变体 plugin 会把变体目录插到 `sys.path[0]`，所以不要在变体目录里放同名的空模块。

## 运行

三个变体的调用方式一致，只是 `model_type` 与插件路径不同（以 `variants/my_qwen3_moe_vl_vggt_xattn_mlp` 为例）：

```bash
swift sft \
  --model /path/to/Qwen3-VL-30B-A3B-Instruct \
  --model_type qwen3_vl_moe_vggt_xattn_mlp \
  --custom_register_path xiaomi_spatialembodied/3DA/variants/my_qwen3_moe_vl_vggt_xattn_mlp/plugin/qwen3_vl_moe_vggt_register.py \
  ...
```

推理入口见 `xiaomi_spatialembodied/infer_demo.py`（默认指向主模型 `my_qwen3_vggt_xattnv2`）。

### VGGT 路径解析

注册文件里有一行默认值：

```python
_DEFAULT_VGGT_REPO_DIR = str(Path(__file__).resolve().parents[3] / "vggt")
```

即按 `plugin → 模型目录 → 上一级 → 上两级` 回溯到 `vggt/`。在本仓库布局下：

| 模型 | `parents[3]` 解析结果 | 是否存在 |
|---|---|---|
| `my_qwen3_vggt_xattnv2/plugin/` | `xiaomi_spatialembodied/vggt` | ✗ |
| `variants/<变体>/plugin/` | `xiaomi_spatialembodied/3DA/vggt` | ✗ |

同时 `modeling_qwen3_vl_moe_vggt.py` 里的 `_DEFAULT_VGGT_REPO_DIR` 是 `模型目录/vggt`——**只有主模型目录下真的有 `vggt/`**，两个变体没有。

因此**统一用环境变量覆盖**，这也是原部署（ms-swift `examples/custom/...` 布局）下的实际做法：

```bash
export VGGT_REPO_DIR=/path/to/vggt-main      # VGGT pip 包源码根（含 vggt/ 包）
export VGGT_DTYPE=bfloat16                   # 可选：VGGT 精度
export VGGT_DEVICE=cuda:0                    # 可选：VGGT 放置设备
```

若希望变体目录自包含，可在变体目录内建软链接复用主模型的 VGGT 代码：

```bash
ln -s ../../my_qwen3_vggt_xattnv2/vggt \
      xiaomi_spatialembodied/3DA/variants/my_qwen3_moe_vl_vggt_xattn_mlp/vggt
```

## VGGT / Qwen grid alignment

连接器把 VGGT 的 patch grid 重采样到 Qwen3 的图像 token 网格上，因此 **VGGT 输入如何取景**直接决定 3D token 与 2D token 是否指向同一物理区域。两个开关控制这件事：

| 开关 | 取值 | 默认 | 含义 |
|---|---|---|---|
| `letterbox` | `True` / `False` | **`True`** | `True`：VGGT 流按长宽比缩放到长边 518 后，**补白成 518×518 正方形**（VGGT 官方 `load_fn.py` 的 `mode="pad"`）。`False`：只做保长宽比的 resize，保留真实网格。 |
| `interp_mode` | `bilinear` / `nearest-exact` / `nearest` / `bicubic` | **`bilinear`** | VGGT grid → Qwen grid 的重采样方式。 |

设置优先级：`vggt_connector_config`（随 checkpoint 的 `config.json` 持久化） > 环境变量 `VGGT_LETTERBOX` / `VGGT_INTERP_MODE` > 默认值。

```bash
# 默认无需设置；仅重训时按需覆盖
export VGGT_LETTERBOX=0
export VGGT_INTERP_MODE=nearest-exact
```

> ⚠️ **默认值就是已发布权重的训练配置，不要随意改动。** 改任意一项都会让按另一配置训练出的权重失效。
>
> 主模型（已发布）用的是 `letterbox=True` + `bilinear`：VGGT 画布被补成正方形，而 Qwen 侧保持原图长宽比、无 padding，两者只在**图像中心附近**对齐。以 16:9 图像为例，Qwen 视觉 token 的上下各约 21.6% 会融合到白边区域派生的 3D 特征。
> 两个**未发布**的消融变体已改用 `letterbox=False`（xattn_mlp）与 `nearest-exact`（两者），所以 `model_tools/alignment_debug/` 里那份"对齐良好"的分析结论**只对变体配置成立，不适用于已发布的主模型**。

## 其他说明

- **输入图片尺寸约束**：构造 VGGT 流时用 `do_resize=False` 调用 image processor，要求图片宽高**都能被 `patch_size × merge_size`（Qwen3-VL 下为 32）整除**，否则 processor 会抛出难以理解的 `shape ... invalid for input of size ...`。三个 register 现已做**前置检查**并给出可操作报错。ms-swift 的 `fetch_image` 会自动满足这一点；自己构造输入时请先用 `qwen_vl_utils.smart_resize` 处理。仓库自带的 `demo/sample.jpg` 已做成 1600×896 以满足该约束。
- **`SWIFT_GROUP_BY_DATA_TYPE` 只对两个变体生效**：该环境变量由 `variants/*/plugin/*_register.py` 在模块顶层读取并 `patch_dataloader_mixin()`；主模型 register **没有**这个分支，export 它对主模型训练无任何影响。变体的 `scripts/sft_full.sh` 默认把它置为 `1`。
- 两个变体目录下的 `configuration_qwen3_vl_moe.py` / `modular_qwen3_vl_moe.py` 曾是 0 字节空文件，已移除；配置类实际由各自的 `modeling_*.py` 自带。
- `vggt/` 下已补齐 `__init__.py`，使其成为常规包而非 namespace package——否则环境里若 pip 装过 `vggt`，site-packages 的版本会**静默优先**于这份 vendored 代码。
