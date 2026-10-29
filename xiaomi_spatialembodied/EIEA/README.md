# EIEA — Efficient Image-Embodied Adapter（推理侧）

本目录提供：

| 内容 | 文件 |
|---|---|
| 一份**预计算的 TOR 嵌入缓存** | `tor_embeds_cache_fb.pt`（默认加载）、`tor_embeds_cache.pt`（另一份，内容不同） |
| 加载该缓存并注入 MLLM 的推理代码 | `models/qwen3_vl_vggt_register.py`、`models/qwen3_vl_moe_vggt.py` |
| 示例入口 | `infer_with_EIEA_3D.py` |
| EIEA 参考编码器 / 训练 / 评测 | `cue_encoder.py`、`corruption.py`、`train_eiea_30b.py`、`train_eiea_box.py`、`eval_eiea_30b.py` |
| 随机掩码 + 加噪训练文档 | `docs/EIEA_TRAINING.md` |
| 架构与序列顺序说明 | `docs/EIEA_ARCHITECTURE.md` |
| CPU 单元测试 | `tests/test_eiea.py` |

> **缓存与 30B 维度不匹配的说明**：仓库自带的 `tor_embeds_cache*.pt` 是
> **10×3584**（早期 Qwen2.5-VL-7B 原型），而 Qwen3-VL-30B-A3B 的词嵌入是
> **2048** 维。注册文件现在会检测宽度：不匹配时打印 WARNING 并跳过自动注入
> （仍可正常跑 3DA 推理），设置 `XEMB_EIEA_STRICT_CACHE=1` 可改为直接报错。
> 需要真实 EIEA 注入时，请用 `XEMB_TOR_CACHE` 指向与 30B 匹配的
> `10×2048` 缓存。
>
> **训练代码范围**：本目录新增的是论文 Stage-4 EIEA 的**参考训练/评测实现**
> （box + BEV map + depth，随机模态掩码、部分线索、box 抖动、BEV/depth 加噪）。
> 论文 Stage-4 产出的 EIEA 权重仍不在本次发布范围内；自带缓存仍是固定软提示，
> 不是按图实时生成的物理先验。

## 快速开始

```bash
# 先在仓库根目录构建一个含 VGGT 权重的融合检查点（见 ../3DA/README.md），
# 或把 CUSTOM_MODEL_CFG["model_path"] 指向 Kaggle 下载的已融合权重。
export VGGT_REPO_DIR=$PWD/xiaomi_spatialembodied/3DA/my_qwen3_vggt_xattnv2
export VGGT_DEVICE=cuda:0
export VGGT_DTYPE=bfloat16

python xiaomi_spatialembodied/EIEA/infer_with_EIEA_3D.py
```

默认测试图 `demo/nuscenes_0578_CAM_BACK.jpg` **未随仓库发布**，需要自行放置，
或把 `INFER_CFG["image_paths"]` 改指到已有图片。

## 训练 / 评测（新增）

```bash
# 1) 可选：把 Ego3D 行渲染成 5 通道 BEV
python xiaomi_spatialembodied/EIEA/build_eiea_bev.py --split train \
  --data results/eiea_box/train.json --out results/eiea_box

# 2) 30B 冻结训练（box + BEV + depth，随机掩码 + 加噪）
python xiaomi_spatialembodied/EIEA/train_eiea_30b.py \
  --model_dir /dev/shm/models/xembodied-v0 \
  --data results/eiea_box/train_bev.json \
  --save results/eiea_box/eiea_30b_box_bev_depth.pt \
  --corruption standard

# 3) 7B SSR 代理版（2×4090 可跑）
python xiaomi_spatialembodied/EIEA/train_eiea_box.py \
  --data results/eiea_box/train_bev.json \
  --save results/eiea_box/eiea_7b_box_bev_depth.pt --corruption standard

# 4) 压力评测：clean / 丢模态 / 部分线索 / 抖动 / 遮挡 / 加噪 / wrong-scene
python xiaomi_spatialembodied/EIEA/eval_eiea_30b.py \
  --ckpt results/eiea_box/eiea_30b_box_bev_depth.pt \
  --data results/eiea_box/val_bev.json --limit 100
```

细节、概率表、环境变量和论文口径差异见 `docs/EIEA_TRAINING.md` 与
`docs/EIEA_ARCHITECTURE.md`。

## 注入是怎么发生的

`models/qwen3_vl_moe_vggt.py` 的注入条件很严格：`input_ids` 中 id 等于 `tor_token_id`
的位置数，必须**恰好等于**缓存的向量个数（10），否则不注入。所以：

- **token id 以缓存为准**：`qwen3_vl_vggt_register.py` 读缓存里的 `tor_token_id_vlm`
  （`tor_embeds_cache*.pt` 中是 **151669**）。模块级常量 `TOR_TOKEN_ID` 仅作为旧版缓存的兜底。
- **提示词需要带占位符**：普通文本提示里不含该 token，因此 `generate()` 会把所需数量的
  占位符**紧跟在图像 token 之后**插入（多图时插入在**最后一张图**之后；若样本没有图则回退到
  序列末尾），并同步扩展 `attention_mask`，让注入真正发生；插入时会打印一行说明。
  **图像保持不变**，2D 与 3D 两路仍然吃同一张图。
  > 注意：模型侧的判定是「整个 batch 中占位符总数 == 向量数」，因此**仅 batch size 为 1 时生效**；
  > batch > 1 时脚本会打印一条 WARNING。
- 若不补齐、位置数又不匹配，模型侧会打印 `TOR embedding injection SKIPPED: found N ... but M ...`
  的明确提示，而不是静默退化成纯 3DA 推理。

### 为什么不用缓存里的 `vlm_inputs` 重放

缓存的 `vlm_inputs` 只含 `input_ids` / `attention_mask` / `pixel_values` / `image_grid_thw`，
**不含 VGGT 的 `image_tchw`**（3D 流由 template 从实际图像现场生成）。若用缓存的 `pixel_values`
替换 2D 输入、而 3D 流仍来自调用方的图像，会直接造成 2D/3D 错配——比不注入更糟。
因此这里采用「保留调用方图像 + 补占位符」的方式。
