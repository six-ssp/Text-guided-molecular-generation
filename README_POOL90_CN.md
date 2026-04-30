# 文本引导分子生成项目（SD-VAE x TGM-DLM）详细说明（简中）

本文档对应目录：`/home/six_ssp/my_project`

项目目标：把文本描述编码成条件信号，在 SD-VAE latent 空间做扩散采样，再解码成 SMILES，实现 `text -> molecule`。

当前实现是可跑通、可续训、可评估、可清理磁盘的一套本地工程方案，重点是工程可用性和实验迭代效率。

说明：这份 README 对应当前本地工程状态；根目录脚本、环境修复逻辑、生成/评估入口均已恢复。

---

## 1. 项目现状与边界

### 1.1 当前做成了什么

- 已打通端到端流程：`准备数据 -> 训练 -> 生成 -> 评估`。
- 已实现两种文本融合：`pooled` 和 `crossattn`。
- 已支持自动续训（自动选最新 checkpoint）。
- 已支持大样本评估（不再固定 24 条）。
- 已支持一键清理无关文件与空间管理（日志、`__pycache__`、可选 checkpoint 裁剪）。

### 1.2 和原始 tgm-dlm 的关系

- 不是对原始 tgm-dlm 的“原样复刻”。
- 当前方案是 **SD-VAE latent 级条件扩散**，优势是实现简单、训练和部署成本低。
- 代价是表达能力上限受 latent MLP 架构限制，不等价于更复杂的 token-level 生成范式。

### 1.3 当前效果快照（2026-03-14，全量 test_pool90）

评估文件：`/home/six_ssp/my_project/ChEBI-20_data/prompt_generated_testpool90_full_3021x8.tsv`

生成规模：

- `3021` 个 prompt（来自 `test_pool90` 唯一描述）；
- 每个 prompt 采样 `8` 条；
- 总样本数 `24168`。

总体指标：

- 有效率 `valid_ratio`：`0.9395`（`22706/24168`）
- 唯一率（有效内）`unique_ratio_among_valid`：`0.1755`（`3985/22706`）
- 新颖性（唯一集）`novelty_ratio_unique_valid`：`0.9972`（`3974/3985`）
- 内部多样性（唯一集）`internal_diversity_unique_valid`：`0.922`（约）

按 prompt 观察：

- `2816/3021` 个 prompt 达到 `8/8` 有效（`93.21%`）
- `158/3021` 个 prompt 是 `0/8` 有效（`5.23%`）
- `2040/3021` 个 prompt 出现“该 prompt 下有效样本全部相同”（`67.53%`，模式坍塌）
- `per_prompt_unique_ratio_mean = 0.1803`
- 跨 prompt 重合分子数 `12`（跨描述混淆不高）

逐参考分子指标（2026-04-28，`scripts/evaluate_text_guided_metrics.py`）：

- `BLEU = 0.2453`
- `Exact match = 0.0000`
- `Levenshtein distance = 40.0526`
- `MACCS FTS = 0.2910`
- `RDK FTS = 0.1995`
- `Morgan FTS = 0.1000`
- `FCD = 19.0648`
- `Text2Mol score`：当前未计算（本地没有 Text2Mol matching model）
- `Validity = 0.9395`

这两类指标口径不同：`valid/unique/novelty/diversity` 评价生成集合本身；BLEU、Exact、指纹相似度和 FCD 更接近“是否生成到参考答案分子”。

### 1.4 当前主要问题

1. **同一 prompt 下多样性不足（主问题）**  
   全量评估下 `67.53%` 的 prompt 出现“有效样本全相同”，说明单 prompt 采样模式坍塌明显。

2. **仍有部分 prompt 完全无效**  
   `5.23%` 的 prompt 为 `0/8` 有效，当前解码鲁棒性不稳定。

3. **分子级无泄露切分尚未默认启用**  
   现有 `train_pool90 / val_pool90 / test_pool90` 在 `CID` 上不重叠，但在 canonical SMILES 上仍有重叠；若做严格泛化评估，建议先做 no-leak 重切分。

4. **全量生成耗时长、内存压力高**  
   这次全量 `3021x8` 生成耗时约 `13h20m`；必须使用分块生成（`WORK_CHUNK_SIZE`）与小解码批次（`DECODE_BATCH_SIZE`）避免被系统 OOM killer 杀进程。

5. **Exact match 受 SD-VAE 重构上限限制**  
   旧实验使用的是 ZINC 预训练权重 `zinc_kl_avg.model`。在 ChEBI-20 分子上，`SMILES -> SD-VAE latent -> SMILES` 不是无损压缩；实测小池子里直接解码真实 latent 的 canonical exact 仍为 `0`。因此后续提升 Exact 的关键不是单纯调 diffusion 采样，而是先提高 SD-VAE latent 的可解码性。

6. **已新增 ChEBI-domain SD-VAE 重训入口**  
   为了统一 SD-VAE 和 diffusion 的数据域，新增了 ChEBI 专用 SD-VAE 训练流程：从 `train_pool90 / validation_pool90` 构建 SD-VAE 训练张量，从随机初始化开始训练 SD-VAE，再用新 SD-VAE 权重重新导出 latent。当前主路线不再默认使用 ZINC warm-start；只有显式设置 `FROM_SCRATCH=0 INIT_MODEL=...` 时才会加载旧权重。

### 1.5 latent inversion 试验结果（2026-04-28，test_pool90 前 128 条）

目的：验证“冻结 SD-VAE decoder，只优化每个分子的 latent”能否提高 SD-VAE 解码上限。

对照设置：同一批 `test_pool90` 前 128 条，实际可编码样本 `125` 条。

| 指标 | 原始 SD-VAE latent | latent inversion 后 |
|---|---:|---:|
| Validity | `0.960` | `0.968` |
| Canonical Exact | `0.024` | `0.664` |
| Levenshtein distance | `27.928` | `13.232` |
| MACCS FTS | `0.411` | `0.886` |
| RDK FTS | `0.273` | `0.844` |
| Morgan FTS | `0.232` | `0.801` |

结论：Exact 的主要瓶颈确实在 SD-VAE latent 可解码性。后续正式提升生成指标时，推荐先对 `train_pool90` 做 latent inversion，再用反演 latent 训练 diffusion。

---

## 2. 目录与核心文件

### 2.1 项目根目录关键文件

- `07_full_pipeline.sh`：总控脚本（推荐入口）。
- `run_local_oneclick.sh`：基础入口（prepare/train/generate/all）。
- `03_train.sh`：训练封装。
- `04_generate.sh`：生成封装。
- `05_evaluate.sh`：评估脚本（有效率/唯一率/新颖性/多样性）。
- `09_optimize_sdvae_latents.sh`：latent inversion 入口；冻结 SD-VAE decoder，只优化每个分子的 latent，使其更容易解码回目标 SMILES。
- `10_train_sdvae_chebi.sh`：构建 ChEBI SD-VAE 训练张量，并从零训练 ChEBI-domain SD-VAE 权重。
- `11_dump_chebi_sdvae_latents.sh`：用 ChEBI-domain SD-VAE 权重重新导出 `*_sdvae_chebi_latents.pt`。
- `scripts/build_sdvae_chebi_dataset.py`：把 ChEBI split 转成 SD-VAE grammar action 张量。
- `scripts/train_sdvae_chebi.py`：ChEBI SD-VAE 训练主程序，默认从零训练；也支持显式 warm-start 对照实验。
- `scripts/evaluate_sdvae_reconstruction.py`：评估 SD-VAE 重构上限。
- `scripts/optimize_sdvae_latents.py`：执行 latent inversion，输出可替换训练用的 `cid -> latent` 文件。
- `scripts/sdvae_utils.py`：SD-VAE 评估/反演共享工具函数。
- `scripts/evaluate_text_guided_metrics.py`：逐参考分子九指标评估脚本（BLEU、Exact、Levenshtein、指纹相似度、FCD、Validity；Text2Mol 支持外部结果导入）。
- `README_POOL90_CN.md`：本说明。

### 2.2 模型与数据关键代码（tgm-dlm 内）

- `tgm-dlm/improved-diffusion/scripts/train_sdvae_latent.py`：训练主程序。
- `tgm-dlm/improved-diffusion/scripts/text_guided_generate.py`：文本引导生成主程序。
- `tgm-dlm/improved-diffusion/scripts/process_text.py`：文本编码预处理。
- `tgm-dlm/improved-diffusion/scripts/dump_sdvae_latents.py`：SD-VAE latent 预处理。
- `tgm-dlm/improved-diffusion/scripts/mydatasets.py`：数据集与 DataLoader。
- `tgm-dlm/improved-diffusion/improved_diffusion/latent_model.py`：条件扩散网络定义。
- `tgm-dlm/improved-diffusion/improved_diffusion/train_util.py`：训练循环、保存、续训。

### 2.3 数据目录（默认）

- `ChEBI-20_data/*.txt`：输入 split（`CID / SMILES / description`）。
- `ChEBI-20_data/*_desc_states.pt`：文本 token states。
- `ChEBI-20_data/*_sdvae_latents.pt`：分子 latent。
- `ChEBI-20_data/prompt_generated*.tsv`：生成结果。

### 2.4 根目录脚本职责总表

- `01_smoke_demo.sh`：最小链路冒烟测试（验证环境和依赖）。
- `02_prepare_full_data.sh`：离线准备全量特征（文本状态 + SD-VAE latent）。
- `03_train.sh`：封装训练参数并启动 `train_sdvae_latent.py`。
- `04_generate.sh`：封装生成参数并启动 `text_guided_generate.py`。
- `05_evaluate.sh`：对生成 TSV 计算 validity/uniqueness/novelty/diversity。
- `06_sweep_checkpoints.sh`：批量扫多个 checkpoint 做对比评估。
- `07_full_pipeline.sh`：总控入口（status/cleanup/prepare/train/generate/evaluate/full）。
- `08_filter_legal.sh`：后处理过滤（合法性/规则过滤场景）。
- `09_optimize_sdvae_latents.sh`：优化 SD-VAE latent，用于提高 SD-VAE 解码上限和后续 Exact 指标。
- `10_train_sdvae_chebi.sh`：从零重训 SD-VAE 到 ChEBI 数据域。
- `11_dump_chebi_sdvae_latents.sh`：用新 SD-VAE 权重导出 ChEBI latent。
- `run_local_oneclick.sh`：底层 one-click 执行器（被 `07` 调用）。

---

## 3. 模型流程与原理

### 3.1 端到端数据流

1. 原始数据 `CID / SMILES / description`。
2. `process_text.py` 用 SciBERT 编码描述，得到 token-level `states + mask`。
3. `dump_sdvae_latents.py` 用 SD-VAE 把 SMILES 映射到 latent 向量。
4. `train_sdvae_latent.py` 在 latent 空间训练扩散模型（MSE 目标）。
5. `text_guided_generate.py` 根据文本条件采样 latent。
6. SD-VAE 解码 latent 得到 SMILES。
7. `05_evaluate.sh` 计算 validity / uniqueness / novelty / diversity。

### 3.2 文本融合方式

- `pooled`：将文本 token states 池化后做条件输入。
- `crossattn`：以 latent/time 为 query，对文本 token states 做 cross-attention，再融合。

经验上 `crossattn` 通常更能利用细粒度文本条件，但也更容易受训练稳定性和数据质量影响。

### 3.3 训练目标

- 扩散模型学习从噪声 latent 还原目标 latent。
- 损失主项是 MSE（`gd.LossType.MSE`）。
- 训练过程维护主参数与 EMA 参数，保存 model/ema/opt/scaler。

### 3.4 技术路线总结（工程视角）

- 路线核心：`文本编码器(SciBERT) -> 条件扩散(在 SD-VAE latent 空间) -> SD-VAE 解码 -> SMILES 评估`。
- 方案取舍：优先保证工程可跑通、可续训、可大样本评估；接受一定表达上限换取实现复杂度与资源成本可控。
- 当前瓶颈：不是“能不能生成”，而是“同一 prompt 下如何提升多样性且保持有效率”。

---

## 4. 环境与前置条件

### 4.1 推荐 Python

默认优先使用：

```bash
/home/six_ssp/my_project/.mamba-tgmsd/bin/python
```

你也可以覆盖：

```bash
PYTHON_BIN=python3 ...
```

### 4.2 GPU 要求

- 脚本支持 `DEVICE=cuda|cpu|auto`。
- 默认 `AUTO_TUNE=1`，会按显存自动调 `BATCH_SIZE / GEN_BATCH_SIZE / NUM_WORKERS / FP16`。
- CUDA 不可用会自动回退 CPU，并关闭 fp16。

### 4.3 必要文件

- SD-VAE 权重：`sdvae/dropbox/results/zinc/zinc_kl_avg.model`
- Grammar：`sdvae/dropbox/context_free_grammars/mol_zinc.grammar`
- SciBERT 本地目录：`tgm-dlm/scibert`（或回退在线模型）

---

## 5. 一键脚本总览

### 5.1 总控脚本（推荐）

`07_full_pipeline.sh` 支持模式：

- `MODE=status`：看磁盘与 checkpoint 状态。
- `MODE=cleanup`：空间清理。
- `MODE=prune`：删除与当前流程无关目录。
- `MODE=prepare`：准备数据。
- `MODE=train`：训练。
- `MODE=generate`：生成。
- `MODE=evaluate`：评估。
- `MODE=full`：`prune(可选)+cleanup(可选)+prepare+train+generate+evaluate`。

### 5.2 基础脚本

`run_local_oneclick.sh` 模式：`prepare | smoke | train | generate | all`。

一般优先用 `07_full_pipeline.sh`，除非你要直接控制底层行为。

---

## 6. 快速开始（推荐顺序）

### 6.1 先看状态

```bash
cd /home/six_ssp/my_project
MODE=status ./07_full_pipeline.sh
```

你会看到：磁盘占用、数据目录体积、checkpoint 数量、最新 checkpoint 路径。

### 6.2 清理空间（建议先执行一次）

```bash
MODE=cleanup KEEP_LATEST_CHECKPOINTS=40 ./07_full_pipeline.sh
```

含义：保留最新 40 组 step，对更老 step 的 `model/ema/opt/scaler` 一并删除。

### 6.3 继续训练（不从头）

```bash
MODE=train \
TEXT_FUSION=crossattn \
CHECKPOINT_PATH=/home/six_ssp/my_project/tgm-dlm/checkpoints_sdvae_latent_crossattn \
TRAIN_STEPS=1400000 \
SAVE_INTERVAL=20000 \
./07_full_pipeline.sh
```

说明：自动续训会从该目录下最新 `PLAIN_model*.pt` 接着跑。

### 6.4 大样本生成与评估

```bash
MODE=generate \
CHECKPOINT_PATH=/home/six_ssp/my_project/tgm-dlm/checkpoints_sdvae_latent_crossattn \
EVAL_NUM_PROMPTS=128 \
NUM_SAMPLES_PER_PROMPT=8 \
GEN_BATCH_SIZE=4 \
DECODE_BATCH_SIZE=8 \
OUTPUT=/home/six_ssp/my_project/ChEBI-20_data/prompt_generated_large.tsv \
./07_full_pipeline.sh

MODE=evaluate \
OUTPUT=/home/six_ssp/my_project/ChEBI-20_data/prompt_generated_large.tsv \
./07_full_pipeline.sh
```

---

## 7. 常用命令模板

### 7.1 全流程一条命令

```bash
MODE=full ./07_full_pipeline.sh
```

### 7.2 只准备数据

```bash
MODE=prepare ./07_full_pipeline.sh
```

### 7.3 只训练（显式 warm-start）

```bash
MODE=train \
TEXT_FUSION=crossattn \
CHECKPOINT_PATH=/home/six_ssp/my_project/tgm-dlm/checkpoints_sdvae_latent_crossattn \
INIT_CHECKPOINT=/home/six_ssp/my_project/tgm-dlm/checkpoints_sdvae_latent_chebi/PLAIN_model800000.pt \
TRAIN_STEPS=1200000 \
./07_full_pipeline.sh
```

### 7.4 只生成（指定模型）

```bash
MODE=generate \
MODEL_PATH=/home/six_ssp/my_project/tgm-dlm/checkpoints_sdvae_latent_crossattn/PLAIN_model1200000.pt \
EVAL_NUM_PROMPTS=256 \
NUM_SAMPLES_PER_PROMPT=8 \
OUTPUT=/home/six_ssp/my_project/ChEBI-20_data/prompt_generated_256x8.tsv \
./07_full_pipeline.sh
```

### 7.5 只评估（JSON 单行输出）

```bash
JSON_ONLY=1 \
GENERATED_FILE=/home/six_ssp/my_project/ChEBI-20_data/prompt_generated_256x8.tsv \
./05_evaluate.sh
```

### 7.6 多样性优化生成

```bash
MODE=generate \
TEXT_FUSION=crossattn \
EVAL_NUM_PROMPTS=128 \
NUM_SAMPLES_PER_PROMPT=8 \
OVERSAMPLE_FACTOR=4 \
SELECT_VALID_UNIQUE=1 \
DECODE_RANDOM=1 \
GEN_BATCH_SIZE=2 \
DECODE_BATCH_SIZE=4 \
DIFFUSION_STEPS=2000 \
OUTPUT=/home/six_ssp/my_project/ChEBI-20_data/prompt_generated_diverse.tsv \
./07_full_pipeline.sh
```

含义：

- `OVERSAMPLE_FACTOR=4`：每个 prompt 先生成 `8*4=32` 个候选，再筛成 8 个输出。
- `SELECT_VALID_UNIQUE=1`：优先选择合法且去重后的分子，缓解同 prompt 重复。
- `DECODE_RANDOM=1`：启用 SD-VAE 随机解码，通常能提高多样性，但候选无效率会上升，所以建议配合过采样和合法过滤。

如果只是做 benchmark 分析，也可以加：

```bash
RERANK_REFERENCE_FILE=/home/six_ssp/my_project/ChEBI-20_data/test_pool90.txt \
RERANK_METRIC=morgan
```

注意：`RERANK_REFERENCE_FILE` 使用真实参考 SMILES，是 oracle rerank，只适合分析上限或 ablation，不适合作为真实生成能力直接报告。

### 7.7 评估 SD-VAE 重构上限

先确认 SD-VAE 自己能不能把真实分子重构回来。这个指标是后续 Exact 的硬上限之一。

```bash
./.mamba-tgmsd/bin/python scripts/evaluate_sdvae_reconstruction.py \
  --split test_pool90 \
  --limit 128 \
  --output-json /home/six_ssp/my_project/ChEBI-20_data/sdvae_recon_test128.json \
  --decoded-tsv /home/six_ssp/my_project/ChEBI-20_data/sdvae_recon_test128.tsv \
  --decode-batch-size 4 \
  --mode gpu
```

如果这里 `exact_match_canonical_smiles` 很低，说明瓶颈在 SD-VAE 重构，而不是 diffusion 采样。

### 7.8 latent inversion 提升 SD-VAE 解码上限

冻结 SD-VAE decoder，只优化每个分子的 latent，使 decoder 对目标 grammar action 的概率更高。小样本验证中，`steps=200` 能把可编码样本的 canonical exact 从 `0` 提到明显非零。

小范围试跑：

```bash
SPLIT=test_pool90 \
LIMIT=128 \
STEPS=200 \
LR=0.03 \
Z_L2=0.00001 \
BATCH_SIZE=2 \
DECODE_BATCH_SIZE=2 \
FINAL_EVAL_LIMIT=128 \
OUTPUT_LATENTS=/home/six_ssp/my_project/ChEBI-20_data/test_pool90_sdvae_latents_inverted_128.pt \
./09_optimize_sdvae_latents.sh
```

训练集正式反演（耗时较长，可断点续跑）：

```bash
SPLIT=train_pool90 \
LIMIT=0 \
STEPS=200 \
LR=0.03 \
Z_L2=0.00001 \
BATCH_SIZE=32 \
DECODE_BATCH_SIZE=2 \
SAVE_EVERY=64 \
FINAL_EVAL_LIMIT=512 \
OUTPUT_LATENTS=/home/six_ssp/my_project/ChEBI-20_data/train_pool90_sdvae_latents_inverted.pt \
./09_optimize_sdvae_latents.sh
```

如果 `BATCH_SIZE=32` OOM，就降到 `16` 或 `8` 后继续跑；脚本会从已有 `OUTPUT_LATENTS` 断点续跑。如果只需要生成训练用 latent、不想在最后解码抽样评估，可以加 `SKIP_FINAL_DECODE=1`。正式记录指标时建议保留 `FINAL_EVAL_LIMIT=512` 或设为 `0` 做全量重构评估。

用反演后的 latent 训练 diffusion：

```bash
MODE=train \
TEXT_FUSION=crossattn \
LATENT_FILE=/home/six_ssp/my_project/ChEBI-20_data/train_pool90_sdvae_latents_inverted.pt \
CHECKPOINT_PATH=/home/six_ssp/my_project/tgm-dlm/checkpoints_sdvae_latent_crossattn_inverted \
INIT_CHECKPOINT=/home/six_ssp/my_project/tgm-dlm/checkpoints_sdvae_latent_crossattn/PLAIN_model1400000.pt \
TRAIN_STEPS=1200000 \
./07_full_pipeline.sh
```

说明：`LATENT_FILE` 现在会从 `07_full_pipeline.sh` 传到训练脚本。反演 latent 改变了 diffusion 的学习目标，建议单独用新 checkpoint 目录，避免和旧实验混在一起。

### 7.9 从零重训 SD-VAE 到 ChEBI 数据域

目的：统一 SD-VAE 和 diffusion 的数据集。SD-VAE 不再依赖 ZINC 预训练权重，而是从随机初始化开始在 `train_pool90` 上训练、用 `validation_pool90` 选 best model。

默认主路线：从零训练 ChEBI-domain SD-VAE。

```bash
BUILD_DATA=missing \
SAVE_DIR=/home/six_ssp/my_project/sdvae/dropbox/results/chebi_pool90_scratch \
EPOCHS=80 \
BATCH_SIZE=128 \
LEARNING_RATE=0.0003 \
./10_train_sdvae_chebi.sh
```

可选对照：如果需要和 ZINC warm-start 比较，才显式打开旧权重初始化。

```bash
FROM_SCRATCH=0 \
INIT_MODEL=/home/six_ssp/my_project/sdvae/dropbox/results/zinc/zinc_kl_avg.model \
BUILD_DATA=missing \
SAVE_DIR=/home/six_ssp/my_project/sdvae/dropbox/results/chebi_pool90_warmstart \
EPOCHS=30 \
BATCH_SIZE=128 \
LEARNING_RATE=0.0003 \
./10_train_sdvae_chebi.sh
```

训练完成后，用新的 SD-VAE 权重重新导出 ChEBI latent：

```bash
SDVAE_SAVED_MODEL=/home/six_ssp/my_project/sdvae/dropbox/results/chebi_pool90_scratch/epoch-best.model \
LATENT_TAG=chebi \
./11_dump_chebi_sdvae_latents.sh
```

然后用新 latent 训练 diffusion：

```bash
MODE=train \
TEXT_FUSION=crossattn \
LATENT_FILE=/home/six_ssp/my_project/ChEBI-20_data/train_pool90_sdvae_chebi_latents.pt \
CHECKPOINT_PATH=/home/six_ssp/my_project/tgm-dlm/checkpoints_sdvae_chebi_crossattn \
INIT_CHECKPOINT=/home/six_ssp/my_project/tgm-dlm/checkpoints_sdvae_latent_crossattn/PLAIN_model1400000.pt \
TRAIN_STEPS=1200000 \
./07_full_pipeline.sh
```

生成时也要指定同一个 SD-VAE 权重，否则 diffusion 生成的是 ChEBI-SD-VAE latent，但 decoder 仍会用旧 ZINC-SD-VAE：

```bash
SDVAE_SAVED_MODEL=/home/six_ssp/my_project/sdvae/dropbox/results/chebi_pool90_scratch/epoch-best.model \
MODEL_PATH=/home/six_ssp/my_project/tgm-dlm/checkpoints_sdvae_chebi_crossattn/PLAIN_model1200000.pt \
TEXT_FUSION=crossattn \
MODE=generate \
./07_full_pipeline.sh
```

---

## 8. 参数说明（高频）

### 8.1 训练/性能参数

| 变量 | 默认 | 含义 |
|---|---:|---|
| `AUTO_TUNE` | `1` | 按显存自动调性能参数 |
| `BATCH_SIZE` | 自动 | 训练 batch size |
| `LATENT_FILE` | 空 | 训练使用的 latent 文件；为空时由 `03_train.sh` 默认使用 `<split>_sdvae_latents.pt` |
| `NUM_WORKERS` | 自动 | DataLoader workers |
| `USE_FP16` | 自动/`1` | 是否启用 fp16 |
| `PIN_MEMORY` | 自动 | DataLoader pin memory |
| `PREFETCH_FACTOR` | `4` | 预取因子（`workers>0` 生效） |
| `PERSISTENT_WORKERS` | `1` | 是否复用 workers |
| `TRAIN_STEPS` | `1200000` | 训练总步数上限（用于 lr anneal） |
| `SAVE_INTERVAL` | `20000` | checkpoint 保存间隔 |
| `LOG_INTERVAL` | `50` | 打印 loss 间隔 |

### 8.2 文本条件参数

| 变量 | 默认 | 含义 |
|---|---:|---|
| `TEXT_FUSION` | `crossattn`（07脚本） | `pooled` 或 `crossattn` |
| `TEXT_ATTN_HEADS` | `8` | cross-attn 头数 |

注意：checkpoint 与融合方式要匹配，否则加载会报错。

### 8.3 生成/评估规模参数

| 变量 | 默认 | 含义 |
|---|---:|---|
| `PROMPT_SOURCE_SPLIT` | `test_pool90` | 自动抽 prompt 的来源 split |
| `EVAL_NUM_PROMPTS` | `128` | 抽取 prompt 数 |
| `NUM_SAMPLES_PER_PROMPT` | `8` | 每个 prompt 采样数 |
| `GEN_BATCH_SIZE` | 自动 | 扩散采样阶段 batch size |
| `WORK_CHUNK_SIZE` | `256` | 分块生成大小，控制主内存峰值 |
| `DECODE_BATCH_SIZE` | `32` | SD-VAE 解码阶段 batch size（OOM 首先调小它） |
| `DIFFUSION_STEPS` | `2000` | 生成采样步数；小池子快速试验可用 `500`，正式结果建议保持 `2000` |
| `OVERSAMPLE_FACTOR` | `1` | 每个 prompt 的候选过采样倍数 |
| `SELECT_VALID_UNIQUE` | `0` | 是否优先输出合法且唯一的候选 |
| `DECODE_RANDOM` | `0` | 是否启用 SD-VAE 随机解码 |
| `CANDIDATE_OUTPUT` | 空 | 可选，保存过采样候选全集 |
| `RERANK_REFERENCE_FILE` | 空 | 可选，benchmark 用真实参考文件 |
| `RERANK_METRIC` | `none` | 可选，`morgan/maccs/rdk` |
| `OUTPUT` | `prompt_generated_large.tsv` | 生成文件路径 |

最终输出样本数约为 `EVAL_NUM_PROMPTS * NUM_SAMPLES_PER_PROMPT`；实际候选数为该值乘以 `OVERSAMPLE_FACTOR`。

### 8.4 空间管理参数

| 变量 | 默认 | 含义 |
|---|---:|---|
| `AUTO_CLEANUP` | `1` | 训练/全流程前自动轻量清理 |
| `CLEAN_PYCACHE` | `1` | 删除 `__pycache__` |
| `CLEAN_LOG_DAYS` | `7` | 删除 N 天前日志 |
| `KEEP_LATEST_CHECKPOINTS` | `0` | 清理时保留最新 N 组 checkpoint |
| `EXTRA_CHECKPOINT_DIRS` | 空 | 额外 checkpoint 目录（逗号分隔） |

---

## 9. 训练与续训机制

### 9.1 自动续训

- 当 `AUTO_RESUME=1` 时，脚本会在 `CHECKPOINT_PATH` 内找最新 `PLAIN_model*.pt`。
- 采用 `sort -V` 进行版本排序，避免字符串排序误选旧模型。

### 9.2 checkpoint 文件命名

- 模型：`PLAIN_modelXXXXXX.pt`
- EMA：`PLAIN_ema_<rate>_XXXXXX.pt`
- 优化器：`PLAIN_optXXXXXX.pt`
- AMP scaler：`PLAIN_scalerXXXXXX.pt`

清理时建议按 step 组一起处理，避免 model 与 opt/scaler 不一致。

---

## 10. 输出文件格式

### 10.1 生成结果 TSV

`text_guided_generate.py` 输出列：

- `prompt_id`
- `sample_idx`
- `prompt`
- `generated_smiles`
- `is_valid`
- `latent`

### 10.2 评估输入要求

`05_evaluate.sh` 默认从 `generated_smiles` 列读 SMILES。

如果你自定义了生成文件，至少要保证存在该列。

---

## 11. 评估指标解释

### 11.1 valid_ratio（有效率）

- 定义：`有效分子数 / 总生成数`
- 有效分子由 RDKit 能否成功解析 SMILES 判定。

### 11.2 unique_ratio_among_valid（唯一率）

- 定义：`去重后的有效分子数 / 有效分子数`
- 只在有效分子集合内去重。
- 唯一率低通常表示模式坍塌或采样温度不足。

### 11.3 novelty_ratio_unique_valid（新颖性）

- 定义：`不在训练集中的去重有效分子 / 去重有效分子`
- 基于 canonical SMILES 对比训练集。

### 11.4 internal_diversity_unique_valid（内部多样性）

- 用 Morgan 指纹 + Tanimoto 距离。
- 取去重有效集合中的成对距离均值。
- 值越高表示结构差异越大。

### 11.5 per_prompt_unique_ratio_mean（prompt 内平均唯一率）

- 定义：对每个 prompt 计算 `该 prompt 下 unique_valid / valid`，再对所有 prompt 取平均。
- 这个指标比全局唯一率更能反映“同一提示下有没有塌缩”。

---

## 12. 日志怎么看

### 12.1 prepare 阶段常见输出

- `skip build (existing *_pool90.txt)`：已存在 split 文件，跳过构建。
- `process_text device=cuda:0`：文本编码用 GPU。
- `resume from ..._desc_states.pt`：断点续算文本状态。
- `Some weights ... were not used when initializing BertModel`：从预训练权重初始化时常见，通常可接受。

### 12.2 train 阶段关注点

- `init from ... missing_keys=...`：warm-start 非严格加载，出现 missing keys 常见。
- `latent dataset train_pool90: xxxx usable rows`：真实参与训练样本数。
- `rank0: step loss`：主进程 loss。
- `saving model ...`：checkpoint 保存点。

### 12.3 generate 阶段关注点

- `device=cuda:0`：生成设备。
- `generate chunks`：正在按块跑全量生成。
- `saved N samples ... valid=... valid_ratio=...`：快速质量概览。

---

## 13. 常见问题与处理

### 13.1 “是不是要从头跑？”

不需要。一般直接续训。

做法：把 `TRAIN_STEPS` 设为大于最新 checkpoint 的步数，比如最新是 `1200000`，就设 `1400000`。

### 13.2 “为什么只有 24 个样本？”

典型原因：只用了示例 prompt（3 个）且每个 `NUM_SAMPLES_PER_PROMPT=8`。

解决：用 `MODE=generate EVAL_NUM_PROMPTS=128 NUM_SAMPLES_PER_PROMPT=8`。

### 13.3 “GPU 不跑/回落 CPU”

检查：

- `nvidia-smi` 是否可用。
- Python 是否能 `import torch; torch.cuda.is_available()`。
- 是否误设 `DEVICE=cpu`。

### 13.4 “checkpoint 加载失败”

先核对 `TEXT_FUSION` 是否匹配模型：

- `pooled` 模型用 `TEXT_FUSION=pooled`
- `crossattn` 模型用 `TEXT_FUSION=crossattn`

### 13.5 “磁盘被 checkpoint 打满”

执行：

```bash
MODE=cleanup KEEP_LATEST_CHECKPOINTS=40 ./07_full_pipeline.sh
```

并适当增大 `SAVE_INTERVAL`，例如 `20000` 或 `40000`。

### 13.6 “生成阶段 CUDA OOM（在 `proxy.decode`）”

先把解码批次降下来：

```bash
MODE=generate \
GEN_BATCH_SIZE=4 \
DECODE_BATCH_SIZE=8 \
... \
./07_full_pipeline.sh
```

如果还 OOM，再降到 `DECODE_BATCH_SIZE=4` 或 `2`。

### 13.7 “生成阶段直接被系统 `Killed`”

这通常不是 Python 异常，而是主内存被打满后被 OOM killer 杀掉。

处理办法：

```bash
MODE=generate \
GEN_BATCH_SIZE=2 \
WORK_CHUNK_SIZE=128 \
DECODE_BATCH_SIZE=4 \
... \
./07_full_pipeline.sh
```

如果还不稳，再降到 `GEN_BATCH_SIZE=1 WORK_CHUNK_SIZE=64 DECODE_BATCH_SIZE=2`。

---

## 14. 推荐实验工作流（可直接照做）

1. 盘点现状：

```bash
MODE=status ./07_full_pipeline.sh
```

2. 先清理：

```bash
MODE=cleanup KEEP_LATEST_CHECKPOINTS=40 ./07_full_pipeline.sh
```

3. 继续训练：

```bash
MODE=train TEXT_FUSION=crossattn TRAIN_STEPS=1400000 SAVE_INTERVAL=20000 ./07_full_pipeline.sh
```

4. 大样本生成：

```bash
MODE=generate \
EVAL_NUM_PROMPTS=128 \
NUM_SAMPLES_PER_PROMPT=8 \
GEN_BATCH_SIZE=4 \
WORK_CHUNK_SIZE=256 \
DECODE_BATCH_SIZE=8 \
OUTPUT=/home/six_ssp/my_project/ChEBI-20_data/prompt_generated_large.tsv \
./07_full_pipeline.sh
```

5. 评估：

```bash
MODE=evaluate OUTPUT=/home/six_ssp/my_project/ChEBI-20_data/prompt_generated_large.tsv ./07_full_pipeline.sh
```

---

## 15. 版本提示

- 本文档面向当前本地工程实现，不保证与上游开源仓库保持同步。
- 你本地脚本和权重路径如果调整过，以本地实际文件为准。
- GitHub 公开仓库默认不包含本地数据、训练日志、权重和 `.mamba-tgmsd` 环境；复现实验前需要按本文档准备数据和模型资源。
