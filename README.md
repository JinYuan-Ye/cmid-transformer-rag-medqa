# Transformer-Enhanced Chinese Medical QA Retrieval

本项目基于 CMID（Chinese Medical Intent Dataset）构建一个“意图理解 + 查询重写 + 检索增强”的中文医疗问答系统。核心思想：用多任务 Transformer 同时完成粗粒度/细粒度医疗意图分类，并将口语化患者问题重写为结构化检索 Query，再进入知识库检索返回回答。

## 功能

- 多粒度意图识别：预测 CMID 的 4 类一级意图与 36 类二级意图。
- 生成式查询重写：用 Decoder 的 Cross-Attention 将非结构化问题重写为标准医学检索指令。
- RAG 检索增强：基于 LLM 扩展的治疗问答知识库，使用 `jieba + TF-IDF + cosine similarity` 检索 Top-K 答案。
- 训练与评估：支持 Accuracy、Macro-F1、BLEU-4、RAG Recall、t-SNE、混淆矩阵、注意力热力图等实验输出。

## 项目结构

```text
CMID-master/
  src/cmid_qa/              # 核心模型、数据处理、检索与可视化代码
  scripts/                  # 训练、演示、知识库构建脚本
  data/raw/                 # 原始 CMID 数据
  data/processed/           # 抽取后的治疗类问题、实体数据
  data/kb/                  # LLM 增强生成的问答知识库
  artifacts/figures/        # 训练曲线、混淆矩阵、t-SNE、注意力图
  reports/                  # 指标报告与历史实验结果
  checkpoints/              # 模型权重
  docs/                     # 汇报 PPT 与项目说明
```

## 环境安装

```bash
pip install -r requirements.txt
pip install -e .
```

如果无法联网下载 `bert-base-chinese`，请先将 BERT 模型下载到本地，并在运行脚本时通过 `--bert-path` 指定本地路径。

## 数据

原始数据位于 `data/raw/CMID.json`，每条样本包含：

- `originalText`：患者原始提问。
- `entities`：医疗实体及位置。
- `label_4class`：一级意图标签。
- `label_36class`：二级细粒度意图标签。
- `seg_result`：分词结果。

知识库位于 `data/kb/generated_answers.json`，格式为：

```json
{"question": "标准问题", "answer": "LLM 生成的谨慎医疗建议"}
```

## 训练

```bash
python scripts/train_eval.py \
  --data-path data/raw/CMID.json \
  --bert-path bert-base-chinese \
  --epochs 10 \
  --batch-size 32 \
  --max-len 128 \
  --checkpoint-path checkpoints/multitask_model.pth
```

训练脚本会联合优化三个任务：

```text
L = w1 * L_4class + w2 * L_36class + w3 * L_generation
```

其中权重由同方差不确定性参数动态学习。

## RAG 演示

```bash
python scripts/demo_rag.py \
  --data-path data/raw/CMID.json \
  --kb-path data/kb/generated_answers.json \
  --model-path checkpoints/multitask_model.pth \
  --top-k 3
```

流程：

1. 输入原始患者问题。
2. Transformer Decoder 生成标准检索 Query。
3. 使用 `jieba` 分词并转为 TF-IDF 向量。
4. 计算 Query 与知识库问题的余弦相似度。
5. 返回 Top-K 匹配问题及答案。

## 构建知识库

只抽取治疗相关问题：

```bash
python scripts/build_knowledge_base.py
```

调用 SiliconFlow 兼容接口生成回答：

```bash
set SILICONFLOW_API_KEY=your_api_key
python scripts/build_knowledge_base.py --generate --max-items 100
```

生成回答时系统提示词要求输出通用、谨慎、非诊断性建议，不承诺疗效，不替代医生诊疗。

## 实验结果

主实验指标：

| Metric | Value |
| --- | ---: |
| Accuracy (L4) | 72.02% |
| Accuracy (L36) | 48.04% |
| BLEU-4 | 0.2407 |
| RAG Recall@1 (Ours) | 28.00% |
| RAG Recall@1 (Raw) | 10.00% |

运行结果和可视化图片，详见 `reports/` 与 `artifacts/figures/`。

## 注意

本项目用于中文医疗问答检索增强研究。输出内容只能作为信息检索和健康科普辅助，不能替代医生诊断、处方或治疗建议。CMID 数据集也仅限科研用途，使用前请遵守原数据集许可与引用要求。
