import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
# ... 原有的 CUDA_VISIBLE_DEVICES 等设置 ...


# ============================================================
# GPT-2 文本分类微调脚本
# 数据集：IMDB 电影评论（二分类：正面 / 负面）
# 环境：本地 Windows 编写，远程 Linux 4090 运行
# ============================================================

# ---------- 第一步：设置环境变量（必须在 import 之前！） ----------
import os

# 指定使用哪张显卡。服务器有两张 4090，显卡 0 被别人占用，用显卡 1
# 这行代码必须在 import torch 之前执行，否则不生效
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 使用 Hugging Face 国内镜像，加速数据集下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 所有 Hugging Face 相关的缓存都指向自己的目录，避免写入别人的空间
os.environ["HF_HOME"] = "/data/yhn/data/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/data/yhn/data/hf_cache/datasets"      # 数据集缓存
os.environ["TRANSFORMERS_CACHE"] = "/data/yhn/data/hf_cache/models"      # 模型缓存
os.environ["TMPDIR"] = "/data/yhn/data/tmp"                              # 临时文件

# ---------- 第二步：导入依赖库 ----------
# 注意：所有 os.environ 设置完，才能 import 这些库，否则环境变量不生效
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,                    # GPT-2 分词器
    GPT2ForSequenceClassification,   # GPT-2 分类模型（带分类头）
    TrainingArguments,               # 训练参数配置
    Trainer,                         # 训练器
)
from sklearn.metrics import accuracy_score, f1_score   # 评估指标


# ---------- 第三步：配置参数 ----------
# 模型路径：你上传到服务器的本地 GPT-2 模型
MODEL_NAME = "/data/yhn/data/models/gpt2-local"

# 数据集名称：Hugging Face 上的 IMDB 数据集
DATASET_NAME = "stanfordnlp/imdb"

# 训练输出目录：训练日志、模型权重都保存在这里
OUTPUT_DIR = "/data/yhn/data/outputs/gpt2-imdb"

# 序列最大长度：每条评论截断/填充到 256 个 token
# 太短会丢失信息，太长会占用更多显存
MAX_LENGTH = 256

# 批次大小：一次处理 16 条数据
# 如果显存不够（报 OOM），改成 8 或 4
BATCH_SIZE = 16

# 训练轮数：整个数据集训练 3 遍
NUM_EPOCHS = 3

# 学习率：微调时常用 5e-5
LEARNING_RATE = 5e-5


# ---------- 第四步：确认环境和 GPU ----------
print("=" * 50)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("=" * 50)


# ---------- 第五步：加载 IMDB 数据集 ----------
# load_dataset 会自动从 Hugging Face（通过镜像）下载数据
# 缓存到 HF_DATASETS_CACHE 指定的目录
print("加载 IMDB 数据集...")
dataset = load_dataset(DATASET_NAME)

# IMDB 有 train 和 test 两个划分，如果没有 test 就用 validation
eval_split = "test" if "test" in dataset else "validation"

# 指定文本列和标签列的名称
text_col = "text"      # 评论内容
label_col = "label"    # 标签（0=负面，1=正面）

# 类别数：根据训练集标签去重计算，IMDB 是 2 类
num_labels = len(set(dataset["train"][label_col]))

print(f"训练集: {len(dataset['train'])} 条")
print(f"评估集: {len(dataset[eval_split])} 条")
print(f"类别数: {num_labels}")


# ---------- 第六步：加载分词器 ----------
# 从本地路径加载 GPT-2 分词器，local_files_only=True 表示不联网
print("加载分词器...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME, local_files_only=True)

# GPT-2 原生没有 pad_token，用 eos_token（结束符）充当
tokenizer.pad_token = tokenizer.eos_token

# 关键：分类任务必须用左填充
# 因为 GPT-2 是自回归模型，取最后一个 token 的表示做分类
# 左填充保证最后一个 token 是真实内容，而不是填充符
tokenizer.padding_side = "left"

# 截断时也从左边截断，保留结尾（结尾通常包含关键信息）
tokenizer.truncation_side = "left"


# 定义分词函数，用于批量处理数据
def tokenize_fn(examples):
    return tokenizer(
        examples[text_col],        # 对文本列进行分词
        truncation=True,           # 超过 max_length 就截断
        padding="max_length",      # 填充到 max_length
        max_length=MAX_LENGTH,     # 最大长度 256
    )


# 对所有数据应用分词函数，batched=True 表示批量处理，速度更快
print("处理数据...")
tokenized_datasets = dataset.map(tokenize_fn, batched=True)


# ---------- 第七步：加载 GPT-2 分类模型 ----------
# GPT2ForSequenceClassification 会在 GPT-2 基础上加一个分类头
# num_labels=2 表示输出 2 个类别（正面/负面）
print("加载 GPT-2 分类模型...")
model = GPT2ForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,                      # 分类类别数
    pad_token_id=tokenizer.eos_token_id,        # 告诉模型 pad_token 的 ID
    local_files_only=True,                      # 只从本地加载，不联网
)


# ---------- 第八步：定义评估指标 ----------
# 训练过程中会自动调用这个函数，计算准确率和 F1
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # predictions 是 logits，取 argmax 得到预测类别
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),           # 准确率
        "f1": f1_score(labels, predictions, average="weighted"),   # 加权 F1
    }


# ---------- 第九步：配置训练参数 ----------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,                       # 输出目录
    num_train_epochs=NUM_EPOCHS,                 # 训练轮数
    per_device_train_batch_size=BATCH_SIZE,      # 训练批次大小
    per_device_eval_batch_size=BATCH_SIZE * 2,   # 评估批次大小（可以更大）
    learning_rate=LEARNING_RATE,                 # 学习率
    warmup_steps=100,                            # 前 100 步学习率逐渐升高
    weight_decay=0.01,                           # 权重衰减，防止过拟合
    logging_steps=50,                            # 每 50 步打印一次日志
    eval_strategy="epoch",                       # 每个 epoch 结束后评估
    save_strategy="epoch",                       # 每个 epoch 结束后保存
    load_best_model_at_end=True,                 # 训练结束加载最好的模型
    bf16=torch.cuda.is_available(),              # 4090 用 bf16 混合精度，更快更省显存
    report_to="none",                            # 不上传到 wandb 等平台
)


# ---------- 第十步：创建 Trainer ----------
trainer = Trainer(
    model=model,                                 # 模型
    args=training_args,                          # 训练参数
    train_dataset=tokenized_datasets["train"],   # 训练集
    eval_dataset=tokenized_datasets[eval_split], # 评估集
    compute_metrics=compute_metrics,             # 评估函数
)


# ---------- 第十一步：开始训练 ----------
print("开始训练...")
trainer.train()


# ---------- 第十二步：最终评估并保存模型 ----------
print("最终评估结果:", trainer.evaluate())
trainer.save_model(OUTPUT_DIR)                   # 保存模型
tokenizer.save_pretrained(OUTPUT_DIR)            # 保存分词器
print(f"模型已保存到: {OUTPUT_DIR}")