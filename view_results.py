# ============================================================
# 训练结果查看脚本
# 作用：读取 Trainer 保存的 trainer_state.json，
#       提取每个 epoch 的 loss、accuracy、F1，
#       以及最终的最优结果。
# ============================================================

import json
import os
import glob


# ---------- 第一步：自动定位 trainer_state.json ----------
# Trainer 保存的 trainer_state.json 可能在两个位置：
#   1. outputs/trainer_state.json（如果没有保存 checkpoint）
#   2. outputs/checkpoint-xxxx/trainer_state.json（保存了 checkpoint 时）
# 这里自动搜索，找到最新的那个

OUTPUT_DIR = "outputs"   # 你的训练输出目录

# 先在 OUTPUT_DIR 下直接找
candidates = glob.glob(os.path.join(OUTPUT_DIR, "trainer_state.json"))

# 如果没有，再去 checkpoint-* 子目录里找
if not candidates:
    candidates = glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*", "trainer_state.json"))

# 如果都没找到，报错退出
if not candidates:
    raise FileNotFoundError(
        f"在 {OUTPUT_DIR} 下找不到 trainer_state.json。\n"
        f"请确认路径是否正确，或训练时是否保存了 checkpoint。"
    )

# 如果有多个，选文件名里 checkpoint 编号最大的（也就是最后一个 epoch 的）
# 例如 checkpoint-4689 比 checkpoint-3126 新
def get_step(path):
    """从路径中提取 checkpoint 的编号，用于排序"""
    parts = path.replace("\\", "/").split("/")
    for p in parts:
        if p.startswith("checkpoint-"):
            return int(p.split("-")[1])
    return 0

# 按 checkpoint 编号从小到大排序，取最后一个
candidates.sort(key=get_step)
state_path = candidates[-1]

print(f"读取训练记录: {state_path}")
print()


# ---------- 第二步：加载 JSON 文件 ----------
with open(state_path, "r", encoding="utf-8") as f:
    state = json.load(f)


# ---------- 第三步：提取训练日志 ----------
# trainer_state.json 里的 log_history 是一个列表，
# 记录了训练过程中的每一步/每个 epoch 的信息：
#   - 训练步：{"step": 50, "loss": 0.45, "epoch": 0.03}
#   - 评估：{"epoch": 1.0, "eval_loss": 0.31, "eval_accuracy": 0.87, "eval_f1": 0.87}
log_history = state.get("log_history", [])

# 分别收集训练记录和评估记录
train_logs = []   # 训练步的 loss
eval_logs = []    # 每个 epoch 的评估结果

for entry in log_history:
    if "eval_accuracy" in entry:
        # 这是评估记录，包含准确率和 F1
        eval_logs.append(entry)
    elif "loss" in entry and "eval_loss" not in entry:
        # 这是训练记录，只有 loss（排除评估时也会有的 loss）
        train_logs.append(entry)


# ---------- 第四步：打印每个 epoch 的评估结果 ----------
print("=" * 70)
print("每个 Epoch 的评估结果")
print("=" * 70)
print(f"{'Epoch':<8}{'Eval Loss':<14}{'Accuracy':<14}{'F1':<14}")
print("-" * 70)

for entry in eval_logs:
    epoch = entry.get("epoch", 0)
    eval_loss = entry.get("eval_loss", 0)
    acc = entry.get("eval_accuracy", 0)
    f1 = entry.get("eval_f1", 0)
    print(f"{epoch:<8.0f}{eval_loss:<14.4f}{acc:<14.4f}{f1:<14.4f}")

print()


# ---------- 第五步：打印训练 loss 的变化（可选） ----------
print("=" * 70)
print("训练 Loss（每 50 步记录一次）")
print("=" * 70)

# 为了不让输出太长，每隔几个记录打印一次
step_interval = max(1, len(train_logs) // 20)   # 最多打印约 20 行

for i, entry in enumerate(train_logs):
    if i % step_interval == 0 or i == len(train_logs) - 1:
        step = entry.get("step", 0)
        loss = entry.get("loss", 0)
        epoch = entry.get("epoch", 0)
        print(f"Step {step:<6}  Epoch {epoch:<6.2f}  Loss: {loss:.4f}")

print()


# ---------- 第六步：打印最优结果 ----------
# Trainer 在 load_best_model_at_end=True 时会记录最优 checkpoint 和最优指标
print("=" * 70)
print("最优结果")
print("=" * 70)

best_checkpoint = state.get("best_model_checkpoint", "N/A")
best_metric = state.get("best_metric", None)

print(f"最优 Checkpoint: {best_checkpoint}")
print(f"最优指标 (Eval Loss): {best_metric}")

# 如果 log_history 里有评估记录，手动找出最好的那个 epoch
if eval_logs:
    best_eval = max(eval_logs, key=lambda x: x.get("eval_accuracy", 0))
    print()
    print("从训练记录中提取的最优 Epoch：")
    print(f"  Epoch     : {best_eval.get('epoch', 0):.0f}")
    print(f"  Accuracy  : {best_eval.get('eval_accuracy', 0):.4f}")
    print(f"  F1        : {best_eval.get('eval_f1', 0):.4f}")
    print(f"  Eval Loss : {best_eval.get('eval_loss', 0):.4f}")

print()
print("=" * 70)
print("查看完成")
print("=" * 70)