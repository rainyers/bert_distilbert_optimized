import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from transformers import DistilBertTokenizer
from models.distilbert_classifier import DistilBERTClassifier
from utils.data_utils import load_ag_news

# 1. 加载模型和tokenizer
model = DistilBERTClassifier(num_labels=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(r"D:\PyCharm_Pros\PycharmProjects\python学习\自然语言处理\bert_distilbert_optimized\results\checkpoint-189\pytorch_model.bin", map_location=device))
model.to(device)
model.eval()

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
dataset = load_ag_news()
dataset = dataset["test"].select(range(500))

def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding=True)

dataset = dataset.map(preprocess, batched=True)
dataset.set_format(type='torch', columns=["input_ids", "attention_mask", "label"])

# 2. 获取预测值与真实值
true_labels = []
pred_labels = []
probs_all = []

with torch.no_grad():
    for batch in torch.utils.data.DataLoader(dataset, batch_size=64):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs[1]
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(preds.cpu().numpy())
        probs_all.extend(probs.cpu().numpy())

# 3. 打印指标报告
print("\n Classification Report:")
print(classification_report(true_labels, pred_labels, digits=4))

# 4. 混淆矩阵
cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["World", "Sports", "Business", "Sci/Tech"], yticklabels=["World", "Sports", "Business", "Sci/Tech"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# 5. ROC 曲线 (One-vs-Rest)
true_onehot = np.eye(4)[true_labels]
probs_all = np.array(probs_all)
plt.figure()
for i in range(4):
    fpr, tpr, _ = roc_curve(true_onehot[:, i], probs_all[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {i} AUC={roc_auc:.2f}")

plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve (Multiclass)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()
