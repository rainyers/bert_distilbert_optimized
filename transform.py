import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from transformers import DistilBertTokenizer
from models.distilbert_classifier import DistilBERTClassifier
from utils.data_utils import load_ag_news

# 加载训练好的模型权重和分词器
model = DistilBERTClassifier(num_labels=4)  # 初始化模型结构（不加载参数）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 判断是否使用GPU

# 加载训练时保存的模型权重文件
model.load_state_dict(torch.load(
    r"D:\PyCharm_Pros\PycharmProjects\python学习\自然语言处理\bert_distilbert_optimized\results\checkpoint-189\pytorch_model.bin",
    map_location=device
))
model.to(device)    # 模型加载到GPU或CPU
model.eval()        # 设置为评估模式（关闭dropout等）

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")  # 加载分词器

# 加载测试数据集，选择前500条以加快评估速度
dataset = load_ag_news()
dataset = dataset["test"].select(range(500))

# 分词预处理函数
def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding=True)

# 对整个测试集应用分词
dataset = dataset.map(preprocess, batched=True)

# 设置数据为PyTorch格式，保留输入ID、注意力掩码和标签字段
dataset.set_format(type='torch', columns=["input_ids", "attention_mask", "label"])

# 获取预测结果和真实标签
true_labels = []    # 真实标签
pred_labels = []    # 预测标签
probs_all = []      # 每个样本对应4类的softmax概率

with torch.no_grad():  # 关闭自动求导，节省显存
    for batch in torch.utils.data.DataLoader(dataset, batch_size=64):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)  # 前向传播
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs[1]  # 兼容模型结构
        probs = torch.softmax(logits, dim=1)  # 将 logits 转为概率
        preds = torch.argmax(probs, dim=1)    # 选取最大概率对应的类别作为预测

        # 将当前batch的结果添加进总列表
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(preds.cpu().numpy())
        probs_all.extend(probs.cpu().numpy())

# 打印分类性能报告
print("\n Classification Report:")
print(classification_report(true_labels, pred_labels, digits=4))  # 精确率、召回率、F1-score、准确率

# 绘制混淆矩阵图
cm = confusion_matrix(true_labels, pred_labels)  # 计算混淆矩阵
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["World", "Sports", "Business", "Sci/Tech"],
            yticklabels=["World", "Sports", "Business", "Sci/Tech"])
plt.xlabel("Predicted")   # 横轴是预测标签
plt.ylabel("True")        # 纵轴是真实标签
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")  # 保存图像为文件
plt.show()

# 多类ROC曲线绘制（One-vs-Rest）
true_onehot = np.eye(4)[true_labels]       # 将真实标签转为one-hot形式（用于计算每类的ROC）
probs_all = np.array(probs_all)            # 预测的概率矩阵
plt.figure()

# 针对每一个类别绘制 ROC 曲线
for i in range(4):
    fpr, tpr, _ = roc_curve(true_onehot[:, i], probs_all[:, i])  # 计算假阳性率和真正率
    roc_auc = auc(fpr, tpr)  # 计算 AUC
    plt.plot(fpr, tpr, label=f"Class {i} AUC={roc_auc:.2f}")  # 绘图

plt.plot([0, 1], [0, 1], 'k--')  # 参考线：随机猜测
plt.title("ROC Curve (Multiclass)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png")  # 保存ROC图像
plt.show()
