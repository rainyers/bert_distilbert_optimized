import torch
from transformers import DistilBertTokenizer, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import evaluate
from models.distilbert_classifier import DistilBERTClassifier
from utils.data_utils import load_ag_news

# 检查 GPU
print("是否支持 CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("当前 GPU:", torch.cuda.get_device_name(0))

# 加载数据
dataset = load_ag_news()

#  限制样本数量，减小训练时间
dataset["train"] = dataset["train"].select(range(2000))
dataset["test"] = dataset["test"].select(range(500))

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding=True)

dataset = dataset.map(preprocess, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

model = DistilBERTClassifier(num_labels=4)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,  # 启用混合精度
    logging_steps=500,
)

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    print("模型是否在GPU:", next(model.parameters()).is_cuda)
    trainer.train()
    trainer.evaluate()
