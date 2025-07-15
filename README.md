# 优化版 AG News 文本分类项目（基于 DistilBERT + GPU）

本项目使用 Hugging Face Transformers 与 PyTorch 实现基于 DistilBERT 的新闻文本分类，并启用 GPU 加速与混合精度训练。

## 特性

- 使用 `distilbert-base-uncased`（轻量版 BERT）
- 支持 GPU 与 fp16 加速
- 适配 RTX 3050 等入门显卡
- 自动从 HuggingFace 下载 AG News 数据集

## 快速使用

```bash
pip install -r requirements.txt
python train.py
```

## 介绍

- 数据集在data文件里，直接使用的arrow文件，csv文件是方便查看的
- train.py为训练代码，训练结果会保存在results文件夹里
- evalute.py为评估代码
- transform.py是可以将数据集转换为csv文件方便查看

## 硬件建议

- 推荐使用 CUDA 11.8 以上，显卡如 RTX 3050+
- 驱动需支持 GPU 计算（非仅图形）
