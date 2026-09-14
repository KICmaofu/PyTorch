from transformers import pipeline
print("transformers导入成功！")
# 测试文本分类小模型
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
res = classifier("I like deep learning")
print(res)