import os
from transformers import Trainer
from transformers import AutoTokenizer, AutoModel

# For MacOS:
os.environ['KMP_DUPLICATE_LIB_OK']='True'
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModel.from_pretrained("bert-base-cased")
input = tokenizer("Masked R-CNN", return_tensors="pt")
output = model(**input)

print(output)