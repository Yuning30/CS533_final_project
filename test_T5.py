import preprocess
import torch

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AdamW, get_cosine_schedule_with_warmup

device = 'cuda'
path = "entailment_bank/data/public_dataset/entailment_trees_emnlp2021_data_v2/dataset/task_1/"

tokenizer = T5Tokenizer.from_pretrained("t5-large")

model = T5ForConditionalGeneration.from_pretrained("./saved_model/T5_large_epoch_2")
model = model.to(device)

dataset_validation = preprocess.entailment_bank_dataset(path + "dev.jsonl")

for i in range(0, 15):
    x, y = dataset_validation[i]
    print(x)
    input_ids = tokenizer(x, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    outputs = model.generate(input_ids, num_beams=4, max_length=128, num_return_sequences=3)
    # import pdb
    # pdb.set_trace()
    print("ground truth\t", y)
    for idx, x in enumerate(outputs):
        print(f"{idx}-th output: {tokenizer.decode(x, skip_special_tokens=True)}")
