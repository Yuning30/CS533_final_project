import preprocess
import torch

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AdamW, get_cosine_schedule_with_warmup

tokenizer = T5Tokenizer.from_pretrained("t5-small")

path = "entailment_bank/data/public_dataset/entailment_trees_emnlp2021_data_v2/dataset/task_1/"

def collate_fn(batch):
    # import pdb
    # pdb.set_trace()
    partial_proofs, labels = zip(*batch)
    proof_encoding = tokenizer(partial_proofs, padding="longest", max_length=512, truncation=True, return_tensors="pt")
    label_encoding = tokenizer(labels, padding="longest", max_length=128, truncation=True, return_tensors="pt")
    input_ids, attention_mask, output_ids = proof_encoding.input_ids, proof_encoding.attention_mask, label_encoding.input_ids
    output_ids[output_ids == tokenizer.pad_token_id] = -100

    return input_ids, attention_mask, output_ids

def finetune_T5(model, dataset_train, dataset_validation, batch_size, num_epoches, verbose=True, device='cuda'):
    model = model.to(device)

    # set up data loader
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # set up optimizer and scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 
                                   'weight_decay': 0.01},
                                  {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                                   'weight_decay': 0.}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)
    schdeuler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=1000)

    for epoch in range(num_epoches):
        # train the model
        model.train()
        loss_total = 0.0
        for input_ids, attention_mask, output_ids in dataloader_train:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            output_ids = output_ids.to(device)

            loss_batch_total = model(input_ids=input_ids, attention_mask=attention_mask, labels=output_ids).loss
            
            loss_batch_avg = loss_batch_total / input_ids.size(0)
            loss_batch_avg.backward()

            optimizer.step()
            schdeuler.step()
            optimizer.zero_grad()

            loss_total += loss_batch_total.item()

        loss_average = loss_total / len(dataloader_train.dataset)

        # eval the model on validation set
        val_loss_total = 0.0
        model.eval()
        with torch.no_grad():
            for input_ids, attention_mask, output_ids in dataloader_validation:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                output_ids = output_ids.to(device)

                val_loss_batch_total = model(input_ids=input_ids, attention_mask=attention_mask, labels=output_ids).loss

                val_loss_total += val_loss_batch_total.item()
        
        val_loss_average = val_loss_total / len(dataloader_validation.dataset)

        # print result for this epoch
        print("Epoch {:3d} | train avg loss {:8.4f} | val avg loss {:8.4f}".format(epoch, loss_average, val_loss_average))

if __name__ == "__main__":
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    dataset_train = preprocess.entailment_bank_dataset(path + "train.jsonl")
    dataset_validation = preprocess.entailment_bank_dataset(path + "dev.jsonl")
    
    finetune_T5(model, dataset_train, dataset_validation, batch_size=32, num_epoches=3, device='cpu')
    