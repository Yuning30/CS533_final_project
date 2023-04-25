import preprocess
import torch
import tqdm

from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AdamW, get_cosine_schedule_with_warmup

tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

path = "entailment_bank/data/public_dataset/entailment_trees_emnlp2021_data_v2/dataset/task_1/"

def collate_fn(batch):
    premises, conclusion, scores = zip(*batch)
    step_encoding = tokenizer(text=premises, text_pair=conclusion, padding="longest", max_length=512, truncation=True, return_tensors="pt")
    input_ids, attention_mask= step_encoding.input_ids, step_encoding.attention_mask

    return input_ids, attention_mask, torch.tensor(scores)

def finetune_roberta(model, dataset_train, dataset_validation, batch_size, num_epoches, verbose=True, device='cuda'):
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
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
    schdeuler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=100000)

    # eval the model on validation set
    val_loss_total = 0.0
    model.eval()
    with torch.no_grad():
        for input_ids, attention_mask, scores in tqdm.tqdm(dataloader_validation):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            scores = scores.to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            predicted_scores = torch.sigmoid(logits).squeeze()

            val_loss_batch_average = torch.nn.MSELoss()(predicted_scores, scores)

            val_loss_total += (val_loss_batch_average.item() * batch_size)
    
    val_loss_average = val_loss_total / len(dataloader_validation.dataset)
    print("initial loss", val_loss_average)

    for epoch in range(num_epoches):
        # train the model
        model.train()
        loss_total = 0.0
        for input_ids, attention_mask, scores in tqdm.tqdm(dataloader_train):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            scores = scores.to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            predicted_scores = torch.sigmoid(logits).squeeze()

            loss_batch_avg = torch.nn.MSELoss()(predicted_scores, scores)
            
            loss_batch_avg.backward()

            optimizer.step()
            schdeuler.step()
            optimizer.zero_grad()

            loss_total += (loss_batch_avg.item() * batch_size)

        loss_average = loss_total / len(dataloader_train.dataset)

        # eval the model on validation set
        val_loss_total = 0.0
        model.eval()
        with torch.no_grad():
            for input_ids, attention_mask, scores in tqdm.tqdm(dataloader_validation):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                scores = scores.to(device)

                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                predicted_scores = torch.sigmoid(logits).squeeze()

                val_loss_batch_average = torch.nn.MSELoss()(predicted_scores, scores)

                val_loss_total += (val_loss_batch_average.item() * batch_size)
        
        val_loss_average = val_loss_total / len(dataloader_validation.dataset)

        # print result for this epoch
        print("Epoch {:3d} | train avg loss {:8.4f} | val avg loss {:8.4f}".format(epoch, loss_average, val_loss_average))

        model.save_pretrained(f"saved_model/roberta_large_epoch_{epoch}")

if __name__ == "__main__":
    model = RobertaForSequenceClassification.from_pretrained('roberta-large', problem_type='regression', num_labels=1)
    dataset_train = preprocess.roberta_dataset(path + "train.jsonl")
    dataset_validation = preprocess.roberta_dataset(path + "dev.jsonl")
    
    finetune_roberta(model, dataset_train, dataset_validation, batch_size=32, num_epoches=20, device='cuda')
    
