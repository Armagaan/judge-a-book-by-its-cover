"""Text classification using bi-directional RNNs"""
import sys
from time import time
from typing import Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe

torch.manual_seed(7)

if len(sys.argv) != 2:
    print(f"USAGE: python rnn.py <dataset_dir_path>")
    exit(1)
DIR_TRAIN = sys.argv[1]

# Download and import GloVe embeddings.
glove = GloVe(name="6B", dim=300, cache="../data/")
glove_embeddings = glove.vectors.clone()
glove_embeddings = torch.cat([glove_embeddings, torch.rand(2, glove_embeddings.shape[1])], axis=0)
glove_vocab = glove.stoi.copy()
# Create tokens for unknown keywords and padding.
glove_vocab["<unk>"] = glove_embeddings.shape[1] - 2
glove_vocab["<pad>"] = glove_embeddings.shape[1] - 1

# df = pd.read_csv(f"{DIR_TRAIN}/train_x.csv")
# lenghts = torch.Tensor([len(i) for i in df["Title"]])
# quantiles = torch.Tensor([0.25, 0.5, 0.75])
# print(torch.quantile(lenghts, quantiles)) # Use the 75% quantile as the max_length of sentences.


# * ---- Preprocessing Pipeline ----
class TextData(Dataset):
    def __init__(self, path_x, path_y) -> None:
        super().__init__()
        self.text = pd.read_csv(path_x).drop(["Cover_image_name"], axis=1)
        self.labels = pd.read_csv(path_y).drop(["Id"], axis=1)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index) -> Tuple[str, int]:
        id = self.text.iloc[index][0]
        text = self.text.iloc[index][1]
        label = self.labels.iloc[index][0]
        return label, text, id

pipeline_label = lambda x: int(x)
tokenizer = get_tokenizer("basic_english")

def pipeline_text(string):
    tokens = tokenizer(string)
    word_to_glove_key_mapping = list()
    for word in tokens:
        if word not in glove_vocab.keys():
            word_to_glove_key_mapping.append(glove_vocab["<unk>"])
        else:
            word_to_glove_key_mapping.append(glove_vocab[word])
    return word_to_glove_key_mapping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_batch(batch):
    max_length = 80
    list_label, list_text, list_id = [], [], []
    for (_label, _text, _id) in batch:
        list_label.append(pipeline_label(_label))
        processed_text = pipeline_text(_text)
        # trim
        if len(processed_text) > max_length:
            processed_text = processed_text[:max_length]
        # pad
        else:
            pad_len = max_length - len(processed_text)
            processed_text = processed_text + ([0] * pad_len)
        list_text.append(torch.tensor(processed_text, dtype=torch.int64))
        list_id.append(_id)
    labels = torch.tensor(list_label, dtype=torch.long)
    texts = torch.stack(list_text)
    ids = torch.tensor(list_id, dtype=torch.int)
    return labels.to(device), texts.to(device), ids.to(device)


# * ---- Model ----
class BiRnn(Module):
    def __init__(self, sentence_len:int) -> None:
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(glove_embeddings)
        self.rnn = nn.RNN(
            input_size=glove_embeddings.shape[1],
            hidden_size=128,
            num_layers=1,
            nonlinearity='tanh',
            batch_first=True,
            bidirectional=True
        )
        self.linear1 = nn.Linear(2 * 128 * sentence_len, 128)
        self.linear2 = nn.Linear(128, 30)
    
    def forward(self, text):
        x = self.embedding(text)
        x, __ = self.rnn(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear1(x).tanh()
        x = self.linear2(x)
        return x


# * ---- Training ----
def evaluate(model, dataloader):
    model.eval()
    acc, count = 0, 0
    list_ids = list()
    predicted_labels = list()

    with torch.no_grad():
        for label, text, ids in dataloader:
            predicted_label = model(text)
            acc += (predicted_label.argmax(1) == label).sum().item()
            count += label.size(0)
            predicted_labels.append(predicted_label.argmax(1))
            list_ids.append(ids)
    predicted_labels = torch.cat(predicted_labels)
    ids = torch.cat(list_ids)
    return predicted_labels, ids, acc/count

def train(model, dataloader_train, dataloader_val, optimizer, criterion, max_epochs:int =10):
    EPOCHS = max_epochs
    train_acc = 0.0
    count = 0
    log_interval = 100

    for epoch in range(1, EPOCHS+1):
        for index, (labels, texts, ids) in enumerate(dataloader_train):
            model.train()
            optimizer.zero_grad()
            predicted_labels = model(texts)
            loss = criterion(predicted_labels, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            train_acc += (predicted_labels.argmax(1) == labels).sum().item()
            count += labels.size(0)
            if index % log_interval == 0:
                __, __, val_acc = evaluate(model, dataloader_val)
                print(
                    f"Epoch: {epoch}/{EPOCHS:2d}"
                    f" | Batch: {index:4d}/{len(dataloader_train)}"
                    f" | Train Acc: {100 * train_acc / count:4.5f} %"
                    f" | Val Acc: {100 * val_acc:4.5f} %"
                )
        print("#" * 75)
    return


if __name__ == "__main__":
    EPOCHS = 50
    LR = 1e-4
    BATCH_SIZE = 64

    model = BiRnn(sentence_len=80).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Preprocess the data
    dataset_train = TextData(
        path_x=f"{DIR_TRAIN}/train_x.csv",
        path_y=f"{DIR_TRAIN}/train_y.csv"
    )
    val_split_idx = int(0.95 * len(dataset_train))
    split_train_, split_val_ = random_split(dataset_train, [val_split_idx, len(dataset_train) - val_split_idx])
    dataloader_train = DataLoader(
        dataset=split_train_,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch
    )
    dataloader_val = DataLoader(
        dataset=split_val_,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch
    )

    # Train the model.
    train(
        model=model,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        optimizer=optimizer,
        criterion=criterion,
        max_epochs=EPOCHS
    )
    torch.save(model, f"rnn_{int(time())}.pt")
    dataset_test = TextData(
        path_x=f"{DIR_TRAIN}/non_comp_test_x.csv",
        path_y=f"{DIR_TRAIN}/non_comp_test_y.csv"
    )

    # Get the test predictions.
    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch
    )
    model.eval()
    preds_test, ids, acc_test = evaluate(model=model, dataloader=dataloader_test)
    print(f"\nTest accuracy: {100 * acc_test:.4f} %")

    # Save output to disk.
    df = pd.DataFrame({
        "Id": ids.to("cpu").to(torch.int),
        "Genre": preds_test.to("cpu").to(torch.int)
    })
    df.sort_values(by=["Id"], inplace=True)
    df.to_csv("non_comp_test_pred_y.csv", index=False)
