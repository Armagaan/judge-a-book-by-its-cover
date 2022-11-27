"""Multimodal book genre classification"""

# * ---- Imports ----
from copy import deepcopy
import os
from PIL import Image
import sys
from time import time
os.environ["TRANSFORMERS_CACHE"]='/home/burouj/work/cache'

import pandas as pd
import torch
import torch.nn as nn
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms
from torchvision.models import resnet152
from transformers import BertTokenizer, BertModel, logging

# Since we won't be using all the ouputs of BERT.
# BertModel prints a message reminding us that we are not using all the outputs.
# We want to ignore that. So, we change the verbosity the default "info" to "error".
logging.set_verbosity_error()
torch.manual_seed(7)

if len(sys.argv) != 2:
    print("Usage: python comp.py <dataset dir path>")
    exit(1)
DIR_TRAIN = sys.argv[1]


# * ---- Preprocessing ----
class MultimodalDataset(Dataset):
    def __init__(self, path_x, path_y, dir_image) -> None:
        super().__init__()
        self.dir_image = dir_image
        self.df = pd.read_csv(path_x).drop(["Id"], axis=1)
        self.labels = pd.read_csv(path_y).drop(["Id"], axis=1)
        return
    
    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index):
        text = self.df["Title"].iloc[index]
        image = Image.open(f"{self.dir_image}/{self.df['Cover_image_name'].iloc[index]}")
        label = self.labels["Genre"].iloc[index]
        return (text, image, label)

pipeline_label = lambda x: int(x)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
pipeline_text = lambda x: tokenizer(x, return_tensors="pt", truncation=True, padding=True)
def pipeline_image(image):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(image)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def collate_batch(batch):
    text = list(map(lambda x:x[0], batch))
    tokenized = pipeline_text(text)
    list_image, list_label = list(), list()
    for (__, image, label) in batch:
        list_image.append(pipeline_image(image))
        list_label.append(pipeline_label(label))
    list_image = torch.stack(list_image)
    list_label = torch.tensor(list_label, dtype=torch.long)
    return (
        tokenized['input_ids'].to(device),
        tokenized['attention_mask'].to(device),
        list_image.to(device),
        list_label.to(device)
    )


# * ---- Model ----
class Book_genre(Module):
    def __init__(self) -> None:
        super().__init__()
        # Text
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False

        # Image
        self.resnet = resnet152(weights="ResNet152_Weights.IMAGENET1K_V1")
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        for param in self.resnet.parameters():
            param.requires_grad = False

        # NN
        self.fc1 = nn.Linear(768+2048, 128) # 768 for bert, 2048 for resnet
        self.fc2 = nn.Linear(128, 30)
        return

    def forward(self, text_input_ids, text_attention, images):
        # Text
        x_text = self.bert(input_ids=text_input_ids, attention_mask=text_attention)
        x_text = x_text.last_hidden_state[:, 0, :] # extract the cls embedding.
        # Image
        x_image = self.resnet(images)
        x_image = x_image.squeeze()
        # NN
        x = torch.cat([x_text, x_image], dim=1)
        x = self.fc1(x).relu()
        x = self.fc2(x)
        return x

def evaluate(model, dataloader):
    model.eval()
    acc, count = 0, 0
    list_predictions = list()
    with torch.no_grad():
        for texts_input_ids, texts_attention, images, labels in dataloader:
            predictions = model(texts_input_ids, texts_attention, images)
            acc += (predictions.argmax(1) == labels).sum().item()
            count += labels.size(0)
            list_predictions.append(predictions.argmax(1))
    return acc/count, torch.cat(list_predictions)

def train(model, dataloader_train, dataloader_val, optimizer, criterion, max_epochs:int =10):
    count = 0
    log_interval = 100
    PATIENCE = 10
    patience = 0
    train_acc = 0
    best_val_acc = 0
    best_model = None

    for epoch in range(1, max_epochs+1):
        for index, (texts_input_ids, texts_attention, images, labels) in enumerate(dataloader_train):
            model.train()
            optimizer.zero_grad()
            predictions = model(texts_input_ids, texts_attention, images)
            loss = criterion(predictions, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            train_acc += (predictions.argmax(1) == labels).sum().item()
            count += labels.size(0)
            if index % log_interval == 0:
                val_acc, __ = evaluate(model, dataloader_val)
                print(
                    f"Epoch: {epoch}/{max_epochs:02d}"
                    f" | Batch: {index:4d}/{len(dataloader_train)}"
                    f" | Train Acc: {100 * train_acc / count:4.5f} %"
                    f" | Val Acc: {100 * val_acc:4.5f} %"
                )
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    patience = 0
                    best_model = deepcopy(model)
                else:
                    patience += 1
                if patience == PATIENCE:
                    print("[STOP] Patience expired.")
                    return best_model
        print("#" * 75)
    print("[STOP] Maximum epochs reached.")
    return best_model


if __name__ == "__main__":
    # * ---- Training ----
    # Initiate the model.
    EPOCHS = 50
    LR = 1e-4
    BATCH_SIZE = 64
    model = Book_genre().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Create combined train_x and train_y files.
    train_x_files = ["train_x", "non_comp_test_x"]
    train_y_files = ["train_y", "non_comp_test_y"]
    df_x1 = pd.read_csv(f"{DIR_TRAIN}/train_x.csv")
    df_x2 = pd.read_csv(f"{DIR_TRAIN}/non_comp_test_x.csv")
    df_x3 = pd.concat([df_x1, df_x2])

    df_y1 = pd.read_csv(f"{DIR_TRAIN}/train_y.csv")
    df_y2 = pd.read_csv(f"{DIR_TRAIN}/non_comp_test_y.csv")
    df_y3 = pd.concat([df_y1, df_y2])
    
    df_x3.to_csv(f"{DIR_TRAIN}/perk_combined_train_x.csv")
    df_y3.to_csv(f"{DIR_TRAIN}/perk_combined_train_y.csv")

    # Prepare the data.
    dataset_train = MultimodalDataset(
        path_x=f"{DIR_TRAIN}/perk_combined_train_x.csv",
        path_y=f"{DIR_TRAIN}/perk_combined_train_y.csv",
        dir_image=f"{DIR_TRAIN}/images"
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

    # Train
    best_model = train(
        model=model,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        optimizer=optimizer,
        criterion=criterion,
        max_epochs=EPOCHS
    )
    torch.save(best_model, f"multimodal_{int(time())}.pt")


    # * ---- Testing ----
    # Our model requires both x and y files to run.
    # Hence, we create a dummy test file.
    df_comp_test_x = pd.read_csv(f"{DIR_TRAIN}/comp_test_x.csv")
    perk_dummpy_comp_test_y = pd.DataFrame({
        "Id": list(range(len(df_comp_test_x))),
        "Genre": [0 for __ in range(len(df_comp_test_x))]
    })
    perk_dummpy_comp_test_y.to_csv(f"{DIR_TRAIN}/perk_dummpy_comp_test_y.csv")

    dataset_test = MultimodalDataset(
        path_x=f"{DIR_TRAIN}/comp_test_x.csv",
        path_y=f"{DIR_TRAIN}/perk_dummpy_comp_test_y.csv",
        dir_image=f"{DIR_TRAIN}/images"
    )
    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch
    )
    __, predictions_test = evaluate(model=best_model, dataloader=dataloader_test)

    # Save output to disk.
    df = pd.DataFrame({
        "Id": torch.Tensor(range(predictions_test.size(0))),
        "Genre": predictions_test.to("cpu")
    })
    df.to_csv("comp_test_pred_y.csv", index=False)
