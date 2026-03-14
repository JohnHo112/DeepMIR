import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import ShortChunkCNN
from dataloader import Artist20Dataset
from ulits import *

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    root = "dataset"
    train_dataset = Artist20Dataset(root, mode="train")
    val_dataset = Artist20Dataset(root, mode="val")
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=8)

    model = ShortChunkCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 100

    best_acc = 0.0
    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        # ---- Training Loop ----
        for wavs, melspecs, mfccs, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            wavs = wavs.to(device)        # shape (B, T)
            melspecs = melspecs.to(device)
            melspecs = torch.sum(melspecs, dim=1).unsqueeze(1)
            # print(melspecs.shape)
            labels = labels.to(device)              # shape (B,)

            optimizer.zero_grad()
            outputs = model(melspecs)              # shape (B, num_classes)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * melspecs.size(0)
            _, preds = outputs.max(1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_avg_loss = train_loss / train_total
        train_acc = train_correct / train_total
        print(f"Train | Loss: {train_avg_loss:.4f} | Acc: {train_acc:.4f}")
        train_loss_list.append(train_avg_loss)
        train_acc_list.append(train_acc)

        # ---- Validation ----
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for wavs, melspecs, mfccs, labels in val_dataloader:
                waves = wavs.to(device)
                melspecs = melspecs.to(device)
                melspecs = torch.sum(melspecs, dim=1).unsqueeze(1)
                labels = labels.to(device)

                outputs = model(melspecs)
                loss = criterion(outputs, labels)
                # print(outputs)
                # print(labels)

                val_loss += loss.item() * melspecs.size(0)
                _, preds = outputs.max(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_avg_loss = val_loss / val_total
        val_acc = val_correct / val_total
        print(f"Valid | Loss: {val_avg_loss:.4f} | Acc: {val_acc:.4f}")
        val_loss_list.append(val_avg_loss)
        val_acc_list.append(val_acc)

        # print(val_loss_list)
        # print(val_acc_list)
    

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "task2/results/model.pth")
            print(f'saving model with acc {best_acc:.5f}')
    
    plot_loss_acc(train_loss_list, train_acc_list, val_loss_list, val_acc_list) 

if __name__ == "__main__":
    train()