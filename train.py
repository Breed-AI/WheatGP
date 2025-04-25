import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pickle
import pandas as pd
import numpy as np
import torch.nn.functional as F
import logging
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

np.random.seed(555)
torch.manual_seed(555)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

with open('./data_example/P_train.pkl', 'rb') as f: #The provided example phenotypes are normalized
    phenotype_train = pickle.load(f)
with open('./data_example/G_train.pkl', 'rb') as f: #
    genotype1d_train = pickle.load(f)

def split_genotype_data(genotype1d_dict, features_per_group):
    G = []
    for sample in genotype1d_dict.values():
        group_samples = []
        for i in range(5):
            start = i * features_per_group
            end = start + features_per_group if i < 4 else len(sample)
            group_samples.append(sample[start:end])
        G.append(group_samples)
    G = np.array(G)
    tensors = [torch.tensor(G[:, i], dtype=torch.float32) for i in range(5)]
    return tensors

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs, patience, scheduler, device):
    best_mse = float('inf')
    best_mae = float('inf')
    best_P = float('-inf')
    p = 0
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(epochs), desc="Epoch Training", leave=False):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            inputs = [b.to(device) for b in batch[:-1]]
            Y_train_batch = batch[-1].to(device)
            optimizer.zero_grad()
            outputs = model(*inputs)
            loss = criterion(outputs, Y_train_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            Y_preds = []
            Y_trues = []
            val_running_loss = 0.0
            for batch in val_loader:
                inputs = [b.to(device) for b in batch[:-1]]
                Y_val_batch = batch[-1].to(device)
                outputs = model(*inputs)
                Y_preds.append(outputs)
                Y_trues.append(Y_val_batch)
                loss = criterion(outputs, Y_val_batch)
                val_running_loss += loss.item()
            epoch_val_loss = val_running_loss / len(val_loader)
            val_losses.append(epoch_val_loss)

            Y_preds = torch.cat(Y_preds).cpu().numpy().flatten()
            Y_trues = torch.cat(Y_trues).cpu().numpy().flatten()

            # if MinMaxScaler
            #with open('scaler.pkl', 'rb') as f:
                #scaler = pickle.load(f)
            #Y_preds = Y_preds.reshape(-1, 1)
            #Y_trues = Y_trues.reshape(-1, 1)
            #Y_preds = scaler.inverse_transform(Y_preds).flatten()
            #Y_trues = scaler.inverse_transform(Y_trues).flatten()

            std_trues = np.std(Y_trues)
            std_preds = np.std(Y_preds)
            if std_trues == 0 or std_preds == 0:
                pearson_r = 0
            else:
                pearson_r = np.corrcoef(Y_trues, Y_preds)[0, 1]
            mae = F.l1_loss(torch.tensor(Y_preds, dtype=torch.float32).to(device),
                            torch.tensor(Y_trues, dtype=torch.float32).to(device)).item()
            mse = F.mse_loss(torch.tensor(Y_preds, dtype=torch.float32).to(device),
                             torch.tensor(Y_trues, dtype=torch.float32).to(device)).item()

            if pearson_r > best_P:
                best_mse = mse
                best_mae = mae
                best_P = pearson_r
                best_model_state = model.state_dict()
                p = 0
            else:
                p += 1
            if p >= patience:
                logging.info(f'Early stopping at epoch {epoch + 1}.')
                break
    print(f'The best pearson_r: {best_P}.')
    return best_P, best_mse, best_mae, best_model_state, Y_trues, Y_preds

script_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(script_dir, 'model')
sys.path.append(module_path)
try:
    from WheatGP_base import wheatGP_base
    print("wheatGP_base")
except ImportError as e:
    print(f"{e}")

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Training using device: {device}")

    features_per_sample = len(next(iter(genotype1d_train.values())))
    features_per_group = features_per_sample // 5
    G1train, G2train, G3train, G4train, G5train = split_genotype_data(genotype1d_train, features_per_group)

    lstm_dim = 10080 # Calculation based on the input size
    learning_rate = 0.005  # 0.0001 - 0.01
    batch_size_train = 64  # 16 - 128
    weight_decay = 0.0001  # 0.00001 - 0.001
    batch_size_val = 1
    epochs = 300  # Over 200
    patience = 50  # 20 - 100

    train_G = [G1train, G2train, G3train, G4train, G5train]
    train_Y = torch.tensor(np.array(list(phenotype_train.values()), dtype=np.float32), dtype=torch.float32).to(device)
    train_dataset = TensorDataset(*train_G, train_Y)
    dataset_size = len(train_dataset)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)

    model = wheatGP_base(lstm_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=90, gamma=0.1)

    logging.info("Training stage")
    best_P, best_mse, best_mae, best_model_state, Y_trues, Y_preds = train_and_validate(
        model, train_loader, val_loader, criterion, optimizer, epochs, patience, scheduler, device)

    best_model_path = f'best_model.ckpt'
    torch.save(best_model_state, best_model_path)
    logging.info(f'torch.save:{best_model_path}')

if __name__ == "__main__":
    main()