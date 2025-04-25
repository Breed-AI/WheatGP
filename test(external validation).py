import sys
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import logging

np.random.seed(555)
torch.manual_seed(555)
lstm_dim = 10080

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
script_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(script_dir, 'model')
sys.path.append(module_path)
try:
    from WheatGP_base import wheatGP_base
    print("wheatGP_base")
except ImportError as e:
    print(f"{e}")

def print_available_gpus():
    if torch.cuda.is_available():
        logging.info(f"Number of available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logging.info("No GPU available, using CPU.")

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

def main():
    print_available_gpus()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Testing using device: {device}")

    try:
        with open('./data_example/P_te.pkl', 'rb') as f: #The provided example phenotypes are normalized
            phenotype_te = pickle.load(f)
        with open('./data_example/G_te.pkl', 'rb') as f: #
            genotype1d_te = pickle.load(f)
    except FileNotFoundError:
        logging.error("NOT FOUND")
        return

    features_per_sample = len(next(iter(genotype1d_te.values())))
    features_per_group = features_per_sample // 5
    G1te, G2te, G3te, G4te, G5te = split_genotype_data(genotype1d_te, features_per_group)


    te_G = [G1te, G2te, G3te, G4te, G5te]
    te_Y = torch.tensor(np.array(list(phenotype_te.values()), dtype=np.float32), dtype=torch.float32).to(device)

    te_dataset = TensorDataset(*te_G, te_Y)
    te_loader = DataLoader(te_dataset, batch_size=1, shuffle=False)

    best_model_path = 'best_model.ckpt'
    try:
        model = wheatGP_base(lstm_dim).to(device)
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        logging.info(f"Loaded best model from {best_model_path}")
    except FileNotFoundError:
        logging.error(f"NOT FOUND")
        return

    with torch.no_grad():
        Y_preds = []
        Y_trues = []
        for batch in te_loader:
            inputs = [b.to(device) for b in batch[:-1]]
            Y_val_batch = batch[-1].to(device)
            outputs = model(*inputs)
            Y_preds.append(outputs)
            Y_trues.append(Y_val_batch)
        Y_preds = torch.cat(Y_preds).cpu().numpy().flatten()
        Y_trues = torch.cat(Y_trues).cpu().numpy().flatten()

        #if MinMaxScaler
        #with open('scaler.pkl', 'rb') as f:
            #scaler = pickle.load(f)
        #Y_preds = Y_preds.reshape(-1, 1)
        #Y_trues = Y_trues.reshape(-1, 1)
        #Y_preds = scaler.inverse_transform(Y_preds).flatten()
        #Y_trues = scaler.inverse_transform(Y_trues).flatten()

    results_df = pd.DataFrame({
        'Observed_Phenotype': Y_trues,
        'Predicted_Phenotype': Y_preds
    })

    # if save
    #csv_path = 'phenotype_predictions.csv'
    #results_df.to_csv(csv_path, index=False)
    #logging.info(f"save to {csv_path}")

    pearson_r = np.corrcoef(Y_trues, Y_preds)[0, 1]
    mse = np.mean((Y_trues - Y_preds) ** 2)
    mae = np.mean(np.abs(Y_trues - Y_preds))

    logging.info(f"Pearson's r: {pearson_r}")
    logging.info(f"MSE: {mse}")
    logging.info(f"MAE: {mae}")

    plt.figure(figsize=(10, 8))
    plt.scatter(Y_trues, Y_preds, alpha=0.5, color='blue', label='Data Points')

    slope, intercept, _, _, _ = linregress(Y_trues, Y_preds)
    line_x = np.linspace(min(Y_trues), max(Y_trues), 100)
    line_y = slope * line_x + intercept
    plt.plot(line_x, line_y, color='red', label=f'Fitted Line (y = {slope:.2f}x + {intercept:.2f})')

    plt.xlabel('Observed Phenotype', fontsize=14)
    plt.ylabel('Predicted Phenotype', fontsize=14)
    plt.title(f'Prediction Results (Pearson\'s r: {pearson_r:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f})', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()