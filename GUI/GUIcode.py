import tkinter as tk
from tkinter import filedialog, messagebox
from ttkbootstrap import Style
from ttkbootstrap import ttk
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
import pickle
import seaborn as sns
from scipy.stats import probplot, linregress
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import logging
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from tkinter.font import Font
from tkinter import font

class ConvPart(nn.Module):
    def __init__(self):
        super(ConvPart, self).__init__()
        self.conv0 = nn.Conv1d(1, 2, 1, padding=1)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv1d(2, 4, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(4, 8, 9, padding=1)
        self.relu2 = nn.ReLU()
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.drop(x)
        return x


class ShapeModule(nn.Module):
    def __init__(self):
        super(ShapeModule, self).__init__()

    def forward(self, x1, x2, x3, x4, x5, adjust_dim=True, concat=True):
        if adjust_dim:
            x1 = x1.unsqueeze(1)
            x2 = x2.unsqueeze(1)
            x3 = x3.unsqueeze(1)
            x4 = x4.unsqueeze(1)
            x5 = x5.unsqueeze(1)
        if concat:
            A_flat = x1.view(x1.size(0), -1)
            B_flat = x2.view(x2.size(0), -1)
            C_flat = x3.view(x3.size(0), -1)
            D_flat = x4.view(x4.size(0), -1)
            E_flat = x5.view(x5.size(0), -1)
            output = torch.cat((A_flat, B_flat, C_flat, D_flat, E_flat), dim=1)
            output = output.reshape(output.shape[0], 1, -1)

            return output
        else:
            return x1, x2, x3, x4, x5


class LSTMModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True):
        super(LSTMModule, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        try:
            lstm_out, (h_n, c_n) = self.lstm(x)
        except RuntimeError as e:
            logging.info(f"Error in LSTMModule forward: {e}")
            raise
        lstm_out = self.drop(lstm_out)
        return lstm_out


class wheatGP_base(nn.Module):
    def __init__(self, lstm_dim):
        super(wheatGP_base, self).__init__()
        self.ConvPart = ConvPart()
        self.lstm = LSTMModule(lstm_dim, 128)
        self.shape_module = ShapeModule()
        self.fc = nn.Sequential(nn.Linear(128, 1))

    def forward(self, x1, x2, x3, x4, x5):
        x1, x2, x3, x4, x5 = self.shape_module(x1, x2, x3, x4, x5, adjust_dim=True, concat=False)
        A = self.ConvPart(x1)
        B = self.ConvPart(x2)
        C = self.ConvPart(x3)
        D = self.ConvPart(x4)
        E = self.ConvPart(x5)

        output = self.shape_module(A, B, C, D, E, adjust_dim=False)
        try:
            output = self.lstm(output)
        except RuntimeError as e:
            logging.info(f"Error in wheatGP_base forward (LSTM part): {e}")
            raise

        output = output[:, -1, :]
        output = self.fc(output)

        return output

    def freeze_layers(self, freeze_conv=True, freeze_lstm=True, freeze_fc=True):
        for param in self.ConvPart.parameters():
            param.requires_grad = not freeze_conv
        for param in self.lstm.parameters():
            param.requires_grad = not freeze_lstm
        for param in self.fc.parameters():
            param.requires_grad = not freeze_fc


def preprocess_data(phenotype_file, genotype_file, seq_length):
    P = pd.read_csv(phenotype_file)
    G = pd.read_csv(genotype_file)

    X = G.to_numpy(dtype=np.int16)
    X[X == 0] = 2
    del G

    Y = P.to_numpy(dtype=np.float16)

    phenotype_dict = {}
    phenotype = Y[:, 1].reshape(-1, 1)

    for index, row in enumerate(phenotype):
        phenotype_dict[index] = row
    del Y, phenotype

    genotype1d_dict = {}
    genotype = X[:, 0:seq_length]

    for index, row in enumerate(genotype):
        genotype1d_dict[index] = row
    del X

    indices = list(phenotype_dict.keys())
    np.random.shuffle(indices)
    split_index = int(len(indices) * 0.9)
    train_indices = indices[:split_index]
    val_indices = indices[split_index:]

    phenotype_train = {i: phenotype_dict[i] for i in train_indices}
    phenotype_val = {i: phenotype_dict[i] for i in val_indices}
    genotype1d_train = {i: genotype1d_dict[i] for i in train_indices}
    genotype1d_val = {i: genotype1d_dict[i] for i in val_indices}

    return phenotype_train, genotype1d_train, phenotype_val, genotype1d_val

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

            std_trues = np.std(Y_trues)
            std_preds = np.std(Y_preds)
            if std_trues == 0 or std_preds == 0:
                pearson_r = 0
            else:
                pearson_r = np.corrcoef(Y_trues, Y_preds)[0, 1]
            mae = nn.functional.l1_loss(torch.tensor(Y_preds, dtype=torch.float32).to(device),
                                        torch.tensor(Y_trues, dtype=torch.float32).to(device)).item()
            mse = nn.functional.mse_loss(torch.tensor(Y_preds, dtype=torch.float32).to(device),
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



def test_model(model, test_loader, device):
    with torch.no_grad():
        Y_preds = []
        Y_trues = []
        for batch in test_loader:
            inputs = [b.to(device) for b in batch[:-1]]
            Y_val_batch = batch[-1].to(device)
            outputs = model(*inputs)
            Y_preds.append(outputs)
            Y_trues.append(Y_val_batch)
        Y_preds = torch.cat(Y_preds).cpu().numpy().flatten()
        Y_trues = torch.cat(Y_trues).cpu().numpy().flatten()

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

    return pearson_r, mse, mae, Y_trues, Y_preds


def save_test_results_to_csv(Y_trues, Y_preds, file_path):
    df = pd.DataFrame({
        'Observed Phenotype': Y_trues,
        'Predicted Phenotype': Y_preds
    })
    df.to_csv(file_path, index=False)

class TextHandler(logging.Handler):
    def __init__(self, text_widget):
        logging.Handler.__init__(self)
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)

        def append():
            self.text_widget.configure(state=tk.NORMAL)
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.configure(state=tk.DISABLED)
            self.text_widget.see(tk.END)

        self.text_widget.after(0, append)

class WheatGPGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("WheatGP")
        self.style = Style(theme='cyborg')

        window_width = 630
        window_height = 800
        self.root.geometry(f"{window_width}x{window_height}")

        self.root.resizable(False, False)

        bold_font = Font(family='Arial', size=10, weight='bold')

        self.style.configure('.', font=bold_font)
        main_frame = ttk.Frame(root, padding=10, style='MainFrame.TFrame')
        main_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        file_frame = ttk.Frame(main_frame, padding=5)
        file_frame.pack(pady=5, fill=tk.X)

        file_frame.columnconfigure(1, weight=1)
        file_frame.columnconfigure(4, weight=1)

        self.phenotype_file_entry = ttk.Entry(file_frame, width=50)
        self.phenotype_file_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        self.phenotype_file_button = ttk.Button(file_frame, text="Phenotype for training (.csv)",
                                                command=self.select_phenotype_file, style='Primary.TButton', width=35)
        self.phenotype_file_button.grid(row=0, column=4, padx=5, pady=5, sticky=tk.NSEW)

        self.genotype_file_entry = ttk.Entry(file_frame, width=50)
        self.genotype_file_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        self.genotype_file_button = ttk.Button(file_frame, text="Genotype for training (.csv)",
                                               command=self.select_genotype_file, style='Primary.TButton', width=35)
        self.genotype_file_button.grid(row=1, column=4, padx=5, pady=5, sticky=tk.NSEW)

        test_file_frame = ttk.Frame(main_frame, padding=5)
        test_file_frame.pack(pady=5, fill=tk.X)

        test_file_frame.columnconfigure(1, weight=1)
        test_file_frame.columnconfigure(4, weight=1)

        self.test_phenotype_file_entry = ttk.Entry(test_file_frame, width=50)
        self.test_phenotype_file_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        self.test_phenotype_file_button = ttk.Button(test_file_frame, text="Phenotype for testing (.pkl)",
                                                     command=self.select_test_phenotype_file, style='Primary.TButton', width=35)
        self.test_phenotype_file_button.grid(row=0, column=4, padx=5, pady=5, sticky=tk.NSEW)

        self.test_genotype_file_entry = ttk.Entry(test_file_frame, width=50)
        self.test_genotype_file_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        self.test_genotype_file_button = ttk.Button(test_file_frame, text="Genotype for testing (.pkl)",
                                                    command=self.select_test_genotype_file, style='Primary.TButton', width=35)
        self.test_genotype_file_button.grid(row=1, column=4, padx=5, pady=5, sticky=tk.NSEW)

        param_frame = ttk.Frame(main_frame, padding=5)
        param_frame.pack(pady=5, fill=tk.X)

        param_labels = [
            ("Sequence Length:", "seq_length_entry", "1280"),
            ("Random Seed:", "random_seed_entry", "555"),
            ("Learning Rate (LR):", "learning_rate_entry", "0.005"),
            ("Batch Size (BS):", "batch_size_train_entry", "64"),
            ("Weight Decay (WD):", "weight_decay_entry", "0.0001"),
            ("Epochs:", "epochs_entry", "300"),
            ("Patience:", "patience_entry", "50"),
            ("LSTM Dim:", "lstm_embedding_units_entry", "10080")
        ]

        for col in range(4):
            param_frame.columnconfigure(col, weight=1)
        for row in range(4):
            param_frame.rowconfigure(row, weight=1)

        for i, (label_text, entry_name, default_value) in enumerate(param_labels):
            row = i // 2
            col = i % 2
            label = ttk.Label(param_frame, text=label_text, font=("Arial", 10, "bold"))
            label.grid(row=row, column=col * 2, padx=5, pady=5, sticky=tk.W)
            entry = ttk.Entry(param_frame, width=20)
            entry.insert(0, default_value)
            entry.grid(row=row, column=col * 2 + 1, padx=5, pady=5, sticky=tk.EW)
            setattr(self, entry_name, entry)

        model_path_frame = ttk.Frame(main_frame, padding=5)
        model_path_frame.pack(pady=5, fill=tk.X)
        model_path_frame.columnconfigure(1, weight=1)
        model_path_frame.columnconfigure(4, weight=1)
        self.model_save_path_entry = ttk.Entry(model_path_frame, width=50)
        self.model_save_path_entry.insert(0, "")
        self.model_save_path_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        self.model_save_path_button = ttk.Button(model_path_frame, text="Select Model Save Path",
                                                 command=self.select_model_save_path, style='Primary.TButton', width=35)
        self.model_save_path_button.grid(row=0, column=4, padx=5, pady=5, sticky=tk.NSEW)

        self.load_model_entry = ttk.Entry(model_path_frame, width=50)
        self.load_model_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        self.load_model_button = ttk.Button(model_path_frame, text="Existing Model for testing",
                                            command=self.select_existing_model, style='Primary.TButton', width=35)
        self.load_model_button.grid(row=1, column=4, padx=5, pady=5, sticky=tk.NSEW)

        button_frame = ttk.Frame(main_frame, padding=5)
        button_frame.pack(pady=10, fill=tk.X)

        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)
        button_frame.columnconfigure(3, weight=1)
        button_frame.columnconfigure(4, weight=1)
        self.preprocess_button = ttk.Button(button_frame, text="Preprocess Data", command=self.preprocess,
                                            style='Primary.TButton', width=15)
        self.preprocess_button.grid(row=0, column=1, padx=5, sticky=tk.EW)

        self.train_button = ttk.Button(button_frame, text="Train Model", command=self.train, state=tk.DISABLED,
                                       style='Primary.TButton', width=15)
        self.train_button.grid(row=0, column=2, padx=5, sticky=tk.EW)

        self.test_button = ttk.Button(button_frame, text="Test Model", command=self.test, state=tk.DISABLED,
                                      style='Primary.TButton', width=15)
        self.test_button.grid(row=0, column=3, padx=5, sticky=tk.EW)

        self.save_results_button = ttk.Button(button_frame, text="Save Test Results", command=self.save_test_results,
                                              state=tk.DISABLED, style='Primary.TButton', width=15)
        self.save_results_button.grid(row=0, column=4, padx=5, sticky=tk.EW)

        self.log_text = tk.Text(main_frame, height=10, width=80, state=tk.DISABLED)

        log_font = font.Font(family='Arial', size=10)
        self.log_text.configure(font=log_font)

        self.log_text.pack(pady=10, fill=tk.BOTH, expand=True)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        text_handler = TextHandler(self.log_text)
        text_handler.setFormatter(formatter)
        logging.getLogger().addHandler(text_handler)
        logging.getLogger().setLevel(logging.INFO)

        self.phenotype_train = None
        self.genotype1d_train = None
        self.phenotype_val = None
        self.genotype1d_val = None
        self.model = None
        self.test_Y_trues = None
        self.test_Y_preds = None

        self._check_train_files_and_enable_button()
        self._check_test_files_and_enable_button()

    def select_phenotype_file(self):
        file_path = filedialog.askopenfilename()
        self.phenotype_file_entry.delete(0, tk.END)
        self.phenotype_file_entry.insert(0, file_path)
        logging.info(f"Selected phenotype file: {file_path}")
        self._check_train_files_and_enable_button()

    def select_genotype_file(self):
        file_path = filedialog.askopenfilename()
        self.genotype_file_entry.delete(0, tk.END)
        self.genotype_file_entry.insert(0, file_path)
        logging.info(f"Selected genotype file: {file_path}")
        self._check_train_files_and_enable_button()

    def select_test_phenotype_file(self):
        file_path = filedialog.askopenfilename()
        self.test_phenotype_file_entry.delete(0, tk.END)
        self.test_phenotype_file_entry.insert(0, file_path)
        logging.info(f"Selected test phenotype file: {file_path}")
        self._check_test_files_and_enable_button()

    def select_test_genotype_file(self):
        file_path = filedialog.askopenfilename()
        self.test_genotype_file_entry.delete(0, tk.END)
        self.test_genotype_file_entry.insert(0, file_path)
        logging.info(f"Selected test genotype file: {file_path}")
        self._check_test_files_and_enable_button()

    def _check_train_files_and_enable_button(self):
        phenotype_file = self.phenotype_file_entry.get()
        genotype_file = self.genotype_file_entry.get()
        if phenotype_file and genotype_file:
            self.train_button.config(state=tk.NORMAL)
        else:
            self.train_button.config(state=tk.DISABLED)

    def _check_test_files_and_enable_button(self):
        test_phenotype_file = self.test_phenotype_file_entry.get()
        test_genotype_file = self.test_genotype_file_entry.get()
        load_model_path = self.load_model_entry.get()
        if (test_phenotype_file and test_genotype_file and load_model_path) or self.model:
            self.test_button.config(state=tk.NORMAL)
        else:
            self.test_button.config(state=tk.DISABLED)

    def select_model_save_path(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".ckpt",
                                                 filetypes=[("Checkpoint files", "*.ckpt")])
        self.model_save_path_entry.delete(0, tk.END)
        self.model_save_path_entry.insert(0, file_path)
        logging.info(f"Selected model save path: {file_path}")

    def select_existing_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("Checkpoint files", "*.ckpt")])
        self.load_model_entry.delete(0, tk.END)
        self.load_model_entry.insert(0, file_path)
        logging.info(f"Selected existing model: {file_path}")
        try:
            lstm_embedding_units = int(self.lstm_embedding_units_entry.get())
            self.model = wheatGP_base(lstm_embedding_units)
            self.model.load_state_dict(torch.load(file_path))
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            logging.info(f"Successfully loaded model from {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            logging.error(f"Failed to load model: {e}")
        self._check_test_files_and_enable_button()

    def preprocess(self):
        phenotype_file = self.phenotype_file_entry.get()
        genotype_file = self.genotype_file_entry.get()
        try:
            seq_length = int(self.seq_length_entry.get())
            random_seed = int(self.random_seed_entry.get())
            np.random.seed(random_seed)
        except ValueError:
            messagebox.showerror("Error", "Please enter valid integers for sequence length and random seed.")
            logging.error("Invalid input for sequence length or random seed.")
            return

        if phenotype_file and genotype_file:
            try:
                self.phenotype_train, self.genotype1d_train, self.phenotype_val, self.genotype1d_val = preprocess_data(
                    phenotype_file, genotype_file, seq_length)

                with open('P_train.pkl', 'wb') as f:
                    pickle.dump(self.phenotype_train, f)
                with open('P_te.pkl', 'wb') as f:
                    pickle.dump(self.phenotype_val, f)
                with open('G_train.pkl', 'wb') as f:
                    pickle.dump(self.genotype1d_train, f)
                with open('G_te.pkl', 'wb') as f:
                    pickle.dump(self.genotype1d_val, f)

                messagebox.showinfo("Success", "Data preprocessing completed and saved.")
                logging.info("Data preprocessing completed and saved.")
            except Exception as e:
                messagebox.showerror("Error", f"Preprocessing error: {e}")
                logging.error(f"Preprocessing error: {e}")
        else:
            messagebox.showerror("Error", "Please select both phenotype and genotype files.")
            logging.error("Both phenotype and genotype files are required.")

    def train(self):
        if self.phenotype_train and self.genotype1d_train and self.phenotype_val and self.genotype1d_val:
            try:
                seq_length = int(self.seq_length_entry.get())
                random_seed = int(self.random_seed_entry.get())
                learning_rate = float(self.learning_rate_entry.get())
                batch_size_train = int(self.batch_size_train_entry.get())
                weight_decay = float(self.weight_decay_entry.get())
                epochs = int(self.epochs_entry.get())
                patience = int(self.patience_entry.get())
                lstm_embedding_units = int(self.lstm_embedding_units_entry.get())
                np.random.seed(random_seed)
                torch.manual_seed(random_seed)
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numerical values for hyperparameters.")
                logging.error("Invalid numerical values for hyperparameters.")
                return

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            logging.info(f"Training using device: {device}")

            features_per_sample = len(next(iter(self.genotype1d_train.values())))
            features_per_group = features_per_sample // 5
            G1train, G2train, G3train, G4train, G5train = split_genotype_data(self.genotype1d_train,
                                                                              features_per_group)
            train_G = [G1train, G2train, G3train, G4train, G5train]
            train_Y = torch.tensor(np.array(list(self.phenotype_train.values()), dtype=np.float32),
                                   dtype=torch.float32).to(device)

            train_dataset = TensorDataset(*train_G, train_Y)
            dataset_size = len(train_dataset)
            train_size = int(0.9 * dataset_size)
            val_size = dataset_size - train_size

            train_dataset, val_dataset = random_split(
                train_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

            train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

            self.model = wheatGP_base(lstm_embedding_units).to(device)

            load_model_path = self.load_model_entry.get()
            if load_model_path:
                try:
                    self.model.load_state_dict(torch.load(load_model_path))
                    logging.info(f"Loaded existing model from {load_model_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load model: {e}")
                    logging.error(f"Failed to load model: {e}")
                    return

            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            scheduler = StepLR(optimizer, step_size=90, gamma=0.1)

            logging.info("Training stage")
            best_P, best_mse, best_mae, best_model_state, _, _ = train_and_validate(
                self.model, train_loader, val_loader, criterion, optimizer, epochs, patience, scheduler, device)

            best_model_path = self.model_save_path_entry.get()
            torch.save(best_model_state, best_model_path)
            logging.info(f'torch.save:{best_model_path}')
            messagebox.showinfo("Success", "Training completed.")
            self.test_button.config(state=tk.NORMAL)
        else:
            messagebox.showerror("Error", "Data preprocessing is not completed.")
            logging.error("Data preprocessing is not completed.")

    def test(self):
        if self.model:
            if self.phenotype_val and self.genotype1d_val:
                phenotype_val = self.phenotype_val
                genotype1d_val = self.genotype1d_val
            else:
                test_phenotype_file = self.test_phenotype_file_entry.get()
                test_genotype_file = self.test_genotype_file_entry.get()
                if test_phenotype_file and test_genotype_file:
                    try:
                        with open(test_phenotype_file, 'rb') as f:
                            phenotype_val = pickle.load(f)
                        with open(test_genotype_file, 'rb') as f:
                            genotype1d_val = pickle.load(f)
                    except FileNotFoundError:
                        messagebox.showerror("Error", "Test data files not found.")
                        logging.error("Test data files not found.")
                        return
                else:
                    messagebox.showerror("Error", "No test data available.")
                    logging.error("No test data available.")
                    return
        else:
            messagebox.showerror("Error", "Model not trained or loaded.")
            logging.error("Model not trained or loaded.")
            return

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f"Testing using device: {device}")

        features_per_sample = len(next(iter(genotype1d_val.values())))
        features_per_group = features_per_sample // 5
        G1te, G2te, G3te, G4te, G5te = split_genotype_data(genotype1d_val, features_per_group)

        te_G = [G1te, G2te, G3te, G4te, G5te]
        te_Y = torch.tensor(np.array(list(phenotype_val.values()), dtype=np.float32), dtype=torch.float32).to(
            device)

        te_dataset = TensorDataset(*te_G, te_Y)
        te_loader = DataLoader(te_dataset, batch_size=1, shuffle=False)

        load_model_path = self.load_model_entry.get()
        if load_model_path:
            model_path = load_model_path
        else:
            model_path = self.model_save_path_entry.get()

        if not model_path:
            messagebox.showerror("Error", "Model path is empty.")
            logging.error("Model path is empty.")
            return
        import os
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Model file not found at {model_path}.")
            logging.error(f"Model file not found at {model_path}.")
            return
        try:
            logging.info(f"Attempting to load model from {model_path}")
            state_dict = torch.load(model_path)
            if state_dict is None:
                messagebox.showerror("Error", f"Failed to load model from {model_path}. Returned None.")
                logging.error(f"Failed to load model from {model_path}. Returned None.")
                return
            self.model.load_state_dict(state_dict)
            logging.info(f"Successfully loaded model from {model_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            logging.error(f"Failed to load model: {e}")
            return

        self.model.eval()

        pearson_r, mse, mae, self.test_Y_trues, self.test_Y_preds = test_model(self.model, te_loader, device)
        messagebox.showinfo("Success",
                            f"Testing completed. Pearson's r: {pearson_r:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
        self.save_results_button.config(state=tk.NORMAL)
        logging.info(f"Testing completed. Pearson's r: {pearson_r:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")

    def save_test_results(self):
        if self.test_Y_trues is not None and self.test_Y_preds is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                     filetypes=[("CSV files", "*.csv")])
            if file_path:
                save_test_results_to_csv(self.test_Y_trues, self.test_Y_preds, file_path)
                messagebox.showinfo("Success", "Test results saved successfully.")
                logging.info("Test results saved successfully.")
        else:
            messagebox.showerror("Error", "No test results available.")
            logging.error("No test results available.")

if __name__ == "__main__":
    root = tk.Tk()
    style = Style()
    style.configure('MainFrame.TFrame', background='#f0f0f0')
    app = WheatGPGUI(root)
    root.mainloop()