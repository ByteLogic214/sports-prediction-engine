import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
import optuna
import numpy as np
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.manual_seed(42)
np.random.seed(42)

MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

class SequenceDataset(Dataset):
    def __init__(self, seqs, targets):
        self.seqs = torch.tensor(seqs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, idx):
        return self.seqs[idx], self.targets[idx]

class DynamicMomentumLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, num_layers=1, dropout=0.1):
        super(DynamicMomentumLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        out = self.fc(hn[-1])
        return self.sigmoid(out)

def generate_synthetic_data(n_samples=300):
    seqs = np.random.randn(n_samples, 5, 4).astype(np.float32)
    targets = np.random.poisson(1.5, n_samples).astype(np.float32)
    return seqs, targets

def objective_lstm(trial, seqs, targets):
    hidden_dim = trial.suggest_categorical('hidden_dim', [8, 16, 32])
    lr = trial.suggest_float('lr', 1e-3, 1e-2, log=True)
    batch_size = 32
    
    split = int(len(seqs) * 0.8)
    train_loader = DataLoader(SequenceDataset(seqs[:split], targets[:split]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SequenceDataset(seqs[split:], targets[split:]), batch_size=batch_size)

    model = DynamicMomentumLSTM(input_dim=4, hidden_dim=hidden_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(5):
        for x_b, y_b in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x_b).squeeze(), y_b)
            loss.backward()
            optimizer.step()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_b, y_b in val_loader:
            val_loss += criterion(model(x_b).squeeze(), y_b).item() * len(y_b)
    return val_loss / (len(seqs) - split)

def main():
    print("[1/3] Optimizando LSTM con Optuna...")
    seqs, targets = generate_synthetic_data()
    study_lstm = optuna.create_study(direction='minimize')
    study_lstm.optimize(lambda t: objective_lstm(t, seqs, targets), n_trials=5)

    best_hidden = study_lstm.best_params['hidden_dim']
    lstm_model = DynamicMomentumLSTM(input_dim=4, hidden_dim=best_hidden)
    
    # 1. Exportar Modelo PyTorch Nativo
    pt_path = os.path.join(MODEL_DIR, "lstm_momentum.pt")
    torch.save(lstm_model.state_dict(), pt_path)
    print(f"[✓] Modelo PyTorch guardado en {pt_path}")

    # 2. Intentar Exportación ONNX Opcional
    try:
        onnx_path = os.path.join(MODEL_DIR, "lstm_momentum.onnx")
        dummy_input = torch.randn(1, 5, 4, dtype=torch.float32)
        torch.onnx.export(
            lstm_model, dummy_input, onnx_path,
            input_names=['sequence_input'], output_names=['momentum_score'],
            dynamic_axes={'sequence_input': {0: 'batch_size'}, 'momentum_score': {0: 'batch_size'}}
        )
        print(f"[✓] Exportación ONNX completada en {onnx_path}")
    except Exception as e:
        print(f"[!] Omitiendo exportación ONNX en este entorno ({e}). La API continuará usando .pt")

    print("[2/3] Generando embeddings y optimizando XGBoost...")
    with torch.no_grad():
        momentum_scores = lstm_model(torch.tensor(seqs)).numpy().flatten()

    X = pd.DataFrame({
        'home_attack_rating': np.random.normal(1.2, 0.3, len(seqs)),
        'away_defense_rating': np.random.normal(0.9, 0.2, len(seqs)),
        'lstm_momentum_home': momentum_scores,
        'lstm_momentum_away': np.roll(momentum_scores, 1)
    })
    
    xgb_model = xgb.XGBRegressor(objective='count:poisson', n_estimators=30, max_depth=3, learning_rate=0.05)
    xgb_model.fit(X, targets)

    xgb_path = os.path.join(MODEL_DIR, "xgboost_poisson.json")
    xgb_model.save_model(xgb_path)

    meta = {"version": "1.0.0", "lstm_hidden": best_hidden, "features": list(X.columns)}
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[3/3] ¡Modelos procesados y guardados exitosamente en ./{MODEL_DIR}/!")

if __name__ == "__main__":
    main()
