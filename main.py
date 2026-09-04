import os
import requests
import numpy as np
import pandas as pd
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import xgboost as xgb
import torch
import torch.nn as nn
from scipy.stats import poisson

# =====================================================================
# CLIENTE THESATSAPI (CONECTOR /LLMS.TXT)
# =====================================================================
class TheStatsAPIClient:
    def __init__(self, api_key: str, base_url: str = "https://api.thestatsapi.com/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "x-api-key": self.api_key,
            "Accept": "application/json",
            "User-Agent": "SportsPredictor/2.0"
        }

    def get_team_lstm_tensor(self, team_id: str, last_n: int = 5) -> torch.Tensor:
        endpoint = f"{self.base_url}/teams/{team_id}/matches/history"
        try:
            res = requests.get(endpoint, headers=self.headers, params={"limit": last_n, "status": "completed"}, timeout=6)
            res.raise_for_status()
            data = res.json()
            matches = data.get("matches", data.get("data", []))
            
            sequence = []
            for m in matches:
                stats = m.get("stats", {})
                sequence.append([
                    float(m.get("goals_scored", 0)),
                    float(m.get("goals_conceded", 0)),
                    float(stats.get("total_shots", stats.get("shots", 10.0))),
                    float(stats.get("corners", 4.0))
                ])
            sequence.reverse()
            while len(sequence) < last_n:
                sequence.insert(0, [1.0, 1.0, 10.0, 4.0])
            
            seq_array = np.array(sequence[:last_n], dtype=np.float32)
            return torch.tensor(seq_array).unsqueeze(0)
        except Exception as e:
            print(f"[!] Warning en TheStatsAPI para '{team_id}': {e}. Usando matriz por defecto.")
            fallback = np.tile([1.0, 1.0, 10.0, 4.0], (last_n, 1)).astype(np.float32)
            return torch.tensor(fallback).unsqueeze(0)


# =====================================================================
# MODELO NEURONAL LSTM
# =====================================================================
class DynamicMomentumLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16):
        super(DynamicMomentumLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return self.sigmoid(out)


# =====================================================================
# INICIALIZACIÓN DE FASTAPI Y MODELOS
# =====================================================================
app = FastAPI(
    title="Match Prediction API + TheStatsAPI Engine",
    description="Inferencia híbrida LSTM + XGBoost + Dixon-Coles con datos de TheStatsAPI.",
    version="2.0.0"
)

class MatchRequest(BaseModel):
    home_team_id: str = Field(..., example="real-madrid")
    away_team_id: str = Field(..., example="barcelona")
    home_attack_rating: float = Field(1.45, example=1.45)
    away_defense_rating: float = Field(0.88, example=0.88)
    odds_home: Optional[float] = Field(None, example=2.10)
    odds_draw: Optional[float] = Field(None, example=3.40)
    odds_away: Optional[float] = Field(None, example=3.60)

lstm_model: Optional[nn.Module] = None
xgb_model: Optional[xgb.XGBRegressor] = None
api_client: Optional[TheStatsAPIClient] = None

MODEL_DIR = "saved_models"

@app.on_event("startup")
def startup_event():
    global lstm_model, xgb_model, api_client
    api_key = os.getenv("THESTATSAPI_KEY", "TU_API_KEY_AQUI")
    api_client = TheStatsAPIClient(api_key=api_key)

    lstm_model = DynamicMomentumLSTM(input_dim=4, hidden_dim=16)
    pt_path = os.path.join(MODEL_DIR, "lstm_momentum.pt")
    if os.path.exists(pt_path):
        try:
            lstm_model.load_state_dict(torch.load(pt_path))
            print("[✓] Modelo LSTM cargado correctamente.")
        except Exception as e:
            print(f"[!] Error cargando LSTM: {e}")
    lstm_model.eval()

    # === CÓDIGO CORREGIDO (sin sklearn) ===
    xgb_model = xgb.XGBRegressor(
        objective='count:poisson',
        n_estimators=30,
        max_depth=3,
        learning_rate=0.05,
        tree_method='hist',
        # device='cuda'  # descomenta si tienes GPU
    )
    xgb_path = os.path.join(MODEL_DIR, "xgboost_poisson.json")
    if os.path.exists(xgb_path):
        try:
            xgb_model.load_model(xgb_path)
            print("[✓] Modelo XGBoost cargado correctamente.")
        except Exception as e:
            print(f"[!] Error cargando XGBoost: {e}")


# =====================================================================
# MATRIZ DIXON-COLES Y EVALUACIÓN DE VALOR (+EV)
# =====================================================================
def dixon_coles_matrix(l_home: float, m_away: float, tau: float = -0.05, max_g: int = 8) -> np.ndarray:
    mat = np.zeros((max_g + 1, max_g + 1))
    for x in range(max_g + 1):
        for y in range(max_g + 1):
            px, py = poisson.pmf(x, l_home), poisson.pmf(y, m_away)
            if x == 0 and y == 0:
                adj = 1.0 - (l_home * m_away * tau)
            elif x == 0 and y == 1:
                adj = 1.0 + (l_home * tau)
            elif x == 1 and y == 0:
                adj = 1.0 + (m_away * tau)
            elif x == 1 and y == 1:
                adj = 1.0 - tau
            else:
                adj = 1.0
            mat[x, y] = px * py * max(0, adj)
    return mat / np.sum(mat)


def calculate_ev(prob: float, odds: Optional[float], kelly_fraction: float = 0.25):
    if not odds or odds <= 1.0:
        return {"ev": None, "stake_pct": 0.0, "is_value": False}
    ev_val = (prob * odds) - 1.0
    b = odds - 1.0
    kelly = ((prob * b) - (1.0 - prob)) / b if (b > 0 and ev_val > 0) else 0.0
    return {
        "ev": round(float(ev_val), 4),
        "stake_pct": round(float(max(0.0, kelly * kelly_fraction) * 100), 2),
        "is_value": bool(ev_val > 0.03)
    }


# =====================================================================
# ENDPOINT POST /predict
# =====================================================================
@app.post("/predict")
def predict_match(payload: MatchRequest):
    if lstm_model is None or api_client is None:
        raise HTTPException(status_code=500, detail="Los servicios no están inicializados.")

    t_home = api_client.get_team_lstm_tensor(payload.home_team_id)
    t_away = api_client.get_team_lstm_tensor(payload.away_team_id)

    with torch.no_grad():
        m_home = float(lstm_model(t_home).numpy().flatten()[0])
        m_away = float(lstm_model(t_away).numpy().flatten()[0])

    X_df = pd.DataFrame([{
        'home_attack_rating': payload.home_attack_rating,
        'away_defense_rating': payload.away_defense_rating,
        'lstm_momentum_home': m_home,
        'lstm_momentum_away': m_away
    }])

    if xgb_model and os.path.exists(os.path.join(MODEL_DIR, "xgboost_poisson.json")):
        l_home = float(xgb_model.predict(X_df)[0])
    else:
        l_home = 1.45

    l_away = float(payload.away_defense_rating * 1.1)

    mat = dixon_coles_matrix(l_home, l_away)
    p_home = float(np.sum(np.tril(mat, -1)))
    p_draw = float(np.sum(np.diag(mat)))
    p_away = float(np.sum(np.triu(mat, 1)))

    return {
        "match": f"{payload.home_team_id} vs {payload.away_team_id}",
        "momentum": {"home": round(m_home, 4), "away": round(m_away, 4)},
        "expected_goals": {"lambda_home": round(l_home, 2), "lambda_away": round(l_away, 2)},
        "probabilities": {"home": round(p_home, 4), "draw": round(p_draw, 4), "away": round(p_away, 4)},
        "value_analysis": {
            "home": calculate_ev(p_home, payload.odds_home),
            "draw": calculate_ev(p_draw, payload.odds_draw),
            "away": calculate_ev(p_away, payload.odds_away)
        }
        }
