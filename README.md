# ⚽ Motor de Predicción Deportiva (LSTM + XGBoost + Dixon-Coles)

Sistema de inferencia y optimización de apuestas con valor (+EV) utilizando integración en vivo con **TheStatsAPI**.

## 🏗️ Arquitectura del Sistema

1. **LSTM (PyTorch)**: Modela el *momentum* reciente de los equipos extrayendo secuencias temporales 3D `[1, 5, 4]`.
2. **XGBoost Regressor**: Predice las expectativas de gol ($\lambda$) con distribución Poisson.
3. **Ajuste Dixon-Coles**: Genera la matriz de probabilidades corregida para empates y marcadores bajos.
4. **Optuna**: Optimización automática de hiperparámetros.

## 🚀 Despliegue con Docker

```bash
# Construir la imagen
docker build -t sports-prediction-api .

# Ejecutar el contenedor
docker run -d -p 8000:8000 -e THESTATSAPI_KEY="TU_KEY" sports-prediction-api
{
  "home_team_id": "real-madrid",
  "away_team_id": "barcelona",
  "home_attack_rating": 1.45,
  "away_defense_rating": 0.88,
  "odds_home": 2.10,
  "odds_draw": 3.40,
  "odds_away": 3.60
}
