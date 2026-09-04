from fastapi.testclient import TestClient
from main import app, dixon_coles_matrix, calculate_ev

client = TestClient(app)

def test_dixon_coles_matrix_sum():
    mat = dixon_coles_matrix(1.5, 1.1)
    assert round(float(mat.sum()), 4) == 1.0

def test_calculate_ev_positive():
    res = calculate_ev(prob=0.50, odds=2.20)
    assert res["is_value"] is True
    assert res["ev"] == 0.1

def test_calculate_ev_negative():
    res = calculate_ev(prob=0.30, odds=2.00)
    assert res["is_value"] is False
    assert res["ev"] == -0.4

def test_predict_endpoint_success():
    payload = {
        "home_team_id": "real-madrid",
        "away_team_id": "barcelona",
        "home_attack_rating": 1.50,
        "away_defense_rating": 0.90,
        "odds_home": 2.10,
        "odds_draw": 3.40,
        "odds_away": 3.50
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "probabilities" in data
    assert "value_analysis" in data
    assert "home" in data["probabilities"]
