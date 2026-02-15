from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

app = FastAPI()

# Synthetic dataset
X = np.array([
    [1, 2, 5],
    [2, 3, 8],
    [3, 2, 12],
    [1, 4, 4],
    [3, 5, 15],
    [2, 5, 9],
    [1, 3, 6]
])

y_duration = np.array([4, 6, 10, 3, 12, 7, 5])
y_delay = np.array([0, 0, 1, 0, 1, 1, 0])

reg_model = RandomForestRegressor()
reg_model.fit(X, y_duration)

clf_model = RandomForestClassifier()
clf_model.fit(X, y_delay)

class ProjectInput(BaseModel):
    complexity: int
    team_size: int
    features: int

@app.get("/")
def home():
    return {"message": "ML Service Running"}

@app.post("/analyze")
def analyze(project: ProjectInput):
    data = np.array([[project.complexity, project.team_size, project.features]])

    duration = reg_model.predict(data)[0]
    delay_prob = clf_model.predict_proba(data)[0][1]

    return {
        "estimated_duration_weeks": round(float(duration), 2),
        "delay_risk_probability": round(float(delay_prob), 2)
    }
