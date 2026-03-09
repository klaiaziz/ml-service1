from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

app = FastAPI()

# Expanded synthetic dataset for better variation
X = np.array([
    [1, 2, 5, 100],  # added description length as 4th feature
    [2, 3, 8, 150],
    [3, 2, 12, 200],
    [1, 4, 4, 80],
    [3, 5, 15, 250],
    [2, 5, 9, 120],
    [1, 3, 6, 90],
    [4, 4, 10, 180],
    [5, 3, 14, 220],
    [2, 6, 7, 110],
    [3, 4, 11, 160]
])

y_duration = np.array([4, 6, 10, 3, 12, 7, 5, 8, 11, 6, 9])
y_delay = np.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1])

reg_model = RandomForestRegressor(random_state=42)
reg_model.fit(X, y_duration)

clf_model = RandomForestClassifier(random_state=42)
clf_model.fit(X, y_delay)

class ProjectInput(BaseModel):
    complexity: int
    team_size: int
    features: int
    description: str = ""  # new field, optional

@app.get("/")
def home():
    return {"message": "ML Service Running"}

@app.post("/analyze")
def analyze(project: ProjectInput):
    # Simple description processing: adjust complexity based on length
    desc_length = len(project.description)
    adjusted_complexity = project.complexity + (desc_length // 100)  # add 1 per 100 chars

    data = np.array([[adjusted_complexity, project.team_size, project.features, desc_length]])

    duration = reg_model.predict(data)[0]
    delay_prob = clf_model.predict_proba(data)[0][1]

    return {
        "estimated_duration_weeks": round(float(duration), 2),
        "delay_risk_probability": round(float(delay_prob), 2)
    }