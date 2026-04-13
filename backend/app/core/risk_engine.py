# app/core/risk_engine.py
import pickle
import numpy as np

class RiskEngine:
    def __init__(self):
        with open('app/ml/random_forest.pkl', 'rb') as f:
            self.model = pickle.load(f)
            
    def calculate_risk(self, vision_data, acoustic_score=0, sdk_score=0):
        # Extract features
        features = np.array([[
            vision_data['density'],
            vision_data['vector_variance'],
            vision_data['compression_score'],
            acoustic_score,
            sdk_score
        ]])
        
        # Predict: 0=Green, 1=Yellow, 2=Red, 3=Critical
        risk_prediction = self.model.predict(features)[0]
        
        state_map = {0: "GREEN", 1: "YELLOW", 2: "RED", 3: "CRITICAL"}
        return state_map.get(risk_prediction, "GREEN")