import shap
import joblib
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBRegressor

# ================= CONFIGURATION ================= #
FEATURE_PATH = Path("data/sip_features_model_ready.csv")
MODEL_PATH = Path("models/final_xgb_model.json")
SCALER_PATH = Path("models/scaler.pkl")
FEATURE_COLUMNS_PATH = Path("models/feature_columns.json")
OUT_DIR = Path("shap_exports")

# ================= THE INTERACTIVE ENGINE ================= #
class ShapEngine:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.df = None
        self.explainer = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads model and data once to save time."""
        if self.model is not None: return

        print("🧠 Loading SHAP Engine Artifacts...")
        try:
            # 1. Load Model
            self.model = XGBRegressor()
            self.model.load_model(MODEL_PATH)
            
            # 2. Load Scaler & Cols
            self.scaler = joblib.load(SCALER_PATH)
            with open(FEATURE_COLUMNS_PATH, "r") as f:
                self.feature_cols = json.load(f)

            # 3. Load Data & SANITIZE IT (Double Safety)
            raw_df = pd.read_csv(FEATURE_PATH)
            
            # ✅ Replace Infinity with NaN, then fill NaN with 0
            self.df = raw_df.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # 4. Initialize Explainer
            self.explainer = shap.TreeExplainer(self.model)
            
            print("✅ SHAP Engine Ready.")
        except Exception as e:
            print(f"❌ Error loading SHAP artifacts: {e}")

    def get_fund_explanation(self, fund_name_snippet):
        """
        Generates a matplotlib figure for the specified fund.
        Used by Streamlit UI.
        """
        # 1. Find the Fund
        mask = self.df["scheme_name"].str.contains(fund_name_snippet, case=False, na=False)
        target_rows = self.df[mask]
        
        if target_rows.empty:
            return None, f"Fund '{fund_name_snippet}' not found."
        
        # Take the first match
        row = target_rows.iloc[[0]]
        full_name = row["scheme_name"].values[0]

        # 2. Prepare Input Data
        X = row.copy()
        for col in self.feature_cols:
            if col not in X.columns: X[col] = 0
        X = X[self.feature_cols]
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # 3. SHAP Values
        shap_values = self.explainer(X_scaled)
        
        # 4. Plot
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        plt.title(f"Return Prediction Factors for: {full_name[:20]}...", fontsize=12)
        plt.tight_layout()
        
        fig = plt.gcf()
        plt.close()
        return fig, full_name

    def get_top_drivers(self, fund_name_snippet):
        """
        Returns the top drivers for the Explanation Agent (The Analyst).
        """
        mask = self.df["scheme_name"].str.contains(fund_name_snippet, case=False, na=False)
        if not mask.any(): return None
        
        row = self.df[mask].iloc[[0]]
        full_name = row["scheme_name"].values[0]

        X = row.copy()
        for col in self.feature_cols:
            if col not in X.columns: X[col] = 0
        X = X[self.feature_cols]
        
        X_scaled = self.scaler.transform(X)
        shap_values = self.explainer(X_scaled)
        values = shap_values[0].values 
        
        feature_map = list(zip(self.feature_cols, values))
        
        # Sort drivers (Positive = Pushing Up, Negative = Dragging Down)
        pos_drivers = sorted([f for f in feature_map if f[1] > 0], key=lambda x: x[1], reverse=True)
        neg_drivers = sorted([f for f in feature_map if f[1] < 0], key=lambda x: x[1])
        
        return {
            "name": full_name,
            "base_value": shap_values[0].base_values,
            "predicted_value": shap_values[0].base_values + values.sum(),
            "top_pos": pos_drivers[:3],
            "top_neg": neg_drivers[:3]
        }

# Singleton
_engine = None
def get_engine():
    global _engine
    if _engine is None:
        _engine = ShapEngine()
    return _engine

# ================= BATCH MODE ================= #
def main():
    print("running Batch SHAP Analysis...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    engine = ShapEngine()
    
    # ✅ FIX: Ensure X is strictly numeric and clean before scaling
    X = engine.df[engine.feature_cols].copy()
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    X_scaled = engine.scaler.transform(X)
    shap_values = engine.explainer.shap_values(X_scaled)

    # 1. Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=engine.feature_cols, show=False)
    plt.savefig(OUT_DIR / "shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Saved: {OUT_DIR / 'shap_summary.png'}")

    # 2. Dependence plots for the top 3 features
    top_features = engine.feature_cols[:3]
    for feat in top_features:
        plt.figure()
        shap.dependence_plot(feat, shap_values, X, feature_names=engine.feature_cols, show=False)
        plt.savefig(OUT_DIR / f"dependence_{feat}.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f" Saved: {OUT_DIR / f'dependence_{feat}.png'}")

if __name__ == "__main__":
    main()