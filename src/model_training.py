import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib
import json

FEATURE_PATH = Path("data/sip_features_model_ready.csv")
MODEL_PATH = Path("models/final_xgb_model.json")
SCALER_PATH = Path("models/scaler.pkl")
FEATURE_COLUMNS_PATH = Path("models/feature_columns.json")

def load_features():
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"❌ Feature file not found at {FEATURE_PATH}. Run Step 2 first!")
        
    df = pd.read_csv(FEATURE_PATH)
    df = df.dropna()
    print(f"   Loaded features: {df.shape}")
    return df

def prepare_data(df: pd.DataFrame):
    y = df["cagr_percent"].astype(float)

    # 🚨 CRITICAL FIX: DROP THE "ANSWER KEY" COLUMNS
    # We must drop anything that allows the model to calculate CAGR directly.
    drop_cols = [
        # IDs and Text
        "scheme_code", "scheme_name", "base_scheme_name", "category", "category_risk_tier",
        "last_nav_date",
        
        # The Target
        "cagr_percent",
        
        # 🚨 LEAKAGE COLUMNS (The Cheats)
        "final_value", "profit", "gain", "return_percent", "absolute_return", 
        "projected_amount", "total_invested", "log_total_invested" 
        # Note: 'total_invested' isn't technically a leak, but it correlates 
        # perfectly with profit if you know the return, so safer to drop for pure rate prediction.
    ]

    # Drop columns that exist in the dataframe
    existing_drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=existing_drop)

    # Keep ONLY numeric columns as features
    X = X.select_dtypes(include=[np.number])

    # Cleanup inf/nan
    X = X.replace([np.inf, -np.inf], np.nan)
    mask_finite = ~X.isna().any(axis=1)
    X = X[mask_finite]
    y = y[mask_finite]

    print(f"   Using {len(X.columns)} features: {X.columns.tolist()}")
    print(f"   Training rows: {len(X)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, X.columns.tolist(), scaler

def evaluate_model_performance(X, y):
    print("\n🔍 Evaluating Real Model Accuracy...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    test_model = XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror",
        random_state=42, n_jobs=-1
    )
    test_model.fit(X_train, y_train)
    preds = test_model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print("-" * 30)
    print(f"   📊 RMSE: {rmse:.2f}%  (Target: < 5.0%)")
    print(f"   📉 MAE:  {mae:.2f}%")
    print(f"   📈 R²:   {r2:.4f}  (Target: 0.60 - 0.85)")
    print("-" * 30)

def train_final_model(X, y):
    print("\n🚀 Training Final Production Model...")
    model = XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror",
        random_state=42, n_jobs=-1
    )
    model.fit(X, y)
    return model

def save_artifacts(model, scaler, feature_cols):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(FEATURE_COLUMNS_PATH, "w") as f:
        json.dump(feature_cols, f)
    print(f"✅ Saved artifacts to {MODEL_PATH.parent}")

def main():
    try:
        df = load_features()
        X, y, feature_cols, scaler = prepare_data(df)
        evaluate_model_performance(X, y)
        model = train_final_model(X, y)
        save_artifacts(model, scaler, feature_cols)
    except Exception as e:
        print(f"\n❌ Training Failed: {e}")

if __name__ == "__main__":
    main()