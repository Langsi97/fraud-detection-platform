import json
from pathlib import Path
import shutil
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_SRC = ROOT / "models" / "best_model_xgboost.pkl"
MODEL_DST = ROOT / "models" / "best_model.pkl"
METRICS_SRC = ROOT / "models" / "best_model_metrics.json"
METRICS_DST = ROOT / "models" / "metrics.json"
FEATURE_IMPORTANCE_DST = ROOT / "models" / "feature_importance.csv"
PREPROCESSOR_PATH = ROOT / "models" / "preprocessor.pkl"

if not MODEL_SRC.exists():
    raise FileNotFoundError(f"Expected source model artifact not found: {MODEL_SRC}")

shutil.copy2(MODEL_SRC, MODEL_DST)
print(f"Copied {MODEL_SRC.name} -> {MODEL_DST.name}")

if METRICS_SRC.exists():
    data = json.loads(METRICS_SRC.read_text(encoding="utf-8"))
    METRICS_DST.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Copied {METRICS_SRC.name} -> {METRICS_DST.name}")
else:
    print(f"No metrics source found at {METRICS_SRC}; skipping metrics copy.")

if PREPROCESSOR_PATH.exists():
    model = joblib.load(MODEL_DST)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
        feature_columns = None

        if hasattr(model, "feature_names_in_"):
            feature_columns = list(model.feature_names_in_)
        elif hasattr(preprocessor, "named_steps") and "preprocessor" in preprocessor.named_steps:
            feature_columns = preprocessor.named_steps["preprocessor"].transformers_[0][2]

        if feature_columns is None:
            raise ValueError("Cannot extract feature columns for feature importance export.")

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_.tolist()
            df = pd.DataFrame({"feature": feature_columns, "importance": importances})
            df.sort_values("importance", ascending=False, inplace=True)
            df.to_csv(FEATURE_IMPORTANCE_DST, index=False)
            print(f"Wrote {FEATURE_IMPORTANCE_DST.name}")
        else:
            print("Model does not expose feature_importances_ attribute; skipping feature importance export.")
