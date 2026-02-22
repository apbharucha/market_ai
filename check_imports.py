
import sys
import traceback

print(f"Python version: {sys.version}")

try:
    import schedule
    print("[OK] schedule imported successfully")
except ImportError as e:
    print(f"[X] schedule import failed: {e}")

try:
    import sklearn
    print(f"[OK] sklearn imported successfully (version {sklearn.__version__})")
    from sklearn.ensemble import RandomForestClassifier
    print("[OK] RandomForestClassifier imported")
except ImportError as e:
    print(f"[X] sklearn import failed: {e}")
    traceback.print_exc()

try:
    import xgboost
    print(f"[OK] xgboost imported successfully (version {xgboost.__version__})")
except ImportError as e:
    print(f"[X] xgboost import failed: {e}")
    traceback.print_exc()

try:
    import lightgbm
    print(f"[OK] lightgbm imported successfully (version {lightgbm.__version__})")
except ImportError as e:
    print(f"[X] lightgbm import failed: {e}")
    traceback.print_exc()
