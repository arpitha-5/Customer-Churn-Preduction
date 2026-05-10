"""
Master script to run the full ML pipeline:
  Phase 2 → Data Preprocessing
  Phase 3 → EDA
  Phase 4 → Model Training
  Phase 5 → Model Optimization
  Phase 6 → Model Saving
"""
import os, sys

# Add paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "model"))

def main():
    print("=" * 60)
    print("   ChurnGuard AI - Full ML Pipeline")
    print("=" * 60)

    # Phase 2: Preprocessing
    from data_preprocessing import preprocess_pipeline
    preprocess_pipeline()

    # Phase 3: EDA
    from eda import run_eda
    run_eda()

    # Phase 4 & 5: Training + Optimization
    from train_models import run_training_pipeline
    run_training_pipeline()

    print("\n" + "="*60)
    print("  🎉 ALL PHASES COMPLETE!")
    print("  Run the app: python app.py")
    print("  Open: http://localhost:5000")
    print("="*60)

if __name__ == "__main__":
    main()
