import os
import sys

import joblib
import pandas as pd
from sklearn_crfsuite import CRF
from tqdm import tqdm

# Add parent directory to the system path to allow for local module imports
# This is often necessary when running scripts from within a package structure.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.feature_extractor import generate_features
from processing.label_generator import generate_labels

# --- Constants ---
TRAINING_DATA_PATH = "data/synthetic_training_data.csv"
SPECIALIST_MODEL_SAVE_PATH = "model/specialist_crf_model.pkl"


def main():
    """
    Trains a Conditional Random Forest (CRF) model to identify merchant names
    in transaction strings.
    """
    print("--- Training the Definitive Specialist CRF Model ---")

    # Load the synthetic training data
    try:
        df = pd.read_csv(TRAINING_DATA_PATH)
        print(f"✅ Successfully loaded {len(df)} rows from {TRAINING_DATA_PATH}")
    except FileNotFoundError:
        print(
            f"❌ CRITICAL: Synthetic training data not found at '{TRAINING_DATA_PATH}'. "
            "Please run the data preparation script first."
        )
        return
    except Exception as e:
        print(f"❌ An error occurred while loading the data: {e}")
        return

    X_train, y_train = [], []

    # Generate features and labels for each sample in the dataset
    print("\nGenerating features and labels from the training data...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Rows"):
        tokens, labels = generate_labels(
            str(row["raw_transaction"]), str(row["clean_merchant"])
        )

        # Only include samples where a valid label sequence was generated
        if tokens and not all(label == "O" for label in labels):
            features = [generate_features(tokens, i) for i in range(len(tokens))]
            X_train.append(features)
            y_train.append(labels)

    if not X_train:
        print("\n❌ CRITICAL: No valid training samples were generated. Check data and labeling logic.")
        return

    print(f"\nGenerated {len(X_train)} valid training samples.")
    print("Training the specialist CRF model... (This may take some time)")

    # Initialize and train the CRF model
    crf = CRF(
        algorithm="lbfgs",
        c1=0.1,  # Coefficient for L1 penalty
        c2=0.1,  # Coefficient for L2 penalty
        max_iterations=100,
        all_possible_transitions=True,
        verbose=False,  # Set to True for detailed training output
    )

    try:
        crf.fit(X_train, y_train)
        print("✅ Specialist model training complete.")
    except Exception as e:
        print(f"❌ An error occurred during model training: {e}")
        return

    # Ensure the model directory exists and save the trained model
    try:
        os.makedirs(os.path.dirname(SPECIALIST_MODEL_SAVE_PATH), exist_ok=True)
        joblib.dump(crf, SPECIALIST_MODEL_SAVE_PATH)
        print(f"✅ Model saved successfully to: {SPECIALIST_MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"❌ An error occurred while saving the model: {e}")


if __name__ == "__main__":
    main()