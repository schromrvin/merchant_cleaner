import os
import sys
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from tqdm import tqdm
from xgboost import XGBClassifier

# Add parent directory to the system path to allow for local module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.tokenizer import custom_tokenize
from processing.xgboost_feature_extractor import generate_cleaner_features

# --- Constants ---
# File paths for input data and output models
TRAINING_DATA_PATH = "data/synthetic_training_data.csv"
CLEANER_MODEL_SAVE_PATH = "model/xgboost_cleaner_model.pkl"
VECTORIZER_SAVE_PATH = "model/xgboost_vectorizer.pkl"
TOKEN_FREQS_SAVE_PATH = "model/token_frequencies.pkl"

# Set of legal suffixes to be handled carefully during labeling
LEGAL_SUFFIXES = {"pte", "ltd", "llp", "lp", "inc", "llc", "corp", "bhd", "sbn"}


def main():
    """
    Trains an XGBoost model to classify tokens from a raw transaction string
    as either part of the clean merchant name or as noise.
    """
    print("--- Training the XGBoost Cleaner Model (Data-Driven Logic) ---")

    # Load the synthetic training data
    try:
        df = pd.read_csv(TRAINING_DATA_PATH)
        print(f"✅ Successfully loaded {len(df)} rows from {TRAINING_DATA_PATH}")
    except FileNotFoundError:
        print(f"❌ CRITICAL: Synthetic training data not found at '{TRAINING_DATA_PATH}'.")
        return
    except Exception as e:
        print(f"❌ An error occurred while loading the data: {e}")
        return

    # --- Step 1: Calculate Global Token Frequencies ---
    print("\nCalculating token frequencies from all raw transactions...")
    all_raw_tokens = [
        t.lower()
        for text in tqdm(df["raw_transaction"].astype(str), desc="Tokenizing")
        for t in custom_tokenize(text)
    ]
    token_frequencies = Counter(all_raw_tokens)
    print(f"✅ Found {len(token_frequencies)} unique tokens.")

    # --- Step 2: Generate Features and Labels for each Token ---
    all_features, all_labels = [], []

    print("\nGenerating data-driven features for model training...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating Features"):
        raw_tokens = custom_tokenize(str(row["raw_transaction"]))
        original_clean_tokens = custom_tokenize(str(row["clean_merchant"]))

        # Create a set of "core" clean tokens, excluding legal suffixes that
        # might appear at the very end of the name.
        core_tokens = [
            t
            for i, t in enumerate(original_clean_tokens)
            if not (
                t.lower() in LEGAL_SUFFIXES and i >= len(original_clean_tokens) - 2
            )
        ]
        clean_tokens_set = set(t.lower() for t in core_tokens)

        if not raw_tokens or not clean_tokens_set:
            continue

        # For each token in the raw transaction, generate its features and a label
        for i, token in enumerate(raw_tokens):
            features = generate_cleaner_features(raw_tokens, i, token_frequencies)
            all_features.append(features)
            # Label is 1 if the token is part of the core merchant name, 0 otherwise
            all_labels.append(1 if token.lower() in clean_tokens_set else 0)

    if not all_features:
        print("❌ CRITICAL: No features were generated. Check the input data and logic.")
        return
    print(f"✅ Generated features for {len(all_features)} total token samples.")
    print(f"   Positive Samples (is_merchant_token=1): {sum(all_labels)}")
    print(f"   Negative Samples (is_merchant_token=0): {len(all_labels) - sum(all_labels)}")


    # --- Step 3: Vectorize Features and Train Model ---
    print("\nVectorizing features for XGBoost...")
    vectorizer = DictVectorizer(sparse=True)
    X = vectorizer.fit_transform(all_features)
    y = np.array(all_labels)
    print("✅ Vectorization complete.")

    print("\nTraining the XGBoost cleaner model...")
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        # Enable parallel processing for speed
        n_jobs=-1,
    )
    xgb.fit(X, y)
    print("✅ Cleaner model training complete.")

    # --- Step 4: Save the Model and Associated Artifacts ---
    try:
        os.makedirs(os.path.dirname(CLEANER_MODEL_SAVE_PATH), exist_ok=True)
        joblib.dump(xgb, CLEANER_MODEL_SAVE_PATH)
        joblib.dump(vectorizer, VECTORIZER_SAVE_PATH)
        joblib.dump(token_frequencies, TOKEN_FREQS_SAVE_PATH)
        print("\n✅ Cleaner model, vectorizer, and token frequencies saved successfully!")
    except Exception as e:
        print(f"\n❌ An error occurred while saving the model files: {e}")


if __name__ == "__main__":
    main()