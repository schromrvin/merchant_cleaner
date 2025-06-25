import os
import re
from collections import Counter

import joblib
import pandas as pd
from tqdm import tqdm

# Direct import from the processing package
from processing.tokenizer import custom_tokenize

# --- Constants ---
TRAINING_DATA_PATH = "data/synthetic_training_data.csv"
TOKEN_FREQS_SAVE_PATH = "model/token_frequencies.pkl"


def clean_token(token: str) -> str:
    """
    Standardizes a token by converting it to lowercase and removing all
    non-alphanumeric characters.
    """
    return re.sub(r"[^a-z0-9]", "", token.lower())


def main():
    """
    Calculates and saves the frequency of each cleaned token found in the
    'raw_transaction' column of the training data.
    """
    print("--- Creating Definitive Token Frequency File on CLEANED Tokens ---")

    # Load the training data from the specified CSV file
    try:
        df = pd.read_csv(TRAINING_DATA_PATH)
        print(f"✅ Successfully loaded {len(df)} rows from {TRAINING_DATA_PATH}")
    except FileNotFoundError:
        print(
            f"❌ CRITICAL: Training data not found at '{TRAINING_DATA_PATH}'. "
            "Please ensure the file exists."
        )
        return
    except Exception as e:
        print(f"❌ An error occurred while loading the data: {e}")
        return

    print("\nCalculating token frequencies on standardized, cleaned tokens...")

    # Process all rows to generate a flat list of cleaned tokens
    all_tokens = [
        clean_token(t)
        for text in tqdm(
            df["raw_transaction"].astype(str), desc="Processing Transactions"
        )
        for t in custom_tokenize(text)
    ]

    # Filter out any empty strings that may have resulted from cleaning
    all_tokens_filtered = [t for t in all_tokens if t]

    # Count the frequency of each unique token
    token_frequencies = Counter(all_tokens_filtered)

    # Ensure the target directory for the model exists
    os.makedirs(os.path.dirname(TOKEN_FREQS_SAVE_PATH), exist_ok=True)

    # Save the token frequencies object to a file
    joblib.dump(token_frequencies, TOKEN_FREQS_SAVE_PATH)

    print(f"\n✅ Token frequencies saved successfully to: {TOKEN_FREQS_SAVE_PATH}")
    print(f"   Counted {len(token_frequencies)} unique tokens.")


if __name__ == "__main__":
    main()