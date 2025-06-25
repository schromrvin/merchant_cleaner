import os
import re
import sys
from collections import defaultdict

import joblib
import pandas as pd
from tqdm import tqdm

# Add parent directory to the system path to allow for local module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from processing.tokenizer import custom_tokenize

# --- Constants ---
# File paths for input data and the output model
ACRA_ENTITIES_PATH = "data/acra_entities.csv"
OTHER_UEN_ENTITIES_PATH = "data/other_uen_entities.csv"
MATCH_INDEX_SAVE_PATH = "model/match_index.pkl"

# A set of common, non-descriptive words to exclude from the index
INDEX_EXCLUSION_LIST = {
    "pte", "ltd", "llp", "lp", "inc", "llc", "corp", "bhd", "sbn", "the",
    "and", "of", "co",
}


def clean_token(token: str) -> str:
    """
    Standardizes a token for reliable matching by converting it to lowercase
    and removing all non-alphanumeric characters.
    """
    return re.sub(r"[^a-z0-9]", "", token.lower())


def main():
    """
    Builds and saves an inverted index from company names for fast merchant matching.
    """
    print("--- Building the DEFINITIVE Matching Index with Standardized Tokens ---")

    all_names_series = []

    # Load company names from the ACRA entities file
    try:
        df_acra = pd.read_csv(
            ACRA_ENTITIES_PATH, on_bad_lines="skip", usecols=["entity_name"]
        )
        all_names_series.append(df_acra["entity_name"].dropna())
        print(f"✅ Successfully loaded {len(df_acra)} rows from {ACRA_ENTITIES_PATH}")
    except FileNotFoundError:
        print(f"⚠️  Warning: ACRA entities file not found at {ACRA_ENTITIES_PATH}")
    except Exception as e:
        print(f"❌ Error loading {ACRA_ENTITIES_PATH}: {e}")

    # Load company names from the other UEN entities file
    try:
        df_other = pd.read_csv(
            OTHER_UEN_ENTITIES_PATH, on_bad_lines="skip", usecols=["entity_name"]
        )
        all_names_series.append(df_other["entity_name"].dropna())
        print(f"✅ Successfully loaded {len(df_other)} rows from {OTHER_UEN_ENTITIES_PATH}")
    except FileNotFoundError:
        print(f"⚠️  Warning: Other UEN entities file not found at {OTHER_UEN_ENTITIES_PATH}")
    except Exception as e:
        print(f"❌ Error loading {OTHER_UEN_ENTITIES_PATH}: {e}")

    # Exit if no data could be loaded
    if not all_names_series:
        print("❌ CRITICAL: No data files found or loaded. Exiting.")
        return

    # Combine all names into a single set of unique, stripped strings
    all_names = pd.concat(all_names_series, ignore_index=True)
    unique_names = set(str(name).strip() for name in all_names.unique())
    print(f"\nProcessing {len(unique_names)} unique entity names.")

    inverted_index = defaultdict(list)

    # Build the inverted index from the unique names
    for name in tqdm(unique_names, desc="Building Standardized Index"):
        raw_tokens = custom_tokenize(name)

        # Clean tokens and filter out excluded and short tokens
        core_tokens = {
            clean_token(t)
            for t in raw_tokens
            if clean_token(t) not in INDEX_EXCLUSION_LIST and len(clean_token(t)) > 1
        }

        # Map each core token to the full entity name
        for token in core_tokens:
            inverted_index[token].append(name)

    # Ensure the target directory exists
    os.makedirs(os.path.dirname(MATCH_INDEX_SAVE_PATH), exist_ok=True)

    # Save the index to a file using joblib for efficient storage
    joblib.dump(dict(inverted_index), MATCH_INDEX_SAVE_PATH)

    print(f"\n✅ Definitive matching index created successfully at: {MATCH_INDEX_SAVE_PATH}")
    print(f"   Indexed {len(inverted_index)} unique tokens.")


if __name__ == "__main__":
    main()