import random
import re
import pandas as pd
from tqdm import tqdm

# --- Constants ---
# File paths for input data and output synthetic data
ACRA_ENTITIES_PATH = "data/acra_entities.csv"
OTHER_UEN_ENTITIES_PATH = "data/other_uen_entities.csv"
SYNTHETIC_DATA_SAVE_PATH = "data/synthetic_training_data.csv"

# A comprehensive list of words to be used as noise in synthetic transactions
COMPREHENSIVE_NOISE_WORDS = [
    # Transaction-related terms
    "payment", "txn", "debit", "credit", "card", "purchase", "store", "shop",
    "online", "bill", "receipt", "charge", "fee", "transfer", "giro", "atm",
    "withdrawal", "singapore", "sg", "suntec", "marina", "tampines", "jurong",

    # Common payment platforms and prefixes
    "paypal", "google", "apple", "amex", "dbs", "posb", "uob", "ocbc", "visa",
    "mastercard", "nets", "grab", "fave", "shopee", "lazada", "gpay",
    "paynow", "paywave", "contactless", "cl",

    # Geographical and address components
    "east", "west", "north", "south", "central", "rd", "road", "st",
    "street", "ave", "avenue", "blvd", "boulevard", "ctrl", "cres",
    "crescent", "dr", "drive", "pl", "place", "link", "walk", "hwy",
    "highway", "bldg", "building", "ctr", "centre", "tower", "point",
    "plaza", "square", "mall", "junction", "complex", "city", "town",
    "view", "park", "hill", "hdb", "hub",

    # Specific Singapore locations
    "bedok", "changi", "pasir", "ris", "punggol", "sengkang", "hougang",
    "amk", "ang", "mo", "kio", "bishan", "toa", "payoh", "novena", "orchard",
    "newton", "bukit", "timah", "clementi", "boon", "lay", "yishun",
    "woodlands", "choa", "chu", "kang", "geylang", "kallang",

    # Singapore Mall Names
    "bedok mall", "century square", "changi city point", "downtown east",
    "eastpoint mall", "jewel", "katong", "i12", "parkway parade",
    "tampines 1", "tampines mall", "white sands", "amk hub", "canberra plaza",
    "causeway point", "northpoint city", "compass one", "heartland mall",
    "hougang mall", "nex", "junction 10", "west mall", "vivocity",
    "harbourfront centre", "alexandra retail centre", "the clementi mall",
    "gek poh", "imm", "jcube", "jem", "jurong point", "pioneer mall",
    "queensway", "the rail mall", "the star vista", "west coast plaza",
    "westgate", "bugis junction", "bugis+", "capitol", "cineleisure",
    "the centrepoint", "city square mall", "citylink mall", "funan",
    "great world city", "holland village", "ion orchard", "junction 8",
    "knightsbridge", "lucky plaza", "marina bay sands", "marina square",
    "millenia walk", "mustafa centre", "ngee ann city", "orchard central",
    "orchard gateway", "palais renaissance", "the paragon", "peoples park",
    "plaza singapura", "raffles city", "shaw house", "sim lim square",
    "suntec city", "tiong bahru plaza", "thomson plaza", "seletar mall",
    "waterway point", "sembawang", "hillion mall", "tanglin mall",
    "united square", "kallang wave mall", "888 plaza", "elias mall",
    "joo chiat complex", "loyang point", "rivervale plaza",
]


def corrupt_word(word: str) -> str:
    """
    Randomly introduces a small corruption (delete, insert, substitute)
    into a word to simulate typos.
    """
    # Only corrupt words that are long enough, and only do so occasionally
    if len(word) < 4 or random.random() > 0.2:
        return word

    action = random.choice(["delete", "insert", "substitute"])
    # Avoid corrupting the first or last character to preserve word shape
    pos = random.randint(1, len(word) - 2)

    if action == "delete":
        return word[:pos] + word[pos + 1 :]
    if action == "insert":
        return word[:pos] + random.choice("abcdefghijklmnopqrstuvwxyz") + word[pos:]
    if action == "substitute":
        return (
            word[:pos] + random.choice("abcdefghijklmnopqrstuvwxyz") + word[pos + 1 :]
        )
    return word


def generate_noisy_transaction(name: str) -> str:
    """
    Generates a realistic, noisy transaction string from a clean merchant name.
    """
    # Step 1: Apply a random case style to the name
    case_style = random.choice(["upper", "lower", "title", "original"])
    if case_style == "upper":
        processed_name = name.upper()
    elif case_style == "lower":
        processed_name = name.lower()
    elif case_style == "title":
        processed_name = name.title()
    else:  # 'original'
        processed_name = name

    # Step 2: Choose a noise template for structuring the transaction
    template_type = random.choice(
        ["simple", "prefix_code", "suffix_code", "complex", "stuck_noise"]
    )

    # Step 3: Generate space-separated noise words for prefixes and suffixes
    prefix = " ".join(random.choices(COMPREHENSIVE_NOISE_WORDS, k=random.randint(1, 2)))
    suffix = " ".join(random.choices(COMPREHENSIVE_NOISE_WORDS, k=random.randint(1, 2)))

    # Step 4: Build the final transaction string based on the chosen template
    if template_type == "stuck_noise":
        # Creates noise directly attached to the name without spaces
        stuck_prefix = random.choice(["WWW.", "TXN*", f"{random.randint(100, 9999)}"])
        stuck_suffix = random.choice([".COM", f"*{random.randint(100, 9999)}", f"#{random.randint(100, 999)}"])
        pattern = random.choice(["prefix", "suffix", "both"])
        if pattern == "prefix":
            transaction = f"{stuck_prefix}{processed_name}"
        elif pattern == "suffix":
            transaction = f"{processed_name}{stuck_suffix}"
        else:  # both
            transaction = f"{stuck_prefix}{processed_name}{stuck_suffix}"

    elif template_type == "simple":
        pos = random.choice(["start", "middle", "end"])
        if pos == "start":
            transaction = f"{processed_name} {suffix}"
        elif pos == "middle":
            transaction = f"{prefix} {processed_name} {suffix}"
        else:  # end
            transaction = f"{prefix} {processed_name}"

    elif template_type == "prefix_code":
        transaction = f"{random.choice(['GPC*', 'VMD*', 'TXN*'])}{processed_name} {suffix}"

    elif template_type == "suffix_code":
        transaction = f"{prefix} {processed_name} #{random.randint(100, 9999)}"

    else:  # complex
        transaction = f"{prefix} {processed_name} on {random.randint(1, 28)}-{random.randint(1, 12)}-2024"

    # Final cleanup: collapse multiple spaces into one and trim whitespace
    return re.sub(r"\s+", " ", transaction).strip()


def main():
    """
    Main function to load entity names, generate synthetic transaction data,
    and save it to a CSV file.
    """
    print("--- Preparing a High-Quality Training Dataset with DYNAMIC NOISE INJECTION ---")
    all_names_list = []

    # Load data from ACRA and other UEN sources
    try:
        df_acra = pd.read_csv(ACRA_ENTITIES_PATH, on_bad_lines="skip", usecols=["entity_name"])
        all_names_list.append(df_acra["entity_name"].dropna())
        print(f"✅ Loaded {len(df_acra)} rows from {ACRA_ENTITIES_PATH}")
    except FileNotFoundError:
        print(f"⚠️  Info: Could not find {ACRA_ENTITIES_PATH}. Skipping.")

    try:
        df_other = pd.read_csv(OTHER_UEN_ENTITIES_PATH, on_bad_lines="skip", usecols=["entity_name"])
        all_names_list.append(df_other["entity_name"].dropna())
        print(f"✅ Loaded {len(df_other)} rows from {OTHER_UEN_ENTITIES_PATH}")
    except FileNotFoundError:
        print(f"⚠️  Info: Could not find {OTHER_UEN_ENTITIES_PATH}. Skipping.")

    # Exit if no data files were found
    if not all_names_list:
        print("❌ CRITICAL: No data files found. Cannot proceed.")
        return

    # Combine, clean, and filter the names
    all_names = pd.concat(all_names_list, ignore_index=True)
    valid_names = [
        str(name).strip()
        for name in all_names.unique()
        if isinstance(name, str) and len(str(name).strip()) > 2
    ]
    print(f"\nFound {len(valid_names)} unique, valid names to process.")

    # Generate synthetic data for each valid name
    synthetic_data = [
        {"raw_transaction": generate_noisy_transaction(name), "clean_merchant": name}
        for name in tqdm(valid_names, desc="Generating Dynamic Noise")
    ]

    # Save the generated data to a CSV file
    pd.DataFrame(synthetic_data).to_csv(SYNTHETIC_DATA_SAVE_PATH, index=False)
    print(f"\n✅ Definitive synthetic dataset created with {len(synthetic_data)} records.")
    print(f"   Saved to: {SYNTHETIC_DATA_SAVE_PATH}")


if __name__ == "__main__":
    main()