import math
import os
import re
import sys
from typing import Set, Tuple

import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from processing.tokenizer import custom_tokenize

# --- Constants ---
MATCH_INDEX_PATH = "model/match_index.pkl"
TOKEN_FREQS_PATH = "model/token_frequencies.pkl"

# --- THE DEFINITIVE STOP WORDS LIST ---
STOP_WORDS = {
    "payment", "txn", "debit", "credit", "card", "purchase", "store", "shop",
    "online", "bill", "receipt", "charge", "fee", "transfer", "giro", "atm",
    "withdrawal", "singapore", "sg", "suntec", "marina", "tampines", "jurong",
    "paypal", "google", "apple", "amex", "dbs", "posb", "uob", "ocbc", "visa",
    "mastercard", "nets", "grab", "fave", "shopee", "lazada", "gpay",
    "paynow", "paywave", "contactless", "cl", "east", "west", "north",
    "south", "central", "rd", "road", "st", "street", "ave", "avenue",
    "blvd", "boulevard", "ctrl", "central", "cres", "crescent", "dr",
    "drive", "pl", "place", "link", "walk", "hwy", "highway", "bldg",
    "building", "ctr", "centre", "tower", "point", "plaza", "square", "mall",
    "junction", "complex", "city", "town", "view", "park", "hill", "hdb",
    "hub", "bedok", "changi", "pasir", "ris", "punggol", "sengkang",
    "hougang", "amk", "ang", "mo", "kio", "bishan", "toa", "payoh", "novena",
    "orchard", "newton", "bukit", "timah", "clementi", "boon", "lay",
    "yishun", "woodlands", "choa", "chu", "kang", "geylang", "kallang",
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
    "joo chiat complex", "loyang point", "rivervale plaza", "pte", "ltd",
    "llp", "lp", "inc", "llc", "corp", "bhd", "sbn", "the", "and", "of",
    "co", "au", "hk", "usa",
}


def clean_token(token: str) -> str:
    """Standardizes a token for reliable matching."""
    return re.sub(r"[^a-z0-9]", "", token.lower())


class MerchantNameProcessor:
    def __init__(self):
        print("Initializing Processor with AGGRESSIVE FILTERING Logic...")
        try:
            self.match_index = joblib.load(MATCH_INDEX_PATH)
            self.token_frequencies = joblib.load(TOKEN_FREQS_PATH)
            self.total_token_count = sum(self.token_frequencies.values())
            print("✅ Index and token frequencies loaded successfully.")
        except FileNotFoundError as e:
            print(f"❌ CRITICAL: File not found: {e.filename}.")
            sys.exit(1)

    def _get_token_weight(self, token: str) -> float:
        """Calculates the IDF-based weight of a token."""
        frequency = self.token_frequencies.get(token, 1)
        return math.log(self.total_token_count / frequency)

    def predict(self, raw_transaction_text: str) -> Tuple[str, float]:
        """Predicts merchant name using aggressive filtering and normalized scoring."""
        all_input_tokens = {
            clean_token(t)
            for t in custom_tokenize(raw_transaction_text)
            if clean_token(t)
        }
        core_input_tokens = {t for t in all_input_tokens if t not in STOP_WORDS}

        if not core_input_tokens:
            return raw_transaction_text, 0.0

        candidate_names = {
            name
            for token in core_input_tokens
            for name in self.match_index.get(token, [])
        }

        if not candidate_names:
            return raw_transaction_text, 0.1

        best_match, highest_score = "", 0.0

        for candidate in candidate_names:
            candidate_tokens = {
                clean_token(t)
                for t in custom_tokenize(candidate)
                if clean_token(t)
            }
            core_candidate_tokens = {
                t for t in candidate_tokens if t not in STOP_WORDS
            }
            if not core_candidate_tokens:
                continue

            intersection = core_input_tokens.intersection(core_candidate_tokens)

            intersection_weight = sum(
                self._get_token_weight(t) for t in intersection
            )
            input_weight = sum(
                self._get_token_weight(t) for t in core_input_tokens
            )
            candidate_weight = sum(
                self._get_token_weight(t) for t in core_candidate_tokens
            )

            denominator = input_weight + candidate_weight
            if denominator == 0:
                continue

            score = (2 * intersection_weight) / denominator

            if score > highest_score:
                best_match, highest_score = candidate, score

        if highest_score > 0.33:
            return best_match, highest_score
        else:
            return raw_transaction_text, highest_score