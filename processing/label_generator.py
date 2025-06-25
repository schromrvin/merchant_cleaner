from typing import List, Tuple

from .tokenizer import custom_tokenize


def generate_labels(
    raw_transaction: str, clean_merchant: str
) -> Tuple[List[str], List[str]]:
    """
    A robust label generator that finds the best possible alignment of
    clean merchant tokens within the raw transaction tokens.
    """
    raw_tokens = custom_tokenize(raw_transaction)
    clean_tokens = custom_tokenize(clean_merchant)

    labels = ["O"] * len(raw_tokens)

    # Return immediately if there are no tokens to process
    if not clean_tokens or not raw_tokens:
        return raw_tokens, labels

    raw_lower = [t.lower() for t in raw_tokens]
    clean_lower = [t.lower() for t in clean_tokens]

    best_match_indices = []
    current_search_start = 0

    # Find the sequence of clean tokens within the raw tokens
    for clean_token in clean_lower:
        try:
            # Search for the next token from where the last one was found
            found_index = raw_lower.index(clean_token, current_search_start)
            best_match_indices.append(found_index)
            current_search_start = found_index + 1
        except ValueError:
            # If a token is not found in sequence, the match is invalid
            best_match_indices = []
            break

    # Apply B-I-E-S (Begin, Inside, End, Single) labeling if a match was found
    if best_match_indices:
        if len(best_match_indices) == 1:
            labels[best_match_indices[0]] = "S"
        else:
            labels[best_match_indices[0]] = "B"
            for i in best_match_indices[1:-1]:
                labels[i] = "I"
            labels[best_match_indices[-1]] = "E"

    return raw_tokens, labels