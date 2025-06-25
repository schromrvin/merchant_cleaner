from typing import Any, Dict, List


def generate_cleaner_features(
    tokens: List[str], index: int, token_frequencies: Dict[str, int]
) -> Dict[str, Any]:
    """
    Generates a dictionary of features for a token, intended for use
    with an XGBoost cleaner model.
    """
    token = tokens[index]
    token_lower = token.lower()

    # Calculate the ratio of the token's position in the list.
    # Avoid division by zero if the token list is empty.
    position_ratio = index / len(tokens) if len(tokens) > 0 else 0

    features = {
        # Frequency-based features
        "token_is_common": token_frequencies.get(token_lower, 0) > 1000,
        "token_freq": token_frequencies.get(token_lower, 0),
        # Case and character type features
        "token_is_upper": token.isupper(),
        "token_is_title": token.istitle(),
        "token_is_digit": token.isdigit(),
        "token_is_alpha": token.isalpha(),
        "token_is_punct": not token.isalnum(),
        # Length and position features
        "token_len": len(token),
        "token_position_ratio": position_ratio,
    }

    return features