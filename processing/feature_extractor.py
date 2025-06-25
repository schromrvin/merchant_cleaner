import re
from typing import Any, Dict, List


def get_token_shape(token: str) -> str:
    """Gets the 'shape' of a token."""
    shape = re.sub(r"\w", "w", token)
    shape = re.sub(r"\d", "d", shape)
    return shape


def generate_features(tokens: List[str], index: int) -> Dict[str, Any]:
    """Generates a dictionary of features for a specific token for the CRF model."""
    token = tokens[index]

    features = {
        "bias": 1.0,
        "token.lower()": token.lower(),
        "token.isupper()": token.isupper(),
        "token.istitle()": token.istitle(),
        "token.isdigit()": token.isdigit(),
        "token.len()": len(token),
        "token.position_ratio": index / len(tokens),
        "token.shape()": get_token_shape(token),
    }

    if index > 0:
        prev_token = tokens[index - 1]
        features.update(
            {
                "-1:token.lower()": prev_token.lower(),
                "-1:token.istitle()": prev_token.istitle(),
                "-1:token.isupper()": prev_token.isupper(),
                "-1:token.isdigit()": prev_token.isdigit(),
                "-1:token.shape()": get_token_shape(prev_token),
            }
        )
    else:
        features["BOS"] = True

    if index < len(tokens) - 1:
        next_token = tokens[index + 1]
        features.update(
            {
                "+1:token.lower()": next_token.lower(),
                "+1:token.istitle()": next_token.istitle(),
                "+1:token.isupper()": next_token.isupper(),
                "+1:token.isdigit()": next_token.isdigit(),
                "+1:token.shape()": get_token_shape(next_token),
            }
        )
    else:
        features["EOS"] = True

    return features