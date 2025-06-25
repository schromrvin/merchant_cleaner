from typing import List
import regex as re


def custom_tokenize(text: str) -> List[str]:
    """
    A comprehensive tokenizer that splits text based on several rules.

    The tokenizer splits on:
    1. Whitespace.
    2. Non-alphanumeric characters (while keeping them as separate tokens).
    3. Transitions between letters and numbers (e.g., "MERCARI456").
    4. Transitions between lowercase and uppercase letters (e.g., "GopayGojek").
    """
    if not text:
        return []

    # Insert spaces at transitions to split concatenated words and numbers
    # Case 1: Lowercase followed by uppercase (e.g., "GopayGojek" -> "Gopay Gojek")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # Case 2: Letter followed by a digit (e.g., "MERCARI456" -> "MERCARI 456")
    text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)
    # Case 3: Digit followed by a letter (e.g., "456MERCARI" -> "456 MERCARI")
    text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)

    # Split the text by whitespace and on any character that is not a word,
    # whitespace, apostrophe, or hyphen, keeping the delimiters.
    tokens = re.split(r"([^\w\s'-]+|\s+)", text)

    # Filter out empty strings and tokens that are only whitespace,
    # and strip whitespace from the remaining tokens.
    return [t.strip() for t in tokens if t and not t.isspace()]