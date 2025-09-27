import re

def normalize_answer(s: str) -> str:
    # lower + trim + collapse spaces
    s = s.lower().strip().replace('\u00a0', ' ')
    s = re.sub(r'\s+', ' ', s)

    # turn "(12,345)" into "-12,345"
    s = re.sub(r'\(([^)]+)\)', r'-\1', s)

    # remove thousands commas
    s = re.sub(r'(?<=\d),(?=\d)', '', s)

    # normalize currency spacing: "$ 123" -> "$123"
    s = re.sub(r'(\$)\s*(\d)', r'\1\2', s)

    # strip spaces entirely to compare compact forms
    s = s.replace(' ', '')

    return s

