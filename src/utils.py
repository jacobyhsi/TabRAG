import re
# def output_sanitizer(text):
#     # leading <think> tags from Qwen3
#     text = re.sub(r'^<think>.*?</think>\s*', '', text, flags=re.DOTALL)
#     # leading new lines
#     text = text.lstrip('\n')
    
#     return text

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

