import re

def normalize_answer(s: str) -> str:
    # lower + trim + collapse spaces
    s = s.lower().strip().replace('\u00a0', ' ')   # was `.replace(' ', ' ')` — no-op, fixed to strip NBSP
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


_NUMBER_TOKEN_RE = re.compile(r'-?\d+\.?\d*')
_YES_WORDS = ('yes', 'true', 'correct', 'affirmative')
_NO_WORDS = ('no', 'false', 'incorrect', 'negative')


def _extract_bool(response: str):
    """Best-effort yes/no verdict from free text. Checks the first sentence
    first, since models usually state their verdict up front and only
    hedge/contradict themselves in the reasoning that follows."""
    text = response.lower().strip()
    text = re.sub(r'[*`]', '', text)
    first = re.split(r'[.\n]', text, maxsplit=1)[0]

    for chunk in (first, text):
        has_yes = any(re.search(rf'\b{w}\b', chunk) for w in _YES_WORDS)
        has_no = any(re.search(rf'\b{w}\b', chunk) for w in _NO_WORDS)
        if has_yes and not has_no:
            return True
        if has_no and not has_yes:
            return False
    return None  # ambiguous — no clear signal either way


def normalize_answer_tablevqa(gt, response) -> bool:
    """TableVQA-specific answer matching. If the ground truth contains a
    number, compare ONLY the numeric value (ignoring units like '$', '%',
    'billion gallons', 'million', etc.). Falls back to substring matching
    for non-numeric (categorical) answers. Ground truth of exactly '0'/'1'
    is treated as a TabFact-style boolean flag: if no literal digit match
    is found, a yes/no/true/false verdict is extracted from the response
    instead of requiring the model to output the raw digit."""

    def prep(s):
        s = str(s).lower().strip().replace('\u00a0', ' ')
        s = re.sub(r'\s+', ' ', s)
        s = re.sub(r'[*`]', '', s)
        s = re.sub(r'\(([^)]+)\)', r'-\1', s)
        s = re.sub(r'(?<=\d),(?=\d)', '', s)
        s = s.replace('$', '').replace('%', '')
        return s

    def numbers(s):
        floats = set()
        for tok in _NUMBER_TOKEN_RE.findall(s):
            try:
                floats.add(float(tok))
            except ValueError:
                pass
        return floats

    gt_prepped = prep(gt)
    gt_numbers = numbers(gt_prepped)

    if gt_numbers:
        resp_numbers = numbers(prep(response))
        if gt_numbers & resp_numbers:
            return True
        if gt_prepped in ('0', '1'):
            resp_bool = _extract_bool(response)
            if resp_bool is not None:
                return resp_bool == (gt_prepped == '1')
        return False

    return gt_prepped.replace(' ', '') in prep(response).replace(' ', '')