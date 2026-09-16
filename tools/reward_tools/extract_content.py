# UPDATE: Replace placeholder paths with your actual paths.
import re


def _extract_answer(text) -> str:
    """
    Extract answer from dict or string.
    - If dict: try common keys ('answer', 'correct_answer', etc.)
    - If string:
        a) First, extract content inside <answer>...</answer>
        b) If not found, try to parse patterns like "Answer(s): A, B."
        c) Otherwise, return cleaned full text.
    """
    if isinstance(text, dict):
        for key in ['answer', 'correct_answer', 'solution', 'label']:
            if key in text:
                ans = text[key]
                if isinstance(ans, (list, dict)):
                    return str(ans)
                return str(ans)
        text = str(text)

    if not isinstance(text, str):
        text = str(text)

    # Step 1: Try <answer> tag
    match_tag = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.IGNORECASE | re.DOTALL)
    if match_tag:
        return match_tag.group(1).strip()

    # Step 2: Try "Answer:" or "Answer(s):" with optional parentheses (English or Chinese)
    answer_match = re.search(
        r'Answer[(（]?(?:s)?[)）]?\s*[:：]?\s*([A-Za-z,\s\.、；;]+?)(?:\s*$|[.\n])',
        text,
        re.IGNORECASE
    )
    if answer_match:
        raw_ans = answer_match.group(1).strip()
        # Remove trailing punctuation at the very end of the extracted part
        raw_ans = re.sub(r'[.。,，;；:：]+\s*$', '', raw_ans)
        return raw_ans

    # Step 3: Fallback
    return text.strip()
print(_extract_answer('Answer(s): A, C.'))


def _extract_option_content(text: str) -> str:
    """
    Extract option content from model response.
    Supports letter or digit prefixed option formats.
    Examples:
      "B: yes" -> "yes"
      "A. Paris" -> "Paris"
      "(C) 42" -> "42"
      "D answer" -> "answer"
      "3. Undeveloped road" -> "Undeveloped road"
      "yes" -> "yes"
    """
    text = text.strip()
    # Match: optional whitespace + optional left parenthesis + letter or digit + optional right parenthesis + optional punctuation(:/.etc) + content
    match = re.match(
        r'''^
            \s*
            [ \(（]?               # optional left parenthesis (English or Chinese)
            ([A-Za-z0-9])          # option label: letter or digit
            [ \)）]?               # optional right parenthesis
            \s*                    # optional spaces
            [:：.．]?              # optional colon or dot (English or Chinese)
            \s*                    # optional spaces
            (.+)                   # the actual content (what we want)
            $''',
        text,
        re.IGNORECASE | re.VERBOSE
    )
    if match:
        return match.group(2).strip()
    return text

# Test
print(_extract_option_content("B: yes"))                # -> "yes"
print(_extract_option_content("A. Paris"))              # -> "Paris"
print(_extract_option_content("3. Undeveloped road"))   # -> "Undeveloped road"
print(_extract_option_content("(C) 42"))                # -> "42"
print(_extract_option_content("（D）answer"))           # -> "answer"
print(_extract_option_content("yes"))                   # -> "yes"
print(_extract_option_content("  E correct  "))         # -> "correct"
