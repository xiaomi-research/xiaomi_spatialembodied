# UPDATE: Replace placeholder paths with your actual paths.
# Option parser for maplmv2 evaluation
import re
from typing import List, Optional


def completion_to_answer(completion: str, choices: List[str]) -> Optional[int]:
    """
    Extract answer index from model completion text.

    Args:
        completion: Model generated text.
        choices: List of choices, which may contain overlapping substrings.

    Returns:
        Choice index (0-based) or None.
    """
    if completion is None:
        return None

    comp = str(completion).strip()
    if not comp:
        return None

    # 1. Try to extract content within <answer> tags
    answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
    answer_match = re.search(answer_pattern, completion, re.IGNORECASE | re.DOTALL)

    comp_content = comp
    if answer_match:
        comp_content = answer_match.group(1).strip()

    # Clean text
    comp_content_low = comp_content.lower().rstrip(".。").strip()

    # 2. Try different matching strategies by priority

    # Strategy 1: Direct digit match
    if comp_content_low.isdigit():
        v = int(comp_content_low)
        # Check 1-based first, then 0-based
        if 1 <= v <= len(choices):
            return v - 1
        if 0 <= v < len(choices):
            return v

    # Strategy 2: Match "digit. description" format
    num_desc_pattern = r'^(\d+)\.?\s*(.*?)$'
    num_match = re.match(num_desc_pattern, comp_content)
    if num_match:
        num = int(num_match.group(1))
        desc = num_match.group(2).strip()
        if 1 <= num <= len(choices):
            if desc:  # Try to match description if present
                desc_low = desc.lower()
                # Try exact match first
                for i, ch in enumerate(choices):
                    if desc_low == str(ch).lower().strip():
                        return i
                # Then try substring match
                for i, ch in enumerate(choices):
                    c_low = str(ch).lower().strip()
                    if c_low and c_low in desc_low:
                        return i
            # Return digit index if no description match
            return num - 1

    # Strategy 3: Exact match
    for i, ch in enumerate(choices):
        c = str(ch).lower().strip()
        if c and c == comp_content_low:
            return i

    # Strategy 4: Handle substring containment
    # Build list of (index, choice, length)
    indexed_choices = [(i, str(ch).lower().strip(), len(str(ch).strip()))
                       for i, ch in enumerate(choices)]

    # Collect all matches
    matches = []

    # 4.1 Collect exact word boundary matches
    for i, c, length in indexed_choices:
        if c and c != comp_content_low:  # Skip exact match (handled above)
            # Check if input fully contains the choice
            if c in comp_content_low:
                matches.append((i, c, length, 1))  # 4th element is match quality score
            # Check if choice fully contains the input
            elif comp_content_low in c:
                matches.append((i, c, length, 2))

    if matches:
        # Sort by match quality, then by choice length
        # Quality 1: input contains choice (more precise)
        # Quality 2: choice contains input
        # For same quality, prefer shorter choice (more precise)
        matches.sort(key=lambda x: (x[3], x[2]))
        return matches[0][0]

    return None


def test_completion_to_answer():
    """Test the function, especially handling of substring containment."""

    # Test case 1: Choices with substring containment
    choices1 = [
        'Very clear',
        'Not clear, road markings are worn',
        'Not clear, road markings are occluded by vehicles',
        'Not clear, road markings are worn and occluded by vehicles',
    ]

    # Test case 2: Choices without substring containment
    choices2 = [
        "Normal city road",
        "Construction area",
        "School zone",
        "Highway",
    ]

    test_cases = [
        # (completion, choices, expected_index, description)
        ("<answer>Not clear, road markings are worn</answer>", choices1, 1, "Match mid-length choice"),
        ("<answer>4. Not clear, road markings are worn and occluded by vehicles</answer>", choices1, 3, "Match longest choice"),
        ("<answer>Very clear</answer>", choices1, 0, "Match short choice"),
        ("<answer>Not clear, road markings are occluded by vehicles</answer>", choices1, 2, "Match medium-length choice"),
        ("worn and occluded", choices1, 3, "Partial match to longest choice"),
        ("worn", choices1, 1, "Partial match to mid choice"),
        ("1. Not clear, road markings are worn", choices1, 1, "Numbered mid choice"),
        ("2", choices1, 1, "Numeric index 1-based"),
        ("<answer>3</answer>", choices1, 2, "Tagged numeric index"),
        ("<answer>1. Normal city road</answer>", choices2, 0, "Normal choice test 1"),
        ("Construction area", choices2, 1, "Normal choice test 2"),
        ("<answer>Highway</answer>", choices2, 3, "Normal choice test 3"),
        ("Some random text", choices1, None, "No match text"),
        ("", choices1, None, "Empty string"),
        (None, choices1, None, "None input"),
    ]

    print("Testing completion_to_answer function (handling substring containment):")
    print("-" * 80)

    all_passed = True
    for i, (completion, choices, expected, description) in enumerate(test_cases, 1):
        result = completion_to_answer(completion, choices)
        passed = result == expected
        status = "PASS" if passed else "FAIL"
        all_passed = all_passed and passed

        print(f"Test {i:2d} {status}: {description}")
        if not passed:
            print(f"    Input: {completion}")
            print(f"    Choices: {choices}")
            print(f"    Expected: {expected}, Got: {result}")
        print()

    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed!")

    return all_passed


def test_edge_cases():
    """Test edge cases."""

    print("Testing edge cases:")
    print("-" * 40)

    # Test numeric boundaries
    choices = ["A", "B", "C"]

    test_cases = [
        ("0", choices, 0, "0-based: 0 -> index 0"),
        ("1", choices, 0, "1-based: 1 -> index 0"),
        ("2", choices, 1, "1-based: 2 -> index 1"),
        ("3", choices, 2, "1-based: 3 -> index 2"),
        ("<answer>2</answer>", choices, 1, "Tagged digit"),
    ]

    for completion, choices, expected, description in test_cases:
        result = completion_to_answer(completion, choices)
        status = "PASS" if result == expected else "FAIL"
        print(f"{status} {description}: input={completion}, result={result}, expected={expected}")


def debug_specific_case():
    """Debug specific cases."""
    choices = [
        'Very clear',
        'Not clear, road markings are worn',
        'Not clear, road markings are occluded by vehicles',
        'Not clear, road markings are worn and occluded by vehicles',
    ]

    test_inputs = [
        ("worn", 1),
        ("2", 1),
        ("3", 2),
    ]

    print("Debugging specific cases:")
    print("Choices:", choices)
    print()

    for test_input, expected in test_inputs:
        result = completion_to_answer(test_input, choices)
        status = "PASS" if result == expected else "FAIL"
        print(f"{status} Input: '{test_input}'")
        print(f"  Result: {result} (expected: {expected})")
        if result is not None:
            print(f"  Matched choice: {choices[result]}")
        print()


if __name__ == "__main__":
    print("=" * 60)
    test_completion_to_answer()
    print("\n" + "=" * 60)
    test_edge_cases()
    print("\n" + "=" * 60)
    debug_specific_case()
