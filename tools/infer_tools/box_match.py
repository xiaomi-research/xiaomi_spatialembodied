# UPDATE: Replace placeholder paths with your actual paths.
# Match flexible box format in the SURDS task
import re


def test_regex_pattern():
    # Original regex (note: pattern has many spaces and $ symbols, may need adjustment)
    original_pattern = r'<answer>\s*  $ \s*(\d+)\s*,\s*(\d+)\s* $  \s*,\s*  $ \s*(\d+)\s*,\s*(\d+)\s* $  \s*</answer>'

    # Fixed regex (removed $ and extra spaces, assuming standard format)
    fixed_pattern = r'<answer>\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*</answer>'

    # Test string
    test_string = "<answer>(361,563),(468,610)</answer>"

    print("Test string:", test_string)
    print("\n1. Testing with original regex:")
    original_match = re.search(original_pattern, test_string, re.IGNORECASE)
    if original_match:
        x1, y1, x2, y2 = map(int, original_match.groups())
        print(f"  Match successful! Extracted coordinates: [{x1}, {y1}, {x2}, {y2}]")
    else:
        print("  Match failed! Original pattern may contain unnecessary $ symbols and spaces")

    print("\n2. Testing with fixed regex:")
    fixed_match = re.search(fixed_pattern, test_string, re.IGNORECASE)
    if fixed_match:
        x1, y1, x2, y2 = map(int, fixed_match.groups())
        print(f"  Match successful! Extracted coordinates: [{x1}, {y1}, {x2}, {y2}]")
    else:
        print("  Match failed!")

    print("\n3. More robust regex (allowing various space and bracket variants):")
    robust_pattern = r'<answer>\s*[\(（]?\s*(\d+)\s*,\s*(\d+)\s*[\)）]?\s*,\s*[\(（]?\s*(\d+)\s*,\s*(\d+)\s*[\)）]?\s*</answer>'
    robust_match = re.search(robust_pattern, test_string, re.IGNORECASE)
    if robust_match:
        x1, y1, x2, y2 = map(int, robust_match.groups())
        print(f"  Match successful! Extracted coordinates: [{x1}, {y1}, {x2}, {y2}]")
    else:
        print("  Match failed!")


if __name__ == "__main__":
    test_regex_pattern()
