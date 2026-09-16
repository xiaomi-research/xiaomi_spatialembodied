# UPDATE: Replace placeholder paths with your actual paths.
# Filter think tag for R1-type models
import json
import re


def preprocess_model_output(model_output, think_start="<tool_call>", think_end="]", clean_xml=True):
    """
    Preprocess model output: remove the thinking section and return the final answer text.

    Args:
        model_output: Model output, can be a string, dict, or JSON string.
        think_start: Start marker for the thinking section, default is '<tool_call>'.
        think_end: End marker for the thinking section, default is ']'.
        clean_xml: Whether to clean XML tags (e.g. <answer>), default is True.

    Returns:
        Cleaned text string.
    """
    # Step 1: Extract text
    text = ""
    if isinstance(model_output, dict):
        if 'response' in model_output:
            text = model_output['response']
        else:
            text = str(model_output)
    elif isinstance(model_output, str):
        try:
            data = json.loads(model_output)
            if isinstance(data, dict) and 'response' in data:
                text = data['response']
            else:
                text = model_output
        except json.JSONDecodeError:
            text = model_output
    else:
        text = str(model_output)

    # Step 2: Remove thinking section
    # Build regex pattern for cross-line matching
    # Use re.DOTALL so . matches all characters including newlines
    pattern = re.escape(think_start) + r'.*?' + re.escape(think_end)
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)

    # Step 3 (optional): Clean XML tags
    # if clean_xml:
    #     cleaned_text = re.sub(r'</?[a-zA-Z]+>', '', cleaned_text)

    return cleaned_text.strip()


# Example 1: Dict input with response field
output1 = {"response": "<tool_call>\nOkay, let's see...\n]\n\n<answer>0</answer>"}
result1 = preprocess_model_output(output1)
print(result1)  # Output: "0"

# Example 2: String input without XML tags
output2 = "<tool_call>\nOkay, let's see...\n]\n\nA"
result2 = preprocess_model_output(output2, clean_xml=False)
print(result2)  # Output: "A"

# Example 3: JSON string input
output3 = '{"response": "	RTE\\nReasoning...\\n]\\n\\nB"}'
result3 = preprocess_model_output(output3)
print(result3)  # Output: "B"

output4 = "<answer>0</answer>"
result4 = preprocess_model_output(output4)
print(result4)  # Output: "0"
