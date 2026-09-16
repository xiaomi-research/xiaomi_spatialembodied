# UPDATE: Replace placeholder paths with your actual paths.
import json
import re
import os

def update_option_format(text):
    """
    Update the multiple-choice option format from A) to A. in the text
    """
    return re.sub(r'([A-Za-z])\)', r'\1.', text)

def clean_final_answer(text):
    """
    Clean the Final Answer section, converting various Final Answer markers to <answer> tag format
    """
    # Define various possible Final Answer prefixes
    patterns = [
        "**Final Answer:**",
        "**Final Answer**:",
        "**Final Answer**",
        "### Final Answer:",
        "## Final Answer:",
        "# Final Answer:",
        "Final Answer:",
        "Final Answer：",  # Full-width colon variant
    ]

    # Try splitting for each pattern
    for pattern in patterns:
        if pattern in text:
            # Split the string
            parts = text.split(pattern, 1)
            if len(parts) > 1:
                # Get the answer part, strip leading/trailing whitespace
                answer = parts[1].strip()

                # Build new content: front part + <answer> tag
                new_text = parts[0] + f"<answer>{answer}</answer>"
                return new_text

    # If no pattern matched, return original text
    return text

def clean_final_answer_advanced(text):
    """
    More powerful Final Answer cleaning function, handling more edge cases
    """
    original_text = text

    # Try using regex to match various Final Answer patterns
    import re

    # Regex patterns matching various Final Answer prefixes
    patterns = [
        r'\*\*Final Answer\*\*\s*[:：]?\s*(.*)',  # **Final Answer** or **Final Answer:**
        r'#{1,3}\s*Final Answer\s*[:：]?\s*(.*)',  # # Final Answer:
        r'Final Answer\s*[:：]\s*(.*)',  # Final Answer:
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            # Get the full matched text
            full_match = match.group(0)
            answer_text = match.group(1).strip()

            # Replace the entire matched part
            new_text = text.replace(full_match, f"<answer>{answer_text}</answer>")
            return new_text

    # If regex didn't match either, try simple string search
    # Find position of "Final Answer" (case-insensitive)
    import re
    final_answer_pattern = re.compile(r'Final Answer', re.IGNORECASE)
    match = final_answer_pattern.search(text)

    if match:
        # Found position
        start_pos = match.start()

        # Find first colon from that position
        colon_pos = text.find(':', start_pos)
        if colon_pos == -1:
            # Try Chinese colon
            colon_pos = text.find('：', start_pos)

        if colon_pos != -1:
            # Get answer part (from after colon to end of string)
            answer = text[colon_pos + 1:].strip()

            # Replace the entire part starting from "Final Answer"
            text_to_replace = text[start_pos:]
            new_text = text[:start_pos] + f"<answer>{answer}</answer>"
            return new_text

    return original_text

def clean_final_answer_combined(text):
    """
    Combined Final Answer cleaning function using both simple and advanced methods
    """
    # Try simple method first
    patterns = [
        "**Final Answer:**",
        "**Final Answer**:",
        "**Final Answer**",
        "### Final Answer:",
        "## Final Answer:",
        "# Final Answer:",
        "Final Answer:",
        "Final Answer：",  # Full-width colon variant
    ]

    for pattern in patterns:
        if pattern in text:
            # Split the string
            parts = text.split(pattern, 1)
            if len(parts) > 1:
                # Get the answer part
                answer = parts[1].strip()

                # If answer starts with "**", may be leftover from previous format
                if answer.startswith("**"):
                    answer = answer[2:].strip()
                if answer.startswith("**"):
                    answer = answer[2:].strip()

                # Build new content
                new_text = parts[0] + f"<answer>{answer}</answer>"
                return new_text

    # If simple method doesn't work, use regex
    return clean_final_answer_advanced(text)

def process_jsonl_file(input_file, output_file):
    """
    Process JSONL file, clean Final Answer format and update option format
    """
    updated_records = []
    processed_count = 0
    modified_count = 0
    conversion_stats = {
        'total_assistants': 0,
        'converted': 0,
        'needs_attention': []
    }

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                processed_count += 1
                record_modified = False

                # Process each message
                for msg_idx, message in enumerate(record.get('messages', [])):
                    if 'content' in message and message.get('role') == 'assistant':
                        content = message['content']
                        original_content = content
                        conversion_stats['total_assistants'] += 1

                        # Check if there is a Final Answer
                        has_final_answer = any(pattern in content.lower() for pattern in
                                             ['final answer', '**final answer**', '### final answer:', '## final answer:'])

                        if has_final_answer:
                            # Clean Final Answer section
                            cleaned_content = clean_final_answer_combined(content)

                            # Check if conversion was successful
                            if "<answer>" in cleaned_content and "</answer>" in cleaned_content:
                                conversion_stats['converted'] += 1
                            elif "Final Answer" in cleaned_content or "final answer" in cleaned_content:
                                # Still has Final Answer but wasn't converted
                                conversion_stats['needs_attention'].append({
                                    'line': line_num,
                                    'message_idx': msg_idx,
                                    'content_preview': content[:200] + "..." if len(content) > 200 else content
                                })

                            content = cleaned_content

                        # Apply option format conversion to all messages
                        content = update_option_format(content)

                        if content != original_content:
                            modified_count += 1
                            record_modified = True
                            message['content'] = content

                updated_records.append(record)

            except json.JSONDecodeError as e:
                print(f"Warning: JSON parse error at line {line_num}: {e}")
                continue

    # Write output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in updated_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # Statistics
    print(f"Processing complete!")
    print(f"Total processed: {processed_count} records")
    print(f"Modified: {modified_count} messages")
    print(f"Conversion statistics:")
    print(f"  - Total assistant messages: {conversion_stats['total_assistants']}")
    print(f"  - Successfully converted messages: {conversion_stats['converted']}")

    if conversion_stats['total_assistants'] > 0:
        conversion_rate = conversion_stats['converted'] / conversion_stats['total_assistants'] * 100
        print(f"  - Conversion success rate: {conversion_rate:.1f}%")

    if conversion_stats['needs_attention']:
        print(f"  - Messages needing attention: {len(conversion_stats['needs_attention'])}")
        print(f"  - First 3 messages needing attention:")
        for i, item in enumerate(conversion_stats['needs_attention'][:3]):
            print(f"    Line {item['line']}, message {item['message_idx']}: {item['content_preview']}")

    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")

    return updated_records

def test_clean_function():
    """
    Test the clean_final_answer_combined function
    """
    test_cases = [
        # Your examples
        """**Step-by-Step Reasoning:**\n\n1. Identify "object <o1,CAM_FRONT_RIGHT,0.855,0.501>" in the "front right" camera view.\n\n2. Considering the ego vehicle's current speed of 23.21 km/h and potential proximity to the object, the vehicle may need to slow down or steer slightly left.\n\n3. Taking these actions helps ensure safety by preventing collisions and maintaining a clear path.\n\n**Final Answer:** The ego vehicle should consider slowing down or steering slightly left to avoid the object detected on the right side. This action ensures safety by preventing potential collisions and allowing more reaction time for any unexpected movements of the object.""",

        # Other possible formats
        """Some reasoning text.**Final Answer** The ego vehicle should stop.""",
        """Some reasoning text.### Final Answer: The ego vehicle should maintain speed.""",
        """Some reasoning text.Final Answer: The ego vehicle should speed up.""",
        """Some reasoning text.**Final Answer**: The ego vehicle should change lanes.""",
    ]

    print("Testing clean_final_answer_combined function:")
    print("=" * 80)

    for i, test_case in enumerate(test_cases):
        print(f"\nTest case {i+1}:")
        print(f"Input: {test_case[:100]}...")
        result = clean_final_answer_combined(test_case)
        print(f"Output: {result[:100]}...")

        # Check if <answer> tag is present
        if "<answer>" in result and "</answer>" in result:
            print("[OK] Successfully converted")
        else:
            print("[FAIL] Conversion failed")

        # Extract answer part
        match = re.search(r'<answer>(.*?)</answer>', result, re.DOTALL)
        if match:
            print(f"Extracted answer: {match.group(1)[:50]}...")

    print("\n" + "=" * 80)

def main():
    # Run tests first
    test_clean_function()

    # Input file path
    input_file = "data/c_rl_data_all/data/DriveLMMo1_swift_TRAIN_train_r1.jsonl"

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file does not exist: {input_file}")
        return

    # Generate output file path (append v2 to original filename)
    base_dir = os.path.dirname(input_file)
    base_name = os.path.basename(input_file)
    name_without_ext, ext = os.path.splitext(base_name)
    output_file = os.path.join(base_dir, f"{name_without_ext}v2{ext}")

    # Process file
    updated_records = process_jsonl_file(input_file, output_file)

    # Show some example comparisons
    print("\n\nExample comparisons (first 3):")
    print("=" * 100)

    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break

            original_record = json.loads(line.strip())
            updated_record = updated_records[i]

            # Find assistant message
            original_assistant = None
            updated_assistant = None

            for msg in original_record.get('messages', []):
                if msg.get('role') == 'assistant':
                    original_assistant = msg['content']
                    break

            for msg in updated_record.get('messages', []):
                if msg.get('role') == 'assistant':
                    updated_assistant = msg['content']
                    break

            if original_assistant and updated_assistant:
                print(f"\nExample {i+1}:")
                print("-" * 50)
                print("Original (last 100 chars):")
                print(original_assistant[-100:] if len(original_assistant) > 100 else original_assistant)
                print("\nAfter conversion (last 100 chars):")
                print(updated_assistant[-100:] if len(updated_assistant) > 100 else updated_assistant)
                print("-" * 50)

if __name__ == "__main__":
    main()
