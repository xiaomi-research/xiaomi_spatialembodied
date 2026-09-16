# UPDATE: Replace placeholder paths with your actual paths.
import json
import os
from tqdm import tqdm

def convert_drivelmm_o1_format(input_path, output_path):
    """
    Convert DriveLMM-o1 data format, add system prompt

    Args:
        input_path: Original JSONL file path
        output_path: Output JSONL file path
    """

    # Define system prompt
    # system_prompt = "When answering the question based on the provided image, follow a structured and logical reasoning process. Organize your response using the format, ensuring each step builds upon the previous one and clearly explains how the image(s) contribute to the solution. Your answer should be structured as Reasoning Steps: (step by step reasoning) Final Answer: (final answer) \n Question: "
    
    system_prompt = """You are a driving scene analysis assistant. When responding, structure your answer in two clear sections:
    **Step-by-Step Reasoning**:
    - Analyze the driving scene step by step, noting important objects, conditions, and potential risks, etc.
    - Reference specific visible elements in your reasoning.

    **Final Answer**:
    - Provide a concise answer based on your reasoning.

    Keep responses safety-focused and grounded in the 6-view image(s). Always begin with **Step-by-Step Reasoning**, followed by **Final Answer**."""
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    # Read original file
    print(f"Reading input file: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Found {len(lines)} lines to process")
    
    # Convert data
    converted_lines = []
    error_count = 0
    
    for i, line in tqdm(enumerate(lines), total=len(lines), desc="Processing"):
        try:
            # Parse JSON
            data = json.loads(line.strip())
            
            # Get original messages
            original_messages = data.get("messages", [])
            
            if not original_messages:
                print(f"Warning: Line {i} has no messages, skipping")
                continue
            
            # Build new message list
            new_messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Add original messages
            new_messages.extend(original_messages)
            
            # Build new data object
            new_data = {
                "messages": new_messages,
                "images": data.get("images", [])
            }
            
            # Add other possible existing fields
            for key, value in data.items():
                if key not in ["messages", "images"]:
                    new_data[key] = value
            
            # Add to result list
            converted_lines.append(json.dumps(new_data, ensure_ascii=False))
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON on line {i}: {e}")
            error_count += 1
        except Exception as e:
            print(f"Error processing line {i}: {e}")
            error_count += 1
    
    # Write output file
    print(f"\nWriting {len(converted_lines)} converted lines to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in converted_lines:
            f.write(line + '\n')
    
    print(f"\nConversion complete!")
    print(f"Successfully converted: {len(converted_lines)} lines")
    print(f"Errors encountered: {error_count} lines")
    
    return converted_lines

# Execute conversion
if __name__ == "__main__":
    # Input/output paths
    input_file = "data/DriveLMM-o1-main/data/DriveLMMo1_swift_TEST.jsonl"
    output_file = "data/preprocess_data/all_eval_data/drivelmm/drivelmm_ms_swift_test.jsonl"
    
    # Execute conversion
    converted_data = convert_drivelmm_o1_format(input_file, output_file)
    
    # Show first 3 converted examples
    print("\n--- First 3 converted examples ---")
    for i, line in enumerate(converted_data[:3]):
        print(f"\nExample {i+1}:")
        data = json.loads(line)
        print(f"Messages count: {len(data['messages'])}")
        print(f"System prompt: {data['messages'][0]['role']}: {data['messages'][0]['content'][:50]}...")
        print(f"User message: {data['messages'][1]['role']}: {data['messages'][1]['content'][:50]}...")
        print(f"Assistant response length: {len(data['messages'][2]['content'])} chars")
        print(f"Has images: {'images' in data and len(data.get('images', [])) > 0}")
        if 'images' in data and data['images']:
            print(f"Image path: {data['images'][0]}")