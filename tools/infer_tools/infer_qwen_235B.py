# UPDATE: Replace placeholder paths with your actual paths.
import requests
import json
import os
import base64
import argparse
from tqdm import tqdm

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Qwen3-VL model inference script')

    parser.add_argument('--input_file', type=str, required=True,
                       help='Input data file path')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output result file path')
    parser.add_argument('--server_url', type=str, default="http://<SERVER_IP>:8000/v1/chat/completions",
                       help='Model server URL')
    parser.add_argument('--model_name', type=str, default="Qwen3_VL_235B_A22B_Thinking",
                       help='Model name')
    parser.add_argument('--timeout', type=int, default=120,
                       help='Request timeout in seconds')
    parser.add_argument('--delay', type=float, default=0.5,
                       help='Delay between requests in seconds')

    return parser.parse_args()

def encode_image_to_base64(image_path):
    """Encode an image to a base64 string"""
    if not os.path.exists(image_path):
        print(f"Warning: Image file does not exist: {image_path}")
        return None
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Failed to read image {image_path}: {e}")
        return None

def build_multimodal_content(user_text, image_paths):
    """Build multimodal content following the SGLang official format"""
    content = [{"type": "text", "text": 'Please answer the question directly and do not overthink!\n' + user_text}]

    # Add image content for each image path
    for image_path in image_paths:
        base64_image = encode_image_to_base64(image_path)
        if base64_image:
            # Build data URL format
            data_url = f"data:image/jpeg;base64,{base64_image}"
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": data_url
                }
            })

    return content

def process_sample(sample, args):
    """Process a single sample and get model response"""
    try:
        # Extract user message and image paths
        user_content = sample["messages"][0]["content"]
        image_paths = sample["images"]

        # Build multimodal message (following SGLang official format)
        multimodal_content = build_multimodal_content(user_content, image_paths)

        # Build request body - using SGLang official example format
        payload = {
            "model": args.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": multimodal_content
                }
            ],
            "max_tokens": 3500
        }

        # Send request
        response = requests.post(args.server_url, json=payload, timeout=args.timeout)

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            print(f"Request failed, status code: {response.status_code}")
            print(f"Response content: {response.text}")
            return None

    except Exception as e:
        print(f"Error occurred while processing sample: {e}")
        return None

def main():
    # Parse command line arguments
    args = parse_arguments()

    print(f"Starting processing...")
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Server: {args.server_url}")
    print(f"Model: {args.model_name}")

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file does not exist: {args.input_file}")
        return

    # Read input dataset
    with open(args.input_file, "r") as f_in:
        samples = [json.loads(line) for line in f_in]

        print(f"Found {len(samples)} samples")

        # Open output file outside the loop
        with open(args.output_file, "w") as f_out:
            for i, sample in enumerate(tqdm(samples, desc="Processing samples")):
                # Get model response
                model_response = process_sample(sample, args)

                if not model_response:
                    model_response = "Request failed or image processing error"
                    print(f"Sample {i} processing failed")

                # Build output record
                output_record = {
                    "response": model_response,
                    "labels": sample["messages"][1]["content"] if len(sample["messages"]) > 1 else "No label",
                    "logprobs": None,
                    "messages": [
                        {
                            "role": "user",
                            "content": sample["messages"][0]["content"],
                            "loss": None
                        },
                        {
                            "role": "assistant",
                            "content": model_response
                        }
                    ],
                    "images": [{"bytes": None, "path": p} for p in sample["images"]]
                }

                # Write output file
                f_out.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                f_out.flush()

                # Add delay to avoid server overload
                import time
                time.sleep(args.delay)

    print(f"Processing complete! Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()
