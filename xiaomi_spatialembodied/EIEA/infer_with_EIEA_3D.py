#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL inference script with EIEA 3D pipeline.
Supports: native Qwen3-VL model + custom qwen3_vl_moe_vggt model with TOR embedding injection.
"""
import os
import logging

# -------------------------- Environment Configuration --------------------------
# Uncomment and modify as needed:
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['MAX_PIXELS'] = '2080000'

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Kept in sync with `TOR_EMBEDS_PATH` in models/qwen3_vl_vggt_register.py.
TOR_EMBEDS_CACHE_NAME = "tor_embeds_cache_fb.pt"

# -------------------------- Model Switch Configuration --------------------------
# Model mode: True = custom VGGT-fused model | False = native Qwen3-VL-MoE
USE_CUSTOM_MODEL = True

# Custom model configuration (your fine-tuned merged checkpoint)
# This file lives in xiaomi_spatialembodied/EIEA/, so both paths need one `..` to reach xiaomi_spatialembodied/.
CUSTOM_MODEL_CFG = {
    # <-- UPDATE: set to your merged checkpoint directory -->
    "model_path": os.path.join(os.path.dirname(__file__), "..", "3DA", "my_qwen3_vggt_xattnv2", "base_ckpt"),
    "model_type": "qwen3_vl_moe_vggt",
    # <-- UPDATE: set to your EIEA register file path -->
    "register_file": os.path.join(os.path.dirname(__file__), "models", "qwen3_vl_vggt_register.py"),
}

# Native Qwen3-VL model configuration
NATIVE_MODEL_CFG = {
    # <-- UPDATE: set to your Qwen3-VL-30B-A3B-Instruct path -->
    "model_path": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "model_type": "qwen3_vl_moe",
}

# Inference configuration
INFER_CFG = {
    "max_tokens": 512,
    "temperature": 0,
    "user_prompt": (
        "Task Description:\n"
        "The primary goal of this task is to determine the relative front-back positioning of the two "
        "objects from the camera's perspective, where the object farther from the camera is considered "
        "to be more forward.\n\n"
        "Question:\n"
        "Is the black truck in front of the white car?\n\n"
        "Options:\n"
        "- Yes\n"
        "- No\n"
        "- Almost the same in terms of front-back position\n\n"
        "Finally, provide a concise and definitive response in the <answer> tag. Use the following format:\n"
        "<answer>[Final answer]</answer>\n"
    ),
    # <-- UPDATE: set to your test image path -->
    "image_paths": [os.path.join(os.path.dirname(__file__), "..", "demo", "nuscenes_0578_CAM_BACK.jpg")],
}

# -------------------------- Dependency Import --------------------------
from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
from swift.utils import import_external_file

# Register custom model architecture if needed
if USE_CUSTOM_MODEL:
    logger.info(f"Loading custom model register file: {CUSTOM_MODEL_CFG['register_file']}")
    import_external_file(CUSTOM_MODEL_CFG["register_file"])

# -------------------------- Inference Engine Initialization --------------------------
def init_engine() -> TransformersEngine:
    """Initialize the inference engine (auto-adapts to native/custom model)."""
    model_cfg = CUSTOM_MODEL_CFG if USE_CUSTOM_MODEL else NATIVE_MODEL_CFG
    model_path = model_cfg["model_path"]
    model_type = model_cfg["model_type"]

    if not os.path.isdir(model_path) and not model_path.startswith("Qwen/"):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    logger.info(f"Initializing model: {model_type} | path: {model_path}")
    engine = TransformersEngine(
        model=model_path,
        model_type=model_type,
        torch_dtype="bfloat16",
        trust_remote_code=True,
    )
    return engine

# -------------------------- Core Inference Logic --------------------------
def infer_image(engine: TransformersEngine, image_path: str, prompt: str, config: RequestConfig):
    """Run inference on a single image."""
    if not os.path.isfile(image_path):
        logger.error(
            "Image not found: %s\n"
            "  The default is a nuScenes back-camera frame, which is NOT bundled with this\n"
            "  repository. Either drop your own frame at that path, or point\n"
            "  INFER_CFG['image_paths'] at an image you do have (e.g. ../demo/sample.jpg).\n"
            "  Note the shipped TOR cache (%s) is tied to one specific question, so the\n"
            "  answer is only meaningful on a matching input.",
            image_path,
            os.path.basename(TOR_EMBEDS_CACHE_NAME),
        )
        return None

    infer_requests = [
        InferRequest(
            messages=[{"role": "user", "content": f"<image>{prompt}"}],
            images=[image_path],
        )
    ]

    resp_list = engine.infer(infer_requests, config)
    answer = resp_list[0].choices[0].message.content
    return answer

# -------------------------- Main --------------------------
if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("Qwen3-VL EIEA 3D inference task started")
    logger.info("=" * 50)

    # 1. Initialize engine
    infer_engine = init_engine()

    # 2. Initialize inference config
    req_config = RequestConfig(
        max_tokens=INFER_CFG["max_tokens"],
        temperature=INFER_CFG["temperature"],
    )

    # 3. Run inference on each image
    logger.info("Starting image inference...")
    for idx, img_path in enumerate(INFER_CFG["image_paths"]):
        logger.info(f"\n=== Inferring image {idx+1} ===")
        result = infer_image(
            engine=infer_engine,
            image_path=img_path,
            prompt=INFER_CFG["user_prompt"],
            config=req_config,
        )
        print(f"\nPrompt: {INFER_CFG['user_prompt']}")
        print(f"Model response: {result}")

    logger.info("\n" + "=" * 50)
    logger.info("Inference task completed!")
    logger.info("=" * 50)
