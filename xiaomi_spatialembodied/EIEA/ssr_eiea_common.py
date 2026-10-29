"""Loaders for the SSR-based EIEA approximation (rebuttal E2).

The released XEmbodied checkpoint has no EIEA cue encoder (only a fixed 10x3584
demo cache), so this module uses SSR's MIDI (Mamba latent-cue interpreter) to
produce the 10 physical-cue tokens and SSR's VLM adapter to consume them.
"""
import os
import sys
import types

import torch
from transformers import AutoTokenizer, CLIPProcessor, CLIPVisionModel, SiglipProcessor, SiglipVisionModel

_DEFAULT_SSR_ROOT = "/home/kaneqian/XEmbodied/code/SSR"
SSR_ROOT = os.environ.get(
    "SSR_ROOT", _DEFAULT_SSR_ROOT if os.path.isdir(_DEFAULT_SSR_ROOT) else "")
if SSR_ROOT and SSR_ROOT not in sys.path:
    sys.path.insert(0, SSR_ROOT)

# SSR modules call `import autoroot` purely for path setup; it fails outside a
# pyrootutils project tree. A stub is enough for inference.
sys.modules.setdefault("autoroot", types.ModuleType("autoroot"))

_SSR_MODELS = os.environ.get("SSR_MODEL_ROOT", "/media/nas_users/kaneqian/model/ssr")
MIDI_DIR = os.environ.get("SSR_MIDI_DIR", os.path.join(_SSR_MODELS, "SSR-MIDI-7B"))
MAMBA_DIR = os.environ.get("SSR_MAMBA_DIR", os.path.join(_SSR_MODELS, "mamba-130m-hf"))
CLIP_DIR = os.environ.get("SSR_CLIP_DIR", os.path.join(_SSR_MODELS, "clip-vit-large-patch14"))
SIGLIP_DIR = os.environ.get("SSR_SIGLIP_DIR", os.path.join(_SSR_MODELS, "siglip-so400m-patch14-384"))
VLM_BASE = os.environ.get("SSR_VLM_BASE",
                          "/media/nas_users/kaneqian/model/Qwen2.5-VL-7B-Instruct")
VLM_ADAPTER = os.environ.get("SSR_VLM_ADAPTER", os.path.join(_SSR_MODELS, "SSR-VLM-7B"))
DEPTH_CKPT = os.environ.get("SSR_DEPTH_CKPT", os.path.join(_SSR_MODELS, "DepthPro/depth_pro.pt"))

TOR = "<tor>"


def _require_dir(path: str, what: str) -> None:
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"{what} not found at {path!r}. Set the corresponding SSR_* "
            "environment variable (see model/eiea/README.md).")


def load_midi(device):
    """Load MIDI without the frozen Qwen2.5-7B teacher (only needed for training)."""
    _require_dir(MIDI_DIR, "SSR-MIDI checkpoint")
    _require_dir(MAMBA_DIR, "mamba-130m-hf checkpoint")
    import ssr.models.midi as midi_mod
    from ssr.models.midi import MIDI, MIDIConfig
    from ssr.utils.prompt import SSRSpecialToken

    cfg = MIDIConfig.from_pretrained(MIDI_DIR)
    cfg.mamba_path_or_name = MAMBA_DIR
    cfg.llm_path_or_name = "dummy-teacher"
    # The config records the teacher width (3584) which defines tor_proj's output.
    teacher_width = 3584

    class _DummyTeacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(hidden_size=teacher_width)

    orig = midi_mod.Qwen2ForCausalLM.from_pretrained
    try:
        midi_mod.Qwen2ForCausalLM.from_pretrained = lambda *a, **k: _DummyTeacher()
        midi = MIDI.from_pretrained(MIDI_DIR, config=cfg, torch_dtype=torch.float32)
    finally:
        midi_mod.Qwen2ForCausalLM.from_pretrained = orig
    if os.environ.get("MIDI_USE_MAMBAPY", "1") == "1":
        try:
            midi.mamba.config.use_mambapy = True
            for layer in midi.mamba.backbone.layers:
                layer.mixer.use_mambapy = True
        except Exception as e:
            print("mambapy backend not enabled:", e)
    midi = midi.to(device).eval()
    for p in midi.parameters():
        p.requires_grad_(False)

    tok = AutoTokenizer.from_pretrained(MAMBA_DIR)
    tok.add_tokens(SSRSpecialToken.TOR_TOKEN, special_tokens=True)
    return midi, tok


def load_visual_encoders(device):
    _require_dir(CLIP_DIR, "CLIP ViT-L checkpoint")
    _require_dir(SIGLIP_DIR, "SigLIP checkpoint")
    clip_proc = CLIPProcessor.from_pretrained(CLIP_DIR)
    clip = CLIPVisionModel.from_pretrained(CLIP_DIR, torch_dtype=torch.float32).to(device).eval()
    sig_proc = SiglipProcessor.from_pretrained(SIGLIP_DIR)
    sig = SiglipVisionModel.from_pretrained(SIGLIP_DIR, torch_dtype=torch.float32).to(device).eval()
    return clip_proc, clip, sig_proc, sig


@torch.no_grad()
def visual_embeds(image_pil, depth_uint8, encoders, device):
    import numpy as np
    clip_proc, clip, sig_proc, sig = encoders
    img = clip_proc(images=image_pil, return_tensors="pt").to(device)
    dep = sig_proc(images=np.asarray(depth_uint8), return_tensors="pt").to(device)
    return (
        clip(**img).last_hidden_state.float(),
        sig(**dep).last_hidden_state.float(),
    )


@torch.no_grad()
def depth_uint8(image_pil, dpro, dtf, device):
    """DepthPro -> uint8 HxWx3, as in SSR's infer.py."""
    import numpy as np
    import depth_pro as depth_pro_pkg

    arr, _, f_px = depth_pro_pkg.load_pil(image_pil)
    x = dtf(arr).to(device)
    depth = dpro.infer(x, f_px=f_px)["depth"].cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255.0
    return np.stack([depth.astype(np.uint8)] * 3, axis=-1)


def load_vlm(device):
    _require_dir(VLM_BASE, "Qwen2.5-VL-7B base checkpoint")
    _require_dir(VLM_ADAPTER, "SSR-VLM LoRA adapter")
    import json as _json

    from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from ssr.models.vlm import SSRVLM

    vlm = SSRVLM.from_pretrained(
        VLM_BASE,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="sdpa",
    )
    # transformers' PreTrainedModel.load_adapter does not inject into this
    # custom subclass under transformers 4.57; inject with PEFT directly.
    ac = _json.load(open(os.path.join(VLM_ADAPTER, "adapter_config.json")))
    lora_cfg = LoraConfig(
        r=ac["r"], lora_alpha=ac["lora_alpha"], lora_dropout=ac["lora_dropout"],
        target_modules=ac["target_modules"], bias=ac["bias"],
        task_type=ac["task_type"], inference_mode=True,
    )
    inject_adapter_in_model(lora_cfg, vlm, adapter_name="default")
    sd = load_file(os.path.join(VLM_ADAPTER, "adapter_model.safetensors"))
    # Keys were saved from a PeftModel wrapper; strip its prefix for a model
    # that was injected in place.
    sd = {k.replace("base_model.model.", "", 1): v for k, v in sd.items()}
    missing, unexpected = set_peft_model_state_dict(vlm, sd, adapter_name="default")
    print(f"adapter loaded: missing={len(missing)} unexpected={len(unexpected)}")
    vlm.eval()
    for p in vlm.parameters():
        p.requires_grad_(False)
    return vlm
