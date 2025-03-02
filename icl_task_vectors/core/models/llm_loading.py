'''
2025/3/2修正
GPU に載せきれない場合に CPU にオフロード (device_map="auto")
メモリ最適化のためのオプション追加 (offload_buffers=True)
VRAM 消費削減のため torch_dtype=torch.float16 を指定
'''

import math
import os
from typing import Literal, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from accelerate import init_empty_weights
from accelerate.utils.modeling import infer_auto_device_map, get_balanced_memory

from core.models.utils.llm_layers import get_layers, get_layers_path

BASE_KWARGS = {
    "torch_dtype": torch.float16,
    "trust_remote_code": True,
}

GPU_KWARGS = {
    **BASE_KWARGS,
    # "load_in_8bit": True,
    # "device_map": "auto",
    "offload_buffers": True,  # ✅ GPU メモリのフラグメント化を防ぐ
}

CPU_KWARGS = {
    **BASE_KWARGS,
}

LlamaVariant = Literal["huggingface", "vicuna"]
LlamaSize = Literal["7B", "13B"]


def _setup_tokenizer(tokenizer: PreTrainedTokenizer) -> None:
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token


def get_model_path(model_type: str, model_variant: str) -> str:
    model_path = MODEL_PATHS[model_type][model_variant]
    return model_path


def _create_device_map(model_path: str) -> dict[str, int]:
    with init_empty_weights():
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    layer_class = get_layers(model)[0].__class__.__name__

    max_memory = get_balanced_memory(model, no_split_module_classes=[layer_class])
    max_memory[0] = 0
    base_device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=[layer_class])

    num_devices = torch.cuda.device_count()

    layers_path = get_layers_path(model)

    device_map_layers = {k: v for k, v in base_device_map.items() if k.startswith(layers_path)}
    device_map_other = {k: 0 for k, v in base_device_map.items() if k not in device_map_layers}

    num_layers = len(device_map_layers)
    num_layers_per_device = math.ceil(num_layers / max(1, (num_devices - 1)))
    device_map_layers = {k: (i // num_layers_per_device + 1) for i, k in enumerate(device_map_layers)}

    device_map = {**device_map_other, **device_map_layers}
    return device_map


def load_model(model_type: str, model_variant: str, load_to_cpu: bool = False):
    model_path = get_model_path(model_type, model_variant)

    kwargs = CPU_KWARGS if load_to_cpu else GPU_KWARGS

    # ✅ 変更: GPU に載せきれない部分を CPU にオフロード
    kwargs["device_map"] = "auto"
    kwargs["torch_dtype"] = torch.float16  # ✅ FP16 でロード

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model = model.eval()

    return model


def load_tokenizer(model_type: str, model_variant: str) -> PreTrainedTokenizer:
    model_path = get_model_path(model_type, model_variant)

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    _setup_tokenizer(tokenizer)

    return tokenizer


def load_model_and_tokenizer(
    model_type: str, model_variant: str, load_to_cpu: bool = False
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    tokenizer = load_tokenizer(model_type, model_variant)
    model = load_model(model_type, model_variant, load_to_cpu=load_to_cpu)

    return model, tokenizer


MODEL_PATHS = {
    "pythia": {
        "1.4B": "EleutherAI/pythia-1.4b",
        "2.8B": "EleutherAI/pythia-2.8b",
        "6.9B": "EleutherAI/pythia-6.9b",
        "12B": "EleutherAI/pythia-12b",
    },
    "meta-llama": {
        "Llama-2-7b-hf": "meta-llama/Llama-2-7b-hf",
    },
    "falcon": {
        "7B": "tiiuae/falcon-7b",
        "40B": "tiiuae/falcon-40b",
    },
    "gpt-j": {
        "6B": "EleutherAI/gpt-j-6B",
    },
    "gpt-2": {
        "0.35B": "gpt2-medium",
        "0.77B": "gpt2-large",
        "1.5B": "gpt2-xl",
    },
    "mpt": {
        "7B": "mosaicml/mpt-7b",
    },
    "gpt-neox": {
        "20B": "EleutherAI/gpt-neox-20b",
    },
    "starcoder": {
        "regular": "bigcode/starcoder",
        "plus": "bigcode/starcoderplus",
    },
    "cerebras-gpt": {
        "6.7B": "cerebras/Cerebras-GPT-6.7B",
        "13B": "cerebras/Cerebras-GPT-13B",
    },
}
