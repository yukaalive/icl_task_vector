from dataclasses import asdict
from typing import ContextManager, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
import transformers  # DynamicCache の判定等で使う場合あり

from core.data.datasets.few_shot_dataset import FewShotDataset
from core.data.datasets.few_shot_format import FewShotFormat
from core.models.context_managers.tracing.forward_trace import ForwardTrace
from core.models.context_managers.tracing.forward_tracer import ForwardTracer
from core.models.context_managers.utils import CombinedContextManager
from core.models.utils.llm_layers import get_lm_pipeline
from core.utils.misc import get_nested_tensor_size
from core.utils.nested import nested_apply, nested_concat


def traced_forward(
    model: PreTrainedModel,
    inputs: Dict,
    forward_kwargs: Optional[dict] = None,
    batch_size: Optional[int] = None,
    forward_modifiers: Optional[Iterable[ContextManager]] = (),
) -> Tuple[CausalLMOutputWithPast, ForwardTrace]:
    context_manager, forward_trace = traced_forward_context_manager(model)
    with context_manager:
        outputs = modified_forward(
            model,
            inputs=inputs,
            forward_kwargs=forward_kwargs,
            batch_size=batch_size,
            forward_modifiers=forward_modifiers,
        )
    return outputs, forward_trace


def modified_forward(
    model: PreTrainedModel,
    inputs: Dict,
    forward_kwargs: Optional[dict] = None,
    batch_size: Optional[int] = None,
    forward_modifiers: Optional[Iterable[ContextManager]] = (),
) -> CausalLMOutputWithPast:
    context_manager = modified_forward_context_manager(model, forward_modifiers=forward_modifiers)
    with context_manager:
        outputs = batch_forward(
            model,
            inputs=inputs,
            forward_kwargs=forward_kwargs,
            batch_size=batch_size,
        )
    return outputs


def get_input_type(inputs: Dict) -> str:
    if "input_ids" not in inputs and "inputs_embeds" not in inputs:
        raise ValueError("inputs must contain either input_ids or inputs_embeds")
    if "input_ids" in inputs and "inputs_embeds" in inputs:
        raise ValueError("inputs must contain either input_ids or inputs_embeds, not both")
    return "input_ids" if "input_ids" in inputs else "inputs_embeds"


def _get_forward_kwargs(forward_kwargs: Optional[Dict] = None) -> Dict:
    """
    forwardやgenerateで使う引数をまとめておきたい場合の簡易的な関数。
    必要に応じてデフォルト引数を足したりします。
    """
    forward_kwargs = forward_kwargs or {}

    # 例: デフォルトで隠れ状態を出す・キャッシュを使わない
    forward_kwargs.setdefault("return_dict", True)
    forward_kwargs.setdefault("output_hidden_states", True)
    forward_kwargs.setdefault("use_cache", False)

    return forward_kwargs


def _get_batches(inputs: Dict, batch_size: int, show_progress: bool = False) -> Iterable[Dict]:
    """
    inputs を batch_size ごとにスライスして返すジェネレータ。
    """
    input_type = get_input_type(inputs)
    num_inputs = len(inputs[input_type])
    indices = range(0, num_inputs, batch_size)

    def _slice_batch(start: int, end: int):
        return nested_apply(inputs, lambda t: t[start:end])

    if show_progress:
        from tqdm import tqdm
        indices = tqdm(indices)

    for i in indices:
        yield _slice_batch(i, i + batch_size)

def batch_forward(
    model: PreTrainedModel,
    inputs: Dict,
    forward_kwargs: Optional[Dict] = None,
    batch_size: int = 100,
    show_progress: bool = False,
) -> CausalLMOutputWithPast:
    """
    大きい入力をバッチに分けて model(**batch_inputs) を実行し、結果を結合して返す。
    """
    forward_kwargs = _get_forward_kwargs(forward_kwargs)

    if batch_size is None or batch_size < 1:
        batch_size = _auto_batch_size(model, inputs)

    # バッチに分ける
    batches = _get_batches(inputs, batch_size, show_progress=show_progress)

    output_all = []
    for batch_inputs in batches:
        batch_inputs = nested_apply(batch_inputs, lambda t: t.to(model.device))
        with torch.no_grad():
            out = model(**batch_inputs, **forward_kwargs)
            # CPUに戻してメモリ節約（バッチ結合時にGPUに置くと大きくなるため）
            out = nested_apply(out, lambda t: t.cpu() if torch.is_tensor(t) else t)
        output_all.append(out)

    # --------------------------------------------------
    # 出力をマージ
    if isinstance(output_all[0], dict):
        # output_all[0] が dict の場合
        merged_output = {}
        output_keys = list(output_all[0].keys())
        for key in output_keys:
            # None以外の値のみ収集
            vals = [o[key] for o in output_all if o[key] is not None]

            # DynamicCache の項目が含まれる場合の処理
            if any(isinstance(v, transformers.cache_utils.DynamicCache) for v in vals):
                if len(vals) > 0:
                    merged_output[key] = vals[-1]  # 最後のバッチを採用
                else:
                    # Noneではなく空のテンソルやデフォルト値を設定
                    merged_output[key] = torch.tensor([]) if key in ['logits'] else None
                continue

            if len(vals) == 0:
                # キーに応じてデフォルト値を設定
                if key in ['logits', 'hidden_states']:
                    merged_output[key] = torch.tensor([])  # 空のテンソル
                else:
                    merged_output[key] = None
            else:
                try:
                    merged_output[key] = nested_concat(vals)
                except Exception as e:
                    print(f"Error concatenating {key}: {e}")
                    # 安全なフォールバック
                    if torch.is_tensor(vals[0]):
                        merged_output[key] = vals[0].clone()
                    else:
                        merged_output[key] = vals[0]

    else:
        # output_all[0] が ModelOutput (CausalLMOutputWithPast 等) の場合
        first_dict = vars(output_all[0])  # or output_all[0].__dict__
        merged_output = {}
        for key in first_dict.keys():
            vals = []
            for o in output_all:
                val = getattr(o, key, None)
                if val is not None:
                    vals.append(val)

            # DynamicCache の項目が含まれる場合、同様に最後のものだけを使う
            if any(isinstance(v, transformers.cache_utils.DynamicCache) for v in vals):
                if len(vals) > 0:
                    merged_output[key] = vals[-1]
                else:
                    # Noneではなく空のテンソルやデフォルト値を設定
                    merged_output[key] = torch.tensor([]) if key in ['logits'] else None
                continue

            if len(vals) == 0:
                # キーに応じてデフォルト値を設定
                if key in ['logits', 'hidden_states']:
                    merged_output[key] = torch.tensor([])  # 空のテンソル
                else:
                    merged_output[key] = None
            else:
                try:
                    merged_output[key] = nested_concat(vals)
                except Exception as e:
                    print(f"Error concatenating {key}: {e}")
                    # 安全なフォールバック
                    if torch.is_tensor(vals[0]):
                        merged_output[key] = vals[0].clone()
                    else:
                        merged_output[key] = vals[0]

    # 最後に CausalLMOutputWithPast に詰め直す
    return CausalLMOutputWithPast(**merged_output)

def _auto_batch_size(model: PreTrainedModel, inputs: Dict) -> int:
    """
    モデルサイズや入力長からバッチサイズを自動推定する例。
    必要なければ固定値や外部設定にしてもOK。
    """
    base_batch_size = 400
    base_model_size_gb = 11.5  # pythia-12b
    base_sequence_length = 50

    model_size_gb = sum(get_nested_tensor_size(t) for t in model.parameters()) / (1024**3)
    input_type = get_input_type(inputs)
    sequence_length = inputs[input_type].shape[1]

    batch_size = int(
        base_batch_size
        * (base_model_size_gb / model_size_gb)
        * (base_sequence_length / sequence_length)
    )
    return max(batch_size, 1)


def batch_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    inputs: Dict,
    generate_kwargs: Optional[Dict] = None,
    batch_size: Optional[int] = None,
    show_progress: bool = False,
) -> torch.Tensor:
    generate_kwargs = _get_forward_kwargs(generate_kwargs)

    # もし generate_kwargs に return_dict があれば削除
    if "return_dict" in generate_kwargs:
        del generate_kwargs["return_dict"]

    # 生成で hidden_states を使う場合
    generate_kwargs.setdefault("return_dict_in_generate", True)

    if batch_size is None or batch_size < 1:
        batch_size = _auto_batch_size(model, inputs)

    input_type = get_input_type(inputs)
    total_length = inputs[input_type].shape[0]

    batches = _get_batches(inputs, batch_size, show_progress=show_progress)
    all_batch_ids = []

    for batch_inputs in batches:
        batch_inputs = nested_apply(batch_inputs, lambda t: t.to(model.device))
        with torch.no_grad():
            batch_ids_output = model.generate(
                **batch_inputs,
                **generate_kwargs,
                pad_token_id=tokenizer.pad_token_id,
            )

        # GenerateDecoderOnlyOutput 等の場合は .sequences を取り出す
        if hasattr(batch_ids_output, "sequences"):
            batch_ids = batch_ids_output.sequences
        else:
            batch_ids = batch_ids_output

        all_batch_ids.append(batch_ids.cpu())

    generate_ids = torch.cat(all_batch_ids, dim=0)

    # 以下、元の入力長だけ切り取って返す
    new_ids = []
    offset = 0
    for batch_inputs in _get_batches(inputs, batch_size, show_progress=False):
        bs = len(batch_inputs[input_type])
        seq_len = batch_inputs[input_type].shape[1]
        new_ids.append(generate_ids[offset : offset + bs, seq_len:])
        offset += bs

    new_ids = torch.cat(new_ids, dim=0)
    return new_ids


def decode_predictions(
    output_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    few_shot_format: FewShotFormat = FewShotFormat()
) -> List[str]:
    new_tokens = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    answers = [tokens.split(few_shot_format.example_separator)[0] for tokens in new_tokens]
    return answers


def tokenize_prompts(tokenizer: PreTrainedTokenizer, prompts: List[str]) -> Dict[str, torch.Tensor]:
    return tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        return_token_type_ids=False
    )


def tokenize_datasets(
    tokenizer: PreTrainedTokenizer,
    datasets: List[FewShotDataset],
    few_shot_format: FewShotFormat = FewShotFormat(),
    format_dataset_kwargs: Optional[dict] = None,
) -> Dict[str, torch.Tensor]:
    if format_dataset_kwargs is None:
        format_dataset_kwargs = {}
    prompts = few_shot_format.format_datasets(datasets, **format_dataset_kwargs)
    return tokenize_prompts(tokenizer, prompts)


def hidden_to_logits(model: PreTrainedModel, hidden: torch.Tensor) -> torch.Tensor:
    device = model.device
    lm_pipeline = get_lm_pipeline(model)

    hidden = hidden.to(device)
    hidden = hidden.type(lm_pipeline.parameters().__next__().dtype)

    with torch.no_grad():
        logits = lm_pipeline(hidden).cpu()
    return logits


def logits_to_tokens(
    logits: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    ignore_ids: Optional[List[int]] = None
) -> List[str]:
    if ignore_ids is not None:
        logits[np.arange(len(logits)), ignore_ids] = -np.inf
    ids = logits.argmax(dim=-1).numpy()
    tokens = np.vectorize(tokenizer.decode)(ids)
    return tokens

# get_logits関数の修正
def get_logits(
    model: PreTrainedModel,
    forward_trace: ForwardTrace,
    position: int,
    layer: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    ForwardTrace に記録された residual_stream などから hidden state を抜き出し、
    それをロジットに変換した結果をまとめて返す例。
    """
    layer_indexer = layer if layer is not None else slice(None, None, None)
    logits = {}
    
    # 安全にresidual_streamからデータを取得
    residual_stream_dict = asdict(forward_trace.residual_stream) if forward_trace.residual_stream else {}
    
    for name, hidden in residual_stream_dict.items():
        if hidden is None or len(hidden) == 0:
            logits[name] = torch.zeros((1, model.config.vocab_size))  # デフォルト値の設定
            continue
            
        try:
            # インデックスエラーを防止
            if position < hidden.shape[1]:
                logit = hidden_to_logits(model, hidden[:, layer_indexer, position])
                logits[name] = logit
            else:
                logits[name] = torch.zeros((1, model.config.vocab_size))
        except Exception as e:
            print(f"Error in get_logits for {name}: {e}")
            logits[name] = torch.zeros((1, model.config.vocab_size))
            
    return logits

def traced_forward_context_manager(model: PreTrainedModel) -> Tuple[ContextManager, ForwardTrace]:
    forward_trace = ForwardTrace()
    context_manager = ForwardTracer(model, forward_trace)
    return context_manager, forward_trace


def modified_forward_context_manager(
    model: PreTrainedModel,
    forward_modifiers: Optional[Iterable[ContextManager]] = ()
) -> ContextManager:
    return CombinedContextManager([*forward_modifiers])
