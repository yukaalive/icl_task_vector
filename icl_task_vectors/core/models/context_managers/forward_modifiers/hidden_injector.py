import torch
from transformers import PreTrainedModel
from core.models.utils.llm_layers import get_layers

class HiddenInjector:
    def __init__(
        self,
        model: PreTrainedModel,
        injection_layers: torch.Tensor,  # (batch_size)
        injection_positions: torch.Tensor,  # (batch_size)
        hiddens_to_inject: torch.Tensor,  # (batch_size, hidden_size)
        debug_level: int = 1,  # デバッグレベル: 0=最小限、1=通常、2=詳細
    ):
        """
        Args:
            model: The model to inject hidden states into
            injection_layers: the layer to inject hidden states into, for each example in the batch (batch_size)
            injection_positions: the position to inject hidden states into, for each example in the batch (batch_size)
            hiddens_to_inject: the hidden states to inject, for each example in the batch (batch_size, hidden_size)
            debug_level: level of debug output (0=minimal, 1=normal, 2=verbose)
        """
        self._model = model
        self._debug_level = debug_level
        
        # 入力チェック - 次元を確認
        if hiddens_to_inject.ndim != 2:
            raise ValueError(f"hiddens_to_inject should be 2D (batch_size, hidden_size), but got shape {hiddens_to_inject.shape}")
        
        # CPUに保存してメモリを節約（必要な時だけGPUに移動）
        if torch.is_tensor(injection_layers):
            self._injection_layer = injection_layers.detach().cpu()
        else:
            self._injection_layer = torch.tensor([injection_layers], device='cpu')
            
        if torch.is_tensor(injection_positions):
            self._injection_position = injection_positions.detach().cpu()
        else:
            self._injection_position = torch.tensor([injection_positions], device='cpu')
            
        # バッチサイズが一致しているか確認
        if self._injection_layer.size(0) != self._injection_position.size(0):
            raise ValueError(f"Batch size mismatch: injection_layers ({self._injection_layer.size(0)}) and "
                             f"injection_positions ({self._injection_position.size(0)})")
        
        if self._injection_layer.size(0) != hiddens_to_inject.size(0):
            raise ValueError(f"Batch size mismatch: injection_layers ({self._injection_layer.size(0)}) and "
                             f"hiddens_to_inject ({hiddens_to_inject.size(0)})")
        
        # hiddens_to_injectは大きいテンソルなのでCPUに保存
        self._hidden_to_inject = hiddens_to_inject.detach().cpu()
        
        # 隠れ状態のサイズをモデルの隠れ状態サイズと比較
        model_hidden_size = model.config.hidden_size
        if self._hidden_to_inject.size(1) != model_hidden_size:
            raise ValueError(f"Hidden size mismatch: hiddens_to_inject ({self._hidden_to_inject.size(1)}) and "
                             f"model ({model_hidden_size})")
        
        self._hooks = []
        
        # モデルレイヤー情報を取得して保存
        self._layers = get_layers(model)
        
        # 注入するレイヤーが有効範囲内か確認
        max_layer_idx = len(self._layers) - 1
        for layer_idx in self._injection_layer:
            if layer_idx < 0 or layer_idx > max_layer_idx:
                raise ValueError(f"Layer index {layer_idx.item()} is out of range [0, {max_layer_idx}]")
        
        # 注入成功フラグ（デバッグ用）
        self._injection_success = [False] * len(self._injection_layer)
        
        # デバッグ用出力
        if self._debug_level >= 1:
            print(f"Debug: HiddenInjector initialized - layers shape: {self._injection_layer.shape}, "
                  f"positions shape: {self._injection_position.shape}, "
                  f"hiddens shape: {self._hidden_to_inject.shape}")
            print(f"Debug: Model has {len(self._layers)} layers, hidden size {model_hidden_size}")

    def __enter__(self):
        self._register_forward_hooks()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # フックを解除してリソースを解放
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        
        # 注入成功率をチェック（デバッグ用）
        if self._debug_level >= 1:
            success_count = sum(self._injection_success)
            total_count = len(self._injection_success)
            print(f"Debug: Injection success rate: {success_count}/{total_count} ({success_count/total_count:.2%})")
        
        # 明示的にメモリを解放
        torch.cuda.empty_cache()

    def _register_forward_hooks(self):
        def inject_hidden_hook(layer_idx):
            def inject_hidden(mod, inp, out):
                try:
                    # 出力の形式を確認（モデル種類によって異なる）
                    if isinstance(out, tuple):
                        hidden_states = out[0]  # 最初の要素を隠れ状態として扱う
                    elif hasattr(out, 'hidden_states'):
                        # transformersの一部のモデル出力
                        hidden_states = out.hidden_states
                    elif isinstance(out, torch.Tensor):
                        hidden_states = out
                    else:
                        # サポートされていない出力形式
                        if self._debug_level >= 2:
                            print(f"警告: 層{layer_idx}の出力形式がサポートされていません: {type(out)}")
                        return out
                    
                    # テンソルでない場合はスキップ
                    if not isinstance(hidden_states, torch.Tensor):
                        if self._debug_level >= 2:
                            print(f"警告: 層{layer_idx}の隠れ状態がテンソルではありません: {type(hidden_states)}")
                        return out
                    
                    # 現在のバッチサイズを取得
                    current_batch_size = hidden_states.shape[0]
                    
                    # このレイヤーに対する注入が必要かどうかを事前にチェック
                    inject_needed = False
                    for i in range(min(current_batch_size, len(self._injection_layer))):
                        if self._injection_layer[i].item() == layer_idx:
                            inject_needed = True
                            break
                            
                    if not inject_needed:
                        return out
                    
                    # 注入前の隠れ状態のコピーを作成（デバッグ用）
                    if self._debug_level >= 2:
                        hidden_states_before = hidden_states.clone()
                    
                    # バッチの各要素に対して個別に処理
                    modified = False
                    for batch_idx in range(current_batch_size):
                        # グローバルなインデックスを計算（現在のバッチ内での位置）
                        global_idx = batch_idx % len(self._injection_layer)
                        
                        # この層に注入するかチェック
                        if self._injection_layer[global_idx].item() == layer_idx:
                            # 位置の取得と適切な変換
                            pos = self._injection_position[global_idx].item()
                            if pos < 0:
                                pos = hidden_states.shape[1] + pos
                            
                            # 範囲内チェック
                            if 0 <= pos < hidden_states.shape[1]:
                                # 隠れ状態を注入
                                # このGPUデータがアクセスされるときだけCPUからコピー
                                hidden_to_inject = self._hidden_to_inject[global_idx].to(
                                    device=hidden_states.device,
                                    dtype=hidden_states.dtype, 
                                    non_blocking=True
                                )
                                
                                # 次元の確認
                                if hidden_states[batch_idx, pos].shape != hidden_to_inject.shape:
                                    if self._debug_level >= 1:
                                        print(f"警告: 隠れ状態の次元が一致しません: "
                                              f"model={hidden_states[batch_idx, pos].shape}, "
                                              f"inject={hidden_to_inject.shape}")
                                    # 次元を合わせる試み
                                    if hidden_states[batch_idx, pos].numel() == hidden_to_inject.numel():
                                        hidden_to_inject = hidden_to_inject.reshape(hidden_states[batch_idx, pos].shape)
                                    else:
                                        continue  # 次元が合わない場合はスキップ
                                
                                # 注入前の隠れ状態情報（デバッグ用）
                                if self._debug_level >= 2:
                                    before_norm = torch.norm(hidden_states[batch_idx, pos]).item()
                                    inject_norm = torch.norm(hidden_to_inject).item()
                                
                                # 実際の注入
                                hidden_states[batch_idx, pos] = hidden_to_inject
                                modified = True
                                self._injection_success[global_idx] = True
                                
                                # 注入の確認（デバッグ用）
                                if self._debug_level >= 2:
                                    after_norm = torch.norm(hidden_states[batch_idx, pos]).item()
                                    print(f"成功: 層{layer_idx}の位置{pos}に隠れ状態を注入しました（バッチ{batch_idx}）")
                                    print(f"  ノルム変化: {before_norm:.4f} -> {after_norm:.4f} (注入: {inject_norm:.4f})")
                            else:
                                if self._debug_level >= 1:
                                    print(f"警告: 位置{pos}が範囲外です（サイズ: {hidden_states.shape[1]}）")
                    
                    # 修正されなかった場合はマーカーを設定（デバッグ用）
                    if not modified and self._debug_level >= 1:
                        print(f"注意: 層{layer_idx}への注入が実行されませんでした")
                    
                    # 修正前後の変化を確認（デバッグ用）
                    if modified and self._debug_level >= 2:
                        # 注入前後の隠れ状態の平均二乗誤差を計算
                        mse = torch.mean((hidden_states - hidden_states_before) ** 2).item()
                        print(f"層{layer_idx}の隠れ状態変化: MSE={mse:.6f}")
                
                except Exception as e:
                    print(f"警告: 隠れ状態注入中にエラーが発生 (層{layer_idx}): {e}")
                
                # 出力形式を維持
                if isinstance(out, tuple):
                    # タプルの最初の要素を更新
                    return (hidden_states,) + out[1:]
                elif hasattr(out, 'hidden_states'):
                    # オブジェクトの場合、hidden_states属性を更新
                    out.hidden_states = hidden_states
                    return out
                else:
                    # その他の場合は直接返す
                    return hidden_states
            
            return inject_hidden
            
        # 各層にフックを登録
        try:
            for i, layer in enumerate(self._layers):
                hook = layer.register_forward_hook(inject_hidden_hook(i))
                self._hooks.append(hook)
                
        except Exception as e:
            print(f"エラー: フック登録中に例外が発生: {e}")
            # 登録済みのフックを解除
            for hook in self._hooks:
                hook.remove()
            self._hooks = []
            raise