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
    ):
        """
        Args:
            model: The model to inject hidden states into
            injection_layers: the layer to inject hidden states into, for each example in the batch (batch_size)
            injection_positions: the position to inject hidden states into, for each example in the batch (batch_size)
            hiddens_to_inject: the hidden states to inject, for each example in the batch (batch_size, hidden_size)
        """
        self._model = model
        
        # CPUに保存してメモリを節約（必要な時だけGPUに移動）
        if torch.is_tensor(injection_layers):
            self._injection_layer = injection_layers.detach().cpu()
        else:
            self._injection_layer = torch.tensor([injection_layers], device='cpu')
            
        if torch.is_tensor(injection_positions):
            self._injection_position = injection_positions.detach().cpu()
        else:
            self._injection_position = torch.tensor([injection_positions], device='cpu')
            
        # hiddens_to_injectは大きいテンソルなのでCPUに保存
        self._hidden_to_inject = hiddens_to_inject.detach().cpu()
        
        self._hooks = []
        
        # デバッグ用出力
        print(f"Debug: HiddenInjector initialized - layers shape: {self._injection_layer.shape}, "
              f"positions shape: {self._injection_position.shape}, "
              f"hiddens shape: {self._hidden_to_inject.shape}")

    def __enter__(self):
        self._register_forward_hooks()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # フックを解除してリソースを解放
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        
        # 明示的にメモリを解放
        torch.cuda.empty_cache()

    def _register_forward_hooks(self):
        def inject_hidden_hook(layer_idx):
            def inject_hidden(mod, inp, out):
                try:
                    # 出力がタプルの場合は最初の要素を取得
                    hidden_states = out[0] if isinstance(out, tuple) else out
                    
                    if not isinstance(hidden_states, torch.Tensor):
                        # 例えばAttentionOutputのようなオブジェクトの場合
                        if hasattr(hidden_states, 'hidden_states'):
                            hidden_states = hidden_states.hidden_states
                        else:
                            # 適切な隠れ状態が見つからなければ何もしない
                            return out
                    
                    # 現在のバッチサイズを取得
                    current_batch_size = hidden_states.shape[0]
                    
                    # このレイヤーに対する注入が必要かどうかを事前にチェック
                    # CPUで計算して、必要な場合のみGPUに移動
                    inject_needed = False
                    for i in range(min(current_batch_size, len(self._injection_layer))):
                        if self._injection_layer[i].item() == layer_idx:
                            inject_needed = True
                            break
                            
                    if not inject_needed:
                        return out
                    
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
                                hidden_states[batch_idx, pos] = hidden_to_inject
                                modified = True
                    
                    # 修正されなかった場合はマーカーを設定（デバッグ用）
                    if not modified and layer_idx % 5 == 0:  # すべての層で出力すると多すぎるため
                        print(f"注意: 層{layer_idx}への注入が実行されませんでした")
                
                except Exception as e:
                    print(f"警告: 隠れ状態注入中にエラーが発生 (層{layer_idx}): {e}")
                
                return out
            
            return inject_hidden
            
        # 各層にフックを登録
        try:
            layers = get_layers(self._model)
            for i, layer in enumerate(layers):
                hook = layer.register_forward_hook(inject_hidden_hook(i))
                self._hooks.append(hook)
                
        except Exception as e:
            print(f"エラー: フック登録中に例外が発生: {e}")
            # 登録済みのフックを解除
            for hook in self._hooks:
                hook.remove()
            self._hooks = []
            raise