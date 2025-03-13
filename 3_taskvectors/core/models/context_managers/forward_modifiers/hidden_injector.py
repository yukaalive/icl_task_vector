import torch
from transformers import PreTrainedModel
from core.models.utils.llm_layers import get_layers

class HiddenInjector:
    def __init__(
        self,
        layers=None,
        positions=None,
        hiddens=None
    ):
        """
        Args:
            layers: 隠れ状態を注入するレイヤーのリスト（バッチ内の全例に適用）
            positions: 隠れ状態を注入する位置のリスト（バッチ内の全例に適用）
            hiddens: 注入する隠れ状態 (batch_size, num_layers, hidden_size)
        """
        self._layers = layers
        self._positions = positions
        self._hiddens = hiddens
        self._hooks = []
        self._model = None

    def apply_to(self, model):
        """モデルに対して注入を適用するためのコンテキストマネージャを返す"""
        self._model = model
        return self

    def __enter__(self):
        print(f"Debug: HiddenInjector initialized - layers shape: {torch.Size(self._layers) if self._layers else None}, positions shape: {torch.Size(self._positions) if self._positions else None}, hiddens shape: {self._hiddens.shape if self._hiddens is not None else None}")
        
        # パラメータチェック
        if self._layers is None or len(self._layers) == 0:
            print("Error: 注入するレイヤーが指定されていません")
            # エラーを発生させず、空のフックリストを返すだけにする
            return self
        
        if self._positions is None or len(self._positions) == 0:
            print("Error: 注入する位置が指定されていません")
            # デフォルト位置を設定（最初のトークン）
            self._positions = [0]
        
        # モデルの層の数とhidden_sizeを取得
        try:
            layers = get_layers(self._model)
            num_layers = len(layers)
            
            # モデルのデータ型を取得（最初の層の重みから）
            model_dtype = next(self._model.parameters()).dtype
            
            # サンプル入力の生成と実行は無視 - トラブルの原因になるため
            print(f"Debug: Model has {num_layers} layers, dtype {model_dtype}")
        except Exception as e:
            print(f"Warning: モデル情報の取得に失敗しましたが、処理を続行します: {e}")
        
        self._register_forward_hooks()
        return self  # 必ずselfを返す

    def __exit__(self, exc_type, exc_value, traceback):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def _register_forward_hooks(self):
        def inject_hidden_hook(layer_idx):
            def inject_hidden(mod, inp, out):
                # layersに指定したレイヤーの場合のみ注入
                if layer_idx not in self._layers:
                    return out
                
                try:
                    hidden_states = out[0] if isinstance(out, tuple) else out
                    batch_size = hidden_states.shape[0]
                    
                    # 注入位置を特定
                    for batch_idx in range(batch_size):
                        pos = self._positions[0]  # すべてのバッチで同じ位置に注入
                        
                        # hiddensが指定されている場合、注入
                        if self._hiddens is not None:
                            # 元のノルムを記録
                            orig_norm = torch.norm(hidden_states[batch_idx, pos]).item()
                            
                            # 注入するhidden state
                            # 形状とデータ型を確認して適切に処理
                            try:
                                if len(self._hiddens.shape) == 4:  # (batch_size, num_layers, seq_len, hidden_size)
                                    # 注入データの選択
                                    inject_hidden = self._hiddens[batch_idx, 0, pos]
                                elif len(self._hiddens.shape) == 3:  # (batch_size, seq_len, hidden_size)
                                    inject_hidden = self._hiddens[batch_idx, pos]
                                else:  # 他の形状の場合
                                    print(f"Warning: 予期しない隠れ状態の形状: {self._hiddens.shape}")
                                    inject_hidden = self._hiddens[batch_idx]
                                
                                # デバイスと型を合わせる
                                inject_hidden = inject_hidden.to(hidden_states.device).to(hidden_states.dtype)
                                
                                # 注入
                                hidden_states[batch_idx, pos] = inject_hidden
                                
                                # 注入後のノルムを記録
                                new_norm = torch.norm(hidden_states[batch_idx, pos]).item()
                                
                                # MSEも計算
                                mse = torch.mean((hidden_states[batch_idx, pos] - inject_hidden) ** 2).item()
                                
                                print(f"成功: 層{layer_idx}の位置{pos}に隠れ状態を注入しました（バッチ{batch_idx}）")
                                print(f"  ノルム変化: {orig_norm:.4f} -> {new_norm:.4f} (注入: {new_norm:.4f})")
                                print(f"層{layer_idx}の隠れ状態変化: MSE={mse:.6f}")
                            except Exception as e:
                                print(f"警告: 層{layer_idx}、バッチ{batch_idx}、位置{pos}での隠れ状態注入に失敗しました: {e}")
                                # エラー詳細のデバッグ情報
                                print(f"  隠れ状態の形状: {self._hiddens.shape}")
                                print(f"  レイヤーインデックス: {self._layers}")
                                print(f"  現在のレイヤー: {layer_idx}")
                except Exception as e:
                    print(f"警告: 層{layer_idx}でのフック実行中にエラーが発生しました: {e}")
                    # 元の出力をそのまま返す
                    
                return out
            return inject_hidden
        
        try:
            # 各層にフックを登録
            for i, layer in enumerate(get_layers(self._model)):
                hook = layer.register_forward_hook(inject_hidden_hook(i))
                self._hooks.append(hook)
            
            # 注入の準備が完了したことを報告
            print(f"Debug: Injection success rate: {len(self._hooks)}/{len(get_layers(self._model))} ({100.0 * len(self._hooks) / len(get_layers(self._model)):.2f}%)")
        except Exception as e:
            print(f"警告: フック登録中にエラーが発生しました: {e}")