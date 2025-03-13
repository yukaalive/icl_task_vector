# 元のコード（もしあれば）
injector = HiddenInjector(
    model=model,
    injection_layers=layer_tensor,
    injection_positions=position_tensor,
    hiddens_to_inject=hidden_tensor
)

# 修正後のコード
injector = HiddenInjector(
    layers=[layer],  # 整数のリスト
    positions=[position],  # 整数のリスト
    hiddens=hidden_tensor  # テンソル
)

# 使用時は
with injector.apply_to(model):
    # モデルを使った処理