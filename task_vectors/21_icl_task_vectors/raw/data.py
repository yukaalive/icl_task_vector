import pandas as pd
# データを読み込み
df = pd.read_csv('raw', sep='\t', header=None, names=['en', 'ja'])

# 英語と日本語の両方が短い文章を抽出（例: 50文字以下）
df_short = df[(df['en'].str.len() <= 50) & (df['ja'].str.len() <= 30)]

# ランダムに200個をサンプリング
df_sample = df_short.sample(n=200, random_state=42)

# 保存
df_sample.to_csv('short_200.tsv', sep='\t', index=False, header=False)

print(f"短い文章の総数: {len(df_short)}")
print(f"抽出した文章数: {len(df_sample)}")
print("\n最初の5つ:")
print(df_sample.head())