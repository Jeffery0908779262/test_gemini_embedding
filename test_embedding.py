import pandas as pd
import numpy as np
import os
from google import genai

# 初始化 Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 1. 載入 xlsx
print("正在讀取 xlsx 檔案...")
try:
    df = pd.read_excel('space_a.xlsx')
except Exception as e:
    print(f"讀取失敗，請檢查檔名或路徑：{e}")
    exit()

print(df.columns.tolist())

# 2. 確認欄位名稱
print("欄位名稱：", df.columns.tolist())
print(f"共 {len(df)} 筆資料\n")

# 3. Gemini embedding 函式
def embed_text(text):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    vector = np.array(response.embeddings[0].values, dtype=np.float32)

    # 👉 加上 normalize（很重要）
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    return vector

# 4. 組合要向量化的文字
df.columns = df.columns.str.strip()
df = df.fillna("")

texts = df['主題'].astype(str) + "：" + df['RAG 優化擴充文本'].astype(str)

# 5. 執行向量化
print(f"開始向量化，共 {len(texts)} 筆資料...")

embeddings = []
for i, text in enumerate(texts.tolist()):
    if i % 10 == 0:
        print(f"進度：{i}/{len(texts)}")

    emb = embed_text(text)
    embeddings.append(emb)

embeddings = np.array(embeddings)

print(f"\n向量化完成！向量維度：{embeddings.shape}")

# 6. 儲存向量
np.save('space_a_embeddings.npy', embeddings)
print("已儲存：space_a_embeddings.npy")

# 7. 儲存對應索引
df.to_csv('space_a_index.csv', encoding='utf-8-sig', index=True)
print("已儲存：space_a_index.csv")