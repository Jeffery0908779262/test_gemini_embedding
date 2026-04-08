import os
import numpy as np
import pandas as pd
from google import genai
from sklearn.metrics.pairwise import cosine_similarity

# 1. 初始化 Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 2. 載入你的資料庫
print("載入資料庫中...")
embeddings = np.load("space_a_embeddings.npy")   # 載入你先前存好的文件向量
df = pd.read_csv("space_a_index.csv", encoding="utf-8-sig")

# 3. 測試問題
test_query = "乙方於終止租約或租賃期滿不交還房屋，自終止租約或租賃期滿之翌日起，乙方應支付按房租壹倍計算之違約金。"
print(f"\n🗣️ 測試問題：{test_query}")

# 4. 用 Gemini API 將問題轉成向量
response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=test_query
)

query_embedding = np.array(response.embeddings[0].values, dtype=np.float32)

# 5. 正規化 query 向量（如果你的文件向量當初也有 normalize，這樣最一致）
norm = np.linalg.norm(query_embedding)
if norm > 0:
    query_embedding = query_embedding / norm

# 6. 計算相似度
# query_embedding shape: (d,)
# embeddings shape: (n, d)
cos_scores = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]

# 7. 找出最高的前 3 名
top_k = 3
top_results_indices = np.argsort(cos_scores)[-top_k:][::-1]

# 8. 印出結果
print("\n🎯 AI 幫你找出的最相關法規 Top 3：\n" + "-" * 40)
for i, idx in enumerate(top_results_indices):
    score = cos_scores[idx]
    topic = df["主題"].iloc[idx]

    print(f"第 {i+1} 名 (相似度分數: {score:.4f})")
    print(f"📌 命中主題：{topic}")

    # 如果你想多印內容，也可以加下面這行
    print(df["RAG 優化擴充文本"].iloc[idx])

    print("-" * 40)