import pandas as pd

# 讀取原始 CSV
data_path = "./data/elder.csv"   # 你的檔案路徑
df = pd.read_csv(data_path)

# 要修改成的 instruction
new_instruction = "你是一位長者，有時固執但可妥協，喜歡自然對話，每次回覆簡短（1–100字），會分享日常、回憶或感受，當志工提出建議時可能堅持自己的想法，但在合理溫和說服下願意妥協，用真實長者語氣回答，讓志工學會耐心傾聽與引導。"

# 統一修改 instruction 欄位
df["instruction"] = new_instruction

# 輸出到新檔案
save_path = "elder.csv"
df.to_csv(save_path, index=False, encoding="utf-8-sig")

print(f"已經修改 instruction 並存成 {save_path}")
