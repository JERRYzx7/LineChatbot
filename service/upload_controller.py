from flask import Blueprint, send_from_directory
import os
from flask import request, jsonify
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    PushMessageRequest,
    TextMessage
)
from repository.user_repo import UserRepo
from utils.textToCSV import LineChatProcessor
# 建立 MessagingApi client

from dotenv import load_dotenv
load_dotenv()

configuration = Configuration(access_token=os.getenv('LINE_CHANNEL_ACCESS_TOKEN'))

def push_message(user_id, text):
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)

        line_bot_api.push_message(
            PushMessageRequest(
                to=user_id,
                messages=[TextMessage(text=text)]
            )
        )


UPLOAD_FOLDER = "uploads"

# Create a Blueprint for the main routes
upload_bp = Blueprint('upload', __name__)

# 確保有 uploads 資料夾
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@upload_bp.post("/upload-file")
def upload_file():
    """
    接收前端傳來的 token 與 txt 檔案，並儲存檔案
    """
    token = request.form.get("token")
    file = request.files.get("file")

    if not token:
        return jsonify({"error": "缺少 token"}), 400
    if not file or file.filename == "":
        return jsonify({"error": "缺少檔案"}), 400
    if not file.filename.lower().endswith(".txt"):
        return jsonify({"error": "只允許上傳 .txt 檔案"}), 400

    # 安全檔名（避免 token 汙染）
    save_path = os.path.join(UPLOAD_FOLDER, file.filename)

    try:
        # 嘗試存檔
        file.save(save_path)

        # 確認存檔成功之後，找 user 並推播訊息
        user = UserRepo().find_user_by_file_token(token)
        if not user:
            return jsonify({"error": "找不到對應使用者"}), 404

        push_message(user.line_id, "📂 檔案已成功上傳 ✅")

        processor = LineChatProcessor(output_name="test", master_name=user.user_nickname, data_dir="./processed_csv")

        with open(save_path, "rb") as f:  # 用 rb，模擬上傳的檔案
            csv_file_name = processor.process(f)

        print("輸出 CSV:", csv_file_name)

        return jsonify({
            "message": "檔案上傳成功",
            "token": token,
            "filename": file.filename,
            "saved_as": file.filename
        }), 200

    except Exception as e:
        # 存檔失敗就不推播
        return jsonify({"error": f"檔案儲存失敗: {str(e)}"}), 500
    
