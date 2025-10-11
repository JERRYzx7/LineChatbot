import os
from dotenv import load_dotenv
from flask import Flask, request, abort, send_from_directory
from linebot.v3 import (
    WebhookHandler
)
from linebot.v3.exceptions import (
    InvalidSignatureError
)
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
    PushMessageRequest
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    FileMessageContent
)
# Import the blueprint
# from extensions import db #, jwt
from routes import main_bp
from service.upload_controller import upload_bp
import uuid
from repository.user_repo import UserRepo
from model.user import User

from training.gemma_inference import gemma_inference

DOMAIN = "https://refutably-cistic-miyoko.ngrok-free.dev"


model_path = f"./training/models/saved-Qwen3-model-8B"
lora_path = f"./training/models/saved-Qwen3-model-8B"
# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config.from_prefixed_env()
# Register the blueprint
app.register_blueprint(main_bp)
app.register_blueprint(upload_bp, url_prefix="/upload")

# @app.before_request
# def create_tables():
#     db.create_all()

# db.init_app(app)
# jwt.init_app(app)

# Get LINE Channel credentials from environment variables
channel_secret = os.getenv('LINE_CHANNEL_SECRET')
channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')

if not channel_secret or not channel_access_token:
    print('Please set LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN in your .env file')
    exit()

handler = WebhookHandler(channel_secret)
configuration = Configuration(access_token=channel_access_token)

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        app.logger.info("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_id = event.source.user_id   # 取得使用者 ID
    user_text = event.message.text.strip()  # 使用者輸入文字
    
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)

        # 取得使用者暱稱
        profile = line_bot_api.get_profile(user_id=user_id)
        user_nickname = profile.display_name

        # Echo the same message back to the user
        if user_text == "上傳":
            token = str(uuid.uuid4())
            user_repo = UserRepo()
            user = user_repo.find_user_by_line_id(line_id=user_id)
            if user:
                user_repo.update_file_token(user, user_nickname, token)
            else:
                user_repo.create_user(user_id, user_nickname, token)
            upload_url = f"{DOMAIN}/upload-page?token={token}"
            reply_text = f"請點擊以下連結上傳檔案：\n{upload_url}"
        else:
            reply_text = gemma_inference(model_path, lora_path, user_text)  # 預設回 echo

        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_text)]
            )
        )

if __name__ == "__main__":
    app.run(port=5000)
