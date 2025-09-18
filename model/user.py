from sqlalchemy import DateTime, func
from extensions import db


class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    line_id = db.Column(db.String(255), nullable=False)  # 假設你有一個 Line id 
    user_nickname = db.Column(db.String(255), nullable=False)
    file_token = db.Column(db.String(255), nullable=True)
    file_name = db.Column(db.Text, nullable=True)
    created_at = db.Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = db.Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)
    
    def __init__(self, line_id,user_nickname, file_token):
        self.line_id = line_id
        self.user_nickname = user_nickname
        self.file_token = file_token
        self.file_name = None