import logging

from sqlalchemy import and_, extract
from extensions import db
from model.user import User
from sqlalchemy.exc import SQLAlchemyError

class UserRepo:

    @staticmethod
    def find_user_by_line_id(line_id):
        try:
            return User.query.filter_by(line_id=line_id).first()
        except SQLAlchemyError as e:
            logging.error(f"Error getting user by line_id: {e}")
            return None
    
    @staticmethod
    def create_user(line_id, user_nickname, file_token):
        try:
            user = User(line_id=line_id, user_nickname=user_nickname, file_token=file_token)
            db.session.add(user)
            db.session.commit()
        except SQLAlchemyError as e:
            logging.error(f"Error creating user: {e}")
            return None
        return user
    
    @staticmethod
    def update_file_token(user, user_nickname, file_token):
        try:
            user.file_token = file_token
            user.user_nickname = user_nickname
            db.session.commit()
            return user
        except SQLAlchemyError as e:
            logging.error(f"Error updating file token: {e}")
            return None
    @staticmethod
    def update_file_name(user, file_name):
        try:
            user.file_name = file_name
            db.session.commit()
            return user
        except SQLAlchemyError as e:
            logging.error(f"Error updating file name: {e}")
            return None

    @staticmethod
    def find_user_by_file_token(file_token):
        try:
            return User.query.filter_by(file_token=file_token).first()
        except SQLAlchemyError as e:
            logging.error(f"Error finding user by file_token: {e}")
            return None
    
    @staticmethod
    def delete_user(line_id):
        try:
            user = User.query.filter_by(line_id=line_id).first()
            db.session.delete(user)
            db.session.commit()
            return user
        except SQLAlchemyError as e:
            logging.error(f"Error deleting user: {e}")
            return None