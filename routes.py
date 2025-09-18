from flask import Blueprint, send_from_directory

# Create a Blueprint for the main routes
main_bp = Blueprint('main', __name__)

@main_bp.route("/upload-page")
def index():
    """Serves the main index.html page."""
    return send_from_directory('static', 'index.html')
