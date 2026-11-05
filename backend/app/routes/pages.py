# app/routes/pages.py
from flask import Blueprint, send_from_directory
import os

pages_bp = Blueprint("pages", __name__)

FRONTEND_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
)

@pages_bp.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")

@pages_bp.route("/matches")
def matches():
    return send_from_directory(FRONTEND_DIR, "matches.html")

@pages_bp.route("/table")
def table():
    return send_from_directory(FRONTEND_DIR, "table.html")

@pages_bp.route("/news")
def news():
    return send_from_directory(FRONTEND_DIR, "news.html")
