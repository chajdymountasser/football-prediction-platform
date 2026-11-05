# app/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
    DEBUG = os.getenv("FLASK_DEBUG", "False") == "True"
