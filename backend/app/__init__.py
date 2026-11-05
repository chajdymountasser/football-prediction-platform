# app/__init__.py
from flask import Flask
from .config import Config
import os

def create_app():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

    app = Flask(
        __name__,
        template_folder=FRONTEND_DIR,
        static_folder=FRONTEND_DIR,
        static_url_path="/static"
    )

    app.config.from_object(Config)

    # API routes
    from .routes.fixtures import fixtures_bp
    from .routes.standings import standings_bp
    from .routes.meta import meta_bp
    from .routes.predict import predict_bp
    from .routes.pages import pages_bp
    from .routes.teams import teams_bp

    app.register_blueprint(fixtures_bp, url_prefix="/api")
    app.register_blueprint(standings_bp, url_prefix="/api")
    app.register_blueprint(meta_bp, url_prefix="/api")
    app.register_blueprint(predict_bp, url_prefix="/api")
    app.register_blueprint(teams_bp, url_prefix="/api")

    # Pages
    app.register_blueprint(pages_bp)

    return app
