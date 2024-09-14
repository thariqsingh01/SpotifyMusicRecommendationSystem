#__init__.py

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import urllib

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'mssql+pyodbc:///?odbc_connect=' + urllib.parse.quote_plus(
        'DRIVER={ODBC Driver 17 for SQL Server};SERVER=MSCS\\SQLEXPRESS;DATABASE=Spotify;Trusted_Connection=yes;')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SQLALCHEMY_ECHO'] = True

    db.init_app(app)

    with app.app_context():
        from .main import bp as main_bp
        from .main import initialize_clustering

        # Register the main blueprint
        app.register_blueprint(main_bp)

        # Optional: Create tables if not exist
        db.create_all()

        # Perform clustering initialization once at startup
        initialize_clustering()

    return app
