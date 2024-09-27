#__init__.py

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import urllib
from sqlalchemy import create_engine

db = SQLAlchemy()

def create_app(uri=None):
    app = Flask(__name__)
    if uri is None:
        uri = 'mssql+pyodbc:///?odbc_connect=' + urllib.parse.quote_plus(
            'DRIVER={ODBC Driver 17 for SQL Server};SERVER=MSCS\\SQLEXPRESS;DATABASE=Spotify;Trusted_Connection=yes;'
        )
    app.config['SQLALCHEMY_DATABASE_URI'] = uri
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SQLALCHEMY_ECHO'] = True

    # Create the engine
    engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'], echo=False)

    db.init_app(app)

    with app.app_context():
        from .main import bp as main_bp
        # Register the main blueprint
        app.register_blueprint(main_bp)

        # Optional: Create tables if not exist
        db.create_all()

    return app, engine  # Return both app and engine
