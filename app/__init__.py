from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import urllib
from sqlalchemy import create_engine

db = SQLAlchemy()  # Define db here, but we won't initialize it yet

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

    db.init_app(app)  # Initialize db with the app

    with app.app_context():
        # Import the models here to avoid circular imports
        from .models import SpotifyData  

        # Optional: Create tables if not exist
        db.create_all()

        # Register the main blueprint
        from .main import bp as main_bp
        app.register_blueprint(main_bp)

    return app, engine  # Return both app and engine
