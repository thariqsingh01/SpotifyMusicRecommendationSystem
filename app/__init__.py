from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import urllib
from sqlalchemy import create_engine
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

db = SQLAlchemy() 
def log_registered_routes(app):
    logging.info("Registered Routes:")
    for rule in app.url_map.iter_rules():
        logging.info(f"{rule}")

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
        # Import the models here to avoid circular imports
        from .models import SpotifyData  
        db.create_all()

        from .main import bp as main_bp
        app.register_blueprint(main_bp)

    return app, engine 

    
