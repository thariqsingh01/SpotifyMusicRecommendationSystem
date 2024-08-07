from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import urllib

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'mssql+pyodbc:///?odbc_connect=' + urllib.parse.quote_plus(
        'DRIVER={ODBC Driver 17 for SQL Server};SERVER=DESKTOP-MS27SES\\SQLEXPRESS;DATABASE=Spotify;Trusted_Connection=yes;')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SQLALCHEMY_ECHO'] = True

    db.init_app(app)

    with app.app_context():
        from . import main
        app.register_blueprint(main.bp)

        # Optional: Create tables if not exist
        db.create_all()

    return app
