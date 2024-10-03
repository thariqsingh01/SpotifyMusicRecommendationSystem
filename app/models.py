#models.py

from app import db

class SpotifyData(db.Model):
    __tablename__ = 'Spotify'
    number = db.Column(db.Integer)
    artist_name = db.Column(db.String)
    track_name = db.Column(db.String)
    track_id = db.Column(db.String, unique=True, primary_key=True)
    popularity = db.Column(db.Integer)
    year = db.Column(db.Integer)
    genre = db.Column(db.String)
    danceability = db.Column(db.Float)
    energy = db.Column(db.Float)
    key = db.Column(db.Integer)
    loudness = db.Column(db.Float)
    mode = db.Column(db.Float)
    speechiness = db.Column(db.Float)
    acousticness = db.Column(db.Float)
    instrumentalness = db.Column(db.Float)
    liveness = db.Column(db.Float)
    valence = db.Column(db.Float)
    tempo = db.Column(db.Float)
    duration_ms = db.Column(db.Integer)
    time_signature = db.Column(db.Integer)
    kmeans = db.Column(db.Integer) 
    dbscan = db.Column(db.Integer)  
    agglomerative = db.Column(db.Integer)  

    def duration_in_minutes_seconds(self):
        if self.duration_ms:
            total_seconds = self.duration_ms // 1000
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes} min {seconds} sec"
        else:
            return None
        
    def __repr__(self):
        return f'<SpotifyData {self.track_name}>'
