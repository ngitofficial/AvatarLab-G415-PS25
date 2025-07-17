from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
import os

app = Flask(__name__)
CORS(app, supports_credentials=True)

app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:iposteatu9@localhost:5432/UserInfo'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'users.login'

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

print(BASE_DIR)

app.config['BASE_DIR'] = os.path.join(BASE_DIR)


app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'files', 'uploads')
app.config['AUDIO_OUTPUT_FOLDER'] = os.path.join(BASE_DIR, 'files', 'outputaudio')
app.config['REFERENCE_AUDIO_FOLDER'] = os.path.join(BASE_DIR, 'files', 'reference_audios')
app.config['DEFAULT_VOICES_FOLDER'] = os.path.join(BASE_DIR, 'files', 'defaultvoices')
app.config['PROCESSED_FOLDER'] = os.path.join(BASE_DIR, 'files', 'processed')
app.config['CHECKPOINTS_BASE'] = os.path.join(BASE_DIR,'files', 'checkpoints', 'base_speakers', 'EN')
app.config['CHECKPOINTS_CONVERTER'] = os.path.join(BASE_DIR, 'files', 'checkpoints', 'converter')
app.config['FINAL_OUTPUT_FOLDER'] = os.path.join(BASE_DIR,'files','output_final')

from flask1.routes.uploadroute import model_routes
from flask1.users.routes import users
from flask1.users.user_history import hist
from flask1.admin.allusers import admin_hist

app.register_blueprint(model_routes)
app.register_blueprint(users)
app.register_blueprint(hist)
app.register_blueprint(admin_hist)