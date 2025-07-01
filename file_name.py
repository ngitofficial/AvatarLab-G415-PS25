import uuid
from datetime import datetime

def generate_audio_filename(prefix = "speech", ext = "wav"):
    unique_id = uuid.uuid4().hex
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}_{unique_id}.{ext}"

def generate_mp4_filename(prefix = "video", ext = "mp4"):
    unique_id = uuid.uuid4().hex
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}_{unique_id}.{ext}"