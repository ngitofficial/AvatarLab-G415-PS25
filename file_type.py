import subprocess

def convert_wav_to_mp3(input_path, output_path):
    if not input_path.lower().endswith('.wav'):
        raise ValueError("Input file must be a .wav file")
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-acodec", "libmp3lame",
        "-q:a", "2",
        output_path
    ]
    subprocess.run(cmd, check=True)

def is_audio_file(filename, allowed_extensions={'mp3', 'wav'}):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def extract_mp3_from_mp4(input_mp4_path, output_mp3_path):
    cmd = [
        "ffmpeg",
        "-i", input_mp4_path,
        "-vn",
        "-acodec", "libmp3lame",
        "-q:a", "2",
        output_mp3_path
    ]
    subprocess.run(cmd, check=True)

def extract_wav_from_mp4(input_mp4_path, output_wav_path):
    cmd = [
        "ffmpeg",
        "-i", input_mp4_path,
        "-vn",  # Disable video
        "-acodec", "pcm_s16le",
        "-ar", "44100",  # Sample rate
        "-ac", "2",      # Channels
        output_wav_path
    ]
    subprocess.run(cmd, check=True)

def remove_audio_from_mp4(input_mp4_path, output_muted_mp4_path):
    cmd = [
        "ffmpeg",
        "-i", input_mp4_path,
        "-an",  # Remove audio
        "-c:v", "copy",
        output_muted_mp4_path
    ]
    subprocess.run(cmd, check=True)
