# from flask import current_app
# import os
# import torch
# from openvoice import se_extractor
# from openvoice.api import BaseSpeakerTTS, ToneColorConverter
# from utils.file_name import generate_audio_filename

# ckpt_base = current_app.config['CHECKPOINTS_BASE'] #'checkpoints/base_speakers/EN'
# ckpt_converter = current_app.config['CHECKPOINTS_CONVERTER'] #'checkpoints/converter'
# aof = current_app.config['AUDIO_OUTPUT_FOLDER'] #'.\\flask1\\files\\outputaudio\\'

# device="cuda:0" if torch.cuda.is_available() else "cpu"

# base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device)
# base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')

# tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
# tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

# source_se = torch.load(f'{ckpt_base}/en_default_se.pth').to(device)

# def generate_speech_with_openvoice(text, reference_audio_path):
#     unique_name = generate_audio_filename()
#     output_path = os.path.join(aof, unique_name)
#     target_se, _ = se_extractor.get_se(reference_audio_path, tone_color_converter, target_dir='processed', vad=True)

#     tmp_path = output_path.replace(".wav", "_tmp.wav")
    
#     base_speaker_tts.tts(text, tmp_path, speaker='default', language='English', speed=1.0)

#     encode_message = "@MyShell"
#     tone_color_converter.convert(
#         audio_src_path=tmp_path,
#         src_se=source_se,
#         tgt_se=target_se,
#         output_path=output_path,
#         message=encode_message
#     )

#     return output_path


from flask import current_app
import os
from flask1.ttsmodel.openvoice import se_extractor
from flask1.utils.file_name import generate_audio_filename
from flask1.ttsmodel.models import get_openvoice_models

def generate_speech_with_openvoice(text, reference_audio_path):
    base_speaker_tts, tone_color_converter, source_se, _ = get_openvoice_models()
    aof = current_app.config['AUDIO_OUTPUT_FOLDER']
    procs_dir = current_app.config['PROCESSED_FOLDER']
    os.makedirs(procs_dir, exist_ok=True)
    
    unique_name = generate_audio_filename()
    output_path = os.path.join(aof, unique_name)

    target_se, _ = se_extractor.get_se(reference_audio_path, tone_color_converter, target_dir=procs_dir, vad=True)
    print(f"audio string: {_}")
    #extract_se(...) gives you the target speaker embedding — you store that as target_se
    #audio_name is a string you don’t need — so you ignore it with _                                        
    tmp_path = output_path.replace(".wav", "_tmp.wav")
    
    base_speaker_tts.tts(text, tmp_path, speaker='default', language='English', speed=1.0)

    tone_color_converter.convert(
        audio_src_path=tmp_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=output_path,
        message="@MyShell"
    )

    return output_path