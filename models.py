import torch
from flask import current_app
from flask1.ttsmodel.openvoice.api import BaseSpeakerTTS, ToneColorConverter

_base_speaker_tts = None
_tone_color_converter = None
_source_se = None
_device = None

def get_openvoice_models():
    global _base_speaker_tts, _tone_color_converter, _source_se, _device

    if _base_speaker_tts is not None and _tone_color_converter is not None and _source_se is not None:
        return _base_speaker_tts, _tone_color_converter, _source_se, _device

    ckpt_base = current_app.config['CHECKPOINTS_BASE']
    ckpt_converter = current_app.config['CHECKPOINTS_CONVERTER']
    _device = "cuda:0" if torch.cuda.is_available() else "cpu"

    _base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=_device)
    _base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')

    _tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=_device)
    _tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

    _source_se = torch.load(f'{ckpt_base}/en_default_se.pth').to(_device)

    return _base_speaker_tts, _tone_color_converter, _source_se, _device
