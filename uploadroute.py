
from flask import Blueprint, request, jsonify, send_file, current_app
import os
import shutil
from flask1.utils.file_type import is_audio_file, convert_wav_to_mp3
from flask1.ttsmodel.opv1 import generate_speech_with_openvoice
from flask1.lsmodel.inference import generate_lipsynced_video

tts_routes = Blueprint('tts_routes', __name__)

@tts_routes.route('/upload', methods=['POST'])
def upload():
    text = request.form.get('text')
    video = request.files.get('video')  #  Now expecting a video instead of image
    audio = request.files.get('audio')
    voice_choice = request.form.get('voice_choice')

    if not text:
        print("Text is required.")
        return jsonify({"error": "Text is required."}), 400
    
    if audio:
        if not is_audio_file(audio.filename):
            print("Unsupported audio format. Use .wav or .mp3")
            return jsonify({"error": "Unsupported audio format. Use .wav or .mp3"}), 400
    elif voice_choice:
        print(f"default {voice_choice} selected")
    else:
        print("Either upload a reference audio or choose male/female voice.")
        return jsonify({"error": "Either upload a reference audio or choose male/female voice."}), 400

    print("Received text:", text)
    if voice_choice:
        print("Voice choice: ", voice_choice)

    upload_folder = current_app.config['UPLOAD_FOLDER']
    output_folder = current_app.config['AUDIO_OUTPUT_FOLDER']
    default_voices = current_app.config['DEFAULT_VOICES_FOLDER']
    ref_audio_folder = current_app.config['REFERENCE_AUDIO_FOLDER']
    #final_output_folder = current_app.config['FINAL_OUTPUT_FOLDER']

    os.makedirs(upload_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(default_voices, exist_ok=True)
    #os.makedirs(final_output_folder, exist_ok=True)

    video_path = None
    audio_path = None

    if video:
        video_path = os.path.join(upload_folder, video.filename)
        video.save(video_path)
        print(f"Saved video to {video_path}")
    else:
        return jsonify({"error": "Video (.mp4) is required."}), 400

    if audio:
        filename = audio.filename
        ext = os.path.splitext(filename)[1].lower()
        save_path = os.path.join(ref_audio_folder, filename)

        if ext == ".wav":
            wav_path = os.path.join(ref_audio_folder, filename)
            audio.save(wav_path)
            mp3_path = wav_path.replace(".wav", ".mp3")
            try:
                convert_wav_to_mp3(wav_path, mp3_path)
                os.remove(wav_path)
                audio_path = mp3_path
                print(f"Converted and saved reference audio to {mp3_path}")
            except Exception as e:
                return jsonify({"error": f"FFmpeg conversion failed: {e}"}), 500
        else:
            audio.save(save_path)
            audio_path = save_path
            print(f"Saved reference audio to {audio_path}")

    elif voice_choice in ['male', 'female']:
        audio_filename = 'male_default.mp3' if voice_choice == 'male' else 'female_default.mp3'
        audio_path = os.path.join(default_voices, audio_filename)
        print(f"Using default reference audio: {audio_path}")
    else:
        return jsonify({"error": "Either upload a reference audio or choose male/female voice."}), 400

    if text:
        #  Generate speech with OpenVoice
        audio_output_path = generate_speech_with_openvoice(text, audio_path)

        # Clean processed folder
        procs_dir = current_app.config['PROCESSED_FOLDER']
        shutil.rmtree(procs_dir, ignore_errors=True)

        #  Generate lipsynced video
        #output_video_path = os.path.join(final_output_folder, "result_synced.mp4")
        output_path = generate_lipsynced_video(face_path=video_path,
                                                audio_path=audio_output_path,
                                                checkpoint_path="./flask1/files/checkpoints/wav2lip_gan.pth"
                                                #output_path=output_video_path
        )

        return send_file(
            output_path,
            mimetype='video/mp4',
            as_attachment=True
        )

    return jsonify({
        "message": "Files received successfully",
        "text": text,
        "video_saved": bool(video),
        "audio_saved": bool(audio)
    })




# from flask import Blueprint, request, jsonify, send_file, current_app
# import os
# import shutil
# from flask1.utils.file_type import is_audio_file, convert_wav_to_mp3
# from flask1.ttsmodel.opv1 import generate_speech_with_openvoice
# from flask1.lipsyncmodel.inference import generate_lipsynced_video

# tts_routes = Blueprint('tts_routes', __name__)

# @tts_routes.route('/upload', methods=['POST'])
# def upload():
#     text = request.form.get('text')
#     image = request.files.get('image')
#     audio = request.files.get('audio')
#     voice_choice = request.form.get('voice_choice')

#     if not text:
#         print("Text is required.")
#         return jsonify({"error": "Text is required."}), 400
    
#     if audio:
#         if not is_audio_file(audio.filename):
#             print("Unsupported audio format. Use .wav or .mp3")
#             return jsonify({"error": "Unsupported audio format. Use .wav or .mp3"}), 400
#     elif voice_choice:
#         print(f"default {voice_choice} selected")
#     else:
#         print("Either upload a reference audio or choose male/female voice.")
#         return jsonify({"error": "Either upload a reference audio or choose male/female voice."}), 400


#     print("Received text:", text)

#     if(voice_choice):
#         print("Voice choice: ", voice_choice)

#     upload_folder = current_app.config['UPLOAD_FOLDER']
#     output_folder = current_app.config['AUDIO_OUTPUT_FOLDER']
#     default_voices = current_app.config['DEFAULT_VOICES_FOLDER']
#     ref_audio_folder = current_app.config['REFERENCE_AUDIO_FOLDER']

#     os.makedirs(upload_folder, exist_ok=True)
#     os.makedirs(output_folder, exist_ok=True)
#     os.makedirs(default_voices, exist_ok=True)

#     image_path = None
#     audio_path = None

#     if image:
#         image_path = os.path.join(upload_folder, image.filename)
#         image.save(image_path)
#         print(f"Saved image to {image_path}")

#     if audio:
#         filename = audio.filename
#         ext = os.path.splitext(filename)[1].lower()

#         save_path = os.path.join(ref_audio_folder, filename)

#         if ext == ".wav":
#             wav_path = os.path.join(ref_audio_folder, filename)
#             audio.save(wav_path)
#             mp3_path = wav_path.replace(".wav", ".mp3")
#             try:
#                 convert_wav_to_mp3(wav_path, mp3_path)
#                 os.remove(wav_path)
#                 audio_path = mp3_path
#                 print(f"Converted and saved reference audio to {mp3_path}")
#             except Exception as e:
#                 return jsonify({"error": f"FFmpeg conversion failed: {e}"}), 500
#         else:
#             audio.save(save_path)
#             audio_path = save_path
#             print(f"Saved reference audio to {audio_path}")

#     elif voice_choice in ['male', 'female']:
#         audio_filename = 'male_default.mp3' if voice_choice == 'male' else 'female_default.mp3'
#         audio_path = os.path.join(default_voices, audio_filename)
#         print(f"Using default reference audio: {audio_path}")
#     else:
#         print("Either upload a reference audio or choose male/female voice.")
#         return jsonify({"error": "Either upload a reference audio or choose male/female voice."}), 400


#     if text:
#         audio_output_path = generate_speech_with_openvoice(text, audio_path)

#         procs_dir = current_app.config['PROCESSED_FOLDER']
#         shutil.rmtree(procs_dir)


#         output_video_path = generate_lipsynced_video(
#         face_path=uploaded_video_path,
#         audio_path=audio_output_path,
#         checkpoint_path="./flask1/files/checkpoints/wav2lip_gan.pth",
#         output_path="./flask1/files/output_final/result_synced.mp4"
#         )

#         return send_file(
#             output_path,
#             mimetype='audio/wav',
#             as_attachment=False
#         )
#     #if not os.path.exists(output_path):
#     #    return jsonify({"error": "Speech file was not created."}), 500

#     return jsonify({
#         "message": "Files received successfully",
#         "text": text,
#         "image_saved": bool(image),
#         "audio_saved": bool(audio)
#     })