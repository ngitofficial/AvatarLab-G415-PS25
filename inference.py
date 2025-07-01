from flask import current_app
from flask1.utils.file_name import generate_mp4_filename
import os
import subprocess
import numpy as np
import cv2
import torch
from tqdm import tqdm
from flask1.lsmodel.audio import load_wav,melspectrogram
from flask1.lsmodel import face_detection
from flask1.lsmodel.models import Wav2Lip

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _load(checkpoint_path):
    return torch.load(checkpoint_path, map_location=device)

def load_model(path):
    print(f"Loading checkpoint from: {path}")
    model = Wav2Lip()
    checkpoint = _load(path)
    s = checkpoint['state_dict']
    new_s = {k.replace('module.', ''): v for k, v in s.items()}
    model.load_state_dict(new_s, strict=False)
    return model.to(device).eval()

def get_smoothened_boxes(boxes, T=5):
    for i in range(len(boxes)):
        window = boxes[i:i+T] if i + T < len(boxes) else boxes[-T:]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def face_detect(images, face_det_batch_size, pads, nosmooth):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=device)
    results = []

    while True:
        predictions = []
        try:
            for i in range(0, len(images), face_det_batch_size):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + face_det_batch_size])))
        except RuntimeError:
            face_det_batch_size = max(1, face_det_batch_size // 2)
            continue
        break

    pady1, pady2, padx1, padx2 = pads
    for rect, image in zip(predictions, images):
        if rect is None:
            raise ValueError("Face not detected.")
        y1, y2 = max(0, rect[1] - pady1), min(image.shape[0], rect[3] + pady2)
        x1, x2 = max(0, rect[0] - padx1), min(image.shape[1], rect[2] + padx2)
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not nosmooth:
        boxes = get_smoothened_boxes(boxes)

    return [[image[y1:y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

def prepare_batch(img_batch, mel_batch, frame_batch, coords_batch, img_size):
    img_batch = np.asarray(img_batch)
    mel_batch = np.asarray(mel_batch)
    img_masked = img_batch.copy()
    img_masked[:, img_size // 2:] = 0
    img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
    mel_batch = np.reshape(mel_batch, (len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1))
    return img_batch, mel_batch, frame_batch, coords_batch

def datagen(frames, mels, static, batch_size, img_size, pads, nosmooth):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    face_det_results = face_detect([frames[0]] if static else frames, batch_size, pads, nosmooth)

    for i, mel in enumerate(mels):
        idx = 0 if static else i % len(frames)
        frame = frames[idx].copy()
        face, coords = face_det_results[idx]
        face = cv2.resize(face, (img_size, img_size))

        img_batch.append(face)
        mel_batch.append(mel)
        frame_batch.append(frame)
        coords_batch.append(coords)

        if len(img_batch) >= batch_size:
            yield prepare_batch(img_batch, mel_batch, frame_batch, coords_batch, img_size)
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if img_batch:
        yield prepare_batch(img_batch, mel_batch, frame_batch, coords_batch, img_size)

def generate_lipsynced_video(
    face_path,
    audio_path,
    checkpoint_path,
    temp_video_path="./flask1/files/temp/result.avi",
    static=False,
    fps=25.,
    pads=[0, 10, 0, 0],
    face_det_batch_size=16,
    wav2lip_batch_size=128,
    resize_factor=1,
    crop=[0, -1, 0, -1],
    rotate=False,
    nosmooth=False
):
    img_size = 96

    video_file_name = generate_mp4_filename()

    final_path = current_app.config['FINAL_OUTPUT_FOLDER']

    output_path = os.path.join(final_path, video_file_name)


    # Load video
    if not os.path.isfile(face_path):
        raise FileNotFoundError(f"Face video not found: {face_path}")
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if face_path.split('.')[-1] in ['jpg', 'png', 'jpeg']:
        static = True
        full_frames = [cv2.imread(face_path)]
    else:
        video_stream = cv2.VideoCapture(face_path)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        full_frames = []
        while True:
            ret, frame = video_stream.read()
            if not ret: break
            if resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1] // resize_factor, frame.shape[0] // resize_factor))
            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            y1, y2, x1, x2 = crop
            y2, x2 = y2 if y2 != -1 else frame.shape[0], x2 if x2 != -1 else frame.shape[1]
            full_frames.append(frame[y1:y2, x1:x2])
        video_stream.release()
    print(f"Number of frames available for inference: {len(full_frames)}")


    wav = load_wav(audio_path, 16000)
    mel = melspectrogram(wav)
    mel_step_size = 16
    mel_chunks = []
    mel_idx_multiplier = 80. / fps
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > mel.shape[1]:
            mel_chunks.append(mel[:, -mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx:start_idx + mel_step_size])
        i += 1


    print(f"Length of mel chunks: {len(mel_chunks)}")

    full_frames = full_frames[:len(mel_chunks)]
    model = load_model(checkpoint_path)

    frame_h, frame_w = full_frames[0].shape[:-1]
    out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

    for img_batch, mel_batch, frames, coords in tqdm(
        datagen(full_frames.copy(), mel_chunks, static, face_det_batch_size, img_size, pads, nosmooth),
        total=int(np.ceil(len(mel_chunks) / float(wav2lip_batch_size)))
    ):
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch).cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            f[y1:y2, x1:x2] = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            out.write(f)

    out.release()

    # 🔶 FINAL OUTPUT PATH USED HERE (you can change it via output_path param)
    command = f'ffmpeg -y -i "{audio_path}" -i "{temp_video_path}" -strict -2 -q:v 1 "{output_path}"'
    subprocess.call(command, shell=True)
    print(f"✅ Final output video saved to: {output_path}")

    return output_path













# import os
# import sys
# import argparse
# import subprocess
# import platform
# import numpy as np
# import cv2
# from tqdm import tqdm
# import torch
# import audio
# import face_detection
# from models import Wav2Lip

# # ------------------------ Argument Parsing ------------------------

# parser = argparse.ArgumentParser()

# parser.add_argument('--checkpoint_path', type=str, required=True)
# parser.add_argument('--face', type=str, required=True)
# parser.add_argument('--audio', type=str, required=True)
# parser.add_argument('--outfile', type=str, default='results/result_voice.mp4')

# parser.add_argument('--static', type=bool, default=False)
# parser.add_argument('--fps', type=float, default=25.)

# parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0])
# parser.add_argument('--face_det_batch_size', type=int, default=16)
# parser.add_argument('--wav2lip_batch_size', type=int, default=128)

# parser.add_argument('--resize_factor', type=int, default=1)
# parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1])
# parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1])
# parser.add_argument('--rotate', default=False, action='store_true')
# parser.add_argument('--nosmooth', default=False, action='store_true')

# args = parser.parse_args()
# args.img_size = 96

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"Using {device} for inference.")

# # ------------------------ Helpers ------------------------

# def _load(checkpoint_path):
#     return torch.load(checkpoint_path, map_location=device)

# def load_model(path):
#     print(f"Loading checkpoint from: {path}")
#     model = Wav2Lip()
#     checkpoint = _load(path)
#     s = checkpoint['state_dict']
#     new_s = {k.replace('module.', ''): v for k, v in s.items()}
#     model.load_state_dict(new_s)
#     return model.to(device).eval()

# def get_smoothened_boxes(boxes, T=5):
#     for i in range(len(boxes)):
#         window = boxes[i:i+T] if i + T < len(boxes) else boxes[-T:]
#         boxes[i] = np.mean(window, axis=0)
#     return boxes

# def face_detect(images):
#     detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=device)
#     batch_size = args.face_det_batch_size
#     results = []

#     while True:
#         predictions = []
#         try:
#             for i in tqdm(range(0, len(images), batch_size)):
#                 predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
#         except RuntimeError:
#             batch_size = max(1, batch_size // 2)
#             print(f"Reduced face detection batch size to {batch_size} due to OOM")
#             continue
#         break

#     pady1, pady2, padx1, padx2 = args.pads
#     for rect, image in zip(predictions, images):
#         if rect is None:
#             cv2.imwrite('temp/faulty_frame.jpg', image)
#             raise ValueError('Face not detected. Ensure all frames contain a visible face.')
#         y1, y2 = max(0, rect[1] - pady1), min(image.shape[0], rect[3] + pady2)
#         x1, x2 = max(0, rect[0] - padx1), min(image.shape[1], rect[2] + padx2)
#         results.append([x1, y1, x2, y2])

#     boxes = np.array(results)
#     if not args.nosmooth: boxes = get_smoothened_boxes(boxes)
#     return [[image[y1:y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

# def datagen(frames, mels):
#     img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

#     face_det_results = face_detect([frames[0]]) if args.static else face_detect(frames)

#     for i, mel in enumerate(mels):
#         idx = 0 if args.static else i % len(frames)
#         frame = frames[idx].copy()
#         face, coords = face_det_results[idx]

#         face = cv2.resize(face, (args.img_size, args.img_size))
#         img_batch.append(face)
#         mel_batch.append(mel)
#         frame_batch.append(frame)
#         coords_batch.append(coords)

#         if len(img_batch) >= args.wav2lip_batch_size:
#             yield prepare_batch(img_batch, mel_batch, frame_batch, coords_batch)
#             img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

#     if img_batch:
#         yield prepare_batch(img_batch, mel_batch, frame_batch, coords_batch)

# def prepare_batch(img_batch, mel_batch, frame_batch, coords_batch):
#     img_batch = np.asarray(img_batch)
#     mel_batch = np.asarray(mel_batch)
#     img_masked = img_batch.copy()
#     img_masked[:, args.img_size // 2:] = 0
#     img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
#     mel_batch = np.reshape(mel_batch, (len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1))
#     return img_batch, mel_batch, frame_batch, coords_batch

# # ------------------------ Main ------------------------

# def main():
#     if not os.path.isfile(args.face): raise FileNotFoundError(f"Face video not found: {args.face}")
#     if not os.path.isfile(args.audio): raise FileNotFoundError(f"Audio file not found: {args.audio}")

#     # Load video
#     if args.face.split('.')[-1] in ['jpg', 'png', 'jpeg']:
#         args.static = True
#         full_frames = [cv2.imread(args.face)]
#         fps = args.fps
#     else:
#         video_stream = cv2.VideoCapture(args.face)
#         fps = video_stream.get(cv2.CAP_PROP_FPS)
#         full_frames = []
#         print("Reading video frames...")
#         while True:
#             ret, frame = video_stream.read()
#             if not ret: break
#             if args.resize_factor > 1:
#                 frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))
#             if args.rotate:
#                 frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
#             y1, y2, x1, x2 = args.crop
#             y2, x2 = y2 if y2 != -1 else frame.shape[0], x2 if x2 != -1 else frame.shape[1]
#             full_frames.append(frame[y1:y2, x1:x2])
#         video_stream.release()

#     print(f"Number of frames available for inference: {len(full_frames)}")

#     # Load audio
#     if not args.audio.endswith('.wav'):
#         print("Extracting raw audio...")
#         args.audio = 'temp/temp.wav'
#         subprocess.call(f'ffmpeg -y -i "{args.audio}" -strict -2 {args.audio}', shell=True)

#     wav = audio.load_wav(args.audio, 16000)
#     mel = audio.melspectrogram(wav)

#     mel_step_size = 16
#     mel_chunks = []
#     mel_idx_multiplier = 80. / fps
#     i = 0
#     while True:
#         start_idx = int(i * mel_idx_multiplier)
#         if start_idx + mel_step_size > mel.shape[1]:
#             mel_chunks.append(mel[:, -mel_step_size:])
#             break
#         mel_chunks.append(mel[:, start_idx:start_idx + mel_step_size])
#         i += 1

#     print(f"Length of mel chunks: {len(mel_chunks)}")
#     full_frames = full_frames[:len(mel_chunks)]
#     model = load_model(args.checkpoint_path)

#     frame_h, frame_w = full_frames[0].shape[:-1]
#     out = cv2.VideoWriter('./Wav2Lip/temp/result.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

#     for img_batch, mel_batch, frames, coords in tqdm(datagen(full_frames.copy(), mel_chunks),
#                                                      total=int(np.ceil(len(mel_chunks) / args.wav2lip_batch_size))):
#         img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
#         mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

#         with torch.no_grad():
#             pred = model(mel_batch, img_batch).cpu().numpy().transpose(0, 2, 3, 1) * 255.

#         for p, f, c in zip(pred, frames, coords):
#             y1, y2, x1, x2 = c
#             f[y1:y2, x1:x2] = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
#             out.write(f)

#     out.release()

#     # Merge audio + video
#     command = f'ffmpeg -y -i "{args.audio}" -i ./Wav2Lip/temp/result.avi -strict -2 -q:v 1 "{args.outfile}"'
#     subprocess.call(command, shell=True)
#     print(f"\n✅ Output saved to: {args.outfile}")

# if __name__ == '__main__':
#     main()
