import os
import uuid
import cv2
import numpy as np
from flask import Flask, request, render_template, send_from_directory
from PIL import Image
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
import subprocess

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

detector = MTCNN()
embedder = FaceNet()

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        for f in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, f))
        for f in os.listdir(OUTPUT_FOLDER):
            os.remove(os.path.join(OUTPUT_FOLDER, f))

        frame_skip = int(request.form.get("frame_skip", 1))
        drop_initial = 'drop_initial' in request.form

        # Download from YouTube or use uploaded video
        video_path = None
        youtube_url = request.form.get("youtube_url", "").strip()
        if youtube_url:
            video_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.mp4")
            try:
                subprocess.run(["yt-dlp","-f","mp4","-o",video_path,youtube_url], check=True)
            except subprocess.CalledProcessError:
                return "Video download failed.", 400
        elif 'video' in request.files:
            video = request.files['video']
            if video.filename:
                video_path = os.path.join(UPLOAD_FOLDER, uuid.uuid4().hex + ".mp4")
                video.save(video_path)
        else:
            return "Provide a video or YouTube link.", 400

        # Get embeddings from uploaded face images
        known_embs = []
        for file in request.files.getlist("faces"):
            if file.filename:
                img = Image.open(file.stream).convert("RGB")
                img_np = np.array(img)
                dets = detector.detect_faces(img_np)
                for d in dets:
                    x,y,w,h = d['box']
                    crop = img_np[y:y+h, x:x+w]
                    if crop.size == 0:
                        continue
                    emb = embedder.embeddings([crop])[0]
                    known_embs.append(emb)
        if not known_embs:
            return "No valid faces uploaded.", 400
        known_embs = np.array(known_embs)

        results = detect_matching_faces(video_path, known_embs, frame_skip, drop_initial)
        return render_template("results.html", results=results)

    return render_template("upload.html")

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

def detect_matching_faces(video_path, known_embs, frame_skip, drop_initial):
    cap = cv2.VideoCapture(video_path)
    count = 0
    match_id = 0
    results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if drop_initial and count < 10:
            count += 1
            continue
        if count % frame_skip == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets = detector.detect_faces(img)
            for d in dets:
                x,y,w,h = d['box']
                crop = img[y:y+h, x:x+w]
                if crop.size == 0:
                    continue
                emb = embedder.embeddings([crop])[0]
                # Cosine similarity (dot product normalized)
                sims = np.dot(known_embs, emb) / (np.linalg.norm(known_embs, axis=1) * np.linalg.norm(emb))
                if np.max(sims) > 0.7:
                    ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    fname = f"match_{match_id}.jpg"
                    cv2.imwrite(os.path.join(OUTPUT_FOLDER, fname), frame)
                    results.append((fname, f"{ts:.2f}s"))
                    match_id += 1
                    break
        count += 1

    cap.release()
    return results

if __name__ == '__main__':
    app.run(debug=True)
