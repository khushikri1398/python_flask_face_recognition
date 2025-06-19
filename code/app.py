import os
import uuid
import cv2
import torch
import numpy as np
from flask import Flask, request, render_template, send_from_directory
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        for f in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, f))
        for f in os.listdir(OUTPUT_FOLDER):
            os.remove(os.path.join(OUTPUT_FOLDER, f))

        frame_skip = int(request.form.get("frame_skip", 1))
        drop_initial = 'drop_initial' in request.form

        # Upload or YouTube download
        video_path = None
        youtube_url = request.form.get("youtube_url", "").strip()
        if youtube_url:
            video_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.mp4")
            import subprocess
            try:
                subprocess.run([
                    "yt-dlp", "-f", "mp4", "-o", video_path, youtube_url
                ], check=True)
            except subprocess.CalledProcessError:
                return "YouTube download failed.", 400
        elif 'video' in request.files:
            video_file = request.files['video']
            if video_file.filename:
                video_path = os.path.join(UPLOAD_FOLDER, uuid.uuid4().hex + ".mp4")
                video_file.save(video_path)
        else:
            return "No video provided.", 400

        # Save faces and compute embeddings
        face_embeddings = []
        for face_file in request.files.getlist("faces"):
            if face_file.filename:
                img = Image.open(face_file.stream).convert("RGB")
                face = mtcnn(img)
                if face is not None:
                    emb = resnet(face.unsqueeze(0).to(device)).detach()
                    face_embeddings.append(emb)

        if not face_embeddings:
            return "No valid faces found.", 400

        face_embeddings = torch.cat(face_embeddings)  # shape: (N, 512)

        # Run detection
        results = detect_matching_faces(video_path, face_embeddings, frame_skip, drop_initial)
        return render_template("results.html", results=results)

    return render_template("upload.html")

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

def detect_matching_faces(video_path, face_embeddings, frame_skip, drop_initial):
    cap = cv2.VideoCapture(video_path)
    count, match_count = 0, 0
    results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if drop_initial and count < 10:
            count += 1
            continue
        if count % frame_skip == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            face = mtcnn(img)
            if face is not None:
                emb = resnet(face.unsqueeze(0).to(device)).detach()
                sim = torch.cosine_similarity(emb, face_embeddings).max().item()
                if sim > 0.7:  # threshold for match
                    ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    filename = f"match_{match_count}.jpg"
                    out_path = os.path.join(OUTPUT_FOLDER, filename)
                    cv2.imwrite(out_path, frame)
                    results.append((filename, f"{ts:.2f}s"))
                    match_count += 1
        count += 1

    cap.release()
    return results

if __name__ == '__main__':
    app.run(debug=True)
