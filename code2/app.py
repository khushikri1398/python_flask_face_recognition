import os, cv2, uuid, subprocess
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
FACES_FOLDER = 'static/faces'
MATCHES_FOLDER = 'static/matches'

for folder in [UPLOAD_FOLDER, FACES_FOLDER, MATCHES_FOLDER]:
    os.makedirs(folder, exist_ok=True)

detector = MTCNN()
embedder = FaceNet()

def extract_faces(video_path, frame_skip=5):
    cap = cv2.VideoCapture(video_path)
    seen_embs = []
    face_id = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(rgb)

        for d in detections:
            x, y, w, h = d['box']
            x, y = max(0, x), max(0, y)
            w, h = max(1, w), max(1, h)

            face = rgb[y:y+h, x:x+w]
            if face.size == 0:
                continue

            face_resized = cv2.resize(face, (160, 160))
            emb = embedder.embeddings([face_resized])[0]
            emb = emb / np.linalg.norm(emb)

            is_new = all(np.dot(emb, seen) < 0.7 for seen in seen_embs)

            if is_new:
                fname = f"{face_id}.jpg"
                cv2.imwrite(os.path.join(FACES_FOLDER, fname), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                seen_embs.append(emb)
                face_id += 1

        frame_count += 1

    cap.release()

def find_face_matches(video_path, selected_faces, frame_skip=5):
    cap = cv2.VideoCapture(video_path)
    known_embs = []

    for fname in selected_faces:
        path = os.path.join(FACES_FOLDER, fname)
        img = np.array(Image.open(path).convert("RGB"))
        dets = detector.detect_faces(img)
        for d in dets:
            x, y, w, h = d['box']
            x, y = max(0, x), max(0, y)
            w, h = max(1, w), max(1, h)
            crop = img[y:y+h, x:x+w]
            if crop.size == 0:
                continue
            crop_resized = cv2.resize(crop, (160, 160))
            emb = embedder.embeddings([crop_resized])[0]
            emb = emb / np.linalg.norm(emb)
            known_embs.append(emb)

    matches = []
    frame_id = 0
    match_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % frame_skip != 0:
            frame_id += 1
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = detector.detect_faces(rgb)
        for d in dets:
            x, y, w, h = d['box']
            x, y = max(0, x), max(0, y)
            w, h = max(1, w), max(1, h)
            face = rgb[y:y+h, x:x+w]
            if face.size == 0:
                continue
            face_resized = cv2.resize(face, (160, 160))
            emb = embedder.embeddings([face_resized])[0]
            emb = emb / np.linalg.norm(emb)
            sims = np.dot(known_embs, emb)
            if np.max(sims) > 0.7:
                ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                out_name = f"match_{match_id}.jpg"
                cv2.imwrite(os.path.join(MATCHES_FOLDER, out_name), frame)
                matches.append((out_name, f"{ts:.2f}s"))
                match_id += 1
                break
        frame_id += 1
    cap.release()
    return matches

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        for f in os.listdir(FACES_FOLDER): os.remove(os.path.join(FACES_FOLDER, f))
        for f in os.listdir(MATCHES_FOLDER): os.remove(os.path.join(MATCHES_FOLDER, f))

        url = request.form.get('youtube_url')
        video = request.files.get('video')
        frame_skip = int(request.form.get("frame_skip") or 5)

        video_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.mp4")
        if url:
            subprocess.run(["yt-dlp", "-f", "mp4", "-o", video_path, url])
        elif video and video.filename:
            video.save(video_path)
        else:
            return "No video provided", 400

        extract_faces(video_path, frame_skip)
        return redirect(url_for('select_faces', video=os.path.basename(video_path), frame_skip=frame_skip))

    return render_template("index.html")

@app.route('/select/<video>', methods=['GET', 'POST'])
def select_faces(video):
    frame_skip = int(request.args.get("frame_skip", 5))
    if request.method == 'POST':
        selected = request.form.getlist("faces")
        matches = find_face_matches(os.path.join(UPLOAD_FOLDER, video), selected, frame_skip)
        return render_template("results.html", results=matches)
    
    faces = sorted(os.listdir(FACES_FOLDER), key=lambda x: int(x.split('.')[0]))
    return render_template("select_faces.html", faces=faces)

if __name__ == '__main__':
    app.run(debug=True)