# Face Detection & Timestamp Matching Web App

This Flask web application allows you to:
1. Upload a video or provide a YouTube URL.
2. Automatically extract unique faces using MTCNN + FaceNet.
3. Select the faces you're interested in.
4. Find the timestamps in the video where the selected faces appear.

---

## 🚀 Features

- Upload video or paste a YouTube URL
- Frame skipping for faster processing
- Unique face extraction with MTCNN + FaceNet
- Face matching using cosine similarity
- Match timestamps with screenshots
- Built-in image previews and TailwindCSS styling

---

## 🧰 Requirements

Install all dependencies:

```bash
pip install -r requirements.txt
pip install tensorflow==2.10.*
and get this https://github.com/BtbN/FFmpeg-Builds/releases (https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1.zip) add to your system env path bin path.
pip install -U yt-dlp



file Structure:
project/
│
├── app.py
├── requirements.txt
├── README.md
├── static/
│   ├── uploads/
│   ├── faces/
│   └── matches/
└── templates/
    ├── index.html
    ├── select_faces.html
    └── results.html
