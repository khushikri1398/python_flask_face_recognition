import os
import uuid
import json
import cv2
import numpy as np
import face_recognition
from flask import Flask, request, render_template
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx, TextClip, CompositeVideoClip, AudioFileClip

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
CLIPS_FOLDER = 'static/clips'
MUSIC_FOLDER = 'static/music'

for folder in [UPLOAD_FOLDER, CLIPS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Visual effects and parameters
effectParams = {
    "fadein": ["duration"],
    "fadeout": ["duration"],
    "mirror_x": [],
    "mirror_y": [],
    "blackwhite": [],
    "invert_colors": [],
    "resize": ["height"],
    "speedx": ["factor"],
    "rotate": ["angle"]
}

# Face-detect and preview clips as .GIF
from face_recognition import face_locations, face_encodings, compare_faces, face_distance

def split_video_to_gif_clips(video_path, duration=1.5, cooldown=2.0, similarity_threshold=0.6):
    clip = VideoFileClip(video_path)
    clips = []
    idx = 0

    frame_interval = 1.0 / clip.fps
    frame_count = int(clip.duration * clip.fps)
    last_detection_time = -cooldown
    saved_encodings = []

    for i in range(frame_count):
        timestamp = i * frame_interval

        if timestamp - last_detection_time < cooldown:
            continue

        try:
            frame = clip.get_frame(timestamp)

            if frame.dtype == np.float32 or frame.dtype == np.float64:
                frame = (frame * 255).astype(np.uint8)
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

            faces = face_locations(frame, model="hog")
            encodings = face_encodings(frame, known_face_locations=faces)

            if encodings:
                new_encoding = encodings[0]

                # Compare with saved encodings
                is_duplicate = False
                for known in saved_encodings:
                    distance = face_distance([known], new_encoding)[0]
                    if distance < similarity_threshold:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    t_start = max(0, timestamp - 0.25)
                    t_end = min(clip.duration, timestamp + 0.25)

                    if t_end - t_start >= 0.5:
                        subclip = clip.subclip(t_start, t_end).resize(height=180)
                        gif_name = f"preview_{idx}_{uuid.uuid4().hex}.gif"
                        gif_path = os.path.join(CLIPS_FOLDER, gif_name)
                        subclip.write_gif(gif_path, fps=10)
                        clips.append((gif_name, video_path, round(t_start, 2)))
                        print(f"[Face match ✅] Unique clip at {timestamp:.2f}s")
                        saved_encodings.append(new_encoding)
                        idx += 1
                        last_detection_time = timestamp
                else:
                    print(f"[Duplicate ❌] Similar face at {timestamp:.2f}s")
        except Exception as e:
            print(f"[Frame error @ {timestamp:.2f}s]: {e}")

    print(f"✅ Total unique face clips: {len(clips)}")
    return clips




def create_final_video(matches, transition, effect, params, with_audio, music_file=None):
    clips = []
    for video_path, ts in matches:
        try:
            base_clip = VideoFileClip(video_path)
            start = ts
            end = min(ts + 1.5, base_clip.duration)
            clip = base_clip.subclip(start, end).set_fps(base_clip.fps)

            if not with_audio:
                clip = clip.without_audio()

            if effect in effectParams:
                try:
                    if effect == "fadein":
                        clip = clip.fadein(float(params.get("duration", 0.5)))
                    elif effect == "fadeout":
                        clip = clip.fadeout(float(params.get("duration", 0.5)))
                    elif effect == "mirror_x":
                        clip = clip.fx(vfx.mirror_x)
                    elif effect == "mirror_y":
                        clip = clip.fx(vfx.mirror_y)
                    elif effect == "blackwhite":
                        clip = clip.fx(vfx.blackwhite)
                    elif effect == "invert_colors":
                        clip = clip.fx(vfx.invert_colors)
                    elif effect == "resize":
                        height = int(params.get("height", 360))
                        clip = clip.resize(height=height)
                    elif effect == "speedx":
                        factor = float(params.get("factor", 1.0))
                        clip = clip.fx(vfx.speedx, factor)
                    elif effect == "rotate":
                        angle = float(params.get("angle", 0))
                        clip = clip.rotate(angle)
                except Exception as e:
                    print(f"[Effect error] {effect} failed: {e}")

            # Timestamp label in large red text with white background
            timestamp_text = f"{ts:.2f}s"
            text_clip = TextClip(timestamp_text, fontsize=70, color='red', font='Arial-Bold')
            text_clip = text_clip.on_color(size=(text_clip.w + 20, text_clip.h + 10), color=(255, 255, 255), col_opacity=1)
            text_clip = text_clip.set_position(("right", "bottom")).set_duration(clip.duration)
            clip = CompositeVideoClip([clip, text_clip])

            if clips and transition == 'fade':
                clip = clip.crossfadein(0.5)

            clips.append(clip)
        except Exception as e:
            print(f"[Clip error] {e}")

    if not clips:
        return None

    final = concatenate_videoclips(clips, method="compose", padding=-0.5 if transition == "fade" else 0)

    # Add music if selected
    if music_file:
        music_path = os.path.join(MUSIC_FOLDER, music_file + ".mp3")
        if os.path.exists(music_path):
            bg_music = AudioFileClip(music_path).subclip(0, final.duration)
            final = final.set_audio(bg_music)

    final_name = f"final_{uuid.uuid4().hex}.mp4"
    final_path = os.path.join(CLIPS_FOLDER, final_name)

    try:
        final.write_videofile(
            final_path,
            codec="libx264",
            fps=final.fps or 30,
            audio_codec="aac",
            bitrate="5000k",
            threads=4,
            preset="ultrafast"
        )
    except Exception as e:
        print(f"[Write error] Video generation failed: {e}")
        return None

    return final_name


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        for f in os.listdir(CLIPS_FOLDER):
            os.remove(os.path.join(CLIPS_FOLDER, f))

        videos = request.files.getlist('videos')
        video_paths = []

        for video in videos:
            if video.filename:
                video_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}_{video.filename}")
                video.save(video_path)
                video_paths.append(video_path)

        previews = []
        for path in video_paths:
            previews.extend(split_video_to_gif_clips(path))

        # Predefined music list
        music_files = [
            "Arabesque", "carnivalrides", "DarkWinds", "FoxieEpic", "Gran Batalla", "Heroic Demise",
            "magical_theme", "No More Magic", "prologue", "radakan - old crypt", "Soliloquy",
            "The Dark Amulet", "the_kings_crowning_v1", "TheLoomingBattle"
        ]

        return render_template("select_clips.html", previews=previews,
                               effectParams=effectParams,
                               matches_json=json.dumps(previews),
                               music_files=music_files)

    return render_template("index.html")


@app.route('/finalize_video', methods=['POST'])
def finalize_video():
    selected_ids = request.form.getlist("selected_clips")
    all_matches = json.loads(request.form.get("matches_json"))
    selected_matches = [
        (m[1], float(m[2])) for idx, m in enumerate(all_matches) if str(idx) in selected_ids
    ]

    transition = request.form.get("transition", "none")
    effect = request.form.get("effect", "none")
    with_audio = request.form.get("with_audio") == "yes"
    music_file = request.form.get("music_file")
    effect_params = {key.split("param_")[1]: request.form[key] for key in request.form if key.startswith("param_")}

    final_name = create_final_video(selected_matches, transition, effect, effect_params, with_audio, music_file)
    if not final_name:
        return render_template("results.html", results=[], error="Final video generation failed.")

    return render_template("results.html", results=[(final_name, '')], error=None)


if __name__ == '__main__':
    app.run(debug=True)
