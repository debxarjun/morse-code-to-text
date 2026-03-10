"""
Morse Code Eye Blink Decoder
============================
Uses webcam + OpenCV + dlib to detect eye blinks.
Short blink  (<300ms) → DOT  (.)
Long blink   (>300ms) → DASH (-)
Pause ~1.5s            → Letter separator
Pause ~3s              → Word separator
"""

import cv2
import dlib
import numpy as np
import time
import threading
from flask import Flask, render_template, Response, jsonify
from scipy.spatial import distance as dist
from collections import deque

app = Flask(__name__)

# ─── Morse Code Dictionary ───────────────────────────────────────────────────
MORSE_CODE = {
    '.-': 'A',   '-...': 'B', '-.-.': 'C', '-..': 'D',  '.': 'E',
    '..-.': 'F', '--.': 'G',  '....': 'H', '..': 'I',   '.---': 'J',
    '-.-': 'K',  '.-..': 'L', '--': 'M',   '-.': 'N',   '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R',  '...': 'S',  '-': 'T',
    '..-': 'U',  '...-': 'V', '.--': 'W',  '-..-': 'X', '-.--': 'Y',
    '--..': 'Z', '-----': '0','----': '1', '..---': '2','...--': '3',
    '....-': '4','.....' : '5','-....': '6','--...': '7','---..': '8',
    '----.': '9'
}

# ─── EAR (Eye Aspect Ratio) ───────────────────────────────────────────────────
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# ─── Global State ─────────────────────────────────────────────────────────────
state = {
    "morse_buffer": [],        # Current letter's dots/dashes
    "current_letter": "",      # Letter being built
    "decoded_word": "",        # Full decoded text so far
    "status": "OPEN",          # OPEN / BLINK
    "blink_start": None,
    "last_open_time": None,
    "ear": 1.0,
    "message": "Look at camera and blink to start!",
    "letter_finalized": "",
}

EAR_THRESHOLD = 0.25
SHORT_BLINK_MAX = 1.0    # seconds → DOT
LONG_BLINK_MIN  = 1.0   # seconds → DASH
LETTER_PAUSE    = 2.0     # seconds → end of letter
WORD_PAUSE      = 3.0     # seconds → space between words

letter_timer = None
word_timer   = None

def finalize_letter():
    """Convert buffered morse symbols to a letter."""
    global letter_timer, word_timer
    code = "".join(state["morse_buffer"])
    if code:
        ch = MORSE_CODE.get(code, "?")
        state["decoded_word"] += ch
        state["letter_finalized"] = ch
        state["morse_buffer"] = []
        state["current_letter"] = ""
        state["message"] = f"Letter added: '{ch}'"

def finalize_word():
    """Add a space after a word pause."""
    if state["decoded_word"] and not state["decoded_word"].endswith(" "):
        state["decoded_word"] += " "
        state["message"] = "Word space added"

def reset_timers():
    global letter_timer, word_timer
    if letter_timer:
        letter_timer.cancel()
    if word_timer:
        word_timer.cancel()

def start_pause_timers():
    global letter_timer, word_timer
    reset_timers()
    letter_timer = threading.Timer(LETTER_PAUSE, finalize_letter)
    word_timer   = threading.Timer(WORD_PAUSE,   finalize_word)
    letter_timer.start()
    word_timer.start()

# ─── Video Stream Generator ───────────────────────────────────────────────────
def generate_frames():
    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    LEFT_EYE  = list(range(42, 48))
    RIGHT_EYE = list(range(36, 42))

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        current_time = time.time()

        for face in faces:
            shape = predictor(gray, face)
            coords = np.array([[shape.part(i).x, shape.part(i).y]
                                for i in range(68)])

            left_eye  = coords[LEFT_EYE]
            right_eye = coords[RIGHT_EYE]
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
            state["ear"] = round(ear, 3)

            # Draw eye contours
            cv2.polylines(frame, [left_eye],  True, (0, 255, 180), 1)
            cv2.polylines(frame, [right_eye], True, (0, 255, 180), 1)

            # ── Blink DETECTED ──────────────────────────────────────────────
            if ear < EAR_THRESHOLD:
                if state["status"] == "OPEN":
                    state["status"]      = "BLINK"
                    state["blink_start"] = current_time
                    reset_timers()

            # ── Eyes OPENED again ───────────────────────────────────────────
            else:
                if state["status"] == "BLINK" and state["blink_start"]:
                    duration = current_time - state["blink_start"]
                    symbol = "." if duration < SHORT_BLINK_MAX else "-"
                    state["morse_buffer"].append(symbol)
                    state["current_letter"] = "".join(state["morse_buffer"])
                    state["message"] = f"{'DOT' if symbol=='.' else 'DASH'} detected ({duration:.2f}s)"
                    state["last_open_time"] = current_time
                    start_pause_timers()

                state["status"]      = "OPEN"
                state["blink_start"] = None

            # Overlay info
            color = (0, 80, 255) if state["status"] == "BLINK" else (0, 255, 100)
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"STATUS: {state['status']}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Morse: {''.join(state['morse_buffer'])}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 0), 2)

        if not faces:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Encode frame
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

    cap.release()

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/state")
def get_state():
    return jsonify({
        "morse_buffer":   state["morse_buffer"],
        "current_letter": state["current_letter"],
        "decoded_word":   state["decoded_word"],
        "status":         state["status"],
        "ear":            state["ear"],
        "message":        state["message"],
        "letter_finalized": state["letter_finalized"],
    })

@app.route("/clear", methods=["POST"])
def clear():
    reset_timers()
    state["morse_buffer"]    = []
    state["current_letter"]  = ""
    state["decoded_word"]    = ""
    state["message"]         = "Cleared! Ready."
    state["letter_finalized"] = ""
    return jsonify({"ok": True})

@app.route("/backspace", methods=["POST"])
def backspace():
    if state["decoded_word"]:
        state["decoded_word"] = state["decoded_word"][:-1]
    return jsonify({"ok": True})

if __name__ == "__main__":
    print("=" * 50)
    print("  Morse Code Eye Blink Decoder")
    print("  Open http://127.0.0.1:5000 in your browser")
    print("=" * 50)
    app.run(debug=False, threaded=True)
