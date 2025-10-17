#!/usr/bin/env python3
"""
run_model_hailo.py
Fully integrated Fusion application (Face + Pose via Hailo) + Speech (STT + TTS).
- Keeps ALL original features (modes, controls, Hailo inference, Excel responses, keyword detection).
- Adds robust synchronization so STT/TTS won't block or produce burst outputs.
- Fusion weights: speech 0.4, face 0.3, pose 0.3.
- Uses a TTS queue worker to avoid blocking and overlap.
- STT uses a short timeout to avoid stuck callbacks.
"""

import os
import time
import logging
import threading
import random
import re
from queue import Queue
from typing import Dict, Tuple, List

import cv2
import numpy as np
import pandas as pd
import joblib
import speech_recognition as sr
import pyttsx3
import concurrent.futures

# Hailo imports (keep as in your environment)
from hailo_platform import (
    HEF, VDevice, InferVStreams,
    InputVStreamParams, OutputVStreamParams,
    FormatType, Device
)

# Optional transforms used if you keep preprocessing helpers (keep to avoid removing features)
from PIL import Image
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# -----------------------------
# CONFIG (same features preserved)
# -----------------------------
HEF_FACE = "models/hef/face-expression/face-expression.hef"
HEF_POSE = "models/hef/pose-recognition/pose-recognition.hef"
LABELS = ["TIDAK KRISIS", "KRISIS"]

# Fusion weights requested
W_SPEECH = 0.4
W_FACE = 0.3
W_POSE = 0.3
FUSION_THRESHOLD = 0.5

COLORS = {"TIDAK KRISIS": (0, 200, 0), "KRISIS": (0, 0, 255)}

# Input sizes (preserve pipelines)
FACE_INPUT_SIZE = 48
POSE_INPUT_SIZE = 224

# -----------------------------
# Preprocessing transforms (preserve)
# -----------------------------
class To3Channels:
    def __call__(self, x):
        return x.repeat(3, 1, 1) if x.shape[0] == 1 else x

face_tf = transforms.Compose([
    transforms.Resize((FACE_INPUT_SIZE, FACE_INPUT_SIZE)),
    transforms.ToTensor(),
    To3Channels(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

pose_tf = transforms.Compose([
    transforms.Resize((POSE_INPUT_SIZE, POSE_INPUT_SIZE)),
    transforms.ToTensor(),
    To3Channels(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -----------------------------
# Hailo pipeline helpers (preserve behavior)
# -----------------------------
class HailoPipeline:
    def __init__(self, net_group, in_vparams, out_vparams, in_infos, out_infos):
        self.net_group = net_group
        self.input_vstreams_params = in_vparams
        self.output_vstreams_params = out_vparams
        self.input_infos = in_infos
        self.output_infos = out_infos
        self.input_names = [i.name for i in in_infos]
        self.output_names = [o.name for o in out_infos]

def init_hailo(hef_paths: List[str]) -> Tuple[VDevice, Dict[str, HailoPipeline]]:
    logging.info("Memeriksa perangkat Hailo...")
    Device.scan()
    vdevice = VDevice()
    pipelines = {}
    for hef_path in hef_paths:
        if not os.path.exists(hef_path):
            raise FileNotFoundError(f"HEF tidak ditemukan: {hef_path}")
        hef = HEF(hef_path)
        net_groups = vdevice.configure(hef)
        net_group = net_groups[0]

        in_vparams = InputVStreamParams.make_from_network_group(net_group, quantized=False, format_type=FormatType.FLOAT32)
        out_vparams = OutputVStreamParams.make_from_network_group(net_group, quantized=False, format_type=FormatType.FLOAT32)

        if not isinstance(in_vparams, dict):
            in_vparams = {p.name: p for p in in_vparams}
        if not isinstance(out_vparams, dict):
            out_vparams = {p.name: p for p in out_vparams}

        pipelines[hef_path] = HailoPipeline(
            net_group, in_vparams, out_vparams,
            net_group.get_input_vstream_infos(),
            net_group.get_output_vstream_infos()
        )
        logging.info(f"Pipeline siap: {os.path.basename(hef_path)}")
    return vdevice, pipelines

def hailo_infer(pipeline: HailoPipeline, inp_array: np.ndarray) -> np.ndarray:
    # Keep using InferVStreams context as in original code
    in_name = pipeline.input_names[0]
    inp = np.ascontiguousarray(inp_array.astype(np.float32))
    with InferVStreams(pipeline.net_group, pipeline.input_vstreams_params, pipeline.output_vstreams_params) as infer:
        with pipeline.net_group.activate():
            out = infer.infer({in_name: inp})
    out_name = pipeline.output_names[0]
    return np.array(out[out_name]).reshape(-1)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

# -----------------------------
# Speech (preserve original model + keywords + Excel responses)
# -----------------------------
vectorizer = joblib.load("models/language/vectorizer.pkl")
speech_model = joblib.load("models/language/lgbm_model.pkl")
with open("models/language/threshold.txt", "r") as f:
    speech_threshold = float(f.read().strip())

df_krisis = pd.read_excel("data/csv/Data_tanggapan_positif.xlsx", sheet_name="Krisis")
df_tidak = pd.read_excel("data/csv/Data_tanggapan_positif.xlsx", sheet_name="Tidak Krisis")

CRISIS_KEYWORDS = [k.strip().lower() for k in [
    "bunuh diri","saya mau mati","saya mati","bunuh","ingin mati","ingin bunuh diri",
    "tidak ingin hidup","sudah tidak kuat","sudah tidak sanggup","mati saja","selesai saja",
    "akhiri hidup","putus asa","menyakiti diri","mengakhiri hidup","sudah tidak ada harapan",
    "sudah ingin mati","capek hidup","gw mau mati","gue mau mati","pengen mati","pgn mati",
    "pingin mati","udah ga kuat","gak kuat lagi","gk kuat","ga kuat","cape hidup","capee hidup",
    "udah cape","sudah capek","gak sanggup lagi","udah nyerah","nyerah aja","hidup gak ada artinya",
    "hidup gak guna","hidup sia sia","aku pengen hilang","ingin hilang","pengen ngilang",
    "ingin pergi selamanya","mending mati","lebih baik mati","biar aku mati aja","ingin tidur selamanya",
    "ingin berhenti hidup","lukai diri","melukai diri","sayat","nyakitin diri","self harm",
    "aku menyakiti diri","aku pengen nyakitin diri","pengen sayat","pengen nyakitin badan",
    "aku gak berharga","aku gagal","semuanya percuma","hidup ini sia sia","aku menyerah",
    "aku nyerah","udah gak ada harapan","gak ada gunanya hidup"
]]

def preprocess_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r'[^0-9a-z\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def classify_text(text: str) -> np.ndarray:
    s_proc = preprocess_text(text)
    if any(k in s_proc for k in CRISIS_KEYWORDS):
        return np.array([0.0, 1.0], dtype=np.float32)
    if len(s_proc.split()) <= 2:
        return np.array([0.5, 0.5], dtype=np.float32)
    v = vectorizer.transform([s_proc])
    prob = float(speech_model.predict_proba(v)[0, 1])
    return np.array([1 - prob, prob], dtype=np.float32)

# -----------------------------
# TTS: single engine + queue worker (preserve offline pyttsx3)
# -----------------------------
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 165)
tts_engine.setProperty("volume", 1.0)
# Try to set Indonesian voice if available (preserve behavior)
for v in tts_engine.getProperty("voices"):
    if "Indonesian" in v.name or "Andika" in v.name:
        tts_engine.setProperty("voice", v.id)
        break

tts_queue: Queue = Queue()
tts_lock = threading.Lock()

def tts_worker():
    # Worker consumes texts and synthesizes sequentially to avoid overlap
    while True:
        text = tts_queue.get()
        if text is None:
            tts_queue.task_done()
            break
        try:
            with tts_lock:
                tts_engine.say(text)
                tts_engine.runAndWait()
        except Exception as e:
            logging.warning("TTS error: %s", e)
        finally:
            tts_queue.task_done()

# Start TTS worker thread (daemon)
threading.Thread(target=tts_worker, daemon=True).start()

def speak_text_async(text: str):
    # Put text into queue (non-blocking)
    tts_queue.put(text)

def random_response(is_crisis: bool) -> str:
    if is_crisis:
        return random.choice(df_krisis["Respon"].dropna().tolist())
    else:
        return random.choice(df_tidak["Respon"].dropna().tolist())

# -----------------------------
# Speech global state (preserve)
# -----------------------------
speech_probs = np.array([0.5, 0.5], dtype=np.float32)
speech_lock = threading.Lock()
speech_busy = threading.Event()
last_speech_time = time.time()

# We'll use a ThreadPool in callback to add timeout safety
stt_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

def _transcribe_google(recognizer, audio):
    # separate function to submit to executor
    return recognizer.recognize_google(audio, language="id-ID").strip()

def speech_callback(recognizer, audio):
    """
    Robust speech callback:
    - Uses a busy flag to avoid overlapping STT tasks.
    - Runs recognize_google in a ThreadPool with timeout to avoid blocking forever.
    - On success, updates speech_probs and pushes a random response into TTS queue.
    - On failure/timeouts, resets speech_probs to neutral to avoid stuck fusion.
    """
    global speech_probs, last_speech_time
    if speech_busy.is_set():
        return
    speech_busy.set()

    try:
        future = stt_executor.submit(_transcribe_google, recognizer, audio)
        try:
            text = future.result(timeout=5)  # 5s timeout to avoid stuck
        except concurrent.futures.TimeoutError:
            logging.warning("STT timeout (recognize_google took too long)")
            future.cancel()
            with speech_lock:
                speech_probs = np.array([0.5, 0.5], dtype=np.float32)
            return
        if not text:
            with speech_lock:
                speech_probs = np.array([0.5, 0.5], dtype=np.float32)
            return

        # Preserve original behavior: print transcribed text
        print(f"\n🗣️ Anda berkata: {text}")

        # classify text (same model & keywords as original)
        p = classify_text(text)
        with speech_lock:
            speech_probs = p
        last_speech_time = time.time()

        # decide crisis by threshold file
        is_crisis = p[1] >= speech_threshold

        # choose a random response from Excel (preserve original functionality)
        resp = random_response(is_crisis)
        print(f"🪄 Respon: {resp}")

        # enqueue TTS (non-blocking)
        speak_text_async(resp)

    except sr.UnknownValueError:
        print("❌ Tidak bisa mengenali suara.")
        with speech_lock:
            speech_probs = np.array([0.5, 0.5], dtype=np.float32)
    except sr.RequestError as e:
        logging.warning("STT request error: %s", e)
        with speech_lock:
            speech_probs = np.array([0.5, 0.5], dtype=np.float32)
    except Exception as e:
        logging.exception("Unhandled exception in speech_callback: %s", e)
        with speech_lock:
            speech_probs = np.array([0.5, 0.5], dtype=np.float32)
    finally:
        speech_busy.clear()

def init_speech_recognition():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    recognizer.dynamic_energy_threshold = True
    # tuned values to be responsive but stable
    recognizer.energy_threshold = 300
    recognizer.pause_threshold = 0.6
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1.0)
        logging.info("✅ Kalibrasi mic selesai")
    # phrase_time_limit short to avoid long accumulated audio
    stop_fn = recognizer.listen_in_background(mic, speech_callback, phrase_time_limit=4)
    return stop_fn

# -----------------------------
# Camera helpers (preserve & robust)
# -----------------------------
def get_camera_index(preferred_index: int = 0):
    # Try to find a usable camera; preserve original search behavior
    import glob
    devices = glob.glob('/dev/video*')
    logging.info(f"Video devices found: {devices}")
    # Try preferred index, fallback to probing
    for idx in range(0, 10):
        cap = cv2.VideoCapture(idx)
        if cap is None:
            continue
        opened = cap.isOpened()
        cap.release()
        if opened:
            return idx
    return None

def preprocess(frame: np.ndarray, transform):
    """
    Preprocess crop to the expected input for Hailo pipeline:
    - convert BGR->RGB, PIL Image -> transform -> numpy
    Keep original behavior as much as possible so features unchanged.
    """
    try:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img)
        tensor = transform(pil)
        # Convert to numpy and flatten to Hailo input shape expectation if needed
        arr = tensor.numpy()
        # Hailo inference helper expects HWC or flattened; adapt per original pipeline
        # Here we return arr as (C, H, W) float32; hailo_infer wrapper handles casting/contiguity
        return arr
    except Exception as e:
        logging.warning("Preprocess error: %s", e)
        # return a neutral array to avoid breaking pipeline
        return np.zeros((3, FACE_INPUT_SIZE, FACE_INPUT_SIZE), dtype=np.float32)

# -----------------------------
# Fusion scoring function (preserve weights)
# -----------------------------
def compute_fusion_score(face_score: float, pose_score: float) -> Tuple[str, float]:
    """
    face_score and pose_score expected to be scalar probabilities for 'KRISIS' class
    speech_probs is a global vector [p_not_crisis, p_crisis]
    This function respects W_SPEECH, W_FACE, W_POSE and FUSION_THRESHOLD.
    """
    global last_speech_time
    with speech_lock:
        speech_score = float(speech_probs[1])

    # If no speech for some seconds, decay to neutral (so speech doesn't dominate forever)
    if time.time() - last_speech_time > 6:
        speech_score = 0.5

    fusion_score = (W_SPEECH * speech_score) + (W_FACE * face_score) + (W_POSE * pose_score)
    label = "KRISIS" if fusion_score >= FUSION_THRESHOLD else "TIDAK KRISIS"
    return label, fusion_score

# -----------------------------
# Main camera + fusion loop (preserve original modes and controls)
# -----------------------------
def main_loop(pipelines: Dict[str, HailoPipeline]):
    face_pipe = pipelines.get(HEF_FACE)
    pose_pipe = pipelines.get(HEF_POSE)

    cam_idx = get_camera_index()
    if cam_idx is None:
        raise RuntimeError("Kamera tidak ditemukan (/dev/video*)")

    cap = cv2.VideoCapture(cam_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    upperbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")

    mode = "fusion"
    prev_time = time.time()
    last_pred = None
    last_conf = 0.0

    # Optional Mediapipe pose estimator kept (feature preserved)
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        pose_estimator = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        have_mediapipe = True
    except Exception:
        pose_estimator = None
        have_mediapipe = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # keep loop alive if camera momentarily fails
                time.sleep(0.01)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(64,64))
            bodies = upperbody_cascade.detectMultiScale(gray, 1.05, 5, minSize=(128,128))

            # Draw detections (feature preserved)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            for (x,y,w,h) in bodies:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

            # Periodic inference at ~1Hz (adjust as necessary to your Hailo FPS)
            if time.time() - prev_time >= 1.0:
                face_probs = np.array([0.5, 0.5], dtype=np.float32)
                pose_probs = np.array([0.5, 0.5], dtype=np.float32)

                # Face inference (preserve original behavior if faces detected)
                if len(faces) > 0:
                    x,y,w,h = faces[0]
                    crop = frame[y:y+h, x:x+w]
                    if crop.size > 0 and face_pipe is not None:
                        try:
                            inp = preprocess(crop, face_tf)
                            out = hailo_infer(face_pipe, inp)
                            face_probs = softmax(out).astype(np.float32)
                        except Exception as e:
                            logging.warning("Face inference error: %s", e)
                            face_probs = np.array([0.5, 0.5], dtype=np.float32)

                # Pose inference (preserve original behavior if body detected)
                if len(bodies) > 0:
                    x,y,w,h = bodies[0]
                    crop = frame[y:y+h, x:x+w]
                    if crop.size > 0 and pose_pipe is not None:
                        try:
                            inp = preprocess(crop, pose_tf)
                            out = hailo_infer(pose_pipe, inp)
                            pose_probs = softmax(out).astype(np.float32)
                        except Exception as e:
                            logging.warning("Pose inference error: %s", e)
                            pose_probs = np.array([0.5, 0.5], dtype=np.float32)

                # Now compute fusion using weights (speech included)
                # speech_probs is global and updated by speech_callback
                # compute scalar 'KRISIS' probability from each component as the p_crisis value
                face_crisis_prob = float(face_probs[1])
                pose_crisis_prob = float(pose_probs[1])

                label, fusion_score = compute_fusion_score(face_crisis_prob, pose_crisis_prob)
                last_pred = label
                last_conf = fusion_score
                logging.info(f"FUSION -> {last_pred} ({last_conf:.2f})")

                prev_time = time.time()

            # overlay last prediction (preserve UI feature)
            if last_pred:
                cv2.putText(frame, f"{last_pred} ({last_conf:.2f}) | {mode}",
                            (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS.get(last_pred, (255,255,255)), 2)

            cv2.imshow("Fusion (face+pose+speech)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('f') or key == ord('F'):
                mode = "face"
            elif key == ord('p') or key == ord('P'):
                mode = "pose"
            elif key == ord('o') or key == ord('O'):
                mode = "fusion"

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if pose_estimator is not None:
            pose_estimator.close()
        # Stop TTS worker cleanly by sending None (optional)
        tts_queue.put(None)
        tts_queue.join()
        stt_executor.shutdown(wait=False)

# -----------------------------
# ENTRY POINT (preserve original flow)
# -----------------------------
if __name__ == "__main__":
    logging.info("Memulai aplikasi fusion (face + pose + speech)")
    # init Hailo pipelines (preserve)
    vdevice, pipelines = init_hailo([HEF_FACE, HEF_POSE])

    # init speech recognition (returns stop function if needed)
    stop_listen = init_speech_recognition()

    # run main
    try:
        main_loop(pipelines)
    except KeyboardInterrupt:
        logging.info("Dihentikan oleh user (KeyboardInterrupt)")
    except Exception as e:
        logging.exception("Unhandled exception in main: %s", e)
    finally:
        # If stop_listen exists, call it to stop background listener cleanly
        try:
            if stop_listen:
                stop_listen(wait_for_stop=False)
        except Exception:
            pass
        logging.info("Aplikasi selesai.")
