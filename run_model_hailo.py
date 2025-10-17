#!/usr/bin/env python3
"""
run_model_fusion_optimized.py
Optimized fusion of Hailo (face + pose) models and speech model
Weight: speech 0.4, pose 0.3, face 0.3
Optimized for Raspberry Pi 5
"""

import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import time
import platform
import logging
import threading
import re
import random
from collections import deque
from typing import Dict, Tuple, List, Optional

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import mediapipe as mp

import joblib
import speech_recognition as sr
import pyttsx3
import pandas as pd

from hailo_platform import (
    HEF,
    VDevice,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    FormatType,
    Device
)

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# ===== CONFIG =====
HEF_FACE = "models/hef/face-expression/face-expression.hef"
HEF_POSE = "models/hef/pose-recognition/pose-recognition.hef"

FACE_INPUT_SIZE = 48
POSE_INPUT_SIZE = 224

LABELS = ["TIDAK KRISIS", "KRISIS"]
COLORS = {"TIDAK KRISIS": (0,128,0), "KRISIS": (0,0,255)}
W_FACE, W_POSE, W_SPEECH = 0.3, 0.3, 0.4

# Performance settings
INFERENCE_INTERVAL = 1.0  # seconds between inferences
FPS_UPDATE_INTERVAL = 30  # frames between FPS updates
FRAME_SKIP = 1  # process every Nth frame for detection
MAX_FRAME_BUFFER = 3  # maximum frames to buffer

# ===== OPTIMIZED TRANSFORMS (singleton pattern) =====
class TransformCache:
    _face_tf = None
    _pose_tf = None
    
    @classmethod
    def get_face_transform(cls):
        if cls._face_tf is None:
            cls._face_tf = transforms.Compose([
                transforms.Resize((FACE_INPUT_SIZE, FACE_INPUT_SIZE), antialias=True),
                transforms.ToTensor(),
                cls.To3Channels(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
            ])
        return cls._face_tf
    
    @classmethod
    def get_pose_transform(cls):
        if cls._pose_tf is None:
            cls._pose_tf = transforms.Compose([
                transforms.Resize((POSE_INPUT_SIZE, POSE_INPUT_SIZE), antialias=True),
                transforms.ToTensor(),
                cls.To3Channels(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
            ])
        return cls._pose_tf
    
    class To3Channels:
        def __call__(self, x):
            return x.repeat(3, 1, 1) if x.shape[0] == 1 else x

# ===== HAILO HELPERS =====
class HailoPipeline:
    def __init__(self, net_group, input_vstreams_params, output_vstreams_params, input_infos, output_infos):
        self.net_group = net_group
        self.input_vstreams_params = input_vstreams_params
        self.output_vstreams_params = output_vstreams_params
        self.input_infos = input_infos
        self.output_infos = output_infos
        self.input_names = [info.name for info in input_infos]
        self.output_names = [info.name for info in output_infos]

def initialize_vdevice_and_pipelines(hef_paths: List[str]) -> Tuple[VDevice, Dict[str, HailoPipeline]]:
    logging.info("Memeriksa perangkat Hailo...")
    try:
        found = Device.scan()
        logging.info("Device.scan() result: %s", found)
    except Exception:
        logging.debug("Device.scan() tidak tersedia atau error, lanjutkan konfigurasi.")

    vdevice = VDevice()
    pipelines: Dict[str, HailoPipeline] = {}

    for hef_path in hef_paths:
        logging.info("Muat HEF: %s", hef_path)
        if not os.path.exists(hef_path):
            raise FileNotFoundError(f"HEF tidak ditemukan: {hef_path}")
        hef = HEF(hef_path)

        network_groups = vdevice.configure(hef)
        if not network_groups:
            raise RuntimeError("vdevice.configure returned empty network_groups")
        net_group = network_groups[0]
        logging.info("Network group ready for HEF: %s", os.path.basename(hef_path))

        input_vstreams_params = InputVStreamParams.make_from_network_group(
            net_group, quantized=False, format_type=FormatType.FLOAT32
        )
        output_vstreams_params = OutputVStreamParams.make_from_network_group(
            net_group, quantized=False, format_type=FormatType.FLOAT32
        )

        if not isinstance(input_vstreams_params, dict):
            input_vstreams_params = {p.name: p for p in input_vstreams_params}
        if not isinstance(output_vstreams_params, dict):
            output_vstreams_params = {p.name: p for p in output_vstreams_params}

        input_infos = net_group.get_input_vstream_infos()
        output_infos = net_group.get_output_vstream_infos()

        pipelines[hef_path] = HailoPipeline(
            net_group, input_vstreams_params, output_vstreams_params, 
            input_infos, output_infos
        )
        logging.info("Pipeline siap untuk %s (inputs: %s outputs: %s)", 
                     os.path.basename(hef_path),
                     [i.name for i in input_infos], 
                     [o.name for o in output_infos])

    return vdevice, pipelines

def hailo_infer(pipeline: HailoPipeline, inp_array: np.ndarray) -> np.ndarray:
    """Optimized inference with minimal allocations"""
    if not pipeline.input_names:
        raise RuntimeError("Pipeline tidak memiliki input stream")
    
    in_name = pipeline.input_names[0]
    # Avoid unnecessary copy if already contiguous and float32
    if inp_array.flags['C_CONTIGUOUS'] and inp_array.dtype == np.float32:
        inp = inp_array
    else:
        inp = np.ascontiguousarray(inp_array, dtype=np.float32)
    
    inputs = {in_name: inp}
    
    with InferVStreams(pipeline.net_group, 
                      pipeline.input_vstreams_params, 
                      pipeline.output_vstreams_params) as infer:
        with pipeline.net_group.activate():
            outputs = infer.infer(inputs)
    
    out_name = pipeline.output_names[0]
    return outputs[out_name].reshape(-1)

# Pre-allocated softmax computation
_softmax_cache = {}
def softmax(x: np.ndarray) -> np.ndarray:
    """Optimized softmax with numerical stability"""
    x_max = np.max(x)
    e = np.exp(x - x_max)
    return e / np.sum(e)

# ===== OPTIMIZED PREPROCESS =====
def preprocess(img_bgr: np.ndarray, transform) -> np.ndarray:
    """Optimized preprocessing with minimal copies"""
    # Direct conversion without intermediate allocation
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    tensor = transform(pil).unsqueeze(0)
    # Direct transpose to NHWC
    arr = tensor.numpy().transpose(0, 2, 3, 1).astype(np.float32)
    return arr

# ===== SPEECH RECOGNITION SECTION =====
class SpeechRecognizer:
    """Optimized speech recognition with caching"""
    def __init__(self):
        self.vectorizer = joblib.load("models/language/vectorizer.pkl")
        self.model = joblib.load("models/language/lgbm_model.pkl")
        with open("models/language/threshold.txt", "r") as f:
            self.threshold = float(f.read().strip())
        
        # Load response tables once
        self.df_krisis = pd.read_excel("data/csv/Data_tanggapan_positif.xlsx", sheet_name="Krisis")
        self.df_tidak = pd.read_excel("data/csv/Data_tanggapan_positif.xlsx", sheet_name="Tidak Krisis")
        self.krisis_responses = self.df_krisis["Respon"].dropna().tolist()
        self.tidak_responses = self.df_tidak["Respon"].dropna().tolist()
        
        # Precompile regex
        self.cleanup_pattern = re.compile(r'[^0-9a-z\s]')
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Preprocess crisis keywords
        self.crisis_keywords = self._load_crisis_keywords()
        
        # State
        self.probs = np.array([0.5, 0.5], dtype=np.float32)
        self.lock = threading.Lock()
        self.busy = threading.Event()
    
    def _load_crisis_keywords(self):
        raw_keywords = [
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
        ]
        return [self.preprocess_text(k) for k in raw_keywords]
    
    def preprocess_text(self, s):
        s = str(s).lower()
        s = self.cleanup_pattern.sub(' ', s)
        s = self.whitespace_pattern.sub(' ', s)
        return s.strip()
    
    def contains_crisis_keyword(self, s_proc):
        for k in self.crisis_keywords:
            if k in s_proc:
                return True
        return False
    
    def classify_text(self, text):
        s_proc = self.preprocess_text(text)
        if self.contains_crisis_keyword(s_proc):
            return np.array([0.0, 1.0], dtype=np.float32)
        if len(s_proc.split()) <= 2:
            return np.array([0.5, 0.5], dtype=np.float32)
        v = self.vectorizer.transform([s_proc])
        prob = float(self.model.predict_proba(v)[0, 1])
        return np.array([1 - prob, prob], dtype=np.float32)
    
    def get_response(self, is_crisis):
        return random.choice(self.krisis_responses if is_crisis else self.tidak_responses)

# TTS Engine (singleton)
class TTSEngine:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 160)
        self.engine.setProperty("volume", 1.0)
        
        indo_voice_id = None
        for v in self.engine.getProperty("voices"):
            if "Andika" in v.name or "Indonesian" in v.name or "Indonesia" in v.name:
                indo_voice_id = v.id
                break
        if indo_voice_id:
            self.engine.setProperty("voice", indo_voice_id)
    
    def speak(self, text):
        def run():
            with self._lock:
                self.engine.say(text)
                self.engine.runAndWait()
        threading.Thread(target=run, daemon=True).start()

# Global instances
speech_recognizer = None
tts_engine = None

def speech_callback(recognizer, audio):
    global speech_recognizer
    if speech_recognizer.busy.is_set():
        return
    speech_recognizer.busy.set()

    try:
        text = recognizer.recognize_google(audio, language="id-ID").strip()
        if text:
            print(f"\n🗣️ Anda berkata: {text}")
            p = speech_recognizer.classify_text(text)
            with speech_recognizer.lock:
                speech_recognizer.probs = p

            is_crisis = p[1] >= speech_recognizer.threshold
            response_text = speech_recognizer.get_response(is_crisis)
            print(f"🪄 Respon: {response_text}")
            tts_engine.speak(response_text)
        else:
            with speech_recognizer.lock:
                speech_recognizer.probs = np.array([0.5, 0.5], dtype=np.float32)

    except sr.UnknownValueError:
        with speech_recognizer.lock:
            speech_recognizer.probs = np.array([0.5, 0.5], dtype=np.float32)
    except sr.RequestError as e:
        print(f"⚠️ Error STT: {e}")
        with speech_recognizer.lock:
            speech_recognizer.probs = np.array([0.5, 0.5], dtype=np.float32)
    finally:
        speech_recognizer.busy.clear()

def init_speech_recognition():
    global speech_recognizer, tts_engine
    speech_recognizer = SpeechRecognizer()
    tts_engine = TTSEngine()
    
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    recognizer.dynamic_energy_threshold = True
    recognizer.energy_threshold = 300
    recognizer.pause_threshold = 0.8
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1.0)
    recognizer.listen_in_background(mic, speech_callback, phrase_time_limit=7)

# ===== CAMERA HELPERS =====
def get_available_camera(max_index=5):  # Reduced from 10
    """Optimized camera detection"""
    system = platform.system()
    
    if system == "Linux":
        import glob
        video_devices = glob.glob('/dev/video*')
        if video_devices:
            logging.info("Video devices found: %s", video_devices)
    
    backends = [cv2.CAP_V4L2, cv2.CAP_ANY] if system != "Windows" else [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    
    for i in range(max_index):
        for b in backends:
            try:
                cap = cv2.VideoCapture(i, b)
                if cap.isOpened():
                    ret, _ = cap.read()
                    cap.release()
                    if ret:
                        return i, b
            except Exception:
                continue
    
    return None, None

# ===== FPS TRACKER =====
class FPSTracker:
    def __init__(self, window_size=30):
        self.timestamps = deque(maxlen=window_size)
    
    def update(self):
        self.timestamps.append(time.time())
    
    def get_fps(self):
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        return len(self.timestamps) / elapsed if elapsed > 0 else 0.0

# ===== MAIN LOOP =====
def main_loop(pipelines: Dict[str, HailoPipeline]):
    face_pipe = pipelines.get(HEF_FACE)
    pose_pipe = pipelines.get(HEF_POSE)

    cam_idx, backend = get_available_camera()
    if cam_idx is None:
        logging.error("=" * 60)
        logging.error("KAMERA TIDAK DITEMUKAN!")
        logging.error("Pastikan kamera terhubung dengan benar")
        logging.error("=" * 60)
        raise RuntimeError("Tidak menemukan kamera.")
    
    logging.info("Kamera ditemukan: index=%s backend=%s", cam_idx, backend)
    cap = cv2.VideoCapture(cam_idx, backend)
    
    # Optimized resolution for Pi 5
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency

    # Load cascades once
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    upperbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")

    # MediaPipe pose with optimized settings for Pi
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose_estimator = mp_pose.Pose(
        static_image_mode=False, 
        model_complexity=0,  # Lighter model for Pi
        enable_segmentation=False, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )

    mode = "fusion"
    logging.info("Mode awal: %s", mode)
    logging.info("=" * 60)
    logging.info("CONTROLS: [F] Face | [P] Pose | [O] Fusion | [Q] Quit")
    logging.info("=" * 60)

    prev_inference_time = time.time()
    last_pred = None
    last_conf = 0.0
    frame_count = 0
    fps_tracker = FPSTracker()
    
    # Get transforms
    face_tf = TransformCache.get_face_transform()
    pose_tf = TransformCache.get_pose_transform()
    
    # Pre-allocate neutral probs
    neutral_probs = np.array([0.5, 0.5], dtype=np.float32)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            frame_count += 1
            fps_tracker.update()
            
            # Only process detection every FRAME_SKIP frames
            process_detection = (frame_count % FRAME_SKIP == 0)
            
            faces, bodies = [], []
            results = None
            
            if process_detection:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if mode in ("face", "fusion"):
                    faces = face_cascade.detectMultiScale(
                        gray, scaleFactor=1.1382, minNeighbors=7, minSize=(128,128)
                    )
                if mode in ("pose", "fusion"):
                    bodies = upperbody_cascade.detectMultiScale(
                        gray, scaleFactor=1.01618, minNeighbors=5, minSize=(256,256)
                    )
                    # Only run MediaPipe if we have detected body
                    if len(bodies) > 0:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        try:
                            results = pose_estimator.process(rgb)
                        except Exception as e:
                            logging.debug("MediaPipe error: %s", e)

            # Draw detections
            for (x,y,w,h) in faces: 
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            for (x,y,w,h) in bodies: 
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            
            if results and results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0,255,255), thickness=2, circle_radius=2
                    ),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0,0,255), thickness=2, circle_radius=2
                    )
                )

            # Inference at specified interval
            curr_time = time.time()
            if curr_time - prev_inference_time >= INFERENCE_INTERVAL:
                face_probs = None
                pose_probs = None

                if mode in ("face", "fusion") and face_pipe and len(faces) > 0:
                    x,y,w,h = faces[0]
                    face_crop = frame[y:y+h, x:x+w]
                    if face_crop.size > 0:
                        try:
                            inp = preprocess(face_crop, face_tf)
                            out = hailo_infer(face_pipe, inp)
                            face_probs = softmax(out)
                        except Exception as e:
                            logging.debug("Face inference error: %s", e)

                if mode in ("pose", "fusion") and pose_pipe and len(bodies) > 0:
                    x,y,w,h = bodies[0]
                    body_crop = frame[y:y+h, x:x+w]
                    if body_crop.size > 0:
                        try:
                            inp = preprocess(body_crop, pose_tf)
                            out = hailo_infer(pose_pipe, inp)
                            pose_probs = softmax(out)
                        except Exception as e:
                            logging.debug("Pose inference error: %s", e)

                # Fusion logic
                if mode == "fusion":
                    with speech_recognizer.lock:
                        speech_component = speech_recognizer.probs.copy()
                    
                    probs_final = (
                        W_FACE * (face_probs if face_probs is not None else neutral_probs) +
                        W_POSE * (pose_probs if pose_probs is not None else neutral_probs) +
                        W_SPEECH * speech_component
                    )

                    pred = int(np.argmax(probs_final))
                    last_pred = LABELS[pred]
                    last_conf = float(probs_final[pred])
                    logging.info("FUSION -> %s (%.2f)", last_pred, last_conf)

                elif mode == "face" and face_probs is not None:
                    pred = int(np.argmax(face_probs))
                    last_pred = LABELS[pred]
                    last_conf = float(face_probs[pred])
                elif mode == "pose" and pose_probs is not None:
                    pred = int(np.argmax(pose_probs))
                    last_pred = LABELS[pred]
                    last_conf = float(pose_probs[pred])

                prev_inference_time = curr_time

            # Overlay label
            if last_pred:
                label = f"{last_pred} ({last_conf:.2f}) | {mode.upper()}"
                color = COLORS.get(last_pred, (0,0,0))
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (5, 20-th-5), (15+tw, 25), color, -1)
                cv2.putText(frame, label, (10, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # FPS display
            if frame_count % FPS_UPDATE_INTERVAL == 0:
                fps = fps_tracker.get_fps()
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            cv2.imshow("Fusion: Face + Pose Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'): 
                break
            elif key == ord('f') or key == ord('F'): 
                mode = "face"
                logging.info("Mode -> FACE")
            elif key == ord('p') or key == ord('P'): 
                mode = "pose"
                logging.info("Mode -> POSE")
            elif key == ord('o') or key == ord('O'): 
                mode = "fusion"
                logging.info("Mode -> FUSION")

    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    except Exception:
        logging.exception("Fatal error in main loop")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose_estimator.close()
        logging.info("Cleanup complete")

# ===== ENTRY POINT =====
if __name__ == "__main__":
    logging.info("Memulai aplikasi fusion (face + pose + speech) - Optimized for Pi 5")
    missing = [p for p in (HEF_FACE, HEF_POSE) if not os.path.exists(p)]
    if missing:
        logging.error("HEF files tidak ditemukan: %s", missing)
        raise SystemExit(1)

    vdevice, pipelines = initialize_vdevice_and_pipelines([HEF_FACE, HEF_POSE])
    init_speech_recognition()
    main_loop(pipelines)
    logging.info("Selesai.")