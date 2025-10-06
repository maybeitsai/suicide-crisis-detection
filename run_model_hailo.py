#!/usr/bin/env python3
"""
run_model_hailo.py
Versi kompatibel HailoRT v4.20 (Raspberry Pi 5 + Hailo-8L)
- Face-expression HEF (48x48)
- Pose-recognition HEF (224x224)
FIXED: Menggunakan ConfiguredNetworkGroup untuk InferVStreams
"""

import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import time
import platform
import logging
from typing import Dict, Any, Tuple, List
import contextlib

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import mediapipe as mp

# Hailo imports (versi 4.20+)
from hailo_platform import (
    HEF,
    VDevice,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    FormatType,
    Device
)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# ===== CONFIG =====
HEF_FACE = "models/hef/face-expression/face-expression.hef"
HEF_POSE = "models/hef/pose-recognition/pose-recognition.hef"

FACE_INPUT_SIZE = 48
POSE_INPUT_SIZE = 224

LABELS = ["KRISIS", "TIDAK KRISIS"]  # pastikan urutan sesuai model
COLORS = {"TIDAK KRISIS": (0,128,0), "KRISIS": (0,0,255)}
W_FACE, W_POSE = 0.6, 0.4

# ===== TRANSFORMS =====
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

# ===== HAILO HELPERS =====
class HailoPipeline:
    def __init__(self, net_group, input_vstreams_params, output_vstreams_params, input_infos, output_infos):
        self.net_group = net_group  # ConfiguredNetworkGroup
        self.input_vstreams_params = input_vstreams_params
        self.output_vstreams_params = output_vstreams_params
        self.input_infos = input_infos
        self.output_infos = output_infos
        # store input/output names for convenience
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

        # configure() (v4.20) returns network_groups list
        network_groups = vdevice.configure(hef)
        if not network_groups:
            raise RuntimeError("vdevice.configure returned empty network_groups")
        net_group = network_groups[0]  # ConfiguredNetworkGroup
        logging.info("Network group ready for HEF: %s", os.path.basename(hef_path))

        # ===== create input/output vstream params using helper =====
        input_vstreams_params = InputVStreamParams.make_from_network_group(
            net_group, quantized=False, format_type=FormatType.FLOAT32
        )
        output_vstreams_params = OutputVStreamParams.make_from_network_group(
            net_group, quantized=False, format_type=FormatType.FLOAT32
        )

        # Convert to dict if needed
        if not isinstance(input_vstreams_params, dict):
            input_vstreams_params = {p.name: p for p in input_vstreams_params}
        if not isinstance(output_vstreams_params, dict):
            output_vstreams_params = {p.name: p for p in output_vstreams_params}

        # get info lists (info objects)
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
    """
    inp_array: numpy float32 NHWC, shape (1,H,W,C)
    returns flattened numpy array (float32)
    """
    if not pipeline.input_names:
        raise RuntimeError("Pipeline tidak memiliki input stream")
    
    in_name = pipeline.input_names[0]
    # ensure contiguous float32
    inp = np.ascontiguousarray(inp_array.astype(np.float32))
    inputs = {in_name: inp}
    
    # FIXED: Gunakan ConfiguredNetworkGroup langsung dengan aktivasi di dalam InferVStreams
    # InferVStreams akan handle aktivasi secara internal
    with InferVStreams(pipeline.net_group, 
                      pipeline.input_vstreams_params, 
                      pipeline.output_vstreams_params) as infer:
        with pipeline.net_group.activate():
            outputs = infer.infer(inputs)
    
    # grab first output
    out_name = pipeline.output_names[0]
    out = outputs[out_name]
    return np.array(out).reshape(-1)

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

# ===== PREPROCESS =====
def preprocess(img_bgr: np.ndarray, transform) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    tensor = transform(pil).unsqueeze(0)  # (1,C,H,W)
    arr = tensor.numpy().astype(np.float32)
    arr = np.transpose(arr, (0, 2, 3, 1))  # -> NHWC
    return arr

# ===== CAMERA HELPERS =====
def get_available_camera(max_index=10):
    """Cari kamera yang tersedia dengan berbagai metode"""
    system = platform.system()
    
    # Cek apakah ada /dev/video*
    if system == "Linux":
        import glob
        video_devices = glob.glob('/dev/video*')
        logging.info("Video devices found: %s", video_devices)
        
        if not video_devices:
            logging.warning("Tidak ada /dev/video* devices!")
    
    if system == "Windows":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY, cv2.CAP_V4L2]  # Coba CAP_ANY dulu
    
    for i in range(max_index):
        for b in backends:
            try:
                cap = cv2.VideoCapture(i, b)
                if cap.isOpened():
                    # Test read frame
                    ret, _ = cap.read()
                    cap.release()
                    if ret:
                        return i, b
            except Exception as e:
                logging.debug(f"Failed to open camera {i} with backend {b}: {e}")
                continue
    
    return None, None

# ===== MAIN LOOP =====
def main_loop(pipelines: Dict[str, HailoPipeline]):
    face_pipe = pipelines.get(HEF_FACE)
    pose_pipe = pipelines.get(HEF_POSE)

    cam_idx, backend = get_available_camera()
    if cam_idx is None:
        logging.error("=" * 60)
        logging.error("KAMERA TIDAK DITEMUKAN!")
        logging.error("Pastikan:")
        logging.error("1. Kamera terhubung dengan benar")
        logging.error("2. Jalankan: ls -la /dev/video*")
        logging.error("3. Cek permission: sudo usermod -a -G video $USER")
        logging.error("4. Atau gunakan: sudo chmod 666 /dev/video*")
        logging.error("=" * 60)
        raise RuntimeError("Tidak menemukan kamera. Pastikan kamera terhubung.")
    
    logging.info("Kamera ditemukan: index=%s backend=%s", cam_idx, backend)
    cap = cv2.VideoCapture(cam_idx, backend)
    
    # Set resolusi jika perlu
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Haar cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    upperbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")

    # MediaPipe pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose_estimator = mp_pose.Pose(
        static_image_mode=False, 
        model_complexity=1,
        enable_segmentation=False, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )

    mode = "fusion"
    logging.info("Mode awal: %s", mode)
    logging.info("=" * 60)
    logging.info("CONTROLS:")
    logging.info("  [F] - Face mode only")
    logging.info("  [P] - Pose mode only")
    logging.info("  [O] - Fusion mode (face + pose)")
    logging.info("  [Q] - Quit")
    logging.info("=" * 60)

    prev_time = time.time()
    last_pred = None
    last_conf = 0.0
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to read frame, skipping...")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces, bodies = [], []
            if mode in ("face", "fusion"):
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.133, minNeighbors=7, minSize=(48,48)
                )
            if mode in ("pose", "fusion"):
                bodies = upperbody_cascade.detectMultiScale(
                    gray, scaleFactor=1.016, minNeighbors=5, minSize=(128,128)
                )
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    results = pose_estimator.process(rgb)
                except Exception as e:
                    logging.warning("MediaPipe crash: %s", e)
                    results = None
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

            for (x,y,w,h) in faces: 
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            for (x,y,w,h) in bodies: 
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

            curr_time = time.time()
            if curr_time - prev_time >= 1.0:  # infer tiap ~1 detik
                face_probs = None
                pose_probs = None

                if mode in ("face", "fusion") and face_pipe is not None and len(faces) > 0:
                    x,y,w,h = faces[0]
                    face_crop = frame[y:y+h, x:x+w]
                    if face_crop.size > 0:
                        try:
                            inp = preprocess(face_crop, face_tf)
                            out = hailo_infer(face_pipe, inp)
                            face_probs = softmax(out)
                            logging.debug("Face inference OK: %s", face_probs)
                        except Exception as e:
                            logging.warning("Face inference error: %s", e)

                if mode in ("pose", "fusion") and pose_pipe is not None and len(bodies) > 0:
                    x,y,w,h = bodies[0]
                    body_crop = frame[y:y+h, x:x+w]
                    if body_crop.size > 0:
                        try:
                            inp = preprocess(body_crop, pose_tf)
                            out = hailo_infer(pose_pipe, inp)
                            pose_probs = softmax(out)
                            logging.debug("Pose inference OK: %s", pose_probs)
                        except Exception as e:
                            logging.warning("Pose inference error: %s", e)

                # fusion logic
                if mode == "fusion":
                    if face_probs is not None or pose_probs is not None:
                        probs_final = np.zeros(len(LABELS), dtype=np.float32)
                        if face_probs is not None: 
                            probs_final += W_FACE * face_probs
                        if pose_probs is not None: 
                            probs_final += W_POSE * pose_probs
                        pred = int(np.argmax(probs_final))
                        last_pred = LABELS[pred]
                        last_conf = float(probs_final[pred])
                        logging.info("FUSION -> %s (%.2f)", last_pred, last_conf)
                elif mode == "face" and face_probs is not None:
                    pred = int(np.argmax(face_probs))
                    last_pred = LABELS[pred]
                    last_conf = float(face_probs[pred])
                    logging.info("FACE -> %s (%.2f)", last_pred, last_conf)
                elif mode == "pose" and pose_probs is not None:
                    pred = int(np.argmax(pose_probs))
                    last_pred = LABELS[pred]
                    last_conf = float(pose_probs[pred])
                    logging.info("POSE -> %s (%.2f)", last_pred, last_conf)

                prev_time = curr_time

            # overlay label
            if last_pred is not None:
                label = f"{last_pred} ({last_conf:.2f}) | MODE: {mode.upper()}"
                color = COLORS.get(last_pred, (0,0,0))
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                x_text, y_text = 10, 40
                cv2.rectangle(frame, (x_text-5, y_text-th-5), 
                            (x_text+tw+5, y_text+5), color, -1)
                cv2.putText(frame, label, (x_text, y_text), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

            # FPS counter
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - prev_time + 0.001)
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

# ===== QUICK TEST =====
def quick_test_pipeline(pipeline: HailoPipeline, input_size: int):
    h = w = input_size
    test_inp = np.random.rand(1, h, w, 3).astype(np.float32)
    try:
        out = hailo_infer(pipeline, test_inp)
        logging.info("âœ“ Quick test success. Output length: %d", out.size)
    except Exception as e:
        logging.error("âœ— Quick test failed: %s", e)
        logging.exception("Full traceback:")

# ===== ENTRY POINT =====
if __name__ == "__main__":
    logging.info("=" * 60)
    logging.info("MEMULAI APLIKASI HAILO FUSION")
    logging.info("Input sizes: face %dx%d, pose %dx%d" % 
                 (FACE_INPUT_SIZE, FACE_INPUT_SIZE, POSE_INPUT_SIZE, POSE_INPUT_SIZE))
    logging.info("=" * 60)

    missing = [p for p in (HEF_FACE, HEF_POSE) if not os.path.exists(p)]
    if missing:
        logging.error("HEF files tidak ditemukan: %s", missing)
        raise SystemExit(1)

    vdevice, pipelines = initialize_vdevice_and_pipelines([HEF_FACE, HEF_POSE])

    # quick test per pipeline (ukuran sesuai model)
    if HEF_FACE in pipelines:
        logging.info("Quick test face pipeline (48x48)")
        quick_test_pipeline(pipelines[HEF_FACE], FACE_INPUT_SIZE)
    if HEF_POSE in pipelines:
        logging.info("Quick test pose pipeline (224x224)")
        quick_test_pipeline(pipelines[HEF_POSE], POSE_INPUT_SIZE)

    logging.info("=" * 60)
    main_loop(pipelines)
    logging.info("Selesai.")
    