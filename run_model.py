import cv2
import time
import torch
import platform
import numpy as np
import mediapipe as mp
from torchvision import transforms
from module.models import face_expression_model, pose_recognition_model
from PIL import Image

# ======================
# Setup device
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Load Face Expression Model
# ======================
face_model = face_expression_model(num_classes=2, variant="large").to(device)
face_state = torch.load("models/face-expression.pt", map_location=device)
face_model.load_state_dict(face_state)
face_model.eval()

# ======================
# Load Pose Recognition Model
# ======================
pose_model = pose_recognition_model(num_classes=2, variant="large").to(device)
pose_state = torch.load("models/pose-recognition.pt", map_location=device)
pose_model.load_state_dict(pose_state)
pose_model.eval()

print("[INFO] Face & Pose models loaded successfully.")

# ======================
# MODE awal
# ======================
mode = "fusion"  
print(f"[INFO] Mode awal: {mode.upper()}")

# ======================
# Kamera (auto detection)
# ======================
def get_available_camera(max_index=5):
    system = platform.system()
    if system == "Windows":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW, cv2.CAP_ANY]
    else:  # Linux / macOS
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

    for i in range(max_index):
        for backend in backends:
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                cap.release()
                return i, backend
    return None, None

camera_index, backend = get_available_camera()
if camera_index is None:
    raise RuntimeError("❌ Kamera tidak tersedia!")
else:
    print(f"[INFO] Kamera ditemukan di index {camera_index} dengan backend {backend}")
    cap = cv2.VideoCapture(camera_index, backend)

# ======================
# Haarcascade Detectors
# ======================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
upperbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")

# ======================
# MediaPipe Pose
# ======================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_estimator = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ======================
# Transforms
# ======================
class To3Channels:
    def __call__(self, x):
        return x.repeat(3, 1, 1) if x.shape[0] == 1 else x

face_tf = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    To3Channels(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

pose_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    To3Channels(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def preprocess(img, transform):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = transform(img).unsqueeze(0)
    return img.to(device)

# ======================
# Labels & Fusion weights
# ======================
labels = ["KRISIS", "TIDAK KRISIS"]
colors = {
    "TIDAK KRISIS": (0, 128, 0),  # hijau gelap
    "KRISIS": (0, 0, 255)         # merah
}
w_face, w_pose = 0.6, 0.4

# ======================
# Realtime loop
# ======================
prev_time = time.time()
last_pred = None
last_conf = 0.0

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARN] Gagal baca frame dari kamera.")
            continue


        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Selalu deteksi sesuai mode ---
        faces, bodies = [], []
        if mode in ["face", "fusion"]:
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.13333,
                minNeighbors=7,
                minSize=(128, 128)
            )
        if mode in ["pose", "fusion"]:
            bodies = upperbody_cascade.detectMultiScale(
                gray,
                scaleFactor=1.01618,
                minNeighbors=5,
                minSize=(256, 256)
            )
                # --- Tambahan: Pose Estimation dengan MediaPipe ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                results = pose_estimator.process(rgb_frame)
            except Exception as e:
                print("[ERROR] MediaPipe crash:", e)
                results = None


            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                )


        # Gambar box deteksi
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        for (x, y, w, h) in bodies:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        curr_time = time.time()

        # --- Prediksi setiap 1.618 detik ---
        if curr_time - prev_time >= 1.618:
            face_probs, pose_probs = None, None

            # Prediksi wajah
            if mode in ["face", "fusion"]:
                for (x, y, w, h) in faces:
                    pad = int(0.1 * w)
                    x1, y1 = max(0, x+pad), max(0, y+pad)
                    x2, y2 = min(frame.shape[1], x+w-pad), min(frame.shape[0], y+h-pad)
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        inp = preprocess(face_crop, face_tf)
                        with torch.no_grad():
                            outputs = face_model(inp)
                            face_probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

            # Prediksi pose
            if mode in ["pose", "fusion"]:
                for (x, y, w, h) in bodies:
                    body_crop = frame[y:y+h, x:x+w]
                    if body_crop.size > 0:
                        inp = preprocess(body_crop, pose_tf)
                        with torch.no_grad():
                            outputs = pose_model(inp)
                            pose_probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

            # Fusion / Single
            if mode == "fusion":
                if face_probs is not None or pose_probs is not None:
                    probs_final = np.zeros(2)
                    if face_probs is not None:
                        probs_final += w_face * face_probs
                    if pose_probs is not None:
                        probs_final += w_pose * pose_probs
                    pred = np.argmax(probs_final)
                    last_pred = labels[pred]
                    last_conf = probs_final[pred]

            elif mode == "face" and face_probs is not None:
                pred = np.argmax(face_probs)
                last_pred = labels[pred]
                last_conf = face_probs[pred]

            elif mode == "pose" and pose_probs is not None:
                pred = np.argmax(pose_probs)
                last_pred = labels[pred]
                last_conf = pose_probs[pred]
                print(pred, last_conf)

            prev_time = curr_time  # reset timer

        # --- Selalu tampilkan hasil terakhir ---
        if last_pred is not None:
            label = f"{last_pred} ({last_conf:.2f}) | MODE: {mode.upper()}"
            color = colors[last_pred]
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            x, y = 10, 40
            cv2.rectangle(frame, (x-5, y-th-5), (x+tw+5, y+5), color, -1)
            cv2.putText(frame, label, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        cv2.imshow("Fusion: Face + Pose Recognition", frame)

        # ======================
        # Keyboard control
        # ======================
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):   # keluar
            break
        elif key == ord('f'):
            mode = "face"
            print("[INFO] Mode diganti ke: FACE")
        elif key == ord('p'):
            mode = "pose"
            print("[INFO] Mode diganti ke: POSE")
        elif key == ord('o'):
            mode = "fusion"
            print("[INFO] Mode diganti ke: FUSION")
except Exception as e:
    print("[FATAL ERROR]", e)

cap.release()
cv2.destroyAllWindows()
