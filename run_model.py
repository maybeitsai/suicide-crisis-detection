import cv2
import time
import torch
import numpy as np
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
# Kamera (auto detection)
# ======================
def get_available_camera(max_index=10):
    backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_VFW, cv2.CAP_ANY]
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
labels = ["TIDAK KRISIS", "KRISIS"]
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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Selalu deteksi wajah & tubuh ---
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1618,     # lebih besar → lebih ketat
        minNeighbors=7,      # lebih tinggi → lebih sedikit false positive
        minSize=(128, 128)     # jangan deteksi wajah terlalu kecil
    )

    bodies = upperbody_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1618,
        minNeighbors=5,
        minSize=(256, 256)
    )


    # Gambar box deteksi
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    curr_time = time.time()

    # --- Prediksi setiap 3 detik ---
    if curr_time - prev_time >= 1.618:
        face_probs, pose_probs = None, None

        # Prediksi wajah
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
        for (x, y, w, h) in bodies:
            body_crop = frame[y:y+h, x:x+w]
            if body_crop.size > 0:
                inp = preprocess(body_crop, pose_tf)
                with torch.no_grad():
                    outputs = pose_model(inp)
                    pose_probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        # Fusion
        if face_probs is not None or pose_probs is not None:
            probs_final = np.zeros(2)
            if face_probs is not None:
                probs_final += w_face * face_probs
            if pose_probs is not None:
                probs_final += w_pose * pose_probs
            pred = np.argmax(probs_final)
            last_pred = labels[pred]
            last_conf = probs_final[pred]

        prev_time = curr_time  # reset timer

    # --- Selalu tampilkan hasil terakhir ---
    if last_pred is not None:
        label = f"{last_pred} ({last_conf:.2f})"
        color = colors[last_pred]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        x, y = 10, 40
        cv2.rectangle(frame, (x-5, y-th-5), (x+tw+5, y+5), color, -1)
        cv2.putText(frame, label, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    cv2.imshow("Fusion: Face + Pose Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
