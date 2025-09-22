import cv2
import time
import torch
import numpy as np
from torchvision import transforms
from models.load_models import face_expression_model
from PIL import Image

# ======================
# Setup device & model
# ======================
# import torch_directml
# device = torch_directml.device()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 2
model = face_expression_model(num_classes=num_classes, variant="large").to(device)
state_dict = torch.load("models/face-expression.pt", map_location=device)
model.load_state_dict(state_dict)
model.eval()
print("[INFO] Model loaded and ready for inference.")

# ======================
# Auto camera detection
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
    raise RuntimeError("❌ Tidak ada kamera yang tersedia!")
else:
    print(f"[INFO] Kamera ditemukan di index {camera_index} dengan backend {backend}")
    cap = cv2.VideoCapture(camera_index, backend)

# ======================
# Face detector
# ======================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ======================
# Transform (sesuai training)
# ======================
class To3Channels:
    def __call__(self, x):
        return x.repeat(3, 1, 1) if x.shape[0] == 1 else x

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    To3Channels(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def preprocess_face(face_img):
    img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = transform(img).unsqueeze(0)
    return img.to(device)

labels = ["1. KRISIS", "2. TIDAK KRISIS"]

# ======================
# Realtime loop + FPS
# ======================
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        # Sedikit kecilkan kotak untuk fokus ke wajah
        pad = int(0.1 * w)
        x1 = max(0, x + pad)
        y1 = max(0, y + pad)
        x2 = min(frame.shape[1], x + w - pad)
        y2 = min(frame.shape[0], y + h - pad)

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size > 0:
            inp = preprocess_face(face_crop)
            with torch.no_grad():
                outputs = model(inp)
                probs = torch.softmax(outputs, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                label = labels[pred]
                conf = probs[0, pred].item()

            # Gambar kotak & label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Hitung FPS
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Face Expression Realtime", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
