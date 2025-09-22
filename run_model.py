import torch
import torch_directml
from models.load_models import face_expression_model

# ======================
# Load kembali model
# ======================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 2
model = face_expression_model(num_classes=num_classes, variant="large").to(device)

# Load state_dict
model.load_state_dict(torch.load("models/face-expression.pt", map_location=device, weights_only=False))

model.eval()
print("[INFO] Model loaded and ready for inference.")