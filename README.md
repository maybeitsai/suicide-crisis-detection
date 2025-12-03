# 🤖 Robot Pencegah Bunuh Diri

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Copyright](https://img.shields.io/badge/Copyright-Program_Komputer-green.svg)](https://www.dgip.go.id/)

Sistem deteksi krisis bunuh diri berbasis AI menggunakan analisis ekspresi wajah dan pose tubuh untuk identifikasi dini kondisi mental remaja.

## 📋 Deskripsi

Proyek ini mengembangkan robot pendamping yang dapat mendeteksi tanda-tanda krisis bunuh diri pada remaja melalui:

- **Deteksi Ekspresi Wajah**: Analisis emosi dan kondisi mental dari ekspresi wajah
- **Analisis Pose Tubuh**: Deteksi bahasa tubuh yang mengindikasikan kondisi krisis
- **Sistem Peringatan Dini**: Memberikan respons dan intervensi yang tepat

## 🚀 Fitur Utama

- ✅ **Deteksi Visual Real-time**: Analisis melalui kamera webcam
- ✅ **Deteksi Audio/Suara**: Analisis percakapan dengan speech recognition
- ✅ Model AI berbasis MobileNetV3 (optimized untuk performa)
- ✅ Analisis ekspresi wajah dan pose tubuh
- ✅ **Analisis Teks/Ucapan**: LightGBM classifier dengan TF-IDF vectorizer
- ✅ Sistem klasifikasi krisis/tidak krisis
- ✅ **Text-to-Speech Response**: Respon otomatis berbahasa Indonesia
- ✅ Interface yang mudah digunakan
- ✅ 3 mode deteksi visual: Face, Pose, dan Fusion
- ✅ Kontrol keyboard dan voice interaktif
- ✅ MediaPipe pose estimation terintegrasi

## ⚡ Quick Start

```bash
# Clone dan masuk ke direktori
git clone https://github.com/maybeitsai/suicide-crisis-detection.git
cd "suicide-crisis-detection"

# Install dependencies dengan UV
uv sync

# Jalankan sistem deteksi (pilih mode sesuai kebutuhan):

# 1. VISUAL DETECTION - Analisis kamera
python run_model.py        # CPU/GPU (Windows/Linux/macOS)
python run_model_hailo.py  # Hailo AI Hat+ (Raspberry Pi 5)

# 2. SPEECH DETECTION - Analisis suara/percakapan
python run_model_speech.py  # Voice recognition + TTS response

# Kontrol keyboard (Visual mode):
F = Face mode | P = Pose mode | O = Fusion mode | Q = Quit

# Voice commands (Speech mode):
Ucapkan "exit" untuk keluar | Bicara normal untuk analisis
```

## 🛠️ Teknologi yang Digunakan

### 🎯 **Core Technologies**

- **Python 3.9+**
- **UV Package Manager** - Dependency Management

### 🤖 **Visual AI Models**

- **PyTorch** - Deep Learning Framework untuk computer vision
- **OpenCV** - Computer Vision processing
- **Torchvision** - Pre-trained Models (MobileNetV3)
- **MediaPipe** - Pose estimation dan computer vision
- **ONNX Runtime** - Cross-platform ML inference

### 🗣️ **Speech & Language AI**

- **LightGBM** - Gradient boosting untuk text classification
- **SpeechRecognition** - Google Speech-to-Text API
- **pyttsx3** - Text-to-Speech engine
- **scikit-learn** - TF-IDF vectorization dan preprocessing
- **pandas** - Data manipulation untuk training dataset

### ⚡ **Hardware Acceleration**

- **Hailo SDK** - Hardware acceleration untuk Hailo AI processors
- **CUDA Support** - GPU acceleration untuk PyTorch models

## 📁 **Model Files Structure**

```
models/
├── 📸 VISUAL MODELS
│   ├── face-expression.pt              # PyTorch face model (CPU/GPU)
│   ├── face-expression-directml.pt     # DirectML optimized
│   ├── face-expression-v2.pt           # Version 2
│   ├── pose-recognition.pt             # PyTorch pose model (CPU/GPU)
│   ├── pose-recognition-directml.pt    # DirectML optimized
│   └── pose-recognition-v2.pt          # Version 2
│
├── 🗣️ SPEECH/LANGUAGE MODELS
│   └── language/
│       ├── vectorizer.pkl              # TF-IDF vectorizer
│       ├── lgbm_model.pkl             # LightGBM classifier
│       └── threshold.txt               # Optimal classification threshold
│
├── 🌐 CROSS-PLATFORM MODELS (ONNX)
│   └── onnx/
│       ├── face-expression.onnx        # Cross-platform face model
│       └── pose-recognition.onnx       # Cross-platform pose model
│
├── ⚡ HAILO AI MODELS
│   ├── har/                           # Hailo Archive (intermediate)
│   │   ├── face-expression.har
│   │   ├── face-expression-opt.har    # Optimized variant
│   │   ├── pose-recognition.har
│   │   └── pose-recognition-opt.har   # Optimized variant
│   │
│   └── hef/                           # Hailo Executable (final)
│       ├── face-expression/
│       │   ├── face-expression.hef    # Ready untuk Hailo-8L
│       │   └── face-expression_compiled.har
│       └── pose-recognition/
│           ├── pose-recognition.hef   # Ready untuk Hailo-8L
│           └── pose-recognition_compiled.har
└──
```

**Model Usage by Platform:**

| Platform                   | Face Model             | Pose Model              | Speech Model     | Format                 |
| -------------------------- | ---------------------- | ----------------------- | ---------------- | ---------------------- |
| **Windows/Linux/macOS**    | `face-expression.pt`   | `pose-recognition.pt`   | `language/*.pkl` | PyTorch + scikit-learn |
| **Cross-Platform**         | `face-expression.onnx` | `pose-recognition.onnx` | `language/*.pkl` | ONNX + scikit-learn    |
| **Raspberry Pi 5 + Hailo** | `face-expression.hef`  | `pose-recognition.hef`  | `language/*.pkl` | HEF + scikit-learn     |

## 🚀 Hardware Acceleration & Deployment Options

Proyek ini mendukung berbagai platform dengan optimasi khusus untuk performa maksimal:

### 🖥️ **CPU Inference (Universal)**

```bash
python run_model.py
```

- **Platform**: Windows, Linux, macOS
- **Model Format**: PyTorch (.pt)
- **Performa**: Good for development & testing
- **Requirements**: Intel/AMD x64 atau ARM64

### 🎮 **GPU Acceleration (CUDA)**

```bash
# Otomatis terdeteksi jika CUDA tersedia
python run_model.py
```

- **Platform**: Windows, Linux dengan GPU NVIDIA
- **Model Format**: PyTorch (.pt)
- **Performa**: ~3-5x lebih cepat dari CPU
- **Requirements**: CUDA-compatible GPU

### ⚡ **Hailo AI Hat+ (Raspberry Pi 5)**

```bash
python run_model_hailo.py
```

- **Platform**: Raspberry Pi 5 + Hailo AI Hat+
- **Model Format**: HEF (Hailo Executable Format)
- **Performa**: ~26 TOPS AI processing power
- **Requirements**: Raspberry Pi 5, Hailo AI Hat+, HailoRT SDK

### 🌐 **ONNX Runtime (Cross-Platform)**

- **Model Format**: ONNX (.onnx)
- **Platform**: Semua platform dengan ONNX Runtime
- **Performa**: Optimized untuk production deployment
- **Benefit**: Platform-agnostic deployment

### 🔧 **HAR Format (Hailo Archive)**

- **Model Format**: HAR (Hailo Archive)
- **Purpose**: Intermediate format untuk kompilasi ke HEF
- **Variants**: Standard (.har) dan Optimized (-opt.har)

## 📦 Instalasi

### Prasyarat

- Python 3.9 atau lebih baru
- Webcam atau kamera eksternal

### Langkah Instalasi

1. **Clone repository**

   ```bash
   git clone https://github.com/maybeitsai/suicide-crisis-detection.git
   cd "Robot Pencegah Bunuh Diri"
   ```

2. **Install UV package manager** (jika belum ada)

   ```bash
   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | more"

   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | less
   ```

3. **Install dependencies menggunakan UV**

   ```bash
   uv sync
   ```

4. **Aktivasi virtual environment**

   ```bash
   # Windows
   .venv\Scripts\activate

   # macOS/Linux
   source .venv/bin/activate
   ```

### 🍓 **Instalasi Khusus Raspberry Pi 5 + Hailo AI Hat+**

**Prerequisites:**

1. **Raspberry Pi 5** dengan **8GB RAM** (recommended)
2. **Hailo AI Hat+** terpasang dengan benar
3. **Raspberry Pi OS 64-bit** atau **Ubuntu 22.04 LTS**

**Langkah instalasi:**

1. **Setup Raspberry Pi 5**

   ```bash
   # Update sistem
   sudo apt update && sudo apt upgrade -y

   # Install dependencies
   sudo apt install -y python3-pip python3-venv git cmake
   ```

2. **Install Hailo SDK**

   ```bash
   # Download dan install HailoRT
   wget https://hailo.ai/downloads/hailort-4.20.0-linux.deb
   sudo dpkg -i hailort-4.20.0-linux.deb

   # Install Python bindings
   pip install hailort
   ```

3. **Verifikasi Hailo device**

   ```bash
   # Check Hailo device detection
   hailortcli scan

   # Test model loading (opsional)
   hailortcli run models/hef/face-expression/face-expression.hef
   ```

4. **Clone dan setup project**

   ```bash
   git clone https://github.com/maybeitsai/suicide-crisis-detection.git
   cd suicide-crisis-detection

   # Install dependencies (tanpa CUDA packages)
   uv sync --no-dev
   ```

5. **Test installation**
   ```bash
   python run_model_hailo.py
   ```

## 🎯 Cara Penggunaan

### 🚀 Menjalankan Sistem Deteksi Real-time

#### Langkah Dasar - CPU/GPU Inference

```bash
python run_model.py
```

#### 🔥 Hailo AI Hat+ Acceleration (Raspberry Pi 5)

**Prerequisites untuk Raspberry Pi 5:**

```bash
# Install Hailo SDK dan dependencies
sudo apt update
sudo apt install hailo-all

# Verifikasi instalasi Hailo
hailortcli fw-control identify
```

**Menjalankan dengan Hailo acceleration:**

```bash
python run_model_hailo.py
```

**Fitur khusus Hailo mode:**

- ⚡ **26 TOPS** AI processing power
- 🔋 **Low power** consumption (~2W)
- 🎯 **Dedicated inference** - tidak mengganggu CPU utama
- 📊 **Real-time performance** - konsisten 30+ FPS
- 🌡️ **Thermal efficient** - pengelolaan panas optimal

**Troubleshooting Hailo:**

```bash
# Check Hailo device status
lsusb | grep Hailo

# Monitor Hailo performance
sudo hailortcli monitor --rate 1000

# Reset Hailo device jika diperlukan
sudo hailortcli reset
```

#### 🗣️ Menjalankan Speech Detection System

**Mode Deteksi Suara & Percakapan:**

```bash
python run_model_speech.py
```

**Fitur Speech System:**

- 🎤 **Real-time Speech Recognition** - Google Speech-to-Text API (Bahasa Indonesia)
- 🧠 **Text Classification** - LightGBM model dengan TF-IDF features
- 🗨️ **Crisis Keyword Detection** - 50+ kata kunci krisis bunuh diri
- 🔊 **Text-to-Speech Response** - Respons otomatis dalam Bahasa Indonesia
- ⚡ **Low Latency** - Response time <2 detik

**Voice Commands:**

| Command           | Action   | Deskripsi                            |
| ----------------- | -------- | ------------------------------------ |
| **Bicara Normal** | Analisis | Sistem akan menganalisis ucapan Anda |
| **"exit"**        | Keluar   | Hentikan program speech detection    |

**Contoh Interaksi:**

```
🗣️ User: "Saya merasa sedih hari ini"
📊 Sistem: TIDAK KRISIS (Prob: 0.23)
🔊 Respon: "Saya mendengarkan kamu, ceritakan lebih lanjut"

🗣️ User: "Saya ingin mati saja"
📊 Sistem: KRISIS (Prob: 1.0) [keyword detected]
🔊 Respon: "Hidup memang berat, tapi kamu tidak sendirian..."
```

#### 🎮 Kontrol Keyboard Visual Detection

Untuk mode visual detection (`run_model.py`), gunakan keyboard:

| Tombol | Mode            | Deskripsi                                |
| ------ | --------------- | ---------------------------------------- |
| `F`    | **Face Only**   | Hanya menganalisis ekspresi wajah        |
| `P`    | **Pose Only**   | Hanya menganalisis pose tubuh            |
| `O`    | **Fusion Mode** | Gabungan analisis wajah + pose (default) |
| `Q`    | **Quit**        | Keluar dari program                      |

#### 🔍 Mode Deteksi Yang Tersedia

**1. Face Expression Mode (`F`)**

- Fokus pada deteksi ekspresi wajah
- Menggunakan Haar Cascade untuk deteksi wajah
- Analisis area wajah 48x48 pixel
- Weight: 60% dalam fusion mode

**2. Pose Recognition Mode (`P`)**

- Fokus pada analisis pose dan bahasa tubuh
- Menggunakan Haar Cascade untuk deteksi tubuh bagian atas
- Dilengkapi MediaPipe Pose Estimation
- Analisis area tubuh 224x224 pixel
- Weight: 40% dalam fusion mode

**3. Fusion Mode (`O`) - Recommended**

- Kombinasi optimal dari kedua mode
- Akurasi tertinggi dengan menggabungkan:
  - 60% Face Expression Analysis
  - 40% Pose Recognition Analysis
- Mode default saat program dimulai

#### ⚙️ Spesifikasi Teknis

**Model Architecture:**

**Visual Models:**

- **Face Model**: MobileNetV3-Large + Custom Classifier
- **Pose Model**: MobileNetV3-Large + Custom Classifier
- **Input Size**:
  - Face: 48x48 RGB
  - Pose: 224x224 RGB
- **Output Classes**: 2 (KRISIS, TIDAK KRISIS)

**Speech/Language Model:**

- **Text Classifier**: LightGBM with TF-IDF features
- **Vectorizer**: TfidfVectorizer (ngram_range=(1,2), max_features=10000)
- **Features**: Sublinear TF, min_df=2, max_df=0.95
- **Crisis Keywords**: 50+ Indonesian crisis keywords
- **Threshold**: Optimized for high recall (crisis detection priority)
- **Input**: Raw text (Indonesian language)
- **Output**: 2 classes + confidence scores

**Detection Pipeline:**

1. **Camera Auto-Detection**: Otomatis mencari kamera yang tersedia
2. **Face Detection**: Haar Cascade (minSize: 128x128)
3. **Body Detection**: Haar Cascade (minSize: 256x256)
4. **Pose Estimation**: MediaPipe Pose (33 landmark points)
5. **Prediction Interval**: Setiap 1.618 detik (Golden Ratio)

**Hardware Requirements:**

#### 🖥️ **Desktop/Laptop (CPU/GPU)**

- **CPU**: Intel i5 atau AMD Ryzen 5 (minimum)
- **RAM**: 8GB (minimum), 16GB (recommended)
- **GPU**: CUDA-compatible (optional, untuk performa lebih cepat)
- **Camera**: Webcam atau kamera eksternal (minimum 720p)

#### 🍓 **Raspberry Pi 5 + Hailo AI Hat+**

- **SBC**: Raspberry Pi 5 (4GB/8GB RAM)
- **AI Accelerator**: Hailo AI Hat+ (Hailo-8L processor)
- **Storage**: MicroSD 32GB+ (Class 10) atau NVMe SSD
- **Power**: 5V/5A USB-C adapter (untuk Hailo + RPi5)
- **Camera**: RPi Camera Module v3 atau USB camera
- **OS**: Raspberry Pi OS (64-bit) atau Ubuntu 22.04 LTS
- **SDK**: HailoRT v4.20+ dan Hailo Dataflow Compiler

#### 🌐 **Edge Deployment (ONNX)**

- **Platforms**: x86_64, ARM64, ARMv7
- **RAM**: 4GB+ (tergantung platform)
- **Runtime**: ONNX Runtime 1.15+

#### 📊 Output Interface

Program menampilkan informasi real-time:

```
[LABEL] (CONFIDENCE) | MODE: [CURRENT_MODE]
```

**Contoh Output:**

- `TIDAK KRISIS (0.87) | MODE: FUSION` - Kondisi normal dengan confidence 87%
- `KRISIS (0.92) | MODE: FACE` - Terdeteksi krisis dengan confidence 92%

**Visual Indicators:**

- 🟢 **Hijau**: TIDAK KRISIS (kondisi normal)
- 🔴 **Merah**: KRISIS (perlu perhatian)
- 📦 **Kotak Hijau**: Deteksi wajah aktif
- 📦 **Kotak Biru**: Deteksi pose aktif
- 🦴 **Skeleton Kuning**: MediaPipe pose landmarks

## 📊 **Performa Model Terlatih**

### **Overview: Training & Evaluation Results**

Kedua model visual (Face-Expression dan Pose-Recognition) telah dilatih dan dievaluasi pada dataset khusus dengan hasil yang sangat memuaskan. Berikut adalah ringkasan lengkap performa model:

---

### **🎯 Face-Expression Model Performance**

**Dataset Overview:**

- Total images: 4,094
- Training set: 3,275 images (80%)
- Validation set: 409 images (10%)
- Test set: 410 images (10%)
- Class distribution: Krisis (56.5%), Tidak Krisis (43.5%)

**Training Configuration:**

- Architecture: MobileNetV3-Large + Custom Classifier
- Epochs: 10
- Batch size: 64
- Learning rate: 1.618e-5 (fine-tuning)
- Optimizer: AdamW
- Loss function: CrossEntropyLoss with class weights
- Early stopping: Patience = 3

**Training History (Per Epoch):**

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Status        |
| ----- | ---------- | --------- | -------- | ------- | ------------- |
| 1     | 0.6392     | 66.81%    | 0.6367   | 87.29%  | ✓ Best        |
| 2     | 0.4912     | 86.63%    | 0.5117   | 90.22%  | ✓ Best        |
| 3     | 0.3194     | 93.16%    | 0.3237   | 93.40%  | ✓ Best        |
| 4     | 0.1969     | 95.36%    | 0.1930   | 95.11%  | ✓ Best        |
| 5     | 0.1254     | 96.95%    | 0.1337   | 96.58%  | ✓ Best        |
| 6     | 0.0956     | 97.25%    | 0.0999   | 97.07%  | ✓ Best        |
| 7     | 0.0670     | 97.92%    | 0.0864   | 97.56%  | ✓ Best        |
| 8     | 0.0566     | 98.44%    | 0.0676   | 98.04%  | ✓ Best        |
| 9     | 0.0394     | 98.84%    | 0.0647   | 98.04%  | ✓ **BEST**    |
| 10    | 0.0315     | 99.30%    | 0.0603   | 97.80%  | ⚠️ Early Stop |

**Best Model:** Epoch 9 (Val Accuracy: 98.04%, Val Loss: 0.0647)

**Test Set Classification Report:**

| Class            | Precision  | Recall     | F1-Score   | Support |
| ---------------- | ---------- | ---------- | ---------- | ------- |
| **Krisis**       | **0.9635** | **0.9814** | **0.9724** | 215     |
| **Tidak Krisis** | **0.9791** | **0.9590** | **0.9689** | 195     |
| **Macro Avg**    | 0.9713     | 0.9702     | 0.9706     | 410     |
| **Weighted Avg** | 0.9709     | 0.9707     | 0.9707     | 410     |

**Overall Test Accuracy:** **97.07%** ✅

**ROC-AUC Score:** **0.9973** (Outstanding discrimination ability)

**Model Strengths:**

- ✅ High recall for crisis detection (98.14%) - minimize false negatives
- ✅ Strong precision (96.35%) - reduce false alarms
- ✅ Consistent performance across epochs
- ✅ Excellent ROC-AUC indicating reliable confidence scores

**Model File:**

- PyTorch: `models/face-expression.pt` (21 MB)
- DirectML: `models/face-expression-directml-v2.pt` (21 MB)
- ONNX: `models/face-expression.onnx` (cross-platform)

---

### **🎯 Pose-Recognition Model Performance**

**Dataset Overview:**

- Total images: 7,237
- Training set: 5,790 images (80%)
- Validation set: 724 images (10%)
- Test set: 723 images (10%)
- Class distribution: Krisis (50.1%), Tidak Krisis (49.9%) - **PERFECTLY BALANCED**

**Training Configuration:**

- Architecture: MobileNetV3-Large + Custom Classifier
- Epochs: 5
- Batch size: 64
- Learning rate: 1.618e-5 (fine-tuning)
- Optimizer: AdamW
- Loss function: CrossEntropyLoss with class weights
- Early stopping: Patience = 3

**Training History (Per Epoch):**

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Status     |
| ----- | ---------- | --------- | -------- | ------- | ---------- |
| 1     | 0.6082     | 79.60%    | 0.5061   | 90.87%  | ✓ Best     |
| 2     | 0.3447     | 93.90%    | 0.2327   | 94.05%  | ✓ Best     |
| 3     | 0.1397     | 96.70%    | 0.1141   | 95.99%  | ✓ Best     |
| 4     | 0.0663     | 98.53%    | 0.0627   | 97.93%  | ✓ Best     |
| 5     | 0.0346     | 99.24%    | 0.0390   | 98.76%  | ✓ **BEST** |

**Best Model:** Epoch 5 (Val Accuracy: 98.76%, Val Loss: 0.0390)

**Test Set Classification Report:**

| Class            | Precision  | Recall     | F1-Score   | Support |
| ---------------- | ---------- | ---------- | ---------- | ------- |
| **Krisis**       | **1.0000** | **0.9828** | **0.9913** | 349     |
| **Tidak Krisis** | **0.9843** | **1.0000** | **0.9921** | 376     |
| **Macro Avg**    | 0.9921     | 0.9914     | 0.9917     | 725     |
| **Weighted Avg** | 0.9919     | 0.9917     | 0.9917     | 725     |

**Overall Test Accuracy:** **99.17%** ✅ (🏆 **OUTSTANDING**)

**ROC-AUC Score:** **0.9994** (Near-perfect discrimination ability)

**Model Strengths:**

- ✅✅ **Perfect precision for crisis detection (100%)** - zero false alarms
- ✅ Excellent recall (98.28%) - catches 98% of crisis cases
- ✅ Perfect specificity (100%) - correctly identifies all non-crisis cases
- ✅ **Faster convergence** (5 epochs vs 10 for Face Model)
- ✅ Larger, more balanced dataset (7,237 vs 4,094)
- ✅ Highest ROC-AUC (0.9994) - extremely reliable confidence scores

**Model File:**

- PyTorch: `models/pose-recognition.pt` (21 MB)
- DirectML: `models/pose-recognition-directml-v2.pt` (21 MB)
- ONNX: `models/pose-recognition.onnx` (cross-platform)

---

### **📈 Comparative Analysis: Face vs Pose Model**

| Metrik                                | Face-Expression    | Pose-Recognition               |
| ------------------------------------- | ------------------ | ------------------------------ | 
| **Test Accuracy**                     | 97.07%             | **99.17%**                     |
| **Precision (Krisis)**                | 96.35%             | **100.00%**                    |
| **Recall (Krisis)**                   | 98.14%             | 98.28%                         |
| **F1-Score (Krisis)**                 | 0.9724             | **0.9913**                     |
| **Specificity (Tidak Krisis Recall)** | 95.90%             | **100.00%**                    |
| **ROC-AUC**                           | 0.9973             | **0.9994**                     |
| **Training Epochs**                   | 10                 | **5**                          |
| **Dataset Size**                      | 4,094              | **7,237**                      |
| **Class Balance**                     | Imbalanced (56:44) | **Perfectly Balanced (50:50)** |

---

### **🎯 Model Selection & Recommendation**

**For Production Deployment:**

**🥇 RECOMMENDED: Pose-Recognition Model**

**Alasan:**

1. **Higher Accuracy (99.17%)** - Best overall classification performance
2. **Perfect Precision (100%)** - Zero false positives (no unnecessary alarms)
3. **Perfect Specificity (100%)** - Catches all non-crisis correctly
4. **Clinical Excellence** - Best precision-recall balance for healthcare
5. **Faster Training** - Converges in 5 epochs (2x lebih cepat)
6. **Larger Dataset** - 77% lebih banyak training data (7,237 vs 4,094)
7. **Better Balanced** - Perfect 50-50 class distribution (tidak perlu weighted sampling)
8. **Privacy-Friendly** - Skeleton keypoints lebih privacy-preserving dibanding wajah

**🥈 Face-Expression Model**

**Strengths:**

- Still excellent (97.07% accuracy)
- Higher recall for crisis (98.14%) - marginally better at catching crisis
- Smaller dataset requirement (4,094 images)

**Use Case:**

- Alternative when facial expressions matter more
- Fusion mode dengan Pose model untuk robustness maksimal

**Fusion Mode (Recommended for Maximum Robustness):**

```
Confidence Score = 0.6 × Pose_Confidence + 0.4 × Face_Confidence
Prediction = argmax(Fusion Score)
```

**Keuntungan Fusion:**

- Combine strengths dari kedua model
- Robust terhadap various head poses dan body orientations
- Better handling of edge cases dan occlusion
- Near-perfect accuracy (~99%+)

---

### **⚙️ Performance Metrics Explanation**

#### **Untuk Crisis Detection (Kelas "Krisis"):**

**Precision: 96.35% (Face) / 100% (Pose)**

- Dari prediksi "KRISIS", berapa % yang benar-benar krisis?
- ✅ Precision tinggi = Reduce false alarms (mental health professional fatigue)
- Artinya: Setiap alarm yang keluar, benar-benar indikasi krisis

**Recall: 98.14% (Face) / 98.28% (Pose)**

- Dari semua kasus krisis sebenarnya, berapa % yang terdeteksi?
- ✅ Recall tinggi = Minimize missed crisis (lives at risk)
- Artinya: Hanya 1-2% crisis cases yang terlewat

**F1-Score: 0.9724 (Face) / 0.9913 (Pose)**

- Harmonic mean Precision & Recall
- ✅ F1-Score tinggi = Balanced performance
- Artinya: Model tidak over-optimize untuk satu metrik

**ROC-AUC: 0.9973 (Face) / 0.9994 (Pose)**

- Area under ROC curve (0-1, max=1.0)
- ✅ ROC-AUC tinggi = Excellent discrimination ability
- Artinya: Model sangat confident dengan prediksinya

#### **Clinical Relevance:**

| Skenario                    | Target   | Face   | Pose   | 
| --------------------------- | -------- | ------ | ------ | 
| **Missed Crisis (FN)**      | Minimize | 1.86%  | 1.72%  |
| **False Alarms (FP)**       | Minimize | 3.65%  | 0%     |
| **Correct Detection (TP)**  | Maximize | 98.14% | 98.28% |
| **Correct Non-Crisis (TN)** | Maximize | 95.90% | 100%   |

---

### **🧪 Validation Strategy**

**Data Splits:**

| Set            | Purpose                               | Size | Accuracy                      |
| -------------- | ------------------------------------- | ---- | ----------------------------- |
| **Training**   | Update model weights                  | 80%  | 99.30% (Face) / 99.24% (Pose) |
| **Validation** | Hyperparameter tuning, early stopping | 10%  | 98.04% (Face) / 98.76% (Pose) |
| **Testing**    | Final unbiased evaluation             | 10%  | 97.07% (Face) / 99.17% (Pose) |

**Validation Techniques:**

- ✅ Random split (stratified by class)
- ✅ Early stopping (patience=3 epochs)
- ✅ Class weights handling
- ✅ WeightedRandomSampler untuk balanced batches
- ✅ No data leakage antara train/val/test

---

#### 🔧 Konfigurasi Lanjutan

**Mengubah Detection Sensitivity:**

```python
# Dalam file run_model.py, modifikasi parameter:

# Face Detection
faces = face_cascade.detectMultiScale(
    scaleFactor=1.13333,  # Sensitivitas scale (1.1-1.3)
    minNeighbors=7,       # Minimum tetangga (5-10)
    minSize=(128, 128)    # Ukuran minimum wajah
)

# Body Detection
bodies = upperbody_cascade.detectMultiScale(
    scaleFactor=1.01618,  # Sensitivitas scale
    minNeighbors=5,       # Minimum tetangga
    minSize=(256, 256)    # Ukuran minimum tubuh
)
```

**Mengubah Fusion Weights:**

```python
# Dalam file run_model.py:
w_face, w_pose = 0.6, 0.4  # Default: 60% wajah, 40% pose
# Contoh alternatif:
# w_face, w_pose = 0.7, 0.3  # Lebih fokus ke wajah
# w_face, w_pose = 0.5, 0.5  # Bobot seimbang
```

#### 🚨 Troubleshooting

**Kamera Tidak Terdeteksi:**

```bash
# Periksa kamera yang tersedia
python -c "import cv2; print([cv2.VideoCapture(i).isOpened() for i in range(5)])"
```

**Model Tidak Ditemukan:**

```bash
# Pastikan file model ada di direktori yang benar
ls models/face-expression.pt
ls models/pose-recognition.pt
```

**Performance Lambat:**

- Gunakan GPU jika tersedia (`CUDA_VISIBLE_DEVICES=0`)
- Turunkan resolusi kamera
- Gunakan mode single (face/pose) bukan fusion

### 🧠 Model Training & Conversion Guide

Panduan lengkap untuk melatih model dari dataset dan mengkonversi ke berbagai format termasuk Hailo HEF.

#### 📚 **Step 1: Data Preparation (Ekstraksi Video ke Gambar)**

**Objective**: Konversi video krisis/non-krisis menjadi gambar wajah dan skeleton pose.

**Gunakan notebook**: `extract_videos.ipynb`

**Proses Ekstraksi Wajah**:

- Deteksi wajah menggunakan MTCNN (confidence: 0.85)
- Filter aspek ratio untuk menghindari wajah yang terdistorsi
- Minimum ukuran wajah: 64×64 piksel
- Normalisasi ke 224×224 piksel
- **Hasil**: ~4,094 face images siap training

**Proses Ekstraksi Pose**:

- Deteksi pose menggunakan MediaPipe (33 keypoint)
- Overlay skeleton dengan visualisasi joints dan connections
- Normalisasi ke 224×224 piksel
- **Hasil**: ~7,237 pose skeleton images siap training

**Output Directory**:

```
data/final/
├── face/
│   ├── krisis/         (2,315 images)
│   └── tidak_krisis/   (1,779 images)
└── pose/
    ├── krisis/         (3,628 images)
    └── tidak_krisis/   (3,609 images)
```

---

#### 🎓 **Step 2: Model Training**

**Objective**: Melatih MobileNetV3 models pada face dan pose datasets.

**Untuk Face Expression Model**:

Gunakan notebook: `face-expression.ipynb`

**Konfigurasi Training**:

- Jumlah epoch: 10 epoch
- Batch size: 64 gambar per batch
- Ukuran gambar input: 224×224 piksel
- Learning rate: 1.618e-5 (fine-tuning pada pretrained model)
- Data split: 80% training, 10% validation, 10% testing

**Dataset Breakdown**:

- Training: 3,275 face images
- Validation: 409 face images
- Testing: 410 face images

**Data Augmentation** (untuk mencegah overfitting):

- Random horizontal flip (memutar gambar secara horizontal)
- Random rotation hingga 90 derajat
- Color jitter untuk simulasi variasi lighting

**Model Architecture**:

- Backbone: MobileNetV3-Large (pretrained pada ImageNet)
- Custom classifier head untuk klasifikasi krisis/tidak krisis

**Output Training**:

- Best model: `models/face-expression-directml-v2.pt`
- Metrik evaluasi: Accuracy, Precision, Recall, F1-score, ROC-AUC

---

**Untuk Pose Recognition Model**:

Gunakan notebook: `pose-recognition.ipynb`

**Konfigurasi Training**:

- Jumlah epoch: 5 epoch (lebih singkat karena dataset lebih besar & seimbang)
- Batch size: 64 gambar per batch
- Learning rate: 1.618e-5
- Data split: 80% training, 10% validation, 10% testing

**Dataset Breakdown**:

- Training: 5,790 pose skeleton images
- Validation: 724 pose skeleton images
- Testing: 723 pose skeleton images
- **Perfect balance**: ~2,895 images per kelas di training set

**Data Augmentation** (spesifik untuk pose):

- Random horizontal flip
- Random vertical flip (unik untuk pose detection)
- Random rotation hingga 90 derajat
- Color jitter

**Model Architecture**:

- Sama dengan Face Model (MobileNetV3-Large)

**Output Training**:

- Best model: `models/pose-recognition-directml-v2.pt`

---

**Untuk Speech/Language Model**:

Gunakan notebook: `listen_respon.ipynb`

**Konfigurasi Training**:

- Algorithm: LightGBM Classifier
- Features: TF-IDF vectorization (1-2 grams, max 10k features)
- Class weight: Balanced (untuk mengatasi imbalanced data)
- Learning rate: 1.618e-2 (Golden ratio)
- Max depth: 7
- N estimators: 300
- Data split: 80% training, 10% validation, 10% testing

**Dataset Sources**:

- Excel file: `data/csv/Data_tanggapan_positif.xlsx`
- Sheet "Krisis": Kalimat-kalimat yang mengindikasikan krisis
- Sheet "Tidak Krisis": Kalimat-kalimat normal/positif
- Total samples: ~1000+ kalimat dalam Bahasa Indonesia

**Text Preprocessing**:

- Lowercase conversion
- Remove special characters (keep alphanumeric + spaces)
- Normalize whitespaces
- Crisis keyword detection (50+ keywords)

**Threshold Optimization**:

- Grid search pada validation set (0.1 - 0.95)
- Optimized untuk high recall (prioritas deteksi krisis)
- Balanced precision untuk mengurangi false positive

**Output Training**:

- Vectorizer: `models/language/vectorizer.pkl`
- Model: `models/language/lgbm_model.pkl`
- Threshold: `models/language/threshold.txt`

---

**Training Workflow**:

1. **Run Data Extraction**

   - Buka `extract_videos.ipynb`
   - Jalankan semua cell untuk mengekstraksi video

2. **Run Face Training**

   - Buka `face-expression.ipynb`
   - Jalankan semua cell untuk melatih face model
   - Output: `models/face-expression-directml-v2.pt`

3. **Run Pose Training**

   - Buka `pose-recognition.ipynb`
   - Jalankan semua cell untuk melatih pose model
   - Output: `models/pose-recognition-directml-v2.pt`

4. **Run Speech Model Training**
   - Buka `listen_respon.ipynb`
   - Pastikan file `data/csv/Data_tanggapan_positif.xlsx` tersedia
   - Jalankan semua cell untuk training text classifier
   - Output: `models/language/` directory dengan 3 files:
     - `vectorizer.pkl` (TF-IDF vectorizer)
     - `lgbm_model.pkl` (LightGBM classifier)
     - `threshold.txt` (optimal threshold)

---

#### 🔄 **Step 3: Model Conversion - PyTorch to ONNX**

**Objective**: Konversi PyTorch models ke ONNX format untuk cross-platform deployment.

**Proses Konversi**:

- Load trained PyTorch model (.pt)
- Set model ke evaluation mode (no training)
- Create dummy input tensor untuk tracing
- Export ke ONNX format dengan dynamic batch size
- Verify ONNX model validity

**Output**:

```
models/onnx/
├── face-expression.onnx        # ~21 MB
└── pose-recognition.onnx       # ~21 MB
```

**Keuntungan ONNX**:

- ✅ Cross-platform compatibility (Windows, Linux, macOS)
- ✅ Framework-agnostic (bukan hanya PyTorch)
- ✅ Hardware acceleration support
- ✅ Standardized format untuk production

---

#### 🎯 **Step 4: ONNX to Hailo Archive (HAR)**

**Objective**: Konversi ONNX ke Hailo Archive format untuk persiapan kompilasi.

**Prerequisites**:

- Install Hailo Dataflow Compiler
- Verifikasi instalasi dengan memeriksa version

**Proses Konversi**:

- Gunakan Hailo CLI tools untuk konversi ONNX ke HAR
- Tersedia dua varian: Standard dan Optimized
- Standard HAR: ukuran normal, untuk testing
- Optimized HAR: ukuran lebih kecil, performa lebih baik (recommended untuk edge devices)

**Optimasi Level**:

- Level 0: No optimization
- Level 1: Light optimization (basic size reduction)
- Level 2: Aggressive optimization (recommended untuk production)

**Output**:

```
models/har/
├── face-expression.har         # Standard variant
├── face-expression-opt.har     # Optimized variant (recommended)
├── pose-recognition.har        # Standard variant
└── pose-recognition-opt.har    # Optimized variant (recommended)
```

**Catatan**: Gunakan optimized variant (-opt.har) untuk hasil terbaik di edge devices seperti Raspberry Pi 5

---

#### ⚡ **Step 5: HAR to Hailo Executable (HEF) - Compilation**

**Objective**: Compile HAR format ke HEF (Hailo Executable Format) untuk eksekusi di Hailo-8L accelerator.

**Prerequisites**:

- Install Hailo Compiler tools
- Hailo device terdeteksi (untuk verification)
- Target platform: Hailo-8L processor

**Proses Kompilasi**:

- Load optimized HAR file
- Compile ke HEF dengan target Hailo-8L
- Apply optimization level 2 untuk performa maksimal
- Optional: Apply 8-bit quantization untuk ukuran lebih kecil

**Output**:

```
models/hef/
├── face-expression/
│   ├── face-expression.hef             # Ready untuk Hailo-8L
│   └── face-expression_compiled.har    # Intermediate file
└── pose-recognition/
    ├── pose-recognition.hef            # Ready untuk Hailo-8L
    └── pose-recognition_compiled.har   # Intermediate file
```

**Catatan**:

- HEF file adalah format final yang ready untuk deployment
- File size biasanya 5-15 MB setelah kompilasi dan quantization
- 8-bit quantization mengurangi akurasi minimal (~1-2%) tapi mengurangi ukuran signifikan

---

#### ✅ **Step 6: Verification & Testing**

**Objective**: Verify HEF file adalah valid dan melakukan inference testing di Hailo-8L device.

**Verifikasi HEF File**:

- Check file integrity dan compatibility
- Validate model structure di dalam HEF
- Ensure quantization parameters correct

**Inference Testing**:

- Run test samples melalui HEF model di Hailo device
- Verify output shape dan data types
- Compare inference latency vs PyTorch model

**Performance Benchmarking**:

- Measure throughput: images per second
- Measure latency: milliseconds per inference
- Power consumption: watts (Hailo-8L ~5W typical)

**Expected Performance Metrics**:

**Face Expression Model**:

- Input: 224×224 RGB image
- Output: 2 classes (krisis, tidak_krisis) dengan confidence scores
- Latency on Hailo-8L: ~5-8ms per image
- Throughput: ~125-200 images/second
- Power: ~5W

**Pose Recognition Model**:

- Input: 224×224 RGB image (skeleton overlay)
- Output: 2 classes dengan confidence scores
- Latency on Hailo-8L: ~5-8ms per image
- Throughput: ~125-200 images/second
- Power: ~5W

**Speech/Language Model**:

- Input: Raw audio → Speech-to-Text → Text preprocessing
- Text Features: TF-IDF vectors (10k max features, 1-2 grams)
- Model: LightGBM classifier (300 estimators)
- Output: 2 classes (krisis, tidak_krisis) dengan confidence scores
- Latency: ~200-500ms per text prediction
- Crisis Keywords: 50+ real-time keyword detection
- Languages: Bahasa Indonesia (primary)

**Deployment Checklist**:

- ✓ HEF files generated successfully
- ✓ File integrity verified
- ✓ Test inference completed
- ✓ Performance benchmarks acceptable
- ✓ Ready untuk production deployment

**Selanjutnya**: HEF files di `models/hef/` siap untuk deployment ke Hailo-8L device atau embedded system (Raspberry Pi 5, Orange Pi, dll)

---

#### 📊 **Complete Model Training & Conversion Workflow**

**Visual Models Pipeline:**

```
┌─────────────────────────────┐
│ 1. Video Data               │  (source: data/2. Data Video/)
│    - Krisis videos          │
│    - Tidak Krisis videos    │
└────────────┬────────────────┘
             │ extract_videos.ipynb
             ▼
┌─────────────────────────────┐
│ 2. Extracted Images         │  (Face: 4,094 | Pose: 7,237)
│    - Face images (224×224)  │
│    - Pose skeleton images   │
└────────────┬────────────────┘
             │ face-expression.ipynb & pose-recognition.ipynb
             ▼
┌─────────────────────────────┐
│ 3. Trained PyTorch Models   │  (.pt format)
│    - face-expression.pt     │
│    - pose-recognition.pt    │
└────────────┬────────────────┘
             │ torch.onnx.export()
             ▼
┌─────────────────────────────┐
│ 4. ONNX Models              │  (cross-platform)
│    - face-expression.onnx   │
│    - pose-recognition.onnx  │
└────────────┬────────────────┘
             │ hailortcli onnx-to-har
             ▼
┌─────────────────────────────┐
│ 5. Hailo Archive (HAR)      │  (intermediate format)
│    - *.har (standard)       │
│    - *-opt.har (optimized)  │
└────────────┬────────────────┘
             │ hailortcli compile
             ▼
┌─────────────────────────────┐
│ 6. Hailo Executable (HEF)   │  (ready for Hailo-8L)
│    - face-expression.hef    │
│    - pose-recognition.hef   │
└─────────────────────────────┘
```

**Speech/Language Model Pipeline:**

```
┌─────────────────────────────┐
│ 1. Text Data Collection     │  (source: data/csv/)
│    - Data_tanggapan_positif │
│    - Krisis & Tidak Krisis  │
└────────────┬────────────────┘
             │ listen_respon.ipynb
             ▼
┌─────────────────────────────┐
│ 2. Text Preprocessing       │  (~1000+ Indonesian sentences)
│    - Lowercasing            │
│    - Special char removal   │
│    - Crisis keyword extract │
└────────────┬────────────────┘
             │ TF-IDF vectorization + LightGBM
             ▼
┌─────────────────────────────┐
│ 3. Trained Language Model   │  (.pkl format)
│    - vectorizer.pkl (TF-IDF)│
│    - lgbm_model.pkl (LightGBM)│
│    - threshold.txt (optimal)│
└─────────────────────────────┘
```

---

### 📓 Menggunakan Notebook untuk Analisis & Training

```bash
# 1. Data Extraction (Video → Images)
jupyter notebook extract_videos.ipynb

# 2. Face Expression Model Training
jupyter notebook face-expression.ipynb

# 3. Pose Recognition Model Training
jupyter notebook pose-recognition.ipynb

# 4. Speech/Language Model Training
jupyter notebook listen_respon.ipynb
```

**Fitur Notebook**:

**Visual Models (`extract_videos.ipynb`, `face-expression.ipynb`, `pose-recognition.ipynb`)**:

- 📊 Visualisasi data training dan distribusi kelas
- 🎓 Training loops dengan progress tracking PyTorch
- 🖼️ Image augmentation dan preprocessing pipeline
- 📈 Real-time loss/accuracy plotting
- 💾 Model checkpoints dan best model saving

**Speech Model (`listen_respon.ipynb`)**:

- 📝 Text preprocessing dan TF-IDF feature extraction
- 🔍 Crisis keyword analysis dan frequency distribution
- ⚖️ Threshold optimization untuk balanced precision/recall
- 📊 Classification report dengan confusion matrix
- 💬 Interactive prediction testing dengan sample texts
- 📈 Evaluasi performa model (accuracy, precision, recall, F1, ROC-AUC)
- 🔍 Analisis confusion matrix
- 🧪 Testing dengan gambar statis
- 💾 Model checkpointing dan early stopping

## 📁 Struktur Proyek

```
├── run_model.py           # Script utama untuk CPU/GPU inference
├── run_model_hailo.py     # Script khusus untuk Hailo AI Hat+ (RPi5)
├── pyproject.toml         # Konfigurasi proyek dan dependencies
├── face-expression.ipynb  # Notebook analisis ekspresi wajah
├── pose-recognition.ipynb # Notebook analisis pose tubuh
├── models/               # Model AI dalam berbagai format
│   ├── face-expression.pt              # PyTorch model (CPU/GPU)
│   ├── face-expression-directml.pt     # DirectML optimized
│   ├── pose-recognition.pt             # PyTorch model (CPU/GPU)
│   ├── pose-recognition-directml.pt    # DirectML optimized
│   ├── onnx/                          # ONNX format (cross-platform)
│   │   ├── face-expression.onnx
│   │   └── pose-recognition.onnx
│   ├── har/                           # Hailo Archive format
│   │   ├── face-expression.har        # Standard HAR
│   │   ├── face-expression-opt.har    # Optimized HAR
│   │   ├── pose-recognition.har       # Standard HAR
│   │   └── pose-recognition-opt.har   # Optimized HAR
│   └── hef/                           # Hailo Executable Format
│       ├── face-expression/
│       │   ├── face-expression.hef    # Compiled for Hailo-8L
│       │   └── face-expression_compiled.har
│       └── pose-recognition/
│           ├── pose-recognition.hef   # Compiled for Hailo-8L
│           └── pose-recognition_compiled.har
├── module/               # Modul utama
│   ├── models.py         # Definisi model AI
│   └── utils.py          # Fungsi utilitas
└── data/                 # Dataset untuk training dan testing
    ├── 1. DATA EKSPRESI WAJAH KRISI BUNUH DIRI/
    └── 2. SKENARIO PSIKOLOGI REMAJA/
```

### 🔍 **Penjelasan Format Model**

| Format       | Extension      | Platform         | Keterangan                                 |
| ------------ | -------------- | ---------------- | ------------------------------------------ |
| **PyTorch**  | `.pt`          | CPU, CUDA GPU    | Format native untuk development & training |
| **DirectML** | `-directml.pt` | Windows DirectML | Optimized untuk GPU Windows                |
| **ONNX**     | `.onnx`        | Universal        | Cross-platform deployment                  |
| **HAR**      | `.har`         | Hailo Platform   | Archive format untuk kompilasi             |
| **HEF**      | `.hef`         | Hailo AI Hat+    | Executable format untuk inference          |

## 🔧 Konfigurasi

Model akan otomatis mendeteksi:

- GPU CUDA (jika tersedia) atau CPU
- Kamera yang tersedia di sistem
- Backend kamera yang optimal

## 📊 Dataset

Proyek ini menggunakan dataset khusus yang berisi:

- Data ekspresi wajah krisis vs tidak krisis
- Skenario psikologi remaja
- Video dan gambar untuk training model

## ⚡ Performance

- **Model**: MobileNetV3 (Large & Small variants)
- **Akurasi**: Optimized untuk deteksi real-time
- **Latency**: Low-latency inference untuk aplikasi real-time|

### 🎯 **Hailo AI Hat+ Advantages**

- ⚡ **26 TOPS** dedicated AI processing
- 🔋 **Power efficient** - hanya 2W untuk AI processing
- 🚀 **Consistent performance** - tidak terpengaruh CPU load
- 🌡️ **Cool operation** - thermal management terintegrasi
- 💰 **Cost effective** - performa tinggi dengan harga terjangkau
- 🔧 **Easy integration** - plug-and-play dengan Raspberry Pi 5

## 🤝 Kontribusi

1. Fork repository ini
2. Buat branch fitur (`git checkout -b fitur-baru`)
3. Commit perubahan (`git commit -am 'Menambah fitur baru'`)
4. Push ke branch (`git push origin fitur-baru`)
5. Buat Pull Request

## 📝 Lisensi & Hak Cipta

### 🔒 Hak Cipta Program Komputer

Program komputer ini dilindungi oleh **Hak Cipta Program Komputer** sesuai dengan:

- **UU No. 28 Tahun 2014** tentang Hak Cipta (Indonesia)
- **Pasal 40 ayat (1) huruf r** - Program Komputer
- **Direktorat Jenderal Kekayaan Intelektual (DJKI)** - Kementerian Hukum dan HAM RI

**Informasi Hak Cipta:**

```
© 2025 Robot Pencegah Bunuh Diri
Sistem Deteksi Krisis Berbasis AI - Ekspresi Wajah & Pose
Terdaftar sebagai Program Komputer di Indonesia
```

### 📄 Apache License 2.0

Proyek ini dilisensikan di bawah [Apache License 2.0](LICENSE) - lihat file [LICENSE](LICENSE) untuk detail lengkap.

### 🎯 Tujuan & Penggunaan

Proyek ini dikembangkan untuk tujuan penelitian dan edukasi dalam pencegahan bunuh diri remaja, dengan harapan dapat memberikan manfaat maksimal bagi masyarakat, institusi pendidikan, dan peneliti di bidang kesehatan mental.

---

**⚠️ Disclaimer**: Sistem ini merupakan alat bantu deteksi dini dan tidak menggantikan konsultasi profesional dengan psikolog atau psikiater.
