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

- ✅ Deteksi real-time melalui kamera
- ✅ Model AI berbasis MobileNetV3 (optimized untuk performa)
- ✅ Analisis ekspresi wajah dan pose tubuh
- ✅ Sistem klasifikasi krisis/tidak krisis
- ✅ Interface yang mudah digunakan
- ✅ 3 mode deteksi: Face, Pose, dan Fusion
- ✅ Kontrol keyboard interaktif
- ✅ MediaPipe pose estimation terintegrasi

## ⚡ Quick Start

```bash
# Clone dan masuk ke direktori
git clone https://github.com/maybeitsai/suicide-crisis-detection.git
cd "suicide-crisis-detection"

# Install dependencies dengan UV
uv sync

# Jalankan sistem deteksi (pilih sesuai platform):
python run_model.py        # CPU/GPU (Windows/Linux/macOS)
python run_model_hailo.py  # Hailo AI Hat+ (Raspberry Pi 5)

# Kontrol keyboard:
F = Face mode | P = Pose mode | O = Fusion mode | Q = Quit
```

## 🛠️ Teknologi yang Digunakan

- **Python 3.9+**
- **PyTorch** - Deep Learning Framework
- **OpenCV** - Computer Vision
- **Torchvision** - Pre-trained Models
- **UV Package Manager** - Dependency Management
- **ONNX Runtime** - Cross-platform ML inference
- **Hailo SDK** - Hardware acceleration untuk Hailo AI processors
- **MediaPipe** - Pose estimation dan computer vision

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

#### 🎮 Kontrol Keyboard Interaktif

Setelah menjalankan program, gunakan keyboard untuk mengontrol mode deteksi:

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

- **Face Model**: MobileNetV3-Large + Custom Classifier
- **Pose Model**: MobileNetV3-Large + Custom Classifier
- **Input Size**:
  - Face: 48x48 RGB
  - Pose: 224x224 RGB
- **Output Classes**: 2 (KRISIS, TIDAK KRISIS)

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

### 📓 Menggunakan Notebook untuk Analisis

```bash
# Face Expression Analysis
jupyter notebook face-expression.ipynb

# Pose Recognition Analysis
jupyter notebook pose-recognition.ipynb
```

**Fitur Notebook:**

- Visualisasi data training
- Evaluasi performa model
- Analisis confusion matrix
- Testing dengan gambar statis

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
