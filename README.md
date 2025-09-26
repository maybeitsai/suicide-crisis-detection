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

# Jalankan sistem deteksi
python run_model.py

# Kontrol keyboard:
F = Face mode | P = Pose mode | O = Fusion mode | Q = Quit
```

## 🛠️ Teknologi yang Digunakan

- **Python 3.9+**
- **PyTorch** - Deep Learning Framework
- **OpenCV** - Computer Vision
- **Torchvision** - Pre-trained Models
- **UV Package Manager** - Dependency Management

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

## 🎯 Cara Penggunaan

### 🚀 Menjalankan Sistem Deteksi Real-time

#### Langkah Dasar

```bash
python run_model.py
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

- **CPU**: Intel i5 atau AMD Ryzen 5 (minimum)
- **RAM**: 8GB (minimum), 16GB (recommended)
- **GPU**: CUDA-compatible (optional, untuk performa lebih cepat)
- **Camera**: Webcam atau kamera eksternal (minimum 720p)

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
├── run_model.py           # Script utama untuk menjalankan sistem
├── pyproject.toml         # Konfigurasi proyek dan dependencies
├── face-expression.ipynb  # Notebook analisis ekspresi wajah
├── pose-recognition.ipynb # Notebook analisis pose tubuh
├── models/               # Model AI yang telah dilatih
│   ├── face-expression.pt
│   └── pose-recognition.pt
├── module/               # Modul utama
│   ├── models.py         # Definisi model AI
│   └── utils.py          # Fungsi utilitas
└── data/                 # Dataset untuk training dan testing
    ├── 1. DATA EKSPRESI WAJAH KRISI BUNUH DIRI/
    └── 2. SKENARIO PSIKOLOGI REMAJA/
```

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
- **Latency**: Low-latency inference untuk aplikasi real-time

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
