# 🤖 Robot Pencegah Bunuh Diri

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

### Menjalankan Sistem Deteksi

```bash
python run_model.py
```

### Menggunakan Notebook untuk Analisis

```bash
# Face Expression Analysis
jupyter notebook face-expression.ipynb

# Pose Recognition Analysis
jupyter notebook pose-recognition.ipynb
```

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

## 📝 Lisensi

Proyek ini dikembangkan untuk tujuan penelitian dan edukasi dalam pencegahan bunuh diri remaja.

---

**⚠️ Disclaimer**: Sistem ini merupakan alat bantu deteksi dini dan tidak menggantikan konsultasi profesional dengan psikolog atau psikiater.
