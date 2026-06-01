# Python-Game

Nama : Muhammad Arifin Umasangadji
NRP  : 5024241083

##  Deskripsi Project

Space Field merupakan game berbasis python bertema space shooter yang dikendalikan menggunakan gerakan tangan secara real time melalui webcam. Pemain mengontrol pesawat luar angkasa dengan menggerakkan tangan di depan kamera dan menembakkan peluru menggunakan gesture tertentu.

Proyek ini dikembangkan menggunakan Python dan OpenCV. Sistem deteksi tangan memanfaatkan segmentasi warna kulit (skin color detection) pada ruang warna HSV untuk melacak posisi tangan pemain. Hasil deteksi digunakan untuk mengontrol pergerakan pesawat dan menghasilkan aksi menembak.

---

## 🎮 Gameplay

Pemain mengontrol pesawat luar angkasa menggunakan tangan yang terdeteksi oleh kamera.

### Mekanisme Permainan

* Gerakan tangan mengontrol posisi pesawat.
* Gesture tangan digunakan untuk menembakkan peluru.
* Pemain harus menghancurkan meteor dan pesawat musuh.
* Setiap objek yang berhasil dihancurkan akan menambah skor.
* Pemain memiliki 3 nyawa.
* Permainan berakhir ketika seluruh nyawa habis.

### Sistem Skor

| Objek         | Poin |
| ------------- | ---- |
| Meteor Kecil  | +10  |
| Mini Meteor   | +15  |
| Pesawat Musuh | +30  |
| Meteor Besar  | +50  |

---

## 🖥️ Fitur Utama

### 1. Hand Tracking

Sistem mendeteksi posisi tangan menggunakan webcam secara real-time.

* Webcam Capture menggunakan OpenCV.
* Region of Interest (ROI) untuk area deteksi tangan.
* Tracking berdasarkan contour terbesar.

### 2. Gesture Detection

Gesture menembak dilakukan dengan mendeteksi jarak antara titik pusat tangan dan titik tertinggi contour.

Jika jarak melebihi threshold tertentu:

```python
if dist > 85:
    bullets.append([plane_x, plane_y - 30])
```

Maka peluru akan ditembakkan.

### 3. Skin Color Segmentation

Deteksi tangan dilakukan menggunakan HSV Color Space.

```python
hsv = cv.cvtColor(roi_frame, cv.COLOR_BGR2HSV)

mask_skin = cv.inRange(
    hsv,
    np.array([0,48,80]),
    np.array([20,255,255])
)
```

### 4. Second Object

Game menggunakan berbagai objek interaktif:

* Pesawat pemain
* Meteor kecil
* Meteor besar
* Mini meteor
* Pesawat musuh
* Peluru pemain
* Peluru musuh

### 5. Score System

Sistem skor bertambah setiap kali pemain menghancurkan objek tertentu.

### 6. Collision Detection

Menggunakan perhitungan jarak Euclidean antara objek.

```python
distance = np.sqrt(
    (x1 - x2)**2 +
    (y1 - y2)**2
)
```

### 7. Sprite Overlay

Objek PNG transparan ditampilkan menggunakan alpha blending manual.

Fitur ini digunakan pada:

* Pesawat pemain
* Pesawat musuh
* Meteor
* Ikon nyawa

### 8. Background Music

Game dilengkapi dengan background music:

* MENU.wav → Main Menu
* INGAME.wav → Gameplay

Audio diputar menggunakan library bawaan Windows:

```python
winsound.PlaySound(...)
```

---

## 🛠️ Teknologi yang Digunakan

* Python 3.x
* OpenCV
* NumPy
* Pillow (PIL)
* WinSound

---

## 📂 Struktur Folder

```text
Space-Field-Adventure/
│
├── main.py
│
├── assets/
│   ├── pesawat.png
│   ├── Enemy.png
│   ├── meteorite.png
│   ├── METEOR.png
│   ├── earth.png
│   ├── space.jpg
│   ├── MENU.wav
│   ├── INGAME.wav
│   └── PixelifySans-VariableFont_wght.ttf
│
├── screenshots/
│   ├── menu.png
│   ├── gameplay.png
│   └── gameover.png
│
└── README.md
```

---

## ▶️ Cara Menjalankan Program

### 1. Clone Repository

```bash
git clone https://github.com/username/Space-Field-Adventure.git
```

### 2. Install Dependency

```bash
pip install opencv-python numpy pillow
```

### 3. Jalankan Program

```bash
python main.py
```

---

## 🎹 Kontrol

| Tombol | Fungsi          |
| ------ | --------------- |
| Enter  | Mulai Game      |
| Space  | Kembali ke Menu |
| Q      | Keluar Game     |

### Kontrol Tangan

* Gerakkan tangan → Menggerakkan pesawat.
* Gesture jari terangkat → Menembakkan peluru.

---

## 📸 Screenshot

### Main Menu

Tambahkan screenshot menu di sini.

### Gameplay

Tambahkan screenshot gameplay di sini.

### Game Over

Tambahkan screenshot game over di sini.

---

## 🎥 Video Demonstrasi

Tambahkan link video demonstrasi:

```text
https://youtu.be/link-video-demo
```

---

## 📋 Kesesuaian dengan Ketentuan Project

| Ketentuan             | Status |
| --------------------- | ------ |
| Webcam Input          | ✅      |
| OpenCV VideoCapture   | ✅      |
| HSV Skin Detection    | ✅      |
| Gesture Detection     | ✅      |
| Second Object         | ✅      |
| Scoring System        | ✅      |
| Real-Time Tracking    | ✅      |
| Alpha Blending Sprite | ✅      |
| Publikasi GitHub      | ✅      |
| README.md             | ✅      |

---

## 👨‍💻 Anggota Kelompok

* Muhammad Arifin Umasangadji
* Nama Anggota 2
* Nama Anggota 3

---

## 📄 Lisensi

Proyek ini dikembangkan untuk memenuhi tugas Project Pengolahan Citra dan Visi Komputer (PCV).

