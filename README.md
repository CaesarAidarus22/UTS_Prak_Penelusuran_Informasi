# 🧠 UTS Praktikum Penelusuran Informasi  
### *CLI-Based Multi-Dataset Information Retrieval System*

---

## 📌 Deskripsi Proyek
Proyek ini merupakan tugas **UTS Praktikum Penelusuran Informasi (PI)** yang bertujuan untuk membangun **sistem Information Retrieval (IR)** berbasis *Command-Line Interface (CLI)*.  
Sistem ini mampu melakukan **pencarian dan perankingan dokumen** dari berbagai sumber teks nyata menggunakan **Vector Space Model** dan **Cosine Similarity**.

---

## 👨‍💻 Anggota Kelompok
| No | Nama Anggota | NIM |
|----|---------------|-----|
| 1 | Caesar Aidarus | 2308107010072 |
| 2 | Muhammad Nazlul Ramadhyan | 2308107010036 |
| 3 | Naufal Farel Syafilan | 2308107010058 |

---

## ⚙️ Fitur Sistem
1. **Load & Index Dataset**  
   Sistem membaca semua file teks dari lima sumber dataset (`etd-ugm`, `etd-usk`, `kompas`, `tempo`, `mojok`) dan melakukan preprocessing (case folding, tokenisasi, dan stopword removal).

2. **Search Query**  
   Pengguna dapat memasukkan kata kunci (query). Sistem akan menghitung *cosine similarity* antara query dan setiap dokumen, lalu menampilkan **5 dokumen paling relevan** dengan skor tertinggi.

3. **Exit Program**  
   Menutup program CLI.

---

## 📂 Struktur Folder
UTS_Praktikum_PI/
│
├── main.py
├── README.md
├── requirements.txt
│
├── datasets/
│ ├── etd-ugm/
│ ├── etd-usk/
│ ├── kompas/
│ ├── tempo/
│ └── mojok/
│
└── modules/
├── preprocessing.py
├── vectorization.py
├── search.py
├── indexing.py
└── utils.py


---

## 🧰 Library yang Digunakan
| Library | Kegunaan |
|----------|-----------|
| `pandas` | Pengolahan data dan pembacaan teks |
| `scikit-learn` | `CountVectorizer` dan `cosine_similarity` |
| `whoosh` | Indexing dokumen (opsional sesuai instruksi) |
| `re` | Pembersihan teks |
| `os` | Navigasi folder dan file |
| `math` | Operasi numerik tambahan |

---

## 🧹 Tahapan Pemrosesan
1. **Preprocessing**
   - Case Folding (huruf kecil semua)
   - Tokenization
   - Stopword Removal  
   - (Opsional) Stemming / Lemmatization

2. **Vectorization**
   - Representasi dokumen menggunakan *Bag of Words (BoW)*  
   - Menghasilkan matriks dokumen-term

3. **Cosine Similarity**
   - Menghitung kemiripan antara query dan setiap dokumen  
   - Menampilkan skor dan ranking hasil pencarian

---

## 💻 Cara Menjalankan Program
1. Pastikan semua dependensi sudah terpasang:
   ```bash
   pip install -r requirements.txt

2. Jalankan program utama:
    python main.py

3. Pilih menu:
    [1] Load & Index Dataset
    [2] Search Query
    [3] Exit

4. Contoh penggunaan:
    Pilih menu: 1
    10 dokumen berhasil dimuat dan diproses.

    Pilih menu: 2
    Masukkan query: pangan

    === Hasil Pencarian ===
    Doc 4 | Skor: 0.3363
    Doc 1 | Skor: 0.1208
    ...




