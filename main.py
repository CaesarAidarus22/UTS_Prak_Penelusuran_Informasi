import os
import pandas as pd
from modules.preprocessing import preprocess_text
from modules.vectorization import vectorize_documents
from modules.search import search_query
from modules.indexing import create_index  # <- panggil Whoosh index

DATA_DIR = "datasets"   # folder berisi *.csv dari asisten lab
TOP_K = 10              # jumlah hasil yang ditampilkan

# Kolom-kolom umum untuk konten & judul (judul tidak ditampilkan, hanya dipakai untuk penggabungan teks)
CANDIDATE_CONTENT_COLS = [
    "text", "content", "abstract", "abstrak", "isi", "body", "article",
    "judul_dan_isi", "clean_text", "dokumen", "teks"
]
CANDIDATE_TITLE_COLS = ["title", "judul", "headline"]

def pick_first_existing(cands, df):
    for c in cands:
        if c in df.columns:
            return c
    return None

def combine_title_content(row, title_col, content_col):
    parts = []
    if title_col and pd.notna(row.get(title_col, None)):
        parts.append(str(row[title_col]))
    if content_col and pd.notna(row.get(content_col, None)):
        parts.append(str(row[content_col]))
    return " - ".join(parts) if parts else ""

def load_documents_from_csvs():
    """
    Baca semua CSV di DATA_DIR.
    Return:
      docs : list[str] (teks sudah di-preprocess)
      meta : list[dict] (dataset, file, row_id)
    """
    docs, meta = [], []
    if not os.path.isdir(DATA_DIR):
        print(f'Folder "{DATA_DIR}" tidak ditemukan.')
        return docs, meta

    csv_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
    if not csv_files:
        print(f"Tidak ada file .csv di folder {DATA_DIR}/")
        return docs, meta

    for csv_name in sorted(csv_files):
        fpath = os.path.join(DATA_DIR, csv_name)
        dataset_name = os.path.splitext(csv_name)[0]  # contoh: 'kompas', 'etd_ugm'
        try:
            try:
                df = pd.read_csv(fpath)
            except UnicodeDecodeError:
                df = pd.read_csv(fpath, encoding="latin-1")

            df.columns = [c.strip().lower() for c in df.columns]
            title_col = pick_first_existing(CANDIDATE_TITLE_COLS, df)
            content_col = pick_first_existing(CANDIDATE_CONTENT_COLS, df)

            if not title_col and not content_col:
                # fallback: gabungkan semua kolom per baris
                for i, row in df.iterrows():
                    row_text = " ".join([str(v) for v in row.values if pd.notna(v)]).strip()
                    if not row_text:
                        continue
                    docs.append(preprocess_text(row_text))
                    meta.append({"dataset": dataset_name, "file": csv_name, "row_id": i})
            else:
                for i, row in df.iterrows():
                    raw = combine_title_content(row, title_col, content_col).strip()
                    if not raw:
                        continue
                    docs.append(preprocess_text(raw))
                    meta.append({"dataset": dataset_name, "file": csv_name, "row_id": i})

        except Exception as e:
            print(f"Gagal memproses {csv_name}: {e}")

    return docs, meta

def main():
    print("=== INFORMATION RETRIEVAL SYSTEM ===")
    print("[1] Load & Index Dataset (CSV)")
    print("[2] Search Query")
    print("[3] Exit")
    print("====================================")

    docs, meta = [], []
    vectorizer = None

    while True:
        choice = input("Pilih menu: ").strip()

        if choice == "1":
            docs, meta = load_documents_from_csvs()
            if not docs:
                print("Dokumen tidak ditemukan / kosong.")
                continue
            _, vectorizer = vectorize_documents(docs)
            print(f"{len(docs)} dokumen berhasil dimuat dan diproses dari CSV.")
            # Buat index Whoosh (memenuhi syarat tugas)
            create_index("whoosh_index", docs)
            print()

        elif choice == "2":
            if not docs or vectorizer is None:
                print("Dataset belum dimuat. Pilih [1] dulu.")
                continue
            query = input("Masukkan query: ").strip()
            qproc = preprocess_text(query)

            # Ambil Top-K hasil
            # (pastikan modules/search.py TIDAK memangkas di dalam; jangan ada '[:5]' di sana)
            results = search_query(qproc, docs, vectorizer)[:TOP_K]

            print(f"\n=== Hasil Pencarian (Top-{TOP_K}) ===")
            if not results:
                print("Tidak ada hasil.")
            for rank, (idx, score) in enumerate(results, start=1):
                ds = meta[idx]["dataset"]
                rid = meta[idx]["row_id"]
                print(f"{rank}. [{ds}] row#{rid} | Skor: {score:.4f}")
            print("================================\n")

        elif choice == "3":
            print("Udah Keluar Nichh...")
            break
        else:
            print("Pilihan tidak valid.")

if __name__ == "__main__":
    main()
