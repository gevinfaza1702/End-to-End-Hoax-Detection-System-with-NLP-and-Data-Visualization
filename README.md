# End-to-End Hoax Detection System with NLP and Data Visualization

Sebuah sistem *pipeline* data otomatis yang dirancang untuk mengumpulkan, mengklasifikasikan, memverifikasi, dan memvisualisasikan potensi hoaks dari berbagai sumber berita online. Proyek ini dibangun untuk membantu menganalisis dan memahami pola penyebaran disinformasi secara *end-to-end*, dengan fokus pada artikel berita dari Google News.

## Key Features

-   **Automated News Scraping**: Mengumpulkan artikel berita secara otomatis dari Google News berdasarkan kata kunci yang relevan.
-   **NLP-Based Classification**: Menggunakan model HuggingFace Transformers (BERT) untuk mengklasifikasikan konten sebagai 'hoaks' atau 'bukan hoaks'.
-   **External Fact-Checking**: Terintegrasi dengan Google Fact Check Tools API untuk validasi dan pengayaan data.
-   **Automated Pipeline**: Dilengkapi dengan *scheduler* untuk menjalankan seluruh proses pengumpulan dan analisis data secara periodik.
-   **Interactive Dashboard**: Menyajikan hasil analisis dalam dasbor interaktif yang dibangun dengan Streamlit, memungkinkan eksplorasi data yang mudah.

## Arsitektur Sistem

Berikut adalah diagram alur sederhana dari cara kerja sistem yang sudah difokuskan:

```
         +----------------------+      +--------------------+     +----------------+
Keywords → GoogleNewsScraper --+  →  +   NewsClassifier    |  →  +    Database    |
         +----------------------+      +--------------------+     +----------------+
                                                      |                     |
                                          +--------------------+     +----------------+
                                          |    FactChecker     |     |    Streamlit   |
                                          +--------------------+     |    Dashboard   |
                                                            \        +----------------+
                                                             \
                                                              + Scheduler (daily)
```

## Teknologi yang Digunakan

-   **Backend & Data Processing**: Python, Pandas
-   **NLP/Machine Learning**: HuggingFace Transformers
-   **Data Storage**: SQLAlchemy, SQLite (dapat dikonfigurasi untuk PostgreSQL)
-   **API & Scraper**: gnews, Requests
-   **Dashboard & Visualization**: Streamlit
-   **Automation**: Schedule
-   **Environment Management**: python-dotenv

## Instalasi & Konfigurasi

Untuk menjalankan proyek ini secara lokal, ikuti langkah-langkah berikut:

1.  **Clone repository ini:**
    ```bash
    git clone [https://github.com/gevinfaza1702/End-to-End-Hoax-Detection-System-with-NLP-and-Data-Visualization.git](https://github.com/gevinfaza1702/End-to-End-Hoax-Detection-System-with-NLP-and-Data-Visualization.git)
    cd End-to-End-Hoax-Detection-System-with-NLP-and-Data-Visualization
    ```

2.  **Buat dan aktifkan virtual environment:**
    ```bash
    # Untuk Windows
    python -m venv venv
    .\venv\Scripts\activate

    # Untuk macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install semua dependensi:**
    *(Pastikan Anda sudah membuat file `requirements.txt` dengan menjalankan `pip freeze > requirements.txt` di terminal Anda)*
    ```bash
    pip install -r requirements.txt
    ```

4.  **Konfigurasi Environment Variables:**
    Buat file bernama `.env` di direktori utama proyek dan isi dengan API Key Anda.

    ```env
    # Google Fact Check API Key
    FACT_CHECK_API_KEY="AIzaSy...Your...Key..."
    ```
    **PENTING:** Pastikan file `.env` sudah Anda tambahkan ke dalam `.gitignore` agar kredensial Anda tidak bocor ke publik!

## Penggunaan

Anda bisa menjalankan agen untuk scraping dan klasifikasi, serta meluncurkan dasbor secara terpisah.

1.  **Menjalankan Agen (Satu Kali):**
    Perintah ini akan menjalankan seluruh pipeline satu kali dan kemudian berhenti.
    ```bash
    python social_media_agent.py --once --fact-check --source google
    ```

2.  **Menjalankan Agen (Terjadwal Harian):**
    Perintah ini akan menjalankan pipeline setiap hari pada waktu yang ditentukan (default: 02:00).
    ```bash
    python social_media_agent.py --daily --fact-check --source google --time 02:00
    ```

3.  **Menjalankan Dasbor Streamlit:**
    Pastikan database (`data.db`) sudah terisi setelah menjalankan agen.
    ```bash
    streamlit run dashboard.py
    ```
