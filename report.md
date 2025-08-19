# Rancangan Agen Cerdas Deteksi Hoaks Media Sosial

## Latar Belakang

Munculnya hoaks di media sosial seperti Twitter dan Reddit dapat mempengaruhi
opini publik dan menimbulkan disinformasi. Untuk membantu mengidentifikasi
konten yang meragukan, diperlukan sebuah agen cerdas yang mampu mengumpulkan
data dari berbagai platform, melakukan klasifikasi hoaks menggunakan teknik
NLP (Natural Language Processing), melakukan verifikasi melalui sumber fakta
terpercaya, serta menyajikan hasilnya dalam sebuah dasbor yang mudah
dianalisis.

## Arsitektur Sistem

Arsitektur agen cerdas yang diusulkan terdiri dari beberapa komponen yang
berinteraksi sebagai berikut:

1. **Scraper** – modul ini bertugas mengumpulkan postingan dari media sosial.
   - *TwitterScraper* menggunakan `snscrape` untuk mencari tweet yang
     mengandung kata kunci tertentu. `snscrape` mengakses data publik
     sehingga tidak memerlukan API key Twitter.
   - *RedditScraper* menggunakan `PRAW`, sebuah wrapper Python untuk API
     Reddit. Untuk menggunakannya diperlukan kredensial aplikasi Reddit
     (client ID, client secret dan user agent). Kata kunci dikirim ke
     metode `subreddit("all").search()` untuk mencari kiriman terbaru.

2. **Klasifikator** – modul ini menerapkan model transformer HuggingFace
   (misalnya `bert‑base‑uncased` atau `roberta‑base`) untuk memprediksi
   apakah sebuah teks merupakan hoaks. Model dimuat ke dalam pipeline
   `text‑classification` dan menghasilkan label (`hoax`/`not_hoax`) beserta
   skor keyakinan. Anda bisa mengganti model dengan yang sudah difine
   (fine‑tuned) khusus untuk deteksi hoaks.

3. **Fakta Checker (opsional)** – modul ini terintegrasi dengan Google
   Fact Check Tools API. API ini menyediakan endpoint `/v1alpha1/claims:search`
   yang menggunakan metode `GET` untuk mencari klaim yang telah di‑fact
   check. Endpoint menerima parameter seperti `query` (teks klaim),
   `languageCode`, `maxAgeDays` dan `pageSize`, lalu mengembalikan daftar
   objek `Claim`【853238102903704†L85-L145】. Setiap objek `Claim` memiliki
   properti `claimReview` yang berisi satu atau lebih review, dengan informasi
   seperti URL, judul, tanggal review dan rating tekstual (contoh: "Mostly
   false")【763788042889412†L84-L178】. Dengan API key Google Anda dapat
   mengirim permintaan ke endpoint ini untuk memeriksa apakah isi postingan
   sudah pernah diverifikasi oleh lembaga pemeriksa fakta.

4. **Database** – semua hasil scraping, label prediksi dan informasi
   fact‑checking disimpan ke dalam database relational (SQLite sebagai
   default, namun dapat diganti PostgreSQL). Tabel `posts` memuat kolom
   seperti `platform`, `keyword`, `content`, `url`, `created_at`,
   `predicted_label`, `prediction_score`, `fact_check_url`,
   `fact_check_rating` dan `fact_check_publisher`.

5. **Scheduler** – untuk menjalankan proses scraping, klasifikasi dan
   verifikasi secara berkala, modul ini menggunakan pustaka `schedule`. Anda
   dapat menjadwalkan job setiap hari pada jam tertentu (misalnya 02:00
   waktu Asia/Jakarta). Modul ini juga menyediakan opsi menjalankan proses
   sekali saja.

6. **Dashboard** – antarmuka berbasis `Streamlit` menampilkan data hasil
   pemrosesan. Pengguna dapat memfilter berdasarkan platform, kata kunci,
   label prediksi dan rentang tanggal. Statistik agregat (misalnya jumlah
   total postingan, jumlah hoaks vs bukan hoaks) dan visualisasi
   distribusi label per waktu juga tersedia. Pengguna dapat mengklik
   tautan fact check untuk melihat artikel verifikasi.

Diagram alur sederhana:

```
         +----------------------+      +--------------------+     +----------------+
Keywords → TwitterScraper ----+  →  +                        
         +----------------------+      |   NewsClassifier    |  →  + Database      |
                                    +--------------------+     +----------------+
Keywords → RedditScraper -----+           |                    
                                    +--------------------+     +----------------+
                                    |    FactChecker      |     |  Streamlit     |
                                    +--------------------+     |   Dashboard    |
                                                      \        +----------------+
                                                       \
                                                        + Scheduler (daily)
```

## Implementasi Dasar

Kode Python berikut (tersedia dalam berkas `social_media_agent.py`) mengatur
semua komponen yang dijelaskan di atas. Berikut ringkasan fungsi
utama:

* **Scraping** – kelas `TwitterScraper` dan `RedditScraper` masing‑masing
  memuat fungsi `fetch()` yang mengembalikan daftar objek `Post`.
* **Klasifikasi** – kelas `NewsClassifier` memuat model Transformer dan
  menyediakan metode `classify(text)` untuk menentukan label dan skor.
* **Fact Checking** – kelas `FactChecker` melakukan permintaan HTTP ke
  endpoint Google Fact Check Tools API menggunakan parameter `query` sesuai
  isi postingan. API ini mengembalikan daftar klaim yang sudah diperiksa,
  lengkap dengan informasi publisher, URL dan rating【853238102903704†L85-L145】.
* **Penyimpanan** – kelas `Database` menggunakan SQLAlchemy untuk membuat
  tabel dan menyimpan data. Metode `insert_posts(posts)` menyisipkan
  baris baru sambil menghindari duplikasi berdasarkan URL.
* **Scheduler** – kelas `Scheduler` mengatur jalannya proses scraping,
  klasifikasi, fact checking, dan penyimpanan. Metode `run_job()` dapat
  dipanggil langsung ataupun dijadwalkan secara periodik menggunakan
  parameter `--daily` saat menjalankan script.

```bash
# Contoh menjalankan proses sekali (tanpa fact checking):
python social_media_agent.py --once --no-fact-check --db sqlite:///data.db

# Menjalankan proses setiap hari pukul 02:00 (zona Asia/Jakarta):
python social_media_agent.py --daily --time 02:00 --db sqlite:///data.db

# Menjalankan dasbor Streamlit:
streamlit run dashboard.py -- --db sqlite:///data.db
```

## Catatan Penggunaan

1. **Kredensial Reddit dan API Fact Check** – pastikan Anda mendaftarkan
   aplikasi di Reddit untuk mendapatkan `client_id`, `client_secret`, dan
   `user_agent`. Variabel ini dapat diset melalui variabel lingkungan
   `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, dan `REDDIT_USER_AGENT`.
   Untuk mengaktifkan integrasi Fact Check Tools API, buat API key melalui
   Google Cloud Console, aktifkan Fact Check Tools API, dan isi variabel
   lingkungan `FACT_CHECK_API_KEY`.

2. **Pembatasan Platform** – `snscrape` hanya dapat mengumpulkan tweet
   publik. Akun yang dilindungi atau diblokir tidak dapat diakses. PRAW
   juga mematuhi aturan rate limit Reddit; perhatikan batasan ini agar
   akun Anda tidak diblokir.

3. **Model NLP** – contoh menggunakan model dasar BERT (`bert‑base‑uncased`).
   Untuk performa lebih baik, sebaiknya melatih (fine‑tuning) model
   khusus untuk deteksi hoaks atau menggunakan dataset berbahasa Indonesia.

4. **Legalitas dan Etika** – pastikan penggunaan data dari media sosial
   mengikuti ketentuan layanan masing‑masing platform dan menjaga privasi
   pengguna. Gunakan sistem ini untuk keperluan penelitian atau edukasi.

## Kesimpulan

Agen cerdas ini menawarkan fondasi yang dapat dikembangkan lebih lanjut
untuk mendeteksi dan memverifikasi informasi palsu di media sosial. Dengan
memanfaatkan scraping otomatis, klasifikasi berbasis NLP, dan verifikasi
melalui sumber fakta terpercaya, pengguna dapat memonitor peredaran hoaks
secara terstruktur dan memperoleh wawasan melalui dasbor interaktif.

### Rujukan API

* Google Fact Check Tools API – endpoint `claims:search` menerima
  parameter `query`, `languageCode`, `maxAgeDays` dan `pageSize` via HTTP
  GET. Respons berisi array `claims` dengan informasi klaim serta
  `claimReview` seperti URL, judul, tanggal review dan rating tekstual【853238102903704†L85-L145】【763788042889412†L84-L178】.
