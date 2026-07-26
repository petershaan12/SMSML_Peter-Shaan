# Monitoring dan Logging

## Menjalankan seluruh layanan

Model dan preprocessor harus tersedia di folder `model/`. Setelah menjalankan `modelling_tuning.py`, salin artefaknya:

```bash
cp ../Membangun_model/artifacts/best_model.joblib model/
cp ../Membangun_model/breast_cancer_preprocessing/{preprocessor.joblib,metadata.json} model/
docker compose up --build
```

Layanan yang tersedia:

- API dan Swagger: <http://127.0.0.1:8000/docs>
- Prometheus: <http://127.0.0.1:9090>
- Grafana: <http://127.0.0.1:3000> (`admin` / `admin`)

Kirim trafik setelah layanan aktif:

```bash
python 7.inference.py --count 100
```

Dashboard Grafana **petershaan12 — ML Monitoring** menyediakan 12 panel. Tiga alert yang diprovisikan adalah model tidak termuat, latensi inferensi tinggi, dan input drift. Tambahkan contact point Grafana secara manual agar notifikasi dapat dikirim ke kanal pilihan Anda.

## Bukti submission

Simpan tangkapan layar ke folder bernomor yang telah disediakan. Bukti tidak dibuat secara otomatis karena harus memperlihatkan layanan yang benar-benar berjalan dan notifikasi yang benar-benar terkirim.

- `1.bukti_serving.jpg`: halaman Swagger `/docs` dan respons sukses dari `/predict`.
- `4.bukti monitoring Prometheus/`: minimal tiga screenshot query metrik berbeda.
- `5.bukti monitoring Grafana/`: screenshot dashboard yang membuktikan minimal sepuluh metrik berbeda.
- `6.bukti alerting Grafana/`: untuk setiap alert, simpan screenshot rule dan notifikasinya; Advanced membutuhkan tiga alert.

Jangan menampilkan token, password, atau secret pada screenshot.
