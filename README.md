# Submission Membangun Sistem Machine Learning — Peter-Shaan

Repository final submission untuk kelas Membangun Sistem Machine Learning.

## Status

- Kriteria 1: selesai (Advanced) — tautan tersedia di `Eksperimen_SML_Peter-Shaan.txt`.
- Kriteria 2: selesai (Advanced), termasuk run DagsHub dan bukti screenshot.
- Kriteria 3: selesai (Advanced) — workflow CI berhasil, artifact tersedia, dan image dipublikasikan ke Docker Hub.
- Kriteria 4: selesai (Advanced) — serving aktif, tiga metrik Prometheus, 12 panel Grafana, tiga alert, dan bukti notifikasi tersedia.

Docker image Kriteria 3: <https://hub.docker.com/r/petershaan/breast-cancer-mlflow>

## Kriteria 2

Notebook menjalankan MLflow autolog secara lokal, kemudian hyperparameter tuning dengan manual logging ke DagsHub:

- <https://dagshub.com/petershaan12/SMSML-Peter-Shaan>
- <https://dagshub.com/petershaan12/SMSML-Peter-Shaan.mlflow>

Sebelum menjalankan notebook di Google Colab, tambahkan secret `DAGSHUB_TOKEN`. Jangan menyimpan token di source code atau commit GitHub.

Run DagsHub dan bukti aslinya tersedia di `Membangun_model/`:

- `screenshoot_dashboard.jpg`
- `screenshoot_artifak.jpg`
