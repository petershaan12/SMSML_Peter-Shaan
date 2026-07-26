# Submission Membangun Sistem Machine Learning — Peter-Shaan

Repository final submission untuk kelas Membangun Sistem Machine Learning.

## Status

- Kriteria 1: selesai (Advanced) — tautan tersedia di `Eksperimen_SML_Peter-Shaan.txt`.
- Kriteria 2: selesai (Advanced), termasuk run DagsHub dan bukti screenshot.
- Kriteria 3: workflow CI tersedia di repository `petershaan12/Workflow-CI`.
- Kriteria 4: kode serving, Prometheus, Grafana, dashboard, dan alerting telah tersedia; bukti screenshot perlu diambil saat layanan dijalankan.

## Kriteria 2

Notebook menjalankan MLflow autolog secara lokal, kemudian hyperparameter tuning dengan manual logging ke DagsHub:

- <https://dagshub.com/petershaan12/SMSML-Peter-Shaan>
- <https://dagshub.com/petershaan12/SMSML-Peter-Shaan.mlflow>

Sebelum menjalankan notebook di Google Colab, tambahkan secret `DAGSHUB_TOKEN`. Jangan menyimpan token di source code atau commit GitHub.

Run DagsHub dan bukti aslinya tersedia di `Membangun_model/`:

- `screenshoot_dashboard.jpg`
- `screenshoot_artifak.jpg`
