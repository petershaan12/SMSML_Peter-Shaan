# Membangun Model

Klasifikasi Wisconsin Diagnostic Breast Cancer menggunakan Logistic Regression dan MLflow 2.19.0.

## Menjalankan melalui Google Colab

Buka `Modelling_Peter-Shaan.ipynb`, tambahkan secret bernama `DAGSHUB_TOKEN`, lalu jalankan sel secara berurutan. Jangan menuliskan token langsung di notebook.

Notebook menjalankan dua tahap:

1. `modelling.py` menggunakan MLflow `autolog` dan tracking lokal untuk memenuhi Basic.
2. `modelling_tuning.py --tracking dagshub` menggunakan GridSearchCV dan manual logging ke DagsHub untuk memenuhi Skilled/Advanced.

## Artefak Advanced

Selain model MLflow, proses tuning mencatat:

- `confusion_matrix.png`
- `roc_curve.png`
- `feature_importance.csv`
- `classification_report.json`
- `best_model.joblib`

Hasil eksperimen daring dapat dilihat di:

- Project: <https://dagshub.com/petershaan12/SMSML-Peter-Shaan>
- MLflow UI: <https://dagshub.com/petershaan12/SMSML-Peter-Shaan.mlflow>

Setelah training selesai, simpan dua bukti berikut:

- `screenshoot_dashboard.jpg`: halaman run yang memperlihatkan parameter dan metrik.
- `screenshoot_artifak.jpg`: tab artifacts yang memperlihatkan model dan folder `evaluation`.

Pastikan token tidak terlihat pada tangkapan layar.
