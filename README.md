# Thesis Severity Data Gathering

Notebook data gathering: `notebook-preprocess/preprocess_data_gathering.ipynb`.

Notebook crop ECG PVC: `notebook-preprocess/preprocess_ecg_pvc_crop.ipynb`.

Notebook training YOLO PVC: `notebook-preprocess/training_yolo_pvc.ipynb`.

## Menjalankan dengan uv

```bash
uv sync
uv run jupyter lab
```

Lalu buka `notebook-preprocess/preprocess_data_gathering.ipynb` dan jalankan sel dari atas ke bawah.

Alternatif eksekusi langsung dari terminal:

```bash
uv run jupyter nbconvert --to notebook --execute notebook-preprocess/preprocess_data_gathering.ipynb --inplace
```
