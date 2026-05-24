# Repository Documentation (README.md) Generation Instructions

You are an AI assistant specializing in Machine Learning Engineering, Digital Forensics, and Medical Data Processing. Your task is to generate a highly comprehensive, professional, and academic `README.md` documentation for my GitHub repository (https://github.com/NugrohoRahmanto/Thesis-Severity).

## PROJECT CONTEXT (CRITICAL)
This project is part of a master's thesis research utilizing a *Multi-Model Input* architecture to predict two primary downstream tasks for cardiology patients:
1. **Ablation Success Classification** 2. **Cardio Severity (%)**

**Overall Pipeline Architecture:**
- **Input 1 (1D Tabular Data):** Consists of Patient Historical Data, Patient Echo Data, and Holter PVC (%). This data goes through a *Cleaning & Normalize* phase to become *Tabular Encoder Features*.
- **Input 2 (2D Image Data):** Raw 12-Lead ECG images. Since raw ECG images cannot be ingested directly by the main model, an *Object Detection* approach is utilized to extract specific spatial information—namely, the location of Premature Ventricular Contractions (PVC).
- **Current Repository Scope:** This repository is strictly focused on the image feature extraction phase. A YOLOv12n model has been trained (utilizing Open Source PVC Data) to detect PVCs on 12-Lead ECG images. The inference output from this YOLO model (*PVC Features*) will subsequently serve as processed inputs, ready for *Feature Fusion* with the 1D tabular data in the main Predictor Model.

## YOUR TASK
Write a complete, ready-to-publish `README.md` file targeted at technical audiences (Machine Learning Engineers, Data Scientists, Medical Researchers) who wish to understand, reproduce, and build upon this YOLO-based PVC detection project.

## REQUIRED DOCUMENT STRUCTURE:

1. **Project Overview**
   - Briefly explain the broader thesis context (*Multimodal feature fusion*) and exactly where this repository fits within that pipeline (as a PVC feature extractor from ECGs using YOLOv12n).
   - Mention the primary goals and object detection success metrics.

2. **Extraction Pipeline Architecture**
   - Briefly explain the transition: ECG Signals -> 12-Lead Images -> PVC Bounding Box Extraction -> Feature Output.

3. **Directory Structure**
   - Map out the crucial directories (`data/`, `patient/`, `notebook-preprocess/`, `runs/`, `pyproject.toml`, etc.) based on standard YOLO/ML project layouts and the provided GitHub link.
   - Provide a brief function description for each folder/file.

4. **Data and Preprocessing**
   - Explain the data sources and formats (ECG to image conversion).
   - Detail the label formatting (YOLO format).
   - Explain the `data.yaml` configuration file used for training/detection.
   - Describe the preprocessing steps located in `notebook-preprocess/`, required dependencies, and how to execute them.

5. **Model Training & Configuration**
   - Detail the usage of the YOLOv12n model.
   - Highlight key hyperparameters.
   - Provide fully copy-pasteable CLI (shell) commands to initiate training, with brief explanations for each argument.

6. **Inference & Evaluation**
   - Provide examples of how to run inference on new patient data (`patient/valid`).
   - Explain the generated output files and how these bounding box results will be passed to the main fusion model.
   - Include commands for running validation/evaluation and note the reported metrics (e.g., mAP).

7. **Reproducibility Guide**
   - Provide step-by-step instructions from environment setup (e.g., `pip install -r requirements.txt` or via `pyproject.toml`) to running the first training/inference script.

8. **Troubleshooting & Tips**
   - Address common issues (e.g., incorrect dataset paths, mismatched label formats, GPU OOM errors) and provide quick solutions.

9. **Contribution & Future Work**
   - Outline the brief roadmap leading toward the upcoming *Feature Fusion* integration with tabular data.

## WRITING TONE & FORMATTING STYLE
- Use professional, academic, and highly technical English.
- Utilize clean Markdown hierarchy (Headings, Bullet points, **Bold** text for emphasis).
- Provide explicit `bash` or `python` code blocks for all installation and execution instructions so they can be easily copied by the reader.
- Whenever referencing paths, explicitly state them relative to the repository root.
- **Do not** include introductory or concluding conversational filler. Output ONLY the raw Markdown text for the `README.md`.