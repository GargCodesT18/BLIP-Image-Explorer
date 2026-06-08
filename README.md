# BLIP Image Explorer

BLIP Image Explorer is a Streamlit-based computer vision application that enables users to upload an image, generate a semantic caption, and ask natural-language questions about the scene. The application leverages Salesforce BLIP (Bootstrapping Language-Image Pre-training) models from Hugging Face for both image captioning and Visual Question Answering (VQA).

---

## Overview

This project demonstrates the integration of Vision-Language Models (VLMs) into an interactive web application. It combines two major computer vision tasks:

* **Image Captioning:** Automatically generates descriptive captions for uploaded images.
* **Visual Question Answering (VQA):** Answers user questions regarding objects, colors, activities, counts, and overall scene context.

The application automatically utilizes GPU acceleration when CUDA is available and seamlessly falls back to CPU execution otherwise.

---

## Features

* Interactive Streamlit-based user interface
* BLIP Large model for high-quality image caption generation
* BLIP VQA model for natural-language image understanding
* Session-based history of previous questions and answers
* Confidence-style score derived from generation probabilities
* Automatic device selection (GPU/CPU)
* Sample test images for quick evaluation

---

## Tech Stack

* Python
* Streamlit
* PyTorch
* Hugging Face Transformers
* Pillow (PIL)
* Salesforce BLIP Models

---

## Project Structure

```text
elc_4thsem/
│
├── app.py
├── pipeline.py
├── test_images/
├── image_understanting.ipynb
├── BLIP-Image-Explorer.pdf
├── report.pdf
└── README.md
```

---

## Model Information

This project utilizes the following pretrained models:

* `Salesforce/blip-image-captioning-large`
* `Salesforce/blip-vqa-base`

**Note:**

The pretrained model weights are **not included** in this repository because they occupy approximately **3 GB** of storage space.

During the first execution, Hugging Face Transformers will automatically download and cache the required models locally. Subsequent executions will reuse the cached files without requiring another download.

If you already have the models downloaded, they can also be stored locally and loaded directly by the application.

---

## How It Works

`pipeline.py` initializes the pretrained BLIP models and performs inference.

`app.py` provides the Streamlit interface and manages:

* Image upload
* Semantic caption generation
* Visual Question Answering
* Session history
* Confidence score estimation

---

## Installation

### 1. Clone the repository

```bash
git clone <repository_url>
cd elc_4thsem
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate the environment and install the required packages:

```bash
pip install streamlit torch transformers pillow
```

The required BLIP models will be downloaded automatically during the first execution if they are not already present in the local Hugging Face cache.

---

## Running the Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to access the application.

---

## Usage

1. Launch the application.
2. Upload a `.jpg`, `.jpeg`, or `.png` image.
3. Click **Execute Semantic Parsing** to generate an image caption.
4. Navigate to the VQA section and ask natural-language questions about the uploaded image.
5. View generated answers along with their confidence-style scores and session history.

---

## Sample Images

The `test_images/` directory contains sample images for testing:

* `anthem.webp`
* `dogs_hanging_out_with_friends.jpg`
* `f1racing.jpg`
* `ridingbicycle.jpg`

---

## Notes

* The first execution may take additional time while the pretrained models are downloaded and loaded into memory.
* GPU execution significantly improves inference performance.
* The displayed confidence score is a heuristic derived from token transition probabilities and should not be interpreted as a calibrated probability estimate.

---

## Troubleshooting

**Slow inference**

* Verify whether CUDA is available and the application is running on GPU.

**Model loading errors**

* Ensure an active internet connection during the first execution.
* Verify that the required model files exist in the local Hugging Face cache if using offline mode.

**Streamlit command not found**

Install Streamlit using:

```bash
pip install streamlit
```

---

## Future Improvements

* Add a `requirements.txt` file
* Support batch image processing
* Export captions and VQA results
* Webcam and live image support
* Improved confidence estimation and explainability
* Docker deployment support

---

## Author

Developed as an educational project demonstrating the practical application of Vision-Language Models using Salesforce BLIP, PyTorch, Hugging Face Transformers, and Streamlit.
