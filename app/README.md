# Cloud Segmentation App

Streamlit app for cloud segmentation using the Attention Residual U-Net model.

## Requirements

- [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html)
- The trained model file (`model.pt`) placed at `app/model/model.pt`

## Setup

**1. Create the conda environment (first time only):**

```bash
conda env create -f environment.yml
```

**2. Activate the environment:**

```bash
conda activate clouds-app
```

## Running the app

From inside the `app/` folder:

```bash
streamlit run main.py
```

The app will open automatically in your browser at `http://localhost:8501`.

## Environment variables

Create a `.env` file inside `app/` if you need to override the default model path:

```env
MODEL_PATH=/path/to/your/model.pt
```

## Updating the environment

If `environment.yml` changes, update your existing env with:

```bash
conda env update -f environment.yml --prune
```
