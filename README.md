# Dog Breed Classifier

FastAPI image-classification service built around a Vision Transformer with 120 output classes. The repository contains the application and model implementation; it does not contain the trained checkpoint or training dataset.

## Run locally

Use Python with the dependencies in `requirements.txt` (the Docker image uses Python 3.11):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

The application looks for a compatible checkpoint at
`output/sample_run_checkpoint.bin`. The checkpoint is not included in this
repository. Create `output/` and place the checkpoint there before requesting
predictions. Without it, the API starts but prediction requests report that the
model is not loaded.

```bash
python app_spaces.py
```

The service listens on port `7860`.

## Docker

```bash
docker build -t dog-breed-classifier .
docker run --rm -p 7860:7860 dog-breed-classifier
```

The same checkpoint must be available at `output/sample_run_checkpoint.bin`
when the image is built, or supplied in the container at that path.

## API

- `GET /` — service status, including whether the model checkpoint loaded.
- `POST /api/predict` — classify an uploaded image or an image URL.
- `POST /api/predict-selected-image` — classify the image at the supplied URL.

## Repository contents and artifacts

`app_spaces.py` serves the API, and `models/` contains the ViT implementation
and configuration. The trained checkpoint and training data are not present in
this mirror; the code does not download a checkpoint automatically. The
project description references the publicly available
[Stanford Dogs dataset](http://vision.stanford.edu/aditya/ImageNetDogs/), but
the dataset files are absent and the checkpoint's training provenance is not
established by this checkout.
