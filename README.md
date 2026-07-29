# CIFAR-10 MLOps Pipeline

An end-to-end image classification system: a CNN trained offline on CIFAR-10, served
behind a FastAPI service, driven from a Streamlit UI (with a sleek dark mode design), containerised with Docker, deployed
on Render, and able to retrain itself on data a user uploads through the browser.

> **TODO before submitting:** replace every `TODO` in this file with your real links and numbers.

- **Live UI:** TODO
- **Live API (Swagger):** TODO `/docs`
- **Video demo:** TODO (YouTube)
- **Repo:** TODO

---

## The problem

CIFAR-10 is a widely used dataset of 60,000 32x32 colour images across 10 classes (`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`). 
This project focuses on building a full MLOps pipeline to serve, monitor, and continuously train a Convolutional Neural Network (CNN) on this dataset.

---

## Results

TODO — fill in from the notebook after training.

| Metric | Vanilla CNN | Optimised CNN |
|---|---|---|
| Accuracy | TODO | TODO |
| Top-5 accuracy | TODO | TODO |
| Loss | TODO | TODO |
| Precision (macro) | TODO | TODO |
| Recall (macro) | TODO | TODO |
| F1 (macro) | TODO | TODO |

The optimised model uses data augmentation, batch normalisation, dropout, L2 weight decay,
Adam with a `ReduceLROnPlateau` schedule, and early stopping. The vanilla model uses none of
these and exists to quantify what they are worth.

---

## Repository layout

```
Summative_MLPipeline/
├── README.md
├── requirements.txt
├── Dockerfile              one image, two roles (API / UI)
├── docker-compose.yml      nginx + N api replicas + ui — used for the load test
├── nginx.conf              load balancer, re-resolves Docker DNS so --scale works
├── render.yaml             Render blueprint (two services, one persistent disk)
├── locustfile.py           flood test
│
├── notebook/
│   └── cifar10_mlops.ipynb     acquisition, EDA, preprocessing, training, evaluation
│
├── src/
│   ├── config.py           every path and hyperparameter, all env-overridable
│   ├── preprocessing.py    pickle loading, scaling, augmentation, zip/image decoding
│   ├── model.py            architecture, training callbacks, 6-metric evaluation
│   ├── prediction.py       the single-image prediction path the API calls
│   ├── database.py         SQLite: uploaded images + retrain run history
│   ├── retrain.py          the fast retraining job on uploaded images
│   └── api.py              FastAPI service
│
├── app/
│   └── streamlit_app.py    Dark-mode UI: Status / Insights / Predict / Retrain
│
├── scripts/
│   └── train.py            train the initial model from the CLI
│
├── data/
│   ├── train/train         CIFAR-10 pickles
│   ├── test/test
│   └── meta
└── models/
    └── cifar10_model.keras     the served model (created by training)
```

---

## Setup

### 1. Local, without Docker

```bash
git clone TODO && cd Summative_MLPipeline
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the API and the UI in two terminals:

```bash
uvicorn src.api:app --reload --port 8000
streamlit run app/streamlit_app.py
```

UI at `http://localhost:8501`, Swagger at `http://localhost:8000/docs`.

### 2. Local, with Docker

```bash
docker compose up --build
```

UI at `http://localhost:8501`, load-balanced API at `http://localhost:8080`.

---

## How the pipeline works

**Prediction.** Upload an image on the Predict page. It is resized to 32x32, scaled to
`[0, 1]`, and passed to the model. The UI shows the predicted class, the confidence, and the
top-5 ranking. 

**Upload.** On the Retrain page, you can dynamically expand the dataset. Use the dropdown to select a class (e.g. `horse`) and upload a single image. The API automatically places it in the correct class folder (`uploads/horse/`) for retraining.

**Retraining.** Press *Start Retraining*. The job:

1. loads the **currently served model** — our own custom model, used as the pre-trained base
2. reads the newly uploaded images from the `/uploads` directory
3. measures baseline accuracy on the new data
4. continues training at a reduced learning rate with early stopping
5. re-evaluates on the data to check for improvement
6. **promotes only if accuracy did not drop by more than 1%**, then hot-reloads the API

Every run is written to the SQLite database with its before/after metrics, and shown in the UI History table.

---

## Deployment (Render)

Two web services from the same Dockerfile.

| Service | Start command | Notes |
|---|---|---|
| `cifar-mlops-api` | `uvicorn src.api:app --host 0.0.0.0 --port $PORT` | needs the persistent disk |
| `cifar-mlops-ui` | `streamlit run app/streamlit_app.py --server.port $PORT --server.address 0.0.0.0` | set `API_URL` to the API's public URL |

**The disk matters.** Render's filesystem is ephemeral: without a persistent disk, every
uploaded image, the SQLite database, and any retrained model are erased on the next restart
or redeploy. Attach a disk to the API service, mount it at `/data`, and set:

```
DATA_DIR=/data
DB_PATH=/data/mlops.db
MODELS_DIR=/data/models
```

`render.yaml` in the repo root declares all of this automatically for Render Blueprints. 

---

## Flood request simulation (Locust)

nginx sits in front of the API containers, so scaling the `api` service changes how many
containers the same load is spread across.

```bash
# 1 container
docker compose up -d --build --scale api=1
locust -f locustfile.py --host http://localhost:8080 --headless -u 100 -r 10 -t 60s --csv results_1

# 3 containers
docker compose up -d --scale api=3
locust -f locustfile.py --host http://localhost:8080 --headless -u 100 -r 10 -t 60s --csv results_3

# 5 containers
docker compose up -d --scale api=5
locust -f locustfile.py --host http://localhost:8080 --headless -u 100 -r 10 -t 60s --csv results_5
```

Read the numbers out of `results_N_stats.csv` and fill in:

| Containers | Users | RPS | Median (ms) | p95 (ms) | Max (ms) | Failures |
|---|---|---|---|---|---|---|
| 1 | 100 | TODO | TODO | TODO | TODO | TODO |
| 3 | 100 | TODO | TODO | TODO | TODO | TODO |
| 5 | 100 | TODO | TODO | TODO | TODO | TODO |

> TODO — write two or three sentences on what happened. Did throughput scale linearly with
> containers, or did it flatten out? If it flattened, what became the bottleneck: CPU on the
> host, nginx, or the model's own inference time? Say what you would change to serve more
> traffic.

---

## API reference

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | uptime, model version, requests served, average latency |
| GET | `/classes` | the label vocabulary |
| POST | `/predict` | one image → one prediction + top-5 |
| POST | `/upload-image` | single image + class label upload for retraining |
| POST | `/retrain` | trigger retraining on everything uploaded |
| GET | `/retrain/status` | in-progress flag + run history |
| GET | `/stats` | upload counts per class and prediction history |

Interactive docs at `/docs`.

---

## Video demo

TODO — YouTube link.

Camera on. Show, in this order: the prediction working on an image whose class you can say
out loud; the Insights page and your three interpretations; uploading a single image in the Retrain tab; pressing
retrain; the run appearing in the history with its before/after metrics.
