# CIFAR-100 MLOps Pipeline

An end-to-end image classification system: a CNN trained offline on CIFAR-100, served
behind a FastAPI service, driven from a Streamlit UI, containerised with Docker, deployed
on Render, and able to retrain itself on data a user uploads through the browser.

> **TODO before submitting:** replace every `TODO` in this file with your real links and numbers.

- **Live UI:** TODO
- **Live API (Swagger):** TODO `/docs`
- **Video demo:** TODO (YouTube)
- **Repo:** https://github.com/h-mutumwinka/Summative_MLPipeline.git

---

## The problem

CIFAR-100 is 60,000 32x32 colour images. Each image carries two labels: one of 100 **fine**
classes (`apple`, `beaver`, `boy`, ...) and one of 20 **coarse** superclasses
(`fruit_and_vegetables`, `aquatic_mammals`, `people`, ...).

The deployed model classifies the **20 coarse superclasses**. That is a deliberate choice.
At 32x32 the fine labels are close to the information limit of the pixels — a strong CNN
reaches roughly 45–55% top-1 there, which makes for a model that is wrong about as often as
it is right. On the coarse labels the same architecture is far more reliable and the system
is actually usable. Fine-label results are reported in the notebook appendix for comparison.

Switching is a one-line change: `LABEL_MODE=fine`.

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
cifar_mlops/
├── README.md
├── requirements.txt
├── Dockerfile              one image, two roles (API / UI)
├── docker-compose.yml      nginx + N api replicas + ui — used for the load test
├── nginx.conf              load balancer, re-resolves Docker DNS so --scale works
├── render.yaml             Render blueprint (two services, one persistent disk)
├── locustfile.py           flood test
│
├── notebook/
│   └── cifar100_mlops.ipynb    acquisition, EDA, preprocessing, training, evaluation
│
├── src/
│   ├── config.py           every path and hyperparameter, all env-overridable
│   ├── preprocessing.py    pickle loading, scaling, augmentation, zip/image decoding
│   ├── model.py            architecture, training callbacks, 6-metric evaluation
│   ├── prediction.py       the single-image prediction path the API calls
│   ├── database.py         SQLite: uploaded images + retrain run history
│   ├── retrain.py          the retraining job
│   └── api.py              FastAPI service
│
├── app/
│   └── streamlit_app.py    Status / Insights / Predict / Retrain
│
├── scripts/
│   ├── train.py            train the initial model from the CLI
│   └── make_demo_zip.py    build an upload zip for the video demo
│
├── data/
│   ├── train/train         CIFAR-100 pickles
│   ├── test/test
│   └── meta
└── models/
    └── cifar_cnn.keras     the served model (created by training)
```

---

## Setup

### 1. Local, without Docker

```bash
git clone TODO && cd cifar_mlops
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m scripts.download_data     # fetches the CIFAR-100 pickles into data/
```

The raw pickles are **not** in this repo: `data/train/train` is 155 MB, over GitHub's 100 MB
per-file limit. `scripts/download_data.py` pulls them from the original source, and the
Docker build runs it automatically.

Train the model. Use a GPU if you have one; on CPU this takes hours, so prefer running the
notebook on Colab with a T4 and downloading `models/cifar_cnn.keras`.

```bash
python -m scripts.train
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
top-5 ranking. The exact same `predict()` function is exercised in the notebook, so what you
test offline is what runs in production.

**Upload.** On the Retrain page, upload a `.zip` whose top-level folders are class names:

```
new_data.zip
  flowers/rose_01.png
  flowers/rose_02.png
  insects/bee_01.png
```

Every image is decoded, resized, and written to SQLite as a PNG blob together with its
label. Loose images with a single chosen label work too. `scripts/make_demo_zip.py` builds a
valid zip from the test split if you need one for the demo.

**Retraining.** Press *Retrain model*. The job:

1. loads the **currently served model** — our own custom model, used as the pre-trained base
2. pulls every upload not yet consumed by a previous run
3. mixes in a replay sample of the original training data so a handful of new images cannot
   make the model forget the other classes
4. continues training at a reduced learning rate with early stopping
5. evaluates on the untouched test set and compares against the model it started from
6. **promotes only if accuracy did not drop by more than 1%**, then hot-reloads the API

Every run is written to the database with its before/after metrics, and shown in the UI.

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

`render.yaml` in the repo root declares all of this. Note that retraining needs enough RAM
to hold TensorFlow plus the replay batch — the free instance will be killed mid-run, so use
at least a Starter instance.

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
| POST | `/upload` | bulk training images (zip of class folders, or loose files + label) |
| POST | `/retrain` | trigger retraining on everything uploaded since the last run |
| GET | `/retrain/status` | in-progress flag + run history |
| GET | `/stats` | upload counts per class |

Interactive docs at `/docs`.

---

## Video demo

TODO — YouTube link.

Camera on. Show, in this order: the prediction working on an image whose class you can say
out loud; the Insights page and your three interpretations; uploading a zip; pressing
retrain; the run appearing in the history with its before/after metrics.
