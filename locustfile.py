"""Flood the API with prediction requests.

    # 1 container
    docker compose up -d --scale api=1
    locust -f locustfile.py --host http://localhost:8080

    # then repeat with --scale api=3 and --scale api=5

Open http://localhost:8089, set users and spawn rate, run for a fixed duration
(60s is enough), then export the CSV and put the numbers in the README table.

Headless, for reproducible runs:
    locust -f locustfile.py --host http://localhost:8080 \
           --headless -u 100 -r 10 -t 60s --csv results_1container
"""
import io
import random

from locust import HttpUser, between, task
from PIL import Image


def random_image_bytes() -> bytes:
    """A synthetic 32x32 RGB image. The point of the test is throughput, not accuracy,
    so we do not need real CIFAR images and we avoid shipping the dataset to the load
    generator."""
    arr = bytes(random.getrandbits(8) for _ in range(32 * 32 * 3))
    img = Image.frombytes("RGB", (32, 32), arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


PAYLOAD = random_image_bytes()


class PredictionUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(10)
    def predict(self):
        self.client.post(
            "/predict",
            files={"file": ("load_test.png", PAYLOAD, "image/png")},
            name="/predict",
        )

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
