FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY credit_risk/ credit_risk/
COPY train.py service.py run_demo.py ./
COPY data/ data/

# Train on the bundled demo sample at build time so the image serves
# predictions immediately on `docker run`, no separate training step.
RUN python train.py

EXPOSE 8000
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
