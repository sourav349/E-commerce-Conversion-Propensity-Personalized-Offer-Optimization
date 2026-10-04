# Next Step: Tests + Docker + CI

Copy these files into the root of your existing project.

Expected structure:

```text
User Conversion Prediction/
├── backend/
│   ├── __init__.py
│   ├── Dockerfile
│   ├── main.py
│   ├── predictor.py
│   └── schemas.py
├── frontend/
│   ├── Dockerfile
│   └── app.py
├── models/
├── tests/
│   └── test_api.py
├── .github/
│   └── workflows/
│       └── tests.yml
├── .dockerignore
├── docker-compose.yml
├── requirements.txt
└── requirements-dev.txt
```

## 1. Install test dependencies

```bash
python3 -m pip install -r requirements-dev.txt
```

## 2. Run automated tests

```bash
python3 -m pytest -q
```

## 3. Build and start Docker services

```bash
docker compose up --build
```

Then open:

- Streamlit: http://localhost:8501
- FastAPI docs: http://localhost:8001/docs
- FastAPI health: http://localhost:8001/health

## 4. Stop Docker

```bash
docker compose down
```

## 5. Git

```bash
git add tests backend/Dockerfile frontend/Dockerfile docker-compose.yml .dockerignore requirements-dev.txt .github/workflows/tests.yml
git commit -m "Add automated API tests Docker deployment and CI"
git push origin main
```

## Important

The frontend must read the backend URL from an environment variable:

```python
import os

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8001",
).rstrip("/")
```

The rewritten frontend supplied earlier already follows this pattern.

The model pickle must be loaded with a compatible scikit-learn environment.
If Docker reports a model-version warning/error, capture your working local package versions with:

```bash
python3 -m pip freeze > requirements-lock.txt
```

Then we can pin the Docker environment to the versions that created/loaded the model successfully.
