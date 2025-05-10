# Consulting GPT Backend

FastAPI backend for the Consulting GPT application.

## Project Structure

```
app/
├── api/
│   ├── endpoints/       # API route handlers
├── core/               # Core functionality, config
├── services/           # External service integrations
├── models/            # Database models
└── db/                # Database configuration
```

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` file in project root and configure environment variables:

```
# Copy from .env.example and update values
cp .env.example .env
```

4. Run the development server:

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`  
Swagger documentation at `http://localhost:8000/docs`  
ReDoc documentation at `http://localhost:8000/redoc`

## Development

### Project Dependencies

All project dependencies are managed in `requirements.txt`. To add new dependencies:

1. Add the package to `requirements.txt`
2. Run `pip install -r requirements.txt`

### Code Style

This project follows PEP 8 guidelines.
