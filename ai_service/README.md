# LMS AI Microservice

Hybrid Recommender (SVD + TF-IDF) and DQN agent for personalized learning paths.

## Setup

1. **Export data** (from project root, once or when `backend/db.js` changes):
   ```bash
   node backend/export-data.js
   ```
   This creates `ai_service/data.json` from the mock database.

2. **Create virtual environment and install dependencies:**
   ```bash
   cd ai_service
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the service:**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```
   Or: `python main.py`

## Endpoints

- **GET /api/recommendations/{user_id}** — Returns top-5 hybrid recommendations and DQN next-step suggestion (format + difficulty).
- **GET /health** — Service health check.

## Algorithm summary

- **TF-IDF:** Content-based scores from course title, description, category (user profile = weighted sum of completed courses by rating).
- **SVD:** Collaborative filtering on user–course ratings (Scikit-Surprise).
- **Hybrid:** `0.7 * SVD + 0.3 * TF-IDF` (scores normalized to [0,1]), top-5 excluding already taken.
- **DQN:** Keras MLP; state = normalized counts of (format × difficulty) from completed courses; 9 actions = next format × difficulty; trained on interaction rewards (rating/5 if completed, else 0).
