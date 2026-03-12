# LMS Thesis Demo

Mini-demo LMS with **Hybrid Recommender (SVD + TF-IDF)** and **DQN** for personalized learning paths.

## Quick start

### 1. AI service (Python, port 8000)

```bash
cd ai_service
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

_(If you changed `backend/db.js`, run from project root: `node backend/export-data.js` to refresh `ai_service/data.json`.)_

### 2. Backend + frontend (Node, port 3000)

```bash
cd backend
npm install
npm start
```

### 3. Open the student LMS

In your browser: **http://localhost:3000**

**Step 1 — Onboarding (Cold Start):**

- Fill out the questionnaire: main goal (IT/Coding, Business, etc.), skill level, preferred format.
- Click **"Discover My Path"**.
- AI uses **0.8 CB + 0.2 CF** (preference-based) to generate initial recommendations.

**Step 2 — Dashboard:**

- Hero: "Based on your goals, here is your AI-optimized Learning Path!"
- Top 5 recommended courses + DQN suggestion.
- Click **"Simulate Completing Course"** — the system saves the interaction and fetches **new** recommendations using **0.7 CF + 0.3 CB** (history-based). The DQN banner and recommendations update dynamically.

## Stack

| Layer    | Tech                                                                |
| -------- | ------------------------------------------------------------------- |
| Frontend | Vanilla HTML/CSS/JS (served by Express)                             |
| Backend  | Node.js + Express                                                   |
| Data     | Mock DB in `backend/db.js`                                          |
| AI       | Python FastAPI — Scikit-learn (TF-IDF), Surprise (SVD), Keras (DQN) |

## Endpoints

- **Backend:** `GET /api/users`, `GET /api/courses`, `GET /api/user-dashboard/:userId`
- **AI service:** `GET http://localhost:8000/api/recommendations/:userId`

# LMS-test-demo
