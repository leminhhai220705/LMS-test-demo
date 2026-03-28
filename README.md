# LMS Thesis Demo

A proof-of-concept web-based LMS demo featuring a **Hybrid Course Recommender System** that combines:

- **Content-Based Filtering (TF-IDF + cosine similarity)**
- **Collaborative Filtering (SVD)**

The system supports both:

- **Cold-start recommendation** for new learners using onboarding preferences
- **Warm-start recommendation** for existing learners using interaction history

---

## Quick Start

### 1. Run the AI service (Python, port 8000)

```bash
cd ai_service
python3 -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

_(If you updated backend/db.js, refresh the AI input data first: `node backend/export-data.js` to refresh `ai_service/data.json`.)_

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

- After onboarding, the learner is redirected to the dashboard, where the system displays:
  - learner profile
  - completed courses
  - Top-5 recommended courses

- When the learner clicks “Simulate Completing Course”, the system creates a new interaction record and refreshes the recommendations using the warm-start hybrid logic:
  - Content-Based shortlist
  - SVD prediction
  - score normalization
  - 0.5 Collaborative Filtering + 0.5 Content-Based Filtering
- This demonstrates how recommendations are updated after user history becomes available.

## System Architecture

| Layer    | Tech                                       |
| -------- | ------------------------------------------ |
| Frontend | Vanilla HTML/CSS/JS (served by Express)    |
| Backend  | Node.js + Express                          |
| Data     | Local prototype dataset in `backend/db.js` |
| AI       | Python FastAPI + Scikit-learn + Surprise   |

## Recommendation Logic

- **Cold-start**:
  - Onboarding preference query
  - TF-IDF matching against course metadata
  - Final score: 0.8 CB + 0.2 neutral CF prior
- **Warm-start**:
  - Build learner profile from previous interactions
  - Generate Content-Based scores
  - Keep a Top-30 Content-Based shortlist
  - Predict SVD scores for shortlisted items
  - Normalize both score types
  - Final score: 0.5 CF + 0.5 CB

## Endpoints

- **Backend:** `GET /api/users`, `GET /api/courses`, `GET /api/user-dashboard/:userId`, `POST /api/onboard`, `POST /api/interact`
- **AI service:** `POST /api/recommendations/onboarding`, `POST /api/recommendations`

## Project Purpose

- This project was developed as **a proof-of-concept thesis demo** to show how a hybrid recommender system can be integrated into a simple LMS-style web application.

- It is intended for:
  - recommendation logic demonstration
  - system integration testing
  - cold-start and warm-start workflow demonstration

## Repository

Project repository: **LMS-test-demo**

## Copyright

This project was developed by **Le Hai** for thesis and demonstration purposes.

Copyright (c) Le Hai. All rights reserved.
Unauthorized copying, reproduction, or redistribution is not permitted without the author's permission.
