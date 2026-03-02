"""
LMS AI Microservice: Hybrid Recommender (SVD + TF-IDF) + DQN for personalized learning paths.
Exposes GET /api/recommendations/{user_id} with top-5 courses and DQN next-step suggestion.
"""

import json
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from surprise import SVD, Dataset, Reader
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- Data loading ---
DATA_PATH = Path(__file__).resolve().parent / "data.json"

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Data file not found: {DATA_PATH}. Run from project root: node backend/export-data.js"
        )
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# --- Globals (filled at startup) ---
data = None
courses_list = None
courses_by_id = None
tfidf_matrix = None
tfidf_vectorizer = None
svd_model = None
svd_trainset = None
dqn_model = None
FORMATS = ["Video", "Reading", "Quiz"]
DIFFICULTIES = ["Beginner", "Intermediate", "Advanced"]
FORMAT_TO_IDX = {f: i for i, f in enumerate(FORMATS)}
DIFFICULTY_TO_IDX = {d: i for i, d in enumerate(DIFFICULTIES)}


def build_tfidf(courses):
    """TF-IDF on course title + description + category for content-based filtering."""
    texts = [
        f"{c['title']} {c['description']} {c['category']} {c.get('difficulty', '')} {c.get('format', '')}"
        for c in courses
    ]
    vectorizer = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer


def build_svd(interactions):
    """Train SVD on user-course ratings (Collaborative Filtering)."""
    reader = Reader(rating_scale=(1, 5))
    df = pd.DataFrame(interactions)[["userId", "courseId", "rating"]].copy()
    df.columns = ["user", "item", "rating"]
    dataset = Dataset.load_from_df(df, reader)
    trainset = dataset.build_full_trainset()
    model = SVD(n_factors=20, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    model.fit(trainset)
    return model, trainset


def get_user_taken_course_ids(interactions, user_id):
    """Set of course IDs the user has already interacted with."""
    return {i["courseId"] for i in interactions if i["userId"] == user_id}


def get_tfidf_scores_for_user(user_id, courses, interactions, tfidf_matrix, vectorizer, courses_by_id):
    """Content-based scores: user profile from completed courses (weighted by rating), similarity to all courses."""
    user_interactions = [i for i in interactions if i["userId"] == user_id]
    if not user_interactions:
        # Cold start: return uniform small scores so hybrid can still use SVD
        return {c["id"]: 0.0 for c in courses}

    # User profile = weighted sum of course vectors (by rating)
    profile_vec = np.zeros(tfidf_matrix.shape[1])
    for i in user_interactions:
        cid = i["courseId"]
        if cid not in courses_by_id:
            continue
        idx = next(k for k, c in enumerate(courses) if c["id"] == cid)
        weight = i["rating"] / 5.0
        profile_vec += weight * tfidf_matrix[idx].toarray().flatten()
    profile_vec = profile_vec.reshape(1, -1)
    sims = cosine_similarity(profile_vec, tfidf_matrix).flatten()
    return {courses[j]["id"]: float(sims[j]) for j in range(len(courses))}


def get_tfidf_scores_from_preferences(category: str, difficulty: str, format_pref: str, courses, tfidf_matrix, vectorizer):
    """Content-based scores from onboarding preferences (query string → TF-IDF → cosine similarity)."""
    query = f"{category} {difficulty} {format_pref}"
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    return {courses[j]["id"]: float(sims[j]) for j in range(len(courses))}


def get_svd_scores_for_user(user_id, courses, interactions, svd_model, trainset, courses_by_id):
    """Predicted ratings from SVD for all courses (only for courses not in train for this user)."""
    inner_uid = trainset.to_inner_uid(user_id) if trainset.knows_user(user_id) else None
    course_ids = [c["id"] for c in courses]
    scores = {}
    for cid in course_ids:
        try:
            if inner_uid is not None and trainset.knows_item(cid):
                pred = svd_model.predict(user_id, cid)
                scores[cid] = pred.est
            else:
                # Unknown user or item: use global mean as fallback
                scores[cid] = svd_model.trainset.global_mean
        except Exception:
            scores[cid] = svd_model.trainset.global_mean
    return scores


def hybrid_top_n(user_id, courses, interactions, n=5, svd_weight=0.7, tfidf_weight=0.3):
    """Combine SVD and TF-IDF (normalize to [0,1]) and return top N not-yet-taken courses."""
    taken = get_user_taken_course_ids(interactions, user_id)
    tfidf_scores = get_tfidf_scores_for_user(
        user_id, courses, interactions, tfidf_matrix, tfidf_vectorizer, courses_by_id
    )
    svd_scores = get_svd_scores_for_user(
        user_id, courses, interactions, svd_model, svd_trainset, courses_by_id
    )

    # Normalize to [0, 1]
    all_svd = list(svd_scores.values())
    all_tf = list(tfidf_scores.values())
    svd_min, svd_max = min(all_svd), max(all_svd)
    tf_min, tf_max = min(all_tf), max(all_tf)
    svd_range = svd_max - svd_min or 1
    tf_range = tf_max - tf_min or 1

    combined = []
    for c in courses:
        cid = c["id"]
        if cid in taken:
            continue
        s = (svd_scores[cid] - svd_min) / svd_range * svd_weight + (
            (tfidf_scores[cid] - tf_min) / tf_range * tfidf_weight
        )
        combined.append((cid, s))

    combined.sort(key=lambda x: x[1], reverse=True)
    top_ids = [x[0] for x in combined[:n]]
    top_scores = {x[0]: round(x[1], 4) for x in combined[:n]}
    return top_ids, top_scores


# --- DQN: state = 9-dim (counts per format x difficulty), action = 9 (next format x difficulty) ---
def state_from_history(interactions, user_id, courses_by_id):
    """Build state vector: counts of (format, difficulty) from user's completed courses, normalized."""
    counts = np.zeros(9)  # 3 format x 3 difficulty
    for i in interactions:
        if i["userId"] != user_id or i.get("status") != "completed":
            continue
        c = courses_by_id.get(i["courseId"])
        if not c:
            continue
        fi = FORMAT_TO_IDX.get(c["format"], 0)
        di = DIFFICULTY_TO_IDX.get(c["difficulty"], 0)
        idx = fi * 3 + di
        counts[idx] += 1
    total = counts.sum()
    if total > 0:
        counts = counts / total
    return counts.astype(np.float32)


def action_to_format_difficulty(action_idx):
    """Map action index 0..8 to (format, difficulty)."""
    fi = action_idx // 3
    di = action_idx % 3
    return FORMATS[fi], DIFFICULTIES[di]


def build_dqn_model(state_dim=9, num_actions=9, hidden=64):
    """Simple DQN: state -> hidden -> Q-values for each action."""
    model = keras.Sequential([
        layers.Input(shape=(state_dim,)),
        layers.Dense(hidden, activation="relu"),
        layers.Dense(hidden, activation="relu"),
        layers.Dense(num_actions),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def train_dqn(courses, interactions, courses_by_id, epochs=30):
    """Train DQN on interaction history: state = history before course, action = course's format-difficulty, reward = rating/5 if completed else 0."""
    samples = []
    for idx, i in enumerate(interactions):
        uid = i["userId"]
        cid = i["courseId"]
        c = courses_by_id.get(cid)
        if not c:
            continue
        action = FORMAT_TO_IDX.get(c["format"], 0) * 3 + DIFFICULTY_TO_IDX.get(c["difficulty"], 0)
        reward = (i["rating"] / 5.0) if i.get("status") == "completed" else 0.0
        # State = user's history BEFORE this interaction (exclude this row by index)
        other = [interactions[j] for j in range(len(interactions)) if j != idx]
        state = state_from_history(other, uid, courses_by_id)
        samples.append((state, action, reward))

    if len(samples) < 10:
        return build_dqn_model(), False

    X = np.array([s[0] for s in samples], dtype=np.float32)
    actions = np.array([s[1] for s in samples], dtype=np.int32)
    rewards = np.array([s[2] for s in samples], dtype=np.float32)

    model = build_dqn_model()
    # Target: Q(s,a) = reward (simplified one-step reward)
    Y = np.zeros((len(samples), 9), dtype=np.float32)
    for i in range(len(samples)):
        Y[i, actions[i]] = rewards[i]
    model.fit(X, Y, epochs=epochs, batch_size=32, verbose=0)
    return model, True


def dqn_suggest(user_id, interactions, courses_by_id):
    """Return suggested (format, difficulty) from DQN given user's current state."""
    state = state_from_history(interactions, user_id, courses_by_id)
    state_batch = np.expand_dims(state, axis=0)
    q_values = dqn_model.predict(state_batch, verbose=0).flatten()
    action_idx = int(np.argmax(q_values))
    return action_to_format_difficulty(action_idx)


# --- FastAPI app ---
app = FastAPI(title="LMS AI Service", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def startup():
    global data, courses_list, courses_by_id, tfidf_matrix, tfidf_vectorizer, svd_model, svd_trainset, dqn_model
    data = load_data()
    courses_list = data["courses"]
    courses_by_id = {c["id"]: c for c in courses_list}
    tfidf_matrix, tfidf_vectorizer = build_tfidf(courses_list)
    svd_model, svd_trainset = build_svd(data["interactions"])
    dqn_model, trained = train_dqn(
        courses_list, data["interactions"], courses_by_id
    )
    if not trained:
        dqn_model = build_dqn_model()  # untrained fallback


@app.get("/api/recommendations/{user_id}")
def get_recommendations(user_id: int):
    """Return top-5 hybrid recommendations and DQN next-step suggestion."""
    if data is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    if user_id not in {u["id"] for u in data["users"]}:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    top_ids, top_scores = hybrid_top_n(
        user_id, courses_list, data["interactions"], n=5
    )
    recommendations = []
    for cid in top_ids:
        c = courses_by_id.get(cid)
        if c:
            recommendations.append({
                "courseId": c["id"],
                "title": c["title"],
                "category": c["category"],
                "difficulty": c["difficulty"],
                "format": c["format"],
                "score": top_scores.get(cid, 0),
            })

    format_rec, difficulty_rec = dqn_suggest(user_id, data["interactions"], courses_by_id)
    message = (
        f"Take an {difficulty_rec} {format_rec} course next to maximize engagement."
    )
    dqn_suggestion = {
        "format": format_rec,
        "difficulty": difficulty_rec,
        "message": message,
    }

    return {
        "recommendations": recommendations,
        "dqn_suggestion": dqn_suggestion,
    }


def compute_recommendations_from_data(user_id: int, courses: list, users: list, interactions: list):
    """Compute recommendations using provided data (for POST with fresh interactions)."""
    courses_by_id = {c["id"]: c for c in courses}
    tfidf_m, tfidf_v = build_tfidf(courses)
    svd_m, svd_ts = build_svd(interactions)
    dqn_m, trained = train_dqn(courses, interactions, courses_by_id, epochs=15)
    if not trained:
        dqn_m = build_dqn_model()

    taken = get_user_taken_course_ids(interactions, user_id)
    tfidf_scores = get_tfidf_scores_for_user(user_id, courses, interactions, tfidf_m, tfidf_v, courses_by_id)
    svd_scores = get_svd_scores_for_user(user_id, courses, interactions, svd_m, svd_ts, courses_by_id)

    all_svd = list(svd_scores.values())
    all_tf = list(tfidf_scores.values())
    svd_min, svd_max = min(all_svd), max(all_svd)
    tf_min, tf_max = min(all_tf), max(all_tf)
    svd_range = svd_max - svd_min or 1
    tf_range = tf_max - tf_min or 1

    combined = []
    for c in courses:
        cid = c["id"]
        if cid in taken:
            continue
        s = (svd_scores[cid] - svd_min) / svd_range * 0.7 + (
            (tfidf_scores[cid] - tf_min) / tf_range * 0.3
        )
        combined.append((cid, s))
    combined.sort(key=lambda x: x[1], reverse=True)
    top_ids = [x[0] for x in combined[:5]]
    top_scores = {x[0]: round(x[1], 4) for x in combined[:5]}

    recommendations = []
    for cid in top_ids:
        c = courses_by_id.get(cid)
        if c:
            recommendations.append({
                "courseId": c["id"],
                "title": c["title"],
                "category": c["category"],
                "difficulty": c["difficulty"],
                "format": c["format"],
                "score": top_scores.get(cid, 0),
            })

    state = state_from_history(interactions, user_id, courses_by_id)
    state_batch = np.expand_dims(state, axis=0)
    q_values = dqn_m.predict(state_batch, verbose=0).flatten()
    action_idx = int(np.argmax(q_values))
    format_rec, difficulty_rec = action_to_format_difficulty(action_idx)
    message = f"Take an {difficulty_rec} {format_rec} course next to maximize engagement."

    return {
        "recommendations": recommendations,
        "dqn_suggestion": {"format": format_rec, "difficulty": difficulty_rec, "message": message},
    }


def compute_onboarding_recommendations(category: str, difficulty: str, format_pref: str):
    """Cold-start: 0.8 CB (from preferences) + 0.2 CF (global mean). DQN suggests from preferences."""
    cb_scores = get_tfidf_scores_from_preferences(
        category, difficulty, format_pref,
        courses_list, tfidf_matrix, tfidf_vectorizer
    )
    all_cb = list(cb_scores.values())
    cb_min, cb_max = min(all_cb), max(all_cb)
    cb_range = cb_max - cb_min or 1

    combined = []
    for c in courses_list:
        cid = c["id"]
        cb_norm = (cb_scores[cid] - cb_min) / cb_range
        cf_norm = 0.5
        s = 0.8 * cb_norm + 0.2 * cf_norm
        combined.append((cid, s))

    combined.sort(key=lambda x: x[1], reverse=True)
    top_ids = [x[0] for x in combined[:5]]
    top_scores = {x[0]: round(x[1], 4) for x in combined[:5]}

    recommendations = []
    for cid in top_ids:
        c = courses_by_id.get(cid)
        if c:
            recommendations.append({
                "courseId": c["id"],
                "title": c["title"],
                "category": c["category"],
                "difficulty": c["difficulty"],
                "format": c["format"],
                "score": top_scores.get(cid, 0),
            })

    format_rec = format_pref if format_pref in FORMATS else FORMATS[0]
    difficulty_rec = difficulty if difficulty in DIFFICULTIES else DIFFICULTIES[0]
    message = f"Take an {difficulty_rec} {format_rec} course next to maximize engagement."

    return {
        "recommendations": recommendations,
        "dqn_suggestion": {"format": format_rec, "difficulty": difficulty_rec, "message": message},
    }


@app.post("/api/recommendations/onboarding")
def post_onboarding_recommendations(body: dict = Body(...)):
    """Accept { category, difficulty, format } for cold-start recommendations (0.8 CB + 0.2 CF)."""
    if data is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    category = body.get("category") or body.get("goal")
    difficulty = body.get("difficulty") or body.get("skillLevel")
    format_pref = body.get("format") or body.get("preferredFormat")
    if not category or not difficulty or not format_pref:
        raise HTTPException(status_code=400, detail="Missing category, difficulty, or format")
    return compute_onboarding_recommendations(category, difficulty, format_pref)


@app.post("/api/recommendations")
def post_recommendations(body: dict = Body(...)):
    """Accept { user_id, courses, users, interactions } and return fresh recommendations."""
    user_id = body.get("user_id")
    courses = body.get("courses")
    users = body.get("users")
    interactions = body.get("interactions")
    if user_id is None or not courses or not interactions:
        raise HTTPException(status_code=400, detail="Missing user_id, courses, or interactions")
    if user_id not in {u["id"] for u in (users or [])}:
        pass  # Allow unknown user for demo
    return compute_recommendations_from_data(user_id, courses, users or [], interactions)


@app.get("/health")
def health():
    return {"status": "ok", "data_loaded": data is not None}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
