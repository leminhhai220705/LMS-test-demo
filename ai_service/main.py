"""
LMS AI Microservice: Hybrid Recommender (SVD + TF-IDF)
Exposes GET /api/recommendations/{user_id} with top-5 courses and DQN next-step suggestion.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader

from tensorflow import keras
from tensorflow.keras import layers

# -------------------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parent / "data.json"


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Data file not found: {DATA_PATH}. Run from project root: node backend/export-data.js"
        )
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------------------------------------------------
# GLOBALS
# -------------------------------------------------------------------
data = None
courses_list = None
courses_by_id = None
course_id_to_idx = None
tfidf_matrix = None
tfidf_vectorizer = None
svd_model = None
svd_trainset = None
dqn_model = None

# -------------------------------------------------------------------
# PHASE 3 - ACTION SPACE
# -------------------------------------------------------------------
FORMATS = ["Video", "Reading", "Quiz"]
DIFFICULTIES = ["Beginner", "Intermediate", "Advanced"]

FORMAT_TO_IDX = {f: i for i, f in enumerate(FORMATS)}
DIFFICULTY_TO_IDX = {d: i for i, d in enumerate(DIFFICULTIES)}


# -------------------------------------------------------------------
# PHASE 2 - CONTENT-BASED FILTERING (TF-IDF)
# -------------------------------------------------------------------
def build_tfidf(courses):
    texts = []
    for c in courses:
        title = str(c.get("title", ""))
        description = str(c.get("description", ""))
        category = str(c.get("category", ""))
        difficulty = str(c.get("difficulty", ""))
        format_ = str(c.get("format", ""))
        institution = str(c.get("institution", ""))

        text = f"{title} {description} {category} {difficulty} {format_} {institution}"
        texts.append(text.strip())

    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words="english",
        ngram_range=(1, 2)
    )
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer


# -------------------------------------------------------------------
# PHASE 2 - COLLABORATIVE FILTERING (SVD)
# -------------------------------------------------------------------
def build_svd(interactions):
    if not interactions or len(interactions) == 0:
        print("Warning: No interactions available for SVD.")
        return None, None

    reader = Reader(rating_scale=(1, 5))
    df = pd.DataFrame(interactions)

    required_cols = {"userId", "courseId", "rating"}
    if not required_cols.issubset(df.columns):
        return None, None

    df = df[["userId", "courseId", "rating"]].copy()
    if df.empty:
        return None, None

    df.columns = ["user", "item", "rating"]
    df = df.groupby(["user", "item"], as_index=False)["rating"].mean()
    dataset = Dataset.load_from_df(df, reader)
    trainset = dataset.build_full_trainset()

    # Đồng bộ với compare.py bản tốt hơn
    model = SVD(
        n_factors=50,
        n_epochs=30,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42
    )
    model.fit(trainset)
    return model, trainset


def get_user_taken_course_ids(interactions, user_id):
    return {i["courseId"] for i in interactions if i["userId"] == user_id}


# -------------------------------------------------------------------
# CONTENT-BASED FOR EXISTING USERS
# -------------------------------------------------------------------
def get_tfidf_scores_for_user(user_id, courses, interactions, tfidf_matrix, vectorizer, courses_by_id, course_id_to_idx):
    user_interactions = [i for i in interactions if i["userId"] == user_id]
    if not user_interactions:
        return {c["id"]: 0.0 for c in courses}

    profile_vec = np.zeros(tfidf_matrix.shape[1], dtype=np.float32)

    for i in user_interactions:
        cid = i["courseId"]
        if cid not in courses_by_id or cid not in course_id_to_idx:
            continue

        idx = course_id_to_idx[cid]
        weight = float(i.get("rating", 0)) / 5.0
        profile_vec += weight * tfidf_matrix[idx].toarray().flatten()

    profile_vec = profile_vec.reshape(1, -1)
    sims = cosine_similarity(profile_vec, tfidf_matrix).flatten()

    return {courses[j]["id"]: float(sims[j]) for j in range(len(courses))}


# -------------------------------------------------------------------
# CONTENT-BASED FOR COLD START
# -------------------------------------------------------------------
def get_tfidf_scores_from_preferences(category, difficulty, format_pref, courses, tfidf_matrix, vectorizer):
    query = f"{category} {difficulty} {format_pref}"
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    return {courses[j]["id"]: float(sims[j]) for j in range(len(courses))}


# -------------------------------------------------------------------
# SVD PREDICTION
# -------------------------------------------------------------------
def get_svd_scores_for_user(user_id, courses, interactions, svd_model, trainset, courses_by_id):
    if svd_model is None or trainset is None:
        return {c["id"]: 3.0 for c in courses}

    inner_uid = trainset.to_inner_uid(user_id) if trainset.knows_user(user_id) else None
    course_ids = [c["id"] for c in courses]
    scores = {}

    for cid in course_ids:
        try:
            if inner_uid is not None and trainset.knows_item(cid):
                pred = svd_model.predict(user_id, cid)
                scores[cid] = float(pred.est)
            else:
                scores[cid] = float(svd_model.trainset.global_mean)
        except Exception:
            try:
                scores[cid] = float(svd_model.trainset.global_mean)
            except Exception:
                scores[cid] = 3.0

    return scores


# -------------------------------------------------------------------
# HYBRID MODEL - CBF SHORTLIST + WEIGHTED RE-RANKING
# -------------------------------------------------------------------
def hybrid_top_n(
    user_id,
    courses,
    interactions,
    n=5,
    svd_weight=0.5,
    tfidf_weight=0.5,
    shortlist_size=30
):
    taken = get_user_taken_course_ids(interactions, user_id)

    tfidf_scores = get_tfidf_scores_for_user(
        user_id, courses, interactions, tfidf_matrix, tfidf_vectorizer, courses_by_id, course_id_to_idx
    )
    svd_scores = get_svd_scores_for_user(
        user_id, courses, interactions, svd_model, svd_trainset, courses_by_id
    )

    candidate_ids = [c["id"] for c in courses if c["id"] not in taken]
    if not candidate_ids:
        return [], {}

    # 1) Content-based shortlist trước
    cbf_ranked = sorted(
        [(cid, tfidf_scores[cid]) for cid in candidate_ids],
        key=lambda x: x[1],
        reverse=True
    )
    shortlist_ids = [cid for cid, _ in cbf_ranked[:shortlist_size]]

    if not shortlist_ids:
        return [], {}

    # 2) Normalize trong shortlist
    shortlist_svd = {cid: svd_scores[cid] for cid in shortlist_ids}
    shortlist_cbf = {cid: tfidf_scores[cid] for cid in shortlist_ids}

    svd_values = list(shortlist_svd.values())
    cbf_values = list(shortlist_cbf.values())

    svd_min, svd_max = min(svd_values), max(svd_values)
    cbf_min, cbf_max = min(cbf_values), max(cbf_values)

    svd_range = (svd_max - svd_min) if (svd_max > svd_min) else 1.0
    cbf_range = (cbf_max - cbf_min) if (cbf_max > cbf_min) else 1.0

    combined = []
    for cid in shortlist_ids:
        svd_norm = (shortlist_svd[cid] - svd_min) / svd_range
        cbf_norm = (shortlist_cbf[cid] - cbf_min) / cbf_range
        score = (svd_weight * svd_norm) + (tfidf_weight * cbf_norm)
        combined.append((cid, score))

    combined.sort(key=lambda x: x[1], reverse=True)
    top_ids = [cid for cid, _ in combined[:n]]
    top_scores = {cid: round(score, 4) for cid, score in combined[:n]}

    return top_ids, top_scores


# -------------------------------------------------------------------
# PHASE 3 - STATE ENGINEERING
# -------------------------------------------------------------------
def state_from_history(interactions, user_id, courses_by_id):
    counts = np.zeros(9, dtype=np.float32)

    for i in interactions:
        if i["userId"] != user_id or i.get("status") != "completed":
            continue

        c = courses_by_id.get(i["courseId"])
        if not c:
            continue

        fi = FORMAT_TO_IDX.get(c.get("format", ""), 0)
        di = DIFFICULTY_TO_IDX.get(c.get("difficulty", ""), 0)
        idx = fi * 3 + di
        counts[idx] += 1

    total = counts.sum()
    if total > 0:
        counts = counts / total

    return counts.astype(np.float32)


def action_to_format_difficulty(action_idx):
    fi = action_idx // 3
    di = action_idx % 3
    return FORMATS[fi], DIFFICULTIES[di]


# -------------------------------------------------------------------
# PHASE 3 - DQN ARCHITECTURE
# -------------------------------------------------------------------
def build_dqn_model(state_dim=9, num_actions=9, hidden=64):
    model = keras.Sequential([
        layers.Input(shape=(state_dim,)),
        layers.Dense(hidden, activation="relu"),
        layers.Dense(hidden, activation="relu"),
        layers.Dense(num_actions),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse"
    )
    return model


# -------------------------------------------------------------------
# PHASE 3 - OFFLINE TRAINING
# -------------------------------------------------------------------
def train_dqn(courses, interactions, courses_by_id, epochs=30):
    samples = []

    for idx, i in enumerate(interactions):
        if "userId" not in i or "courseId" not in i or "rating" not in i:
            continue

        uid = i["userId"]
        cid = i["courseId"]
        c = courses_by_id.get(cid)
        if not c:
            continue

        action = FORMAT_TO_IDX.get(c.get("format", ""), 0) * 3 + DIFFICULTY_TO_IDX.get(c.get("difficulty", ""), 0)
        reward = (float(i["rating"]) / 5.0) if i.get("status") == "completed" else 0.0

        other = [interactions[j] for j in range(len(interactions)) if j != idx]
        state = state_from_history(other, uid, courses_by_id)
        samples.append((state, action, reward))

    if len(samples) < 10:
        return build_dqn_model(), False

    X = np.array([s[0] for s in samples], dtype=np.float32)
    actions = np.array([s[1] for s in samples], dtype=np.int32)
    rewards = np.array([s[2] for s in samples], dtype=np.float32)

    model = build_dqn_model()
    Y = np.zeros((len(samples), 9), dtype=np.float32)

    for i in range(len(samples)):
        Y[i, actions[i]] = rewards[i]

    model.fit(X, Y, epochs=epochs, batch_size=32, verbose=0)
    return model, True


# -------------------------------------------------------------------
# PHASE 3 - REAL-TIME INFERENCE
# -------------------------------------------------------------------
def dqn_suggest(user_id, interactions, courses_by_id):
    if dqn_model is None:
        return FORMATS[0], DIFFICULTIES[0]

    try:
        state = state_from_history(interactions, user_id, courses_by_id)
        state_batch = np.expand_dims(state, axis=0)
        q_values = dqn_model.predict(state_batch, verbose=0).flatten()
        action_idx = int(np.argmax(q_values))
        return action_to_format_difficulty(action_idx)
    except Exception as e:
        print(f"DQN Error: {e}")
        return FORMATS[0], DIFFICULTIES[0]


# -------------------------------------------------------------------
# FASTAPI APP
# -------------------------------------------------------------------
app = FastAPI(title="LMS AI Service", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# -------------------------------------------------------------------
# STARTUP
# -------------------------------------------------------------------
@app.on_event("startup")
def startup():
    global data, courses_list, courses_by_id, course_id_to_idx
    global tfidf_matrix, tfidf_vectorizer, svd_model, svd_trainset, dqn_model

    try:
        data = load_data()
        courses_list = data.get("courses", [])
        interactions = data.get("interactions", [])

        courses_by_id = {c["id"]: c for c in courses_list}
        course_id_to_idx = {c["id"]: idx for idx, c in enumerate(courses_list)}

        if courses_list:
            tfidf_matrix, tfidf_vectorizer = build_tfidf(courses_list)

        svd_model, svd_trainset = build_svd(interactions)

        dqn_model, trained = train_dqn(courses_list, interactions, courses_by_id)
        if not trained:
            dqn_model = build_dqn_model()

        print("✅ AI Service started successfully!")
    except Exception as e:
        print(f"❌ Startup Error (App will still run but with limited features): {e}")


# -------------------------------------------------------------------
# MAIN ENDPOINT
# -------------------------------------------------------------------
@app.get("/api/recommendations/{user_id}")
def get_recommendations(user_id: int):
    if data is None:
        raise HTTPException(status_code=503, detail="Service not initialized completely due to startup error.")

    if user_id not in {u["id"] for u in data.get("users", [])}:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    top_ids, top_scores = hybrid_top_n(
        user_id=user_id,
        courses=courses_list,
        interactions=data.get("interactions", []),
        n=5,
        svd_weight=0.5,
        tfidf_weight=0.5,
        shortlist_size=30
    )

    recommendations = []
    for cid in top_ids:
        c = courses_by_id.get(cid)
        if c:
            recommendations.append({
                "courseId": c["id"],
                "title": c.get("title", ""),
                "category": c.get("category", ""),
                "difficulty": c.get("difficulty", ""),
                "format": c.get("format", ""),
                "score": top_scores.get(cid, 0),
            })

    format_rec, difficulty_rec = dqn_suggest(user_id, data.get("interactions", []), courses_by_id)
    message = f"Take an {difficulty_rec} {format_rec} course next to maximize engagement."

    return {
        "recommendations": recommendations,
        "dqn_suggestion": {
            "format": format_rec,
            "difficulty": difficulty_rec,
            "message": message,
        },
    }


# -------------------------------------------------------------------
# HELPER FOR DIRECT DATA POSTING
# -------------------------------------------------------------------
def compute_recommendations_from_data(user_id: int, courses: list, users: list, interactions: list):
    local_courses_by_id = {c["id"]: c for c in courses}
    local_course_id_to_idx = {c["id"]: idx for idx, c in enumerate(courses)}

    tfidf_m, tfidf_v = build_tfidf(courses)
    svd_m, svd_ts = build_svd(interactions)
    dqn_m, trained = train_dqn(courses, interactions, local_courses_by_id, epochs=15)
    if not trained:
        dqn_m = build_dqn_model()

    taken = get_user_taken_course_ids(interactions, user_id)

    tfidf_scores = get_tfidf_scores_for_user(
        user_id, courses, interactions, tfidf_m, tfidf_v, local_courses_by_id, local_course_id_to_idx
    )
    svd_scores = get_svd_scores_for_user(
        user_id, courses, interactions, svd_m, svd_ts, local_courses_by_id
    )

    candidate_ids = [c["id"] for c in courses if c["id"] not in taken]
    cbf_ranked = sorted(
        [(cid, tfidf_scores[cid]) for cid in candidate_ids],
        key=lambda x: x[1],
        reverse=True
    )
    shortlist_ids = [cid for cid, _ in cbf_ranked[:30]]

    if shortlist_ids:
        shortlist_svd = {cid: svd_scores[cid] for cid in shortlist_ids}
        shortlist_cbf = {cid: tfidf_scores[cid] for cid in shortlist_ids}

        svd_values = list(shortlist_svd.values())
        cbf_values = list(shortlist_cbf.values())

        svd_min, svd_max = min(svd_values), max(svd_values)
        cbf_min, cbf_max = min(cbf_values), max(cbf_values)

        svd_range = (svd_max - svd_min) if (svd_max > svd_min) else 1.0
        cbf_range = (cbf_max - cbf_min) if (cbf_max > cbf_min) else 1.0

        combined = []
        for cid in shortlist_ids:
            svd_norm = (shortlist_svd[cid] - svd_min) / svd_range
            cbf_norm = (shortlist_cbf[cid] - cbf_min) / cbf_range
            score = (0.5 * svd_norm) + (0.5 * cbf_norm)
            combined.append((cid, score))

        combined.sort(key=lambda x: x[1], reverse=True)
    else:
        combined = []

    top_ids = [cid for cid, _ in combined[:5]]
    top_scores = {cid: round(score, 4) for cid, score in combined[:5]}

    recommendations = []
    for cid in top_ids:
        c = local_courses_by_id.get(cid)
        if c:
            recommendations.append({
                "courseId": c["id"],
                "title": c.get("title", ""),
                "category": c.get("category", ""),
                "difficulty": c.get("difficulty", ""),
                "format": c.get("format", ""),
                "score": top_scores.get(cid, 0),
            })

    try:
        state = state_from_history(interactions, user_id, local_courses_by_id)
        state_batch = np.expand_dims(state, axis=0)
        q_values = dqn_m.predict(state_batch, verbose=0).flatten()
        action_idx = int(np.argmax(q_values))
        format_rec, difficulty_rec = action_to_format_difficulty(action_idx)
    except Exception:
        format_rec, difficulty_rec = FORMATS[0], DIFFICULTIES[0]

    message = f"Take an {difficulty_rec} {format_rec} course next to maximize engagement."

    return {
        "recommendations": recommendations,
        "dqn_suggestion": {
            "format": format_rec,
            "difficulty": difficulty_rec,
            "message": message
        },
    }


# -------------------------------------------------------------------
# ONBOARDING COLD-START
# -------------------------------------------------------------------
def compute_onboarding_recommendations(category: str, difficulty: str, format_pref: str):
    cb_scores = get_tfidf_scores_from_preferences(
        category, difficulty, format_pref,
        courses_list, tfidf_matrix, tfidf_vectorizer
    )

    all_cb = list(cb_scores.values())
    cb_min, cb_max = min(all_cb) if all_cb else 0, max(all_cb) if all_cb else 1
    cb_range = (cb_max - cb_min) if (cb_max > cb_min) else 1.0

    combined = []
    for c in courses_list:
        cid = c["id"]
        cb_norm = (cb_scores[cid] - cb_min) / cb_range
        cf_norm = 0.5
        score = 0.8 * cb_norm + 0.2 * cf_norm
        combined.append((cid, score))

    combined.sort(key=lambda x: x[1], reverse=True)
    top_ids = [x[0] for x in combined[:5]]
    top_scores = {x[0]: round(x[1], 4) for x in combined[:5]}

    recommendations = []
    for cid in top_ids:
        c = courses_by_id.get(cid)
        if c:
            recommendations.append({
                "courseId": c["id"],
                "title": c.get("title", ""),
                "category": c.get("category", ""),
                "difficulty": c.get("difficulty", ""),
                "format": c.get("format", ""),
                "score": top_scores.get(cid, 0),
            })

    format_rec = format_pref if format_pref in FORMATS else FORMATS[0]
    difficulty_rec = difficulty if difficulty in DIFFICULTIES else DIFFICULTIES[0]
    message = f"Take an {difficulty_rec} {format_rec} course next to maximize engagement."

    return {
        "recommendations": recommendations,
        "dqn_suggestion": {
            "format": format_rec,
            "difficulty": difficulty_rec,
            "message": message
        },
    }


# -------------------------------------------------------------------
# POST ENDPOINTS
# -------------------------------------------------------------------
@app.post("/api/recommendations/onboarding")
def post_onboarding_recommendations(body: dict = Body(...)):
    if data is None:
        raise HTTPException(status_code=503, detail="Service not initialized completely.")

    category = body.get("category") or body.get("goal")
    difficulty = body.get("difficulty") or body.get("skillLevel")
    format_pref = body.get("format") or body.get("preferredFormat")

    if not category or not difficulty or not format_pref:
        raise HTTPException(status_code=400, detail="Missing category, difficulty, or format")

    return compute_onboarding_recommendations(category, difficulty, format_pref)


@app.post("/api/recommendations")
def post_recommendations(body: dict = Body(...)):
    user_id = body.get("user_id")
    courses = body.get("courses")
    users = body.get("users")
    interactions = body.get("interactions")

    if user_id is None or not courses or not interactions:
        raise HTTPException(status_code=400, detail="Missing user_id, courses, or interactions")

    return compute_recommendations_from_data(user_id, courses, users or [], interactions)


# -------------------------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------------------------
@app.get("/")
def root_health_check():
    return {
        "status": "ok",
        "message": "AI Microservice is running.",
        "version": "1.0"
    }


@app.get("/health")
def health():
    return {"status": "ok", "data_loaded": data is not None}


# -------------------------------------------------------------------
# RUN
# -------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)