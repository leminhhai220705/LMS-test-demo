

# """
# LMS AI Microservice: Hybrid Recommender (SVD + TF-IDF) + DQN for personalized learning paths.
# Exposes GET /api/recommendations/{user_id} with top-5 courses and DQN next-step suggestion.
# """

# import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# import json
# from pathlib import Path

# import numpy as np
# from fastapi import FastAPI, HTTPException, Body
# from fastapi.middleware.cors import CORSMiddleware
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import pandas as pd
# from surprise import SVD, Dataset, Reader

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

# # --- Data loading ---
# DATA_PATH = Path(__file__).resolve().parent / "data.json"

# def load_data():
#     if not DATA_PATH.exists():
#         raise FileNotFoundError(
#             f"Data file not found: {DATA_PATH}. Run from project root: node backend/export-data.js"
#         )
#     with open(DATA_PATH, "r", encoding="utf-8") as f:
#         return json.load(f)

# # --- Globals (filled at startup) ---
# data = None
# courses_list = None
# courses_by_id = None
# tfidf_matrix = None
# tfidf_vectorizer = None
# svd_model = None
# svd_trainset = None
# dqn_model = None
# FORMATS = ["Video", "Reading", "Quiz"]
# DIFFICULTIES = ["Beginner", "Intermediate", "Advanced"]
# FORMAT_TO_IDX = {f: i for i, f in enumerate(FORMATS)}
# DIFFICULTY_TO_IDX = {d: i for i, d in enumerate(DIFFICULTIES)}


# # Phase 1.3.3 Đoạn code thể hiện "Xử lý dữ liệu văn bản thành Vector" + Phase 2.2 (TF-IDF)
# def build_tfidf(courses):
#     # Gom metadata (Text)
#     texts = [
#         f"{c['title']} {c['description']} {c['category']} {c.get('difficulty', '')} {c.get('format', '')}"
#         for c in courses
#     ]
#     vectorizer = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 2))
#     matrix = vectorizer.fit_transform(texts)
#     return matrix, vectorizer


# # Phase 1.3.1 Đoạn code thể hiện "Dùng Pandas để Clean raw data (Lọc rác, Missing values), là một module ETL thu nhỏ (Extract - Transform - Load):
# def build_svd(interactions):
#     # --- BẢO VỆ KHI DỮ LIỆU RỖNG ---
#     if not interactions or len(interactions) == 0:
#         print("Warning: No interactions available for SVD.")
#         return None, None
        
#     reader = Reader(rating_scale=(1, 5))
#     # 1. EXTRACT: Hút mảng JSON thô (raw data) vào Pandas DataFrame
#     df = pd.DataFrame(interactions)
    
#     # 2. TRANSFORM & CLEANING (Xử lý Missing Values):
#     # Đảm bảo có đủ cột
#     # Nếu data bị lỗi, mất cột userId, courseId hoặc rating -> Báo lỗi và dừng lại (Lọc rác)
#     if "userId" not in df.columns or "courseId" not in df.columns or "rating" not in df.columns:
#         return None, None
        
#     # Lọc bỏ các cột không cần thiết, chỉ giữ lại đúng 3 cột lõi
#     df = df[["userId", "courseId", "rating"]].copy()
#     if df.empty:
#         return None, None
        
#     df.columns = ["user", "item", "rating"]
#     # 3. LOAD: Nạp dữ liệu đã sạch vào Dataset của thuật toán SVD
#     dataset = Dataset.load_from_df(df, reader)
    
#     trainset = dataset.build_full_trainset()
#     model = SVD(n_factors=20, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
#     model.fit(trainset)
#     return model, trainset


# def get_user_taken_course_ids(interactions, user_id):
#     return {i["courseId"] for i in interactions if i["userId"] == user_id}


# def get_tfidf_scores_for_user(user_id, courses, interactions, tfidf_matrix, vectorizer, courses_by_id):
#     user_interactions = [i for i in interactions if i["userId"] == user_id]
#     if not user_interactions:
#         return {c["id"]: 0.0 for c in courses}

#     profile_vec = np.zeros(tfidf_matrix.shape[1])
#     for i in user_interactions:
#         cid = i["courseId"]
#         if cid not in courses_by_id:
#             continue
#         idx = next(k for k, c in enumerate(courses) if c["id"] == cid)
#         weight = i["rating"] / 5.0
#         profile_vec += weight * tfidf_matrix[idx].toarray().flatten()
#     profile_vec = profile_vec.reshape(1, -1)
#     sims = cosine_similarity(profile_vec, tfidf_matrix).flatten()
#     return {courses[j]["id"]: float(sims[j]) for j in range(len(courses))}


# def get_tfidf_scores_from_preferences(category: str, difficulty: str, format_pref: str, courses, tfidf_matrix, vectorizer):
#     query = f"{category} {difficulty} {format_pref}"
#     query_vec = vectorizer.transform([query])
#     sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
#     return {courses[j]["id"]: float(sims[j]) for j in range(len(courses))}


# def get_svd_scores_for_user(user_id, courses, interactions, svd_model, trainset, courses_by_id):
#     # --- BẢO VỆ KHI SVD BỊ LỖI ---
#     if svd_model is None or trainset is None:
#         return {c["id"]: 3.0 for c in courses}
        
#     inner_uid = trainset.to_inner_uid(user_id) if trainset.knows_user(user_id) else None
#     course_ids = [c["id"] for c in courses]
#     scores = {}
#     for cid in course_ids:
#         try:
#             if inner_uid is not None and trainset.knows_item(cid):
#                 pred = svd_model.predict(user_id, cid)
#                 scores[cid] = pred.est
#             else:
#                 scores[cid] = svd_model.trainset.global_mean
#         except Exception:
#             try:
#                 scores[cid] = svd_model.trainset.global_mean
#             except:
#                 scores[cid] = 3.0
#     return scores


# def hybrid_top_n(user_id, courses, interactions, n=5, svd_weight=0.7, tfidf_weight=0.3):
#     taken = get_user_taken_course_ids(interactions, user_id)
#     tfidf_scores = get_tfidf_scores_for_user(
#         user_id, courses, interactions, tfidf_matrix, tfidf_vectorizer, courses_by_id
#     )
#     svd_scores = get_svd_scores_for_user(
#         user_id, courses, interactions, svd_model, svd_trainset, courses_by_id
#     )

#     all_svd = list(svd_scores.values())
#     all_tf = list(tfidf_scores.values())
#     svd_min, svd_max = min(all_svd) if all_svd else 0, max(all_svd) if all_svd else 1
#     tf_min, tf_max = min(all_tf) if all_tf else 0, max(all_tf) if all_tf else 1
#     svd_range = svd_max - svd_min or 1
#     tf_range = tf_max - tf_min or 1

#     combined = []
#     for c in courses:
#         cid = c["id"]
#         if cid in taken:
#             continue
#         s = (svd_scores[cid] - svd_min) / svd_range * svd_weight + (
#             (tfidf_scores[cid] - tf_min) / tf_range * tfidf_weight
#         )
#         combined.append((cid, s))

#     combined.sort(key=lambda x: x[1], reverse=True)
#     top_ids = [x[0] for x in combined[:n]]
#     top_scores = {x[0]: round(x[1], 4) for x in combined[:n]}
#     return top_ids, top_scores

# # Phase 1.3.2 Đoạn code thể hiện "Biến JSON logs thành Normalized Arithmetic Vectors" (Đỉnh cao của EDM)
# def state_from_history(interactions, user_id, courses_by_id):
#     counts = np.zeros(9) # Tạo 1 vector mảng 9 chiều (Arithmetic Vector) ban đầu toàn số 0
#     for i in interactions:
#         # BƯỚC LỌC NHIỄU (Noise Filtering): Bỏ qua khóa học bị 'dropped' hoặc của user khác
#         if i["userId"] != user_id or i.get("status") != "completed":
#             continue
#         c = courses_by_id.get(i["courseId"])
#         if not c:
#             continue
#         fi = FORMAT_TO_IDX.get(c["format"], 0)
#         di = DIFFICULTY_TO_IDX.get(c["difficulty"], 0)
#         idx = fi * 3 + di
#         counts[idx] += 1
#     total = counts.sum()
#     if total > 0:
#         # BƯỚC CHUẨN HÓA (NORMALIZATION):
#         # Chia tổng để ép tất cả các con số về khoảng [0, 1]
#         counts = counts / total
#     # Ép kiểu về Float32 (Chuẩn toán học cho Mạng Nơ-ron)
#     return counts.astype(np.float32)


# def action_to_format_difficulty(action_idx):
#     fi = action_idx // 3
#     di = action_idx % 3
#     return FORMATS[fi], DIFFICULTIES[di]


# def build_dqn_model(state_dim=9, num_actions=9, hidden=64):
#     model = keras.Sequential([
#         layers.Input(shape=(state_dim,)),
#         layers.Dense(hidden, activation="relu"),
#         layers.Dense(hidden, activation="relu"),
#         layers.Dense(num_actions),
#     ])
#     model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
#     return model


# def train_dqn(courses, interactions, courses_by_id, epochs=30):
#     samples = []
#     for idx, i in enumerate(interactions):
#         if "userId" not in i or "courseId" not in i or "rating" not in i:
#             continue
#         uid = i["userId"]
#         cid = i["courseId"]
#         c = courses_by_id.get(cid)
#         if not c:
#             continue
#         action = FORMAT_TO_IDX.get(c["format"], 0) * 3 + DIFFICULTY_TO_IDX.get(c["difficulty"], 0)
#         reward = (i["rating"] / 5.0) if i.get("status") == "completed" else 0.0
#         other = [interactions[j] for j in range(len(interactions)) if j != idx]
#         state = state_from_history(other, uid, courses_by_id)
#         samples.append((state, action, reward))

#     if len(samples) < 10:
#         return build_dqn_model(), False

#     X = np.array([s[0] for s in samples], dtype=np.float32)
#     actions = np.array([s[1] for s in samples], dtype=np.int32)
#     rewards = np.array([s[2] for s in samples], dtype=np.float32)

#     model = build_dqn_model()
#     Y = np.zeros((len(samples), 9), dtype=np.float32)
#     for i in range(len(samples)):
#         Y[i, actions[i]] = rewards[i]
#     model.fit(X, Y, epochs=epochs, batch_size=32, verbose=0)
#     return model, True


# def dqn_suggest(user_id, interactions, courses_by_id):
#     if dqn_model is None:
#         return FORMATS[0], DIFFICULTIES[0]
#     try:
#         state = state_from_history(interactions, user_id, courses_by_id)
#         state_batch = np.expand_dims(state, axis=0)
#         q_values = dqn_model.predict(state_batch, verbose=0).flatten()
#         action_idx = int(np.argmax(q_values))
#         return action_to_format_difficulty(action_idx)
#     except Exception as e:
#         print(f"DQN Error: {e}")
#         return FORMATS[0], DIFFICULTIES[0]


# # --- FastAPI app ---
# app = FastAPI(title="LMS AI Service", version="1.0")
# app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# # Phase 2.1 Data Loading & Initialization
# @app.on_event("startup")
# def startup():
#     global data, courses_list, courses_by_id, tfidf_matrix, tfidf_vectorizer, svd_model, svd_trainset, dqn_model
#     # --- BẢO VỆ TOÀN BỘ QUÁ TRÌNH KHỞI ĐỘNG ---
#     try:
#         data = load_data()
#         courses_list = data.get("courses", [])
#         interactions = data.get("interactions", [])
#         courses_by_id = {c["id"]: c for c in courses_list}
        
#         if courses_list:
#             tfidf_matrix, tfidf_vectorizer = build_tfidf(courses_list)
            
#         svd_model, svd_trainset = build_svd(interactions)
        
#         dqn_model, trained = train_dqn(courses_list, interactions, courses_by_id)
#         if not trained:
#             dqn_model = build_dqn_model()
            
#         print("✅ AI Service started successfully!")
#     except Exception as e:
#         print(f"❌ Startup Error (App will still run but with limited features): {e}")


# @app.get("/api/recommendations/{user_id}")
# def get_recommendations(user_id: int):
#     if data is None:
#         raise HTTPException(status_code=503, detail="Service not initialized completely due to startup error.")
#     if user_id not in {u["id"] for u in data.get("users", [])}:
#         raise HTTPException(status_code=404, detail=f"User {user_id} not found")

#     top_ids, top_scores = hybrid_top_n(
#         user_id, courses_list, data.get("interactions", []), n=5
#     )
#     recommendations = []
#     for cid in top_ids:
#         c = courses_by_id.get(cid)
#         if c:
#             recommendations.append({
#                 "courseId": c["id"],
#                 "title": c["title"],
#                 "category": c["category"],
#                 "difficulty": c["difficulty"],
#                 "format": c["format"],
#                 "score": top_scores.get(cid, 0),
#             })

#     format_rec, difficulty_rec = dqn_suggest(user_id, data.get("interactions", []), courses_by_id)
#     message = f"Take an {difficulty_rec} {format_rec} course next to maximize engagement."
#     dqn_suggestion = {
#         "format": format_rec,
#         "difficulty": difficulty_rec,
#         "message": message,
#     }

#     return {
#         "recommendations": recommendations,
#         "dqn_suggestion": dqn_suggestion,
#     }


# def compute_recommendations_from_data(user_id: int, courses: list, users: list, interactions: list):
#     courses_by_id = {c["id"]: c for c in courses}
#     tfidf_m, tfidf_v = build_tfidf(courses)
#     svd_m, svd_ts = build_svd(interactions)
#     dqn_m, trained = train_dqn(courses, interactions, courses_by_id, epochs=15)
#     if not trained:
#         dqn_m = build_dqn_model()

#     taken = get_user_taken_course_ids(interactions, user_id)
#     tfidf_scores = get_tfidf_scores_for_user(user_id, courses, interactions, tfidf_m, tfidf_v, courses_by_id)
#     svd_scores = get_svd_scores_for_user(user_id, courses, interactions, svd_m, svd_ts, courses_by_id)

#     all_svd = list(svd_scores.values())
#     all_tf = list(tfidf_scores.values())
#     svd_min, svd_max = min(all_svd) if all_svd else 0, max(all_svd) if all_svd else 1
#     tf_min, tf_max = min(all_tf) if all_tf else 0, max(all_tf) if all_tf else 1
#     svd_range = svd_max - svd_min or 1
#     tf_range = tf_max - tf_min or 1

#     combined = []
#     for c in courses:
#         cid = c["id"]
#         if cid in taken:
#             continue
#         s = (svd_scores[cid] - svd_min) / svd_range * 0.7 + (
#             (tfidf_scores[cid] - tf_min) / tf_range * 0.3
#         )
#         combined.append((cid, s))
#     combined.sort(key=lambda x: x[1], reverse=True)
#     top_ids = [x[0] for x in combined[:5]]
#     top_scores = {x[0]: round(x[1], 4) for x in combined[:5]}

#     recommendations = []
#     for cid in top_ids:
#         c = courses_by_id.get(cid)
#         if c:
#             recommendations.append({
#                 "courseId": c["id"],
#                 "title": c["title"],
#                 "category": c["category"],
#                 "difficulty": c["difficulty"],
#                 "format": c["format"],
#                 "score": top_scores.get(cid, 0),
#             })

#     try:
#         state = state_from_history(interactions, user_id, courses_by_id)
#         state_batch = np.expand_dims(state, axis=0)
#         q_values = dqn_m.predict(state_batch, verbose=0).flatten()
#         action_idx = int(np.argmax(q_values))
#         format_rec, difficulty_rec = action_to_format_difficulty(action_idx)
#     except:
#         format_rec, difficulty_rec = FORMATS[0], DIFFICULTIES[0]
        
#     message = f"Take an {difficulty_rec} {format_rec} course next to maximize engagement."

#     return {
#         "recommendations": recommendations,
#         "dqn_suggestion": {"format": format_rec, "difficulty": difficulty_rec, "message": message},
#     }


# def compute_onboarding_recommendations(category: str, difficulty: str, format_pref: str):
#     cb_scores = get_tfidf_scores_from_preferences(
#         category, difficulty, format_pref,
#         courses_list, tfidf_matrix, tfidf_vectorizer
#     )
#     all_cb = list(cb_scores.values())
#     cb_min, cb_max = min(all_cb) if all_cb else 0, max(all_cb) if all_cb else 1
#     cb_range = cb_max - cb_min or 1

#     combined = []
#     for c in courses_list:
#         cid = c["id"]
#         cb_norm = (cb_scores[cid] - cb_min) / cb_range
#         cf_norm = 0.5
#         s = 0.8 * cb_norm + 0.2 * cf_norm
#         combined.append((cid, s))

#     combined.sort(key=lambda x: x[1], reverse=True)
#     top_ids = [x[0] for x in combined[:5]]
#     top_scores = {x[0]: round(x[1], 4) for x in combined[:5]}

#     recommendations = []
#     for cid in top_ids:
#         c = courses_by_id.get(cid)
#         if c:
#             recommendations.append({
#                 "courseId": c["id"],
#                 "title": c["title"],
#                 "category": c["category"],
#                 "difficulty": c["difficulty"],
#                 "format": c["format"],
#                 "score": top_scores.get(cid, 0),
#             })

#     format_rec = format_pref if format_pref in FORMATS else FORMATS[0]
#     difficulty_rec = difficulty if difficulty in DIFFICULTIES else DIFFICULTIES[0]
#     message = f"Take an {difficulty_rec} {format_rec} course next to maximize engagement."

#     return {
#         "recommendations": recommendations,
#         "dqn_suggestion": {"format": format_rec, "difficulty": difficulty_rec, "message": message},
#     }


# @app.post("/api/recommendations/onboarding")
# def post_onboarding_recommendations(body: dict = Body(...)):
#     if data is None:
#         raise HTTPException(status_code=503, detail="Service not initialized completely.")
#     category = body.get("category") or body.get("goal")
#     difficulty = body.get("difficulty") or body.get("skillLevel")
#     format_pref = body.get("format") or body.get("preferredFormat")
#     if not category or not difficulty or not format_pref:
#         raise HTTPException(status_code=400, detail="Missing category, difficulty, or format")
#     return compute_onboarding_recommendations(category, difficulty, format_pref)


# @app.post("/api/recommendations")
# def post_recommendations(body: dict = Body(...)):
#     user_id = body.get("user_id")
#     courses = body.get("courses")
#     users = body.get("users")
#     interactions = body.get("interactions")
#     if user_id is None or not courses or not interactions:
#         raise HTTPException(status_code=400, detail="Missing user_id, courses, or interactions")
#     return compute_recommendations_from_data(user_id, courses, users or [], interactions)


# @app.get("/health")
# def health():
#     return {"status": "ok", "data_loaded": data is not None}


# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.environ.get("PORT", "10000")) # Mặc định port 10000 cho Render
#     uvicorn.run(app, host="0.0.0.0", port=port)

"""
LMS AI Microservice: Hybrid Recommender (SVD + TF-IDF) + DQN for personalized learning paths.
Exposes GET /api/recommendations/{user_id} with top-5 courses and DQN next-step suggestion.
"""

import os

# Tắt các cảnh báo log không cần thiết của TensorFlow để Terminal sạch sẽ
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
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
    # Kiểm tra xem file data có tồn tại không, nếu không thì báo lỗi kèm hướng dẫn
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Data file not found: {DATA_PATH}. Run from project root: node backend/export-data.js"
        )
    # Đọc file JSON chứa toàn bộ dữ liệu (courses, users, interactions)
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

# ==========================================
# PHASE 3 - TASK 1: ĐỊNH NGHĨA KHÔNG GIAN HÀNH ĐỘNG (ACTION SPACE)
# ==========================================
FORMATS = ["Video", "Reading", "Quiz"]
DIFFICULTIES = ["Beginner", "Intermediate", "Advanced"]
# Tạo từ điển ánh xạ chuỗi thành Index (vd: "Video" -> 0)
FORMAT_TO_IDX = {f: i for i, f in enumerate(FORMATS)}
DIFFICULTY_TO_IDX = {d: i for i, d in enumerate(DIFFICULTIES)}


# ==========================================
# PHASE 2: CONTENT-BASED FILTERING (TF-IDF)
# Trích xuất đặc trưng văn bản giải quyết Cold-Start
# ==========================================
def build_tfidf(courses):
    # Gom siêu dữ liệu (Metadata) thành Text để AI đọc hiểu văn cảnh
    texts = [
        f"{c['title']} {c['description']} {c['category']} {c.get('difficulty', '')} {c.get('format', '')}"
        for c in courses
    ]
    # Khởi tạo Vectorizer: giới hạn 500 từ khóa quan trọng, loại bỏ từ nối tiếng Anh (stop_words)
    vectorizer = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 2))
    # Ép danh sách text thành Ma trận Toán học thưa (TF-IDF Matrix)
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer


# ==========================================
# PHASE 2: COLLABORATIVE FILTERING (SVD)
# ==========================================
def build_svd(interactions):
    # --- BẢO VỆ KHI DỮ LIỆU RỖNG ---
    if not interactions or len(interactions) == 0:
        print("Warning: No interactions available for SVD.")
        return None, None
        
    # Định nghĩa thang điểm từ 1-5 sao cho thư viện Surprise
    reader = Reader(rating_scale=(1, 5))
    
    # 1. EXTRACT: Hút JSON thô vào Pandas DataFrame
    df = pd.DataFrame(interactions)
    
    # 2. TRANSFORM & CLEANING (Lọc rác): Chỉ giữ lại bản ghi có đủ 3 cột lõi
    if "userId" not in df.columns or "courseId" not in df.columns or "rating" not in df.columns:
        return None, None
        
    df = df[["userId", "courseId", "rating"]].copy()
    if df.empty:
        return None, None
        
    # Đổi tên cột cho chuẩn format của thư viện Surprise (user, item, rating)
    df.columns = ["user", "item", "rating"]
    
    # 3. LOAD: Nạp data vào Dataset và xây dựng tập Train
    dataset = Dataset.load_from_df(df, reader)
    trainset = dataset.build_full_trainset()
    
    # Khởi tạo thuật toán Matrix Factorization (Funk SVD): 20 Latent Features, học 20 Epochs, dùng L2 Regularization
    model = SVD(n_factors=20, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    # Tiến hành phân rã ma trận và tìm Đặc trưng ẩn
    model.fit(trainset)
    return model, trainset


def get_user_taken_course_ids(interactions, user_id):
    # Lọc ra các ID khóa học user ĐÃ học để loại trừ lúc gợi ý
    return {i["courseId"] for i in interactions if i["userId"] == user_id}


# --- PHASE 2 - CONTENT-BASED (Dành cho User ĐÃ CÓ lịch sử) ---
def get_tfidf_scores_for_user(user_id, courses, interactions, tfidf_matrix, vectorizer, courses_by_id):
    # Lấy lịch sử tương tác của User
    user_interactions = [i for i in interactions if i["userId"] == user_id]
    if not user_interactions:
        return {c["id"]: 0.0 for c in courses}

    # Khởi tạo Vector Hồ sơ (Profile Vector) bằng 0
    profile_vec = np.zeros(tfidf_matrix.shape[1])
    for i in user_interactions:
        cid = i["courseId"]
        if cid not in courses_by_id:
            continue
        idx = next(k for k, c in enumerate(courses) if c["id"] == cid) # lấy index của khóa học trong danh sách courses
        # Dùng Rating của User làm Trọng số (Weight)
        weight = i["rating"] / 5.0 
        # Cộng dồn vector để tạo thành "Hồ sơ sở thích"
        profile_vec += weight * tfidf_matrix[idx].toarray().flatten()
    
    profile_vec = profile_vec.reshape(1, -1) 
    # Đo khoảng cách Cosine giữa Profile Vector và toàn bộ khóa học khác
    sims = cosine_similarity(profile_vec, tfidf_matrix).flatten()
    return {courses[j]["id"]: float(sims[j]) for j in range(len(courses))}


# --- PHASE 2 - CONTENT-BASED (Dành cho User MỚI TINH - Giải quyết Cold Start) ---
def get_tfidf_scores_from_preferences(category: str, difficulty: str, format_pref: str, courses, tfidf_matrix, vectorizer):
    # Ghép 3 tham số từ Form Onboarding thành 1 Câu Query
    query = f"{category} {difficulty} {format_pref}"
    # Ép Query thành Vector bằng TF-IDF
    query_vec = vectorizer.transform([query])
    # Đo độ tương đồng Cosine trực tiếp với cơ sở dữ liệu khóa học
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    return {courses[j]["id"]: float(sims[j]) for j in range(len(courses))}


# --- PHASE 2 - COLLABORATIVE FILTERING (SVD Prediction) ---
def get_svd_scores_for_user(user_id, courses, interactions, svd_model, trainset, courses_by_id):
    # CƠ CHẾ DỰ PHÒNG LỖI (FAULT TOLERANCE): Nếu SVD bị lỗi hoặc chưa load, trả về điểm trung bình 3.0
    if svd_model is None or trainset is None:
        return {c["id"]: 3.0 for c in courses}
        
    # XỬ LÝ COLD-START CẤP ĐỘ THƯ VIỆN:
    # Lấy ID nội bộ (Inner UID) của User trong ma trận Train. Nếu user mới tinh -> trả về None.
    inner_uid = trainset.to_inner_uid(user_id) if trainset.knows_user(user_id) else None
    
    course_ids = [c["id"] for c in courses]
    scores = {}
    
    for cid in course_ids:
        try:
            # Nếu User ĐÃ CÓ lịch sử VÀ Khóa học ĐÃ TỒN TẠI trong ma trận
            if inner_uid is not None and trainset.knows_item(cid):
                # Dự đoán điểm bằng phép nhân vô hướng các Latent Features
                pred = svd_model.predict(user_id, cid)
                scores[cid] = pred.est
            else:
                # Nếu User hoặc Item mới tinh (Cold-start): Gán điểm trung bình toàn cục (Global Mean)
                scores[cid] = svd_model.trainset.global_mean
        except Exception:
            try:
                # Xử lý ngoại lệ tầng 2: Cố gắng lấy điểm trung bình toàn cục
                scores[cid] = svd_model.trainset.global_mean
            except:
                # Xử lý ngoại lệ tầng cuối: Lấy 3.0 làm chuẩn an toàn
                scores[cid] = 3.0
    return scores


# ==========================================
# PHASE 2: BỘ ĐỊNH TUYẾN LAI (WEIGHTED HYBRID MODEL)
# ==========================================
def hybrid_top_n(user_id, courses, interactions, n=5, svd_weight=0.7, tfidf_weight=0.3):
    taken = get_user_taken_course_ids(interactions, user_id)
    
    # 1. Chạy song song 2 thuật toán để lấy điểm dự đoán
    tfidf_scores = get_tfidf_scores_for_user(
        user_id, courses, interactions, tfidf_matrix, tfidf_vectorizer, courses_by_id
    )
    svd_scores = get_svd_scores_for_user(
        user_id, courses, interactions, svd_model, svd_trainset, courses_by_id
    )

    # 2. Chuẩn hóa thang điểm (Min-Max Normalization) để đưa về khoảng [0, 1]
    all_svd = list(svd_scores.values())
    all_tf = list(tfidf_scores.values())
    svd_min, svd_max = min(all_svd) if all_svd else 0, max(all_svd) if all_svd else 1
    tf_min, tf_max = min(all_tf) if all_tf else 0, max(all_tf) if all_tf else 1
    svd_range = svd_max - svd_min or 1
    tf_range = tf_max - tf_min or 1

    combined = []
    for c in courses:
        cid = c["id"]
        # Bỏ qua các khóa học đã học
        if cid in taken:
            continue
        # 3. Tính điểm tổng hợp (Weighted Average): 70% SVD + 30% CB
        s = (svd_scores[cid] - svd_min) / svd_range * svd_weight + (
            (tfidf_scores[cid] - tf_min) / tf_range * tfidf_weight
        )
        combined.append((cid, s))

    # Xếp hạng từ cao xuống thấp và cắt Top N (mặc định là Top 5)
    combined.sort(key=lambda x: x[1], reverse=True)
    top_ids = [x[0] for x in combined[:n]]
    top_scores = {x[0]: round(x[1], 4) for x in combined[:n]}
    return top_ids, top_scores


# ==========================================
# PHASE 3 - TASK 1: ĐỊNH NGHĨA TRẠNG THÁI (STATE ENGINEERING)
# ==========================================
def state_from_history(interactions, user_id, courses_by_id):
    # Khởi tạo vector mảng 9 chiều (Arithmetic Vector) ban đầu bằng 0
    counts = np.zeros(9) 
    for i in interactions:
        # Lọc nhiễu (Noise Filtering): Bỏ qua khóa học bị bỏ dở (dropped)
        if i["userId"] != user_id or i.get("status") != "completed":
            continue
        c = courses_by_id.get(i["courseId"])
        if not c:
            continue
            
        # Đếm tần suất Format và Difficulty để tìm ra index (từ 0 đến 8)
        fi = FORMAT_TO_IDX.get(c["format"], 0)
        di = DIFFICULTY_TO_IDX.get(c["difficulty"], 0)
        idx = fi * 3 + di
        counts[idx] += 1
        
    total = counts.sum()
    if total > 0:
        # CHUẨN HÓA (NORMALIZATION): Ép tất cả các số đếm về khoảng [0, 1] để tránh bùng nổ gradient
        counts = counts / total
        
    # Trả về kiểu Float32 chuẩn toán học cho Mạng Nơ-ron Tensor
    return counts.astype(np.float32)


# --- PHASE 3 - TASK 1: GIẢI MÃ HÀNH ĐỘNG (ACTION DECODING) ---
def action_to_format_difficulty(action_idx):
    # Dịch ngược index (0-8) ra tổ hợp ngôn ngữ con người (vd: Video + Advanced)
    fi = action_idx // 3
    di = action_idx % 3
    return FORMATS[fi], DIFFICULTIES[di]


# ==========================================
# PHASE 3 - TASK 3: KIẾN TRÚC MẠNG NƠ-RON SÂU (DQN ARCHITECTURE)
# ==========================================
def build_dqn_model(state_dim=9, num_actions=9, hidden=64):
    # Khởi tạo mô hình Sequential của Keras
    model = keras.Sequential([
        layers.Input(shape=(state_dim,)), # Lớp Input: Nhận Vector State 9 chiều
        layers.Dense(hidden, activation="relu"), # Lớp Ẩn 1: 64 nodes, hàm kích hoạt phi tuyến tính ReLU
        layers.Dense(hidden, activation="relu"), # Lớp Ẩn 2: 64 nodes, hàm kích hoạt phi tuyến tính ReLU
        layers.Dense(num_actions), # Lớp Output Linear: Xuất ra 9 giá trị Q-Value
    ])
    # Tối ưu hóa: Dùng Adam Optimizer (tốc độ học 0.001), đo sai số bằng MSE
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


# ==========================================
# PHASE 3 - TASK 4 & 5: HUẤN LUYỆN DQN OFFLINE (OFFLINE TRAINING)
# ==========================================
def train_dqn(courses, interactions, courses_by_id, epochs=30):
    # TASK 4: Experience Replay Buffer (Kho chứa kinh nghiệm để lấy mẫu)
    samples = [] 
    
    for idx, i in enumerate(interactions):
        if "userId" not in i or "courseId" not in i or "rating" not in i:
            continue
        uid = i["userId"]
        cid = i["courseId"]
        c = courses_by_id.get(cid)
        if not c:
            continue
            
        # Xác định Action mà User đã thực hiện trong quá khứ
        action = FORMAT_TO_IDX.get(c["format"], 0) * 3 + DIFFICULTY_TO_IDX.get(c["difficulty"], 0)
        
        # TASK 2: REWARD SHAPING (Cơ chế Khen thưởng)
        # Thưởng (Reward) = Điểm rating / 5.0 nếu hoàn thành. Phạt = 0.0 nếu bỏ dở.
        reward = (i["rating"] / 5.0) if i.get("status") == "completed" else 0.0
        
        # Tính toán State của User TRƯỚC KHI thực hiện hành động này
        other = [interactions[j] for j in range(len(interactions)) if j != idx]
        state = state_from_history(other, uid, courses_by_id)
        
        # Nạp (State, Action, Reward) vào Buffer
        samples.append((state, action, reward))

    # Nếu dữ liệu quá ít (< 10 mẫu), dừng train và trả về model rỗng
    if len(samples) < 10:
        return build_dqn_model(), False

    # Chuyển đổi samples thành Ma trận đặc trưng (X) và Ma trận nhãn (Y) để nạp vào Keras
    X = np.array([s[0] for s in samples], dtype=np.float32)
    actions = np.array([s[1] for s in samples], dtype=np.int32)
    rewards = np.array([s[2] for s in samples], dtype=np.float32)

    model = build_dqn_model()
    Y = np.zeros((len(samples), 9), dtype=np.float32)
    
    # Gán Reward thực tế vào đúng vị trí Action trong ma trận Target (Y)
    for i in range(len(samples)):
        Y[i, actions[i]] = rewards[i]
        
    # TASK 5: Huấn luyện Offline qua nhiều Kỷ nguyên (Epochs)
    model.fit(X, Y, epochs=epochs, batch_size=32, verbose=0)
    return model, True


# ==========================================
# PHASE 3 - TASK 6: SUY LUẬN THỜI GIAN THỰC (REAL-TIME INFERENCE)
# ==========================================
def dqn_suggest(user_id, interactions, courses_by_id):
    if dqn_model is None:
        # Nếu model lỗi, fallback về mặc định
        return FORMATS[0], DIFFICULTIES[0]
    try:
        # Lượng hóa State hiện tại của User
        state = state_from_history(interactions, user_id, courses_by_id)
        state_batch = np.expand_dims(state, axis=0)
        
        # Nạp State vào mạng DQN để nội suy ra 9 giá trị Q-Values
        q_values = dqn_model.predict(state_batch, verbose=0).flatten()
        
        # Lấy Action có Q-Value cao nhất (Hành động tối ưu nhất)
        action_idx = int(np.argmax(q_values))
        return action_to_format_difficulty(action_idx)
    except Exception as e:
        print(f"DQN Error: {e}")
        return FORMATS[0], DIFFICULTIES[0]


# --- FastAPI app ---
app = FastAPI(title="LMS AI Service", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ==========================================
# KHỞI TẠO HỆ THỐNG API (SYSTEM STARTUP)
# Nạp Data và Train Model vào RAM
# ==========================================
@app.on_event("startup")
def startup():
    global data, courses_list, courses_by_id, tfidf_matrix, tfidf_vectorizer, svd_model, svd_trainset, dqn_model
    # --- BẢO VỆ TOÀN BỘ QUÁ TRÌNH KHỞI ĐỘNG CỦA MICROSERVICE ---
    try:
        data = load_data()
        courses_list = data.get("courses", [])
        interactions = data.get("interactions", [])
        courses_by_id = {c["id"]: c for c in courses_list}
        
        if courses_list:
            # Khởi tạo mô hình Phase 2: CB
            tfidf_matrix, tfidf_vectorizer = build_tfidf(courses_list)
            
        # Khởi tạo mô hình Phase 2: CF
        svd_model, svd_trainset = build_svd(interactions)
        
        # Khởi tạo và Huấn luyện mô hình Phase 3: RL (DQN)
        dqn_model, trained = train_dqn(courses_list, interactions, courses_by_id)
        if not trained:
            dqn_model = build_dqn_model()
            
        print("✅ AI Service started successfully!")
    except Exception as e:
        print(f"❌ Startup Error (App will still run but with limited features): {e}")


# ==========================================
# ENDPOINT CHÍNH CỦA MICROSERVICE: LẤY GỢI Ý CÁ NHÂN HÓA
# ==========================================
@app.get("/api/recommendations/{user_id}")
def get_recommendations(user_id: int):
    # Kiểm tra tính hợp lệ của hệ thống và User ID
    if data is None:
        raise HTTPException(status_code=503, detail="Service not initialized completely due to startup error.")
    if user_id not in {u["id"] for u in data.get("users", [])}:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    # Gọi mô hình Hybrid Phase 2 để lấy danh sách Top 5 khóa học (Gợi ý đơn lẻ)
    top_ids, top_scores = hybrid_top_n(
        user_id, courses_list, data.get("interactions", []), n=5
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

    # Gọi mô hình DQN Phase 3 để lấy Định hướng Lộ trình dài hạn
    format_rec, difficulty_rec = dqn_suggest(user_id, data.get("interactions", []), courses_by_id)
    message = f"Take an {difficulty_rec} {format_rec} course next to maximize engagement."
    dqn_suggestion = {
        "format": format_rec,
        "difficulty": difficulty_rec,
        "message": message,
    }

    # Trả về chuỗi JSON chứa cả kết quả Phase 2 và Phase 3
    return {
        "recommendations": recommendations,
        "dqn_suggestion": dqn_suggestion,
    }


# ==========================================
# CÁC ENDPOINT VÀ HÀM PHỤ TRỢ DÙNG ĐỂ TÍNH TOÁN POST TRỰC TIẾP
# ==========================================
def compute_recommendations_from_data(user_id: int, courses: list, users: list, interactions: list):
    # Hàm Helper: Chạy lại toàn bộ flow Train/Test khi nhận dữ liệu truyền thẳng vào thay vì đọc từ file JSON
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
    svd_min, svd_max = min(all_svd) if all_svd else 0, max(all_svd) if all_svd else 1
    tf_min, tf_max = min(all_tf) if all_tf else 0, max(all_tf) if all_tf else 1
    svd_range = svd_max - svd_min or 1
    tf_range = tf_max - tf_min or 1

    combined = []
    for c in courses:
        cid = c["id"]
        if cid in taken:
            continue
        # Áp dụng Hybrid 70/30
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

    try:
        # Đoán State thời gian thực
        state = state_from_history(interactions, user_id, courses_by_id)
        state_batch = np.expand_dims(state, axis=0)
        q_values = dqn_m.predict(state_batch, verbose=0).flatten()
        action_idx = int(np.argmax(q_values))
        format_rec, difficulty_rec = action_to_format_difficulty(action_idx)
    except:
        format_rec, difficulty_rec = FORMATS[0], DIFFICULTIES[0]
        
    message = f"Take an {difficulty_rec} {format_rec} course next to maximize engagement."

    return {
        "recommendations": recommendations,
        "dqn_suggestion": {"format": format_rec, "difficulty": difficulty_rec, "message": message},
    }


def compute_onboarding_recommendations(category: str, difficulty: str, format_pref: str):
    # Hàm Helper: Xử lý kịch bản Cold-Start khi User mới tinh submit Form
    cb_scores = get_tfidf_scores_from_preferences(
        category, difficulty, format_pref,
        courses_list, tfidf_matrix, tfidf_vectorizer
    )
    all_cb = list(cb_scores.values())
    cb_min, cb_max = min(all_cb) if all_cb else 0, max(all_cb) if all_cb else 1
    cb_range = cb_max - cb_min or 1

    combined = []
    for c in courses_list:
        cid = c["id"]
        cb_norm = (cb_scores[cid] - cb_min) / cb_range
        # Vì User chưa có lịch sử, điểm CF (SVD) được chốt tĩnh ở mức 0.5 (Neutral)
        cf_norm = 0.5
        # Đẩy trọng số của Content-Based lên 80% do SVD lúc này không có tác dụng
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
    # API xử lý luồng Submit Form dành cho User chưa có lịch sử (Cold-Start)
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
    # API nhận trực tiếp Dữ liệu từ Request (thay vì đọc file) và trả ra Gợi ý
    user_id = body.get("user_id")
    courses = body.get("courses")
    users = body.get("users")
    interactions = body.get("interactions")
    if user_id is None or not courses or not interactions:
        raise HTTPException(status_code=400, detail="Missing user_id, courses, or interactions")
    return compute_recommendations_from_data(user_id, courses, users or [], interactions)


# --- THÊM ĐOẠN NÀY ĐỂ RENDER HEALTH CHECK THÀNH CÔNG ---
@app.get("/")
def root_health_check():
    return {
        "status": "ok", 
        "message": "AI Microservice is running perfectly on Render!",
        "version": "1.0"
    }

@app.get("/health")
def health():
    # API kiểm tra trạng thái "sống/chết" của Microservice
    return {"status": "ok", "data_loaded": data is not None}


if __name__ == "__main__":
    import uvicorn
    # Mặc định port 10000 để phù hợp với chuẩn Deploy lên Cloud Platform (như Render/Heroku)
    port = int(os.environ.get("PORT", "10000")) 
    uvicorn.run(app, host="0.0.0.0", port=port)