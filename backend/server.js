/**
 * LMS Thesis Demo - Node.js/Express Backend
 * Serves /users, /courses, /user-dashboard/:userId (combined with AI recommendations).
 */

const path = require("path");
const express = require("express");
const cors = require("cors");
const axios = require("axios");
const { courses, users, interactions } = require("./db.js");

const app = express();
const PORT = process.env.PORT || 5000;
const AI_SERVICE_URL = process.env.AI_SERVICE_URL || "http://localhost:8000";
const FRONTEND_URL = process.env.FRONTEND_URL;

// const NEW_STUDENT_ID = 999;
const onboardedUsers = new Map();

app.use(
  cors({
    origin: FRONTEND_URL || "*",
    methods: ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"],
  }),
);
app.use(express.json());

function getUser(uid) {
  return onboardedUsers.get(uid) || users.find((u) => u.id === uid);
}

// Serve frontend static files (from ../frontend when running from backend/)
const frontendPath = path.join(__dirname, "..", "frontend");
app.use(express.static(frontendPath));

// --- API routes ---

// --- MÔ PHỎNG MONGODB SHARDING ARCHITECTURE ---
// Giả lập hệ thống có 3 Server (Shards) độc lập
const shard_0 = []; // Chứa data của User có ID chia hết cho 3
const shard_1 = []; // Chứa data của User có ID chia cho 3 dư 1
const shard_2 = []; // Chứa data của User có ID chia cho 3 dư 2

// Phase 1.2 Hàm giả lập bộ định tuyến "mongos" của MongoDB
// Shard Key ở đây chính là: user_id
function routeToShard(userId, interactionData) {
  const shardKey = userId % 3; // Thuật toán Hashed Sharding đơn giản

  if (shardKey === 0) {
    shard_0.push(interactionData);
    console.log(`[MongoDB Router] Data of User ${userId} routed to SHARD_0`);
  } else if (shardKey === 1) {
    shard_1.push(interactionData);
    console.log(`[MongoDB Router] Data of User ${userId} routed to SHARD_1`);
  } else {
    shard_2.push(interactionData);
    console.log(`[MongoDB Router] Data of User ${userId} routed to SHARD_2`);
  }
}

// simulate users access to the system
/**
 * // Mô phỏng 10 user khác nhau cùng gửi dữ liệu lên LRS đồng thời
for (let i = 1; i <= 10; i++) {
  fetch('https://lms-backend-hf26.onrender.com/api/interact', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      userId: i, 
      courseId: Math.floor(Math.random() * 50) + 1, // Random khóa học từ 1-50
      rating: Math.floor(Math.random() * 3) + 3,    // Random điểm 3-5
      status: 'completed'
    })
  }).then(() => console.log(`Đã gửi data cho User ${i}`));
 */
// ----------------------------------------------

// --- API routes ---

app.get("/api/users", (req, res) => {
  res.json(users);
});

app.get("/api/courses", (req, res) => {
  res.json(courses);
});

app.get("/api/onboarding-options", (req, res) => {
  const categories = [...new Set(courses.map((c) => c.category))].sort();
  const difficulties = ["Beginner", "Intermediate", "Advanced"];
  const formats = ["Video", "Reading", "Quiz"];
  res.json({ categories, difficulties, formats });
});

/**
 * New student onboarding. Creates user 999, fetches AI recommendations from preferences.
 */
app.post("/api/onboard", async (req, res) => {
  const { category, difficulty, format } = req.body;
  if (!category || !difficulty || !format) {
    return res.status(400).json({ error: "category, difficulty, and format required" });
  }

  // --- SỬA Ở ĐÂY: SINH ID NGẪU NHIÊN THAY VÌ FIX CỨNG 999 ---
  // Tạo ID ngẫu nhiên từ 1000 đến 9999
  const dynamicStudentId = Math.floor(Math.random() * 9000) + 1000;

  const profile = `Goal: ${category} · Level: ${difficulty} · Prefers: ${format}`;
  const newUser = {
    id: dynamicStudentId,
    name: "New Student " + dynamicStudentId,
    profile,
    preferences: { category, difficulty, format },
  };

  onboardedUsers.set(dynamicStudentId, newUser);

  let recommendations = [];
  let dqn_suggestion = {
    format: null,
    difficulty: null,
    message: "AI service unavailable.",
  };

  try {
    const { data } = await axios.post(
      `${AI_SERVICE_URL}/api/recommendations/onboarding`,
      { category, difficulty, format },
      { timeout: 15000 },
    );
    recommendations = data.recommendations || [];
    dqn_suggestion = data.dqn_suggestion || dqn_suggestion;
  } catch (err) {
    console.warn("AI onboarding request failed:", err.message);
  }

  res.json({
    user: { id: newUser.id, name: newUser.name, profile: newUser.profile },
    completedCourses: [],
    recommendations,
    dqn_suggestion,
  });
});

/**
 * Simulate completing a course. Pushes new interaction to in-memory db,
 * fetches fresh AI recommendations from Python, returns updated dashboard.
 */

// Phase 1.1 Built an LRS (Learning Record Store) endpoint to capture real-time user micro-interactions using the xAPI standard
app.post("/api/interact", async (req, res) => {
  const { userId, courseId, rating, status } = req.body;
  if (userId == null || courseId == null) {
    return res.status(400).json({ error: "userId and courseId required" });
  }
  const uid = parseInt(userId, 10);
  const cid = parseInt(courseId, 10);
  if (isNaN(uid) || isNaN(cid)) {
    return res.status(400).json({ error: "Invalid userId or courseId" });
  }

  const user = getUser(uid);
  if (!user) return res.status(404).json({ error: "User not found" });
  const course = courses.find((c) => c.id === cid);
  if (!course) return res.status(404).json({ error: "Course not found" });

  const newInteraction = {
    userId: uid,
    courseId: cid,
    rating: typeof rating === "number" ? Math.max(1, Math.min(5, rating)) : 5,
    status: status === "dropped" ? "dropped" : "completed",
  };
  interactions.push(newInteraction);

  // Gọi hàm giả lập Sharding để log ra Terminal cho Thầy xem
  routeToShard(uid, newInteraction);

  const coursesById = Object.fromEntries(courses.map((c) => [c.id, c]));
  const userInteractions = interactions
    .filter((i) => i.userId === uid)
    .map((i) => ({ ...i, course: coursesById[i.courseId] }))
    .filter((i) => i.course);
  const completedCourses = userInteractions
    .filter((i) => i.status === "completed")
    .map((i) => ({
      courseId: i.courseId,
      title: i.course.title,
      category: i.course.category,
      difficulty: i.course.difficulty,
      format: i.course.format,
      rating: i.rating,
      status: i.status,
    }));

  let recommendations = [];
  let dqn_suggestion = {
    format: null,
    difficulty: null,
    message: "AI service unavailable.",
  };

  try {
    const { data } = await axios.post(
      `${AI_SERVICE_URL}/api/recommendations`,
      { user_id: uid, courses, users, interactions },
      { timeout: 30000 },
    );
    recommendations = data.recommendations || [];
    dqn_suggestion = data.dqn_suggestion || dqn_suggestion;
  } catch (err) {
    console.warn("AI service request failed:", err.message);
  }

  res.json({
    user: { id: user.id, name: user.name, profile: user.profile },
    completedCourses,
    recommendations,
    dqn_suggestion,
  });
});

/**
 * User dashboard: profile + completed courses from db + AI recommendations + DQN suggestion.
 * Fetches from Python AI service and combines with db data.
 */
app.get("/api/user-dashboard/:userId", async (req, res) => {
  const userId = parseInt(req.params.userId, 10);
  if (isNaN(userId)) {
    return res.status(400).json({ error: "Invalid user ID" });
  }

  const user = getUser(userId);
  if (!user) {
    return res.status(404).json({ error: "User not found" });
  }

  const coursesById = Object.fromEntries(courses.map((c) => [c.id, c]));

  const userInteractions = interactions
    .filter((i) => i.userId === userId)
    .map((i) => ({
      ...i,
      course: coursesById[i.courseId],
    }))
    .filter((i) => i.course);

  const completedCourses = userInteractions
    .filter((i) => i.status === "completed")
    .map((i) => ({
      courseId: i.courseId,
      title: i.course.title,
      category: i.course.category,
      difficulty: i.course.difficulty,
      format: i.course.format,
      rating: i.rating,
      status: i.status,
    }));

  let recommendations = [];
  let dqn_suggestion = {
    format: null,
    difficulty: null,
    message: "AI service unavailable.",
  };

  try {
    const allUsers = [...users, ...Array.from(onboardedUsers.values())];
    // --- ĐÃ SỬA: Kiểm tra xem user có nằm trong map onboardedUsers không ---
    const { data } = onboardedUsers.has(userId)
      ? await axios.post(
          `${AI_SERVICE_URL}/api/recommendations`,
          { user_id: userId, courses, users: allUsers, interactions },
          { timeout: 30000 },
        )
      : await axios.get(`${AI_SERVICE_URL}/api/recommendations/${userId}`, {
          timeout: 15000,
        });
    recommendations = data.recommendations || [];
    dqn_suggestion = data.dqn_suggestion || dqn_suggestion;
  } catch (err) {
    console.warn("AI service request failed:", err.message);
  }

  res.json({
    user: {
      id: user.id,
      name: user.name,
      profile: user.profile,
    },
    completedCourses,
    recommendations,
    dqn_suggestion,
  });
});

// SPA fallback: serve index.html for non-API routes
app.get("*", (req, res) => {
  if (req.path.startsWith("/api/")) return res.status(404).json({ error: "Not found" });
  res.sendFile(path.join(frontendPath, "index.html"));
});

app.listen(PORT, () => {
  console.log(`LMS Backend running on port ${PORT}`);
  console.log(`AI service expected at ${AI_SERVICE_URL}`);
});
