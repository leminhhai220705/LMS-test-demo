/**
 * LMS Student Experience — Onboarding flow + interactive dashboard.
 * Step 1: Onboarding questionnaire → AI recommendations (0.8 CB + 0.2 CF).
 * Step 2: Dashboard with Simulate Complete → updates to 0.7 CF + 0.3 CB.
 */

const API_BASE = window.APP_API_BASE_URL || "";
let STUDENT_ID = null;

const $ = (sel, root = document) => root.querySelector(sel);

const onboarding = $("#onboarding");
const dashboardEl = $("#dashboard");
const loadingState = $("#loading-state");
const errorState = $("#error-state");
const errorMessage = $("#error-message");
const userBadge = $("#user-badge");
const heroProfile = $("#hero-profile");
const dqnMessage = $("#dqn-message");
const completedList = $("#completed-list");
const completedNone = $("#completed-none");
const recommendationsGrid = $("#recommendations-grid");
const recsNone = $("#recs-none");
const onboardingForm = $("#onboarding-form");
const btnDiscover = $("#btn-discover");

// ==========================================
// --- THÊM ĐOẠN NÀY ĐỂ XÓA TRÍ NHỚ TRÌNH DUYỆT ---
// ==========================================
function resetUserSession() {
  // Đảm bảo STUDENT_ID luôn bắt đầu bằng null khi F5
  STUDENT_ID = null;

  // Hiển thị lại màn hình Onboarding
  showView(onboarding);

  // (Tùy chọn) Reset lại Form lựa chọn
  if (onboardingForm) {
    onboardingForm.reset();
  }
}

// Gọi hàm này ngay lập tức khi file JS được load (tức là mỗi khi F5)
resetUserSession();
// ==========================================

function showView(view) {
  onboarding.classList.add("hidden");
  dashboardEl.classList.add("hidden");
  loadingState.classList.add("hidden");
  errorState.classList.add("hidden");
  if (view) view.classList.remove("hidden");
}

function setError(msg) {
  errorMessage.textContent = msg;
  showView(errorState);
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

async function loadOnboardingOptions() {
  try {
    const res = await fetch(`${API_BASE}/api/onboarding-options`);
    if (!res.ok) throw new Error("Failed to load options");
    const { categories } = await res.json();
    const goalSelect = $("#goal");
    goalSelect.innerHTML =
      '<option value="">— Select your goal —</option>' +
      categories.map((c) => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join("");
  } catch (e) {
    const goalSelect = $("#goal");
    goalSelect.innerHTML =
      '<option value="">— Select your goal —</option>' +
      '<option value="IT/Coding">IT/Coding</option>' +
      '<option value="Business">Business</option>' +
      '<option value="Graphic Design">Graphic Design</option>' +
      '<option value="Mathematics">Mathematics</option>' +
      '<option value="Soft Skills">Soft Skills</option>';
  }
}

function renderCompleted(courses) {
  completedList.innerHTML = "";
  if (!courses || courses.length === 0) {
    completedNone.classList.remove("hidden");
    return;
  }
  completedNone.classList.add("hidden");
  courses.forEach((c) => {
    const el = document.createElement("div");
    el.className = "completed-item";
    el.innerHTML = `
      <div>
        <span class="title">${escapeHtml(c.title)}</span>
        <div class="meta">${escapeHtml(c.category)} · ${escapeHtml(c.format)} · ${escapeHtml(c.difficulty)}</div>
      </div>
      <span class="badge">✓ Completed</span>
    `;
    completedList.appendChild(el);
  });
}

function renderRecommendations(recs, onComplete) {
  recommendationsGrid.innerHTML = "";
  if (!recs || recs.length === 0) {
    recsNone.classList.remove("hidden");
    return;
  }
  recsNone.classList.add("hidden");
  recs.forEach((r) => {
    const card = document.createElement("div");
    card.className = "rec-card";
    const scorePct = Math.round((r.score || 0) * 100);
    card.innerHTML = `
      <div class="rec-title">${escapeHtml(r.title)}</div>
      <div class="rec-meta">${escapeHtml(r.category)} · ${escapeHtml(r.format)} · ${escapeHtml(r.difficulty)}</div>
      <div class="rec-score">AI match <span class="score-value">${scorePct}%</span></div>
      <button class="btn-complete" data-course-id="${r.courseId}">
        Simulate Completing Course
      </button>
    `;
    const btn = card.querySelector(".btn-complete");
    btn.addEventListener("click", () => onComplete(r.courseId, btn));
    recommendationsGrid.appendChild(card);
  });
}

function applyDashboardData(data) {
  heroProfile.textContent = data.user?.profile ?? "";
  userBadge.textContent = data.user?.name ?? "Student";
  userBadge.classList.remove("hidden");

  const dqn = data.dqn_suggestion;
  dqnMessage.textContent = dqn?.message ?? "No suggestion available.";

  renderCompleted(data.completedCourses ?? []);
  renderRecommendations(data.recommendations ?? [], simulateComplete);
}

async function submitOnboarding(e) {
  e.preventDefault();
  const category = $("#goal").value;
  const difficulty = $("#skillLevel").value;
  const format = $("#preferredFormat").value;
  if (!category || !difficulty || !format) return;

  showView(loadingState);
  btnDiscover.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/api/onboard`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ category, difficulty, format }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `Server error: ${res.status}`);
    }

    const data = await res.json();
    STUDENT_ID = data.user?.id ?? 999;
    applyDashboardData(data);
    showView(dashboardEl);
  } catch (e) {
    setError(e.message || "Failed to discover your path. Is the backend running?");
  } finally {
    btnDiscover.disabled = false;
  }
}

async function simulateComplete(courseId, btn) {
  if (!btn || !STUDENT_ID) return;
  btn.disabled = true;
  btn.classList.add("loading");
  btn.textContent = "Updating…";

  try {
    const res = await fetch(`${API_BASE}/api/interact`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        userId: STUDENT_ID,
        courseId,
        rating: Math.floor(Math.random() * 3) + 3,
        status: "completed",
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `Server error: ${res.status}`);
    }

    const data = await res.json();
    applyDashboardData(data);
  } catch (e) {
    btn.disabled = false;
    btn.classList.remove("loading");
    btn.textContent = "Simulate Completing Course";
    setError(e.message || "Failed to complete course.");
  }
}

onboardingForm.addEventListener("submit", submitOnboarding);
loadOnboardingOptions();
