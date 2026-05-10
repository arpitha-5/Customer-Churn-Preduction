/* ==========================================================
   ChurnGuard AI - Frontend JavaScript (PRODUCTION-READY)
   ==========================================================
   Features:
   - Toast notifications (success/error)
   - Loading spinner on predict
   - Public history (no login needed)
   - History filter (all / churn / retained)
   - Clear history
   - Auto-refresh history after predict
   - Mini churn chart on history page
   - Fixed quickChat bug
   ========================================================== */

const API = "";  // Same origin

// ── State ─────────────────────────────────
let token = localStorage.getItem("churn_token");
let currentUser = JSON.parse(localStorage.getItem("churn_user") || "null");
let chartInstances = {};

// ══════════════════════════════════════════
// BULK PREDICTION
// ══════════════════════════════════════════
let latestBulkResults = [];

const bulkInput = document.getElementById('bulkFileInput');
if (bulkInput) {
    bulkInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            document.getElementById('bulkFileName').textContent = file.name;
            // Automatically run prediction upon file select
            runBulkPrediction();
        }
    });
}

async function runBulkPrediction() {
    const fileInput = document.getElementById('bulkFileInput');
    const file = fileInput.files[0];
    if (!file) return;

    const btn = document.getElementById('bulkPredictBtn');
    btn.innerHTML = '<span class="spinner"></span> Processing...';
    btn.disabled = true;

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch(`${API}/api/predict/bulk`, {
            method: "POST",
            body: formData
        });
        
        const data = await res.json();
        if (!res.ok) {
            throw new Error(data.error || "Bulk prediction failed");
        }
        
        latestBulkResults = data.results;
        
        // Render results in table
        const tbody = document.getElementById("bulkResultsBody");
        tbody.innerHTML = "";
        
        latestBulkResults.forEach((r, i) => {
            const isChurn = r.Prediction === "Churn";
            const badgeClass = isChurn ? "badge-danger" : "badge-success";
            
            tbody.innerHTML += `
                <tr>
                    <td>${i + 1}</td>
                    <td><span class="badge ${badgeClass}">${r.Prediction}</span></td>
                    <td>${r["Risk Score"] || "-"}</td>
                    <td>${r.Contract || "-"}</td>
                    <td>${r.tenure !== undefined ? r.tenure + " mo" : "-"}</td>
                    <td>$${r.MonthlyCharges || "-"}</td>
                    <td>${r.Segment || "-"}</td>
                </tr>
            `;
        });
        
        document.getElementById("bulkResultsContainer").style.display = "block";
        document.getElementById("downloadBulkBtn").style.display = "inline-block";
        showToast("success", data.message || "Bulk prediction completed!");
    } catch (err) {
        showToast("error", err.message);
    } finally {
        btn.innerHTML = 'Run Bulk Prediction';
        btn.disabled = false;
        fileInput.value = ""; // reset file input to allow re-uploading same file if needed
    }
}

function downloadBulkCSV() {
    if (!latestBulkResults || latestBulkResults.length === 0) return;
    
    // Convert JSON to CSV
    const headers = Object.keys(latestBulkResults[0]);
    const csvRows = [];
    csvRows.push(headers.join(","));
    
    for (const row of latestBulkResults) {
        const values = headers.map(h => {
            const val = row[h];
            // Escape quotes and wrap strings with commas
            if (typeof val === 'string' && (val.includes(',') || val.includes('"'))) {
                return `"${val.replace(/"/g, '""')}"`;
            }
            return val;
        });
        csvRows.push(values.join(","));
    }
    
    const blob = new Blob([csvRows.join("\n")], { type: "text/csv" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `bulk_predictions_${new Date().getTime()}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
}

// ══════════════════════════════════════════
// INIT
// ══════════════════════════════════════════
document.addEventListener("DOMContentLoaded", () => {
    updateAuthUI();
    navigate("predict");
});

// ══════════════════════════════════════════
// TOAST NOTIFICATIONS
// ══════════════════════════════════════════
function showToast(message, type = "success") {
    const container = document.getElementById("toastContainer");
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;

    const icons = { success: "\u2705", error: "\u274C", info: "\u2139\uFE0F", warning: "\u26A0\uFE0F" };
    toast.innerHTML = `
        <span class="toast-icon">${icons[type] || icons.info}</span>
        <span class="toast-message">${message}</span>
        <button class="toast-close" onclick="this.parentElement.remove()">&times;</button>
    `;

    container.appendChild(toast);

    // Animate in
    requestAnimationFrame(() => toast.classList.add("show"));

    // Auto-remove after 4s
    setTimeout(() => {
        toast.classList.remove("show");
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ══════════════════════════════════════════
// NAVIGATION
// ══════════════════════════════════════════
function navigate(page) {
    document.querySelectorAll(".page").forEach((p) => p.classList.remove("active"));
    document.querySelectorAll(".nav-item").forEach((n) => n.classList.remove("active"));

    const target = document.getElementById(`page-${page}`);
    const nav = document.querySelector(`.nav-item[data-page="${page}"]`);
    if (target) target.classList.add("active");
    if (nav) nav.classList.add("active");

    // Load page data on navigate
    if (page === "dashboard") loadDashboard();
    if (page === "eda") loadEDA();
    if (page === "history") loadHistory();

    // Close mobile sidebar
    document.getElementById("sidebar").classList.remove("open");
}

function toggleSidebar() {
    document.getElementById("sidebar").classList.toggle("open");
}

// ══════════════════════════════════════════
// AUTH
// ══════════════════════════════════════════
function showAuthModal() {
    document.getElementById("authModal").classList.add("active");
}

function closeAuthModal() {
    document.getElementById("authModal").classList.remove("active");
    document.getElementById("loginError").textContent = "";
    document.getElementById("signupError").textContent = "";
}

function switchAuth(mode) {
    document.getElementById("loginTab").classList.toggle("active", mode === "login");
    document.getElementById("signupTab").classList.toggle("active", mode === "signup");
    document.getElementById("loginForm").classList.toggle("hidden", mode !== "login");
    document.getElementById("signupForm").classList.toggle("hidden", mode !== "signup");
}

async function handleLogin(e) {
    e.preventDefault();
    const btn = document.getElementById("loginBtn");
    btn.textContent = "Logging in...";
    btn.disabled = true;
    try {
        const res = await fetch(`${API}/api/login`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                username: document.getElementById("loginUsername").value,
                password: document.getElementById("loginPassword").value,
            }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error);
        token = data.token;
        currentUser = data.user;
        localStorage.setItem("churn_token", token);
        localStorage.setItem("churn_user", JSON.stringify(currentUser));
        updateAuthUI();
        closeAuthModal();
        showToast(`Welcome back, ${currentUser.username}!`, "success");
    } catch (err) {
        document.getElementById("loginError").textContent = err.message;
    }
    btn.textContent = "Login";
    btn.disabled = false;
}

async function handleSignup(e) {
    e.preventDefault();
    const btn = document.getElementById("signupBtn");
    btn.textContent = "Creating...";
    btn.disabled = true;
    try {
        const res = await fetch(`${API}/api/signup`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                username: document.getElementById("signupUsername").value,
                email: document.getElementById("signupEmail").value,
                password: document.getElementById("signupPassword").value,
            }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error);
        token = data.token;
        currentUser = data.user;
        localStorage.setItem("churn_token", token);
        localStorage.setItem("churn_user", JSON.stringify(currentUser));
        updateAuthUI();
        closeAuthModal();
        showToast(`Account created! Welcome, ${currentUser.username}!`, "success");
    } catch (err) {
        document.getElementById("signupError").textContent = err.message;
    }
    btn.textContent = "Create Account";
    btn.disabled = false;
}

function updateAuthUI() {
    const btn = document.getElementById("authBtn");
    const info = document.getElementById("userInfo");
    if (token && currentUser) {
        btn.textContent = "Logout";
        btn.onclick = logout;
        info.style.display = "flex";
        document.getElementById("userAvatar").textContent = currentUser.username[0].toUpperCase();
        document.getElementById("userName").textContent = currentUser.username;
    } else {
        btn.textContent = "Login";
        btn.onclick = showAuthModal;
        info.style.display = "none";
    }
}

function logout() {
    token = null;
    currentUser = null;
    localStorage.removeItem("churn_token");
    localStorage.removeItem("churn_user");
    updateAuthUI();
    showToast("Logged out successfully.", "info");
}

// ══════════════════════════════════════════
// PREDICTION
// ══════════════════════════════════════════
async function handlePredict(e) {
    e.preventDefault();
    const btn = document.getElementById("predictBtn");
    const btnText = document.getElementById("predictBtnText");
    const spinner = document.getElementById("predictSpinner");

    // Show loading spinner
    btnText.style.display = "none";
    spinner.style.display = "inline-block";
    btn.disabled = true;

    const fields = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges",
    ];

    const input = {};
    fields.forEach((f) => {
        const el = document.getElementById(f);
        if (el) {
            let val = el.value;
            if (["tenure", "SeniorCitizen"].includes(f)) val = parseInt(val);
            else if (["MonthlyCharges", "TotalCharges"].includes(f)) val = parseFloat(val);
            input[f] = val;
        }
    });

    try {
        const headers = { "Content-Type": "application/json" };
        if (token) headers["Authorization"] = `Bearer ${token}`;

        const res = await fetch(`${API}/api/predict`, {
            method: "POST",
            headers,
            body: JSON.stringify(input),
        });
        const data = await res.json();
        if (!res.ok) {
            const errMsg = data.details ? data.details.join(", ") : data.error;
            throw new Error(errMsg);
        }

        showResults(data);
        showToast(
            data.prediction === 1
                ? `High churn risk detected (${data.probability}%)`
                : `Low churn risk (${data.probability}%)`,
            data.prediction === 1 ? "warning" : "success"
        );
    } catch (err) {
        showToast("Prediction failed: " + err.message, "error");
    }

    // Reset button
    btnText.style.display = "inline";
    spinner.style.display = "none";
    btn.disabled = false;
}

function showResults(data) {
    const panel = document.getElementById("predictResults");
    panel.style.display = "block";

    const card = document.getElementById("resultCard");
    const isChurn = data.prediction === 1;

    card.className = `result-card glass-card ${isChurn ? "churn" : "no-churn"}`;
    const warnSvg = '<svg viewBox="0 0 24 24" width="40" height="40" fill="none" stroke="var(--danger)" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>';
    const okSvg = '<svg viewBox="0 0 24 24" width="40" height="40" fill="none" stroke="var(--success)" stroke-width="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>';
    document.getElementById("resultIcon").innerHTML = isChurn ? warnSvg : okSvg;
    document.getElementById("resultTitle").textContent = isChurn ? "High Churn Risk" : "Low Churn Risk";
    document.getElementById("resultTitle").style.color = isChurn ? "var(--danger)" : "var(--success)";
    document.getElementById("resultSubtitle").textContent = isChurn
        ? "This customer is likely to churn. Take action now!"
        : "This customer is likely to stay. Keep up the good work!";

    // Animate probability ring
    const pct = data.probability;
    const ring = document.getElementById("ringProgress");
    const circumference = 326.73;
    const offset = circumference - (pct / 100) * circumference;
    ring.style.stroke = isChurn ? "var(--danger)" : "var(--success)";
    ring.style.transition = "none";
    ring.style.strokeDashoffset = circumference;
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            ring.style.transition = "stroke-dashoffset 1.5s ease, stroke 0.3s";
            ring.style.strokeDashoffset = offset;
        });
    });
    document.getElementById("ringText").textContent = `${pct}%`;
    document.getElementById("ringText").style.color = isChurn ? "var(--danger)" : "var(--success)";

    // Risk Score Badge
    if (data.risk_score !== undefined) {
        const wrap = document.getElementById("riskBadgeWrap");
        const badge = document.getElementById("riskBadge");
        const label = document.getElementById("riskScoreLabel");
        wrap.style.display = "flex";
        badge.textContent = `${data.risk_level} Risk`;
        badge.className = `risk-badge risk-${(data.risk_level || "low").toLowerCase()}`;
        label.textContent = `Risk Score: ${data.risk_score}/100`;
        
        if (data.segment) {
            document.getElementById("segmentLabel").textContent = `Segment: ${data.segment}`;
        }
    }

    // Alerts
    const alertsEl = document.getElementById("alertsContainer");
    if (data.alerts && data.alerts.length > 0) {
        alertsEl.style.display = "block";
        alertsEl.innerHTML = data.alerts.map(a => `
            <div class="alert-banner alert-${a.type}">
                <strong>${a.title}</strong>
                <span>${a.message}</span>
            </div>
        `).join("");
    } else {
        alertsEl.style.display = "none";
    }

    // SHAP Feature Importance Chart
    const shapCard = document.getElementById("shapCard");
    if (data.shap_values && data.shap_values.length > 0) {
        shapCard.style.display = "block";
        const labels = data.shap_values.map(s => s.feature.replace(/_/g, " "));
        const values = data.shap_values.map(s => s.value);
        const colors = values.map(v => v > 0 ? "rgba(239, 68, 68, 0.7)" : "rgba(16, 185, 129, 0.7)");
        renderChart("shapChart", "bar", {
            labels,
            datasets: [{ label: "SHAP Value", data: values, backgroundColor: colors, borderRadius: 4 }]
        }, {
            indexAxis: "y",
            scales: {
                x: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8" } },
                y: { grid: { display: false }, ticks: { color: "#94a3b8", font: { size: 10 } } }
            },
            plugins: { legend: { display: false } }
        });
    } else {
        shapCard.style.display = "none";
    }

    // Explanations
    const expCard = document.getElementById("explanationCard");
    if (data.explanation && data.explanation.length > 0) {
        expCard.style.display = "block";
        document.getElementById("explanationList").innerHTML = data.explanation
            .map(e => `
            <div class="exp-item ${e.impact}">
                <span class="exp-badge ${e.impact}">${e.impact.replace("_", " ")}</span>
                <div><strong>${e.feature}:</strong> ${e.reason}</div>
            </div>
        `).join("");
    } else {
        expCard.style.display = "none";
    }

    // Recommendations
    const recsCard = document.getElementById("recsCard");
    if (data.recommendations && data.recommendations.length > 0) {
        recsCard.style.display = "block";
        document.getElementById("recsList").innerHTML = data.recommendations
            .map(r => `
            <div class="rec-item">
                <h4>${r.action} <span class="rec-impact ${r.impact}">${r.impact}</span></h4>
                <p>${r.detail}</p>
            </div>
        `).join("");
    } else {
        recsCard.style.display = "none";
    }

    panel.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ══════════════════════════════════════════
// DASHBOARD
// ══════════════════════════════════════════
async function loadDashboard() {
    try {
        const res = await fetch(`${API}/api/dashboard`);
        const data = await res.json();

        document.getElementById("statTotal").textContent = data.total_predictions || 0;
        document.getElementById("statChurn").textContent = data.total_churn || 0;
        document.getElementById("statRetained").textContent = data.total_no_churn || 0;
        document.getElementById("statRate").textContent = `${data.churn_rate || 0}%`;

        // Churn Pie Chart
        renderChart("churnPieChart", "doughnut", {
            labels: ["Retained", "Churned"],
            datasets: [
                {
                    data: [data.total_no_churn || 0, data.total_churn || 0],
                    backgroundColor: ["#10b981", "#ef4444"],
                    borderWidth: 0,
                    hoverOffset: 8,
                },
            ],
        });

        // Model Comparison
        if (data.model_comparison) {
            const names = data.model_comparison.map((m) => m.model);
            const f1s = data.model_comparison.map((m) => m.f1_score);
            const accs = data.model_comparison.map((m) => m.accuracy);
            renderChart(
                "modelBarChart",
                "bar",
                {
                    labels: names,
                    datasets: [
                        { label: "F1 Score", data: f1s, backgroundColor: "rgba(99, 102, 241, 0.7)", borderRadius: 6 },
                        { label: "Accuracy", data: accs, backgroundColor: "rgba(139, 92, 246, 0.5)", borderRadius: 6 },
                    ],
                },
                {
                    scales: {
                        y: { beginAtZero: true, max: 1, grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8" } },
                        x: { grid: { display: false }, ticks: { color: "#94a3b8", maxRotation: 30 } },
                    },
                }
            );
        }

        // High-Risk Customers List
        const hrList = document.getElementById("highRiskList");
        if (data.high_risk_customers && data.high_risk_customers.length > 0) {
            hrList.innerHTML = `<table class="history-table"><thead><tr>
                <th>#</th><th>Contract</th><th>Tenure</th><th>Monthly $</th><th>Risk %</th><th>Date</th>
            </tr></thead><tbody>${data.high_risk_customers.map((c, i) => `<tr>
                <td>${i + 1}</td>
                <td>${c.contract}</td>
                <td>${c.tenure} mo</td>
                <td>$${c.monthly}</td>
                <td><span class="badge badge-danger">${c.probability}%</span></td>
                <td>${c.date}</td>
            </tr>`).join("")}</tbody></table>`;
        } else {
            hrList.innerHTML = '<p class="loading-text">No high-risk customers detected yet. Make some predictions first.</p>';
        }
    } catch (err) {
        console.error("Dashboard error:", err);
    }
}

function renderChart(canvasId, type, data, extraOptions = {}) {
    if (chartInstances[canvasId]) chartInstances[canvasId].destroy();
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    chartInstances[canvasId] = new Chart(ctx, {
        type,
        data,
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { labels: { color: "#94a3b8", font: { family: "'Inter', sans-serif" } } },
            },
            ...extraOptions,
        },
    });
}

// ══════════════════════════════════════════
// EDA
// ══════════════════════════════════════════
async function loadEDA() {
    // Chart descriptions and insights — suitable for report/viva
    const edaCharts = [
        {
            file: "churn_distribution.png",
            title: "1. Churn Distribution",
            desc: "This bar chart shows the count of customers who churned vs those who stayed.",
            insight: "The dataset is imbalanced — about 73% customers stayed and 27% churned. This tells us churn is a minority class and we may need techniques like oversampling during model training."
        },
        {
            file: "tenure_vs_churn.png",
            title: "2. Tenure vs Churn",
            desc: "This boxplot compares the tenure (months) of churned and retained customers.",
            insight: "Customers who churn have significantly lower tenure (median ~10 months) compared to retained customers (median ~38 months). New customers are more likely to leave."
        },
        {
            file: "monthly_charges_vs_churn.png",
            title: "3. Monthly Charges vs Churn",
            desc: "This boxplot compares monthly charges between churned and retained customers.",
            insight: "Churned customers tend to have higher monthly charges (median ~$80) compared to retained customers (median ~$65). Higher bills may drive customers away."
        },
        {
            file: "contract_vs_churn.png",
            title: "4. Contract Type vs Churn",
            desc: "This countplot shows how different contract types relate to churn.",
            insight: "Month-to-month contract customers churn the most. One-year and two-year contract customers rarely churn. Long-term contracts help retain customers."
        },
        {
            file: "correlation_heatmap.png",
            title: "5. Correlation Heatmap",
            desc: "This heatmap shows the correlation between all numeric features in the dataset.",
            insight: "TotalCharges is strongly correlated with tenure (0.83). MonthlyCharges has a moderate positive correlation with churn, confirming higher charges increase churn risk."
        }
    ];

    const grid = document.getElementById("edaGrid");
    try {
        const res = await fetch(`${API}/api/eda-images`);
        const data = await res.json();

        if (data.images && data.images.length > 0) {
            grid.innerHTML = edaCharts.map(chart => {
                // Find matching image URL from API response
                const imgUrl = data.images.find(img => img.includes(chart.file));
                if (!imgUrl) return '';
                return `
                    <div class="eda-card glass-card">
                        <h4 style="margin:0 0 4px 0;font-size:0.95rem;color:var(--text-primary)">${chart.title}</h4>
                        <p style="font-size:0.78rem;color:var(--text-secondary);margin:0 0 8px 0;">${chart.desc}</p>
                        <img src="${imgUrl}" alt="${chart.title}" loading="lazy">
                        <p style="font-size:0.78rem;color:var(--text-secondary);margin:8px 0 0 0;line-height:1.4;"><strong>Insight:</strong> ${chart.insight}</p>
                    </div>`;
            }).join('');
        } else {
            grid.innerHTML = '<p class="loading-text">No EDA images found. Run the training pipeline first.</p>';
        }
    } catch (err) {
        grid.innerHTML = '<p class="loading-text">Failed to load EDA images.</p>';
    }
}


// ══════════════════════════════════════════
// HISTORY (PUBLIC - NO LOGIN REQUIRED)
// ══════════════════════════════════════════
async function loadHistory() {
    const container = document.getElementById("historyContainer");
    const filterVal = document.getElementById("historyFilter").value;

    container.innerHTML = '<p class="loading-text">Loading history...</p>';

    try {
        const res = await fetch(`${API}/api/history?filter=${filterVal}&limit=100`);
        const data = await res.json();
        const records = data.records || [];

        // Update mini stats
        const allRes = await fetch(`${API}/api/history?filter=all&limit=500`);
        const allData = await allRes.json();
        const allRecords = allData.records || [];
        const churnCount = allRecords.filter((r) => r.prediction === 1).length;
        const retainedCount = allRecords.filter((r) => r.prediction === 0).length;

        document.getElementById("hStatTotal").textContent = allRecords.length;
        document.getElementById("hStatChurn").textContent = churnCount;
        document.getElementById("hStatRetained").textContent = retainedCount;

        // Mini pie chart
        renderChart("historyPieChart", "doughnut", {
            labels: ["Retained", "Churned"],
            datasets: [
                {
                    data: [retainedCount, churnCount],
                    backgroundColor: ["#10b981", "#ef4444"],
                    borderWidth: 0,
                    hoverOffset: 4,
                },
            ],
        });

        if (records.length === 0) {
            container.innerHTML = `<p class="loading-text">${
                filterVal !== "all" ? "No matching predictions found." : "No predictions yet. Make your first prediction!"
            }</p>`;
            return;
        }

        container.innerHTML = `
            <div style="overflow-x: auto;">
                <table class="history-table">
                    <thead><tr>
                        <th>#</th>
                        <th>Result</th>
                        <th>Probability</th>
                        <th>Contract</th>
                        <th>Tenure</th>
                        <th>Monthly ($)</th>
                        <th>Internet</th>
                        <th>Date & Time</th>
                    </tr></thead>
                    <tbody>${records
                        .map((r, i) => {
                            const inp = r.input_data || {};
                            return `
                        <tr>
                            <td>${i + 1}</td>
                            <td><span class="badge ${r.prediction === 1 ? "badge-danger" : "badge-success"}">${r.prediction === 1 ? "Churn" : "Retained"}</span></td>
                            <td>${typeof r.probability === "number" ? (r.probability < 1 ? (r.probability * 100).toFixed(1) : r.probability.toFixed(1)) : "N/A"}%</td>
                            <td>${inp.Contract || "-"}</td>
                            <td>${inp.tenure !== undefined ? inp.tenure + " mo" : "-"}</td>
                            <td>$${inp.MonthlyCharges || "-"}</td>
                            <td>${inp.InternetService || "-"}</td>
                            <td>${r.display_time || "-"}</td>
                        </tr>`;
                        })
                        .join("")}
                    </tbody>
                </table>
            </div>`;
    } catch (err) {
        container.innerHTML = '<p class="loading-text">Failed to load history.</p>';
        console.error("History error:", err);
    }
}

async function clearHistory() {
    if (!token) {
        showToast("Please login to clear history.", "warning");
        showAuthModal();
        return;
    }
    if (!confirm("Are you sure you want to clear all prediction history?")) return;

    try {
        const res = await fetch(`${API}/api/history?scope=all`, {
            method: "DELETE",
            headers: { Authorization: `Bearer ${token}` },
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error);
        showToast(data.message, "success");
        loadHistory();
    } catch (err) {
        showToast("Failed to clear history: " + err.message, "error");
    }
}

// ══════════════════════════════════════════
// CHATBOT
// ══════════════════════════════════════════
async function sendChat(e) {
    if (e && e.preventDefault) e.preventDefault();
    const input = document.getElementById("chatInput");
    const msg = input.value.trim();
    if (!msg) return;
    input.value = "";

    appendChat("user", msg);

    try {
        const headers = { "Content-Type": "application/json" };
        if (token) headers["Authorization"] = `Bearer ${token}`;
        const res = await fetch(`${API}/api/chat`, {
            method: "POST",
            headers,
            body: JSON.stringify({ message: msg }),
        });
        const data = await res.json();
        appendChat("bot", data.response || data.error);
    } catch (err) {
        appendChat("bot", "Sorry, something went wrong. Please try again.");
    }
}

function quickChat(msg) {
    document.getElementById("chatInput").value = msg;
    // Create a proper submit event
    const form = document.querySelector(".chat-input-form");
    const event = new Event("submit", { cancelable: true });
    form.dispatchEvent(event);
}

function appendChat(role, text) {
    const container = document.getElementById("chatMessages");
    const div = document.createElement("div");
    div.className = `chat-msg ${role}`;
    const botIcon = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>';
    const userIcon = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>';
    div.innerHTML = `
        <div class="msg-avatar ${role === "bot" ? "bot-avatar" : ""}">${role === "bot" ? botIcon : userIcon}</div>
        <div class="msg-content">${text}</div>
    `;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

// ══════════════════════════════════════════
// INIT
// ══════════════════════════════════════════
document.addEventListener("DOMContentLoaded", () => {
    initAuth();
});
