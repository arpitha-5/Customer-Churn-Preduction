"""
==========================================================
ChurnGuard AI - Flask Backend (ADVANCED)
==========================================================
Endpoints:
  POST /api/predict      - Predict churn with SHAP + risk score
  GET  /api/history      - All prediction history (public)
  GET  /api/my-history   - Logged-in user's history (auth)
  DELETE /api/history    - Clear all history (auth)
  POST /api/login        - User login (JWT)
  POST /api/signup       - User registration
  GET  /api/dashboard    - Analytics stats + high-risk list
  POST /api/chat         - Smart AI Chatbot
  GET  /api/eda-images   - EDA chart images
  GET  /api/export/csv   - Export predictions as CSV
  GET  /api/export/pdf   - Export predictions as PDF
  GET  /api/models       - Model comparison data
==========================================================
"""
import os, sys, json, io, csv
from datetime import datetime, timedelta
from functools import wraps

from flask import Flask, request, jsonify, send_from_directory, Response, make_response
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import jwt
import bcrypt

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[WARN] SHAP not installed. Using rule-based explanations.")

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False
    print("[WARN] fpdf2 not installed. PDF export disabled.")

# ── Project paths ──────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from database.db import (
    create_user, get_user_by_username, save_prediction,
    get_prediction_history, get_all_prediction_history,
    get_dashboard_stats, save_chat, clear_all_history,
    clear_user_history
)

app = Flask(__name__, static_folder="frontend/static", static_url_path="/static")
CORS(app)

SECRET_KEY = os.environ.get("SECRET_KEY", "churn-prediction-secret-key-2024")
MODEL_DIR = os.path.join(PROJECT_DIR, "model", "saved_models")

# ══════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════
model = None
scaler = None
feature_names = None
shap_explainer = None
kmeans_model = None


def load_ml_artifacts():
    """Load the trained model, scaler, feature names, SHAP, and KMeans."""
    global model, scaler, feature_names, shap_explainer, kmeans_model
    try:
        model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
        print("[INFO] ML artifacts loaded successfully")
        
        # Load KMeans model
        try:
            kmeans_model = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))
            print("[INFO] KMeans segmentation model loaded")
        except FileNotFoundError:
            print("[WARN] KMeans model not found. Run: python model/train_kmeans.py")

        # Initialize SHAP explainer
        if HAS_SHAP and model is not None:
            try:
                # Use KernelExplainer background data for efficiency
                bg_path = os.path.join(os.path.dirname(MODEL_DIR), "..", "data", "processed", "processed_data.pkl")
                if os.path.exists(bg_path):
                    processed = joblib.load(bg_path)
                    bg_data = processed["X_train"][:100]
                else:
                    bg_data = np.zeros((1, len(feature_names)))
                shap_explainer = shap.KernelExplainer(model.predict_proba, bg_data)
                print("[INFO] SHAP explainer initialized")
            except Exception as e:
                print(f"[WARN] SHAP init failed: {e}. Using rule-based explanations.")
                shap_explainer = None
    except FileNotFoundError:
        print("[WARN] ML artifacts not found. Run: python run_pipeline.py")


load_ml_artifacts()


# ══════════════════════════════════════════
# AUTH HELPERS
# ══════════════════════════════════════════
def token_required(f):
    """Decorator: reject requests without a valid JWT token."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not token:
            return jsonify({"error": "Token required. Please login first."}), 401
        try:
            data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            request.user_id = data["user_id"]
            request.username = data["username"]
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired. Please login again."}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token."}), 401
        return f(*args, **kwargs)
    return decorated


def extract_user_id_optional():
    """Try to extract user_id from token, return None if not present."""
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if token:
        try:
            decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            return decoded.get("user_id")
        except Exception:
            pass
    return None


# ══════════════════════════════════════════
# SERVE FRONTEND
# ══════════════════════════════════════════
@app.route("/")
def serve_frontend():
    return send_from_directory("frontend", "index.html")


@app.route("/<path:path>")
def serve_static(path):
    full_path = os.path.join(PROJECT_DIR, "frontend", path)
    if os.path.exists(full_path):
        return send_from_directory("frontend", path)
    return jsonify({"error": "Not found"}), 404


# ══════════════════════════════════════════
# AUTH ENDPOINTS
# ══════════════════════════════════════════
@app.route("/api/signup", methods=["POST"])
def signup():
    data = request.json
    if not data or not all(k in data for k in ["username", "email", "password"]):
        return jsonify({"error": "Username, email, and password are required."}), 400
    if len(data["username"]) < 3:
        return jsonify({"error": "Username must be at least 3 characters."}), 400
    if len(data["password"]) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400
    try:
        pw_hash = bcrypt.hashpw(data["password"].encode(), bcrypt.gensalt()).decode()
        user = create_user(data["username"], data["email"], pw_hash)
        token = jwt.encode(
            {"user_id": user["id"], "username": user["username"],
             "exp": datetime.utcnow() + timedelta(days=7)},
            SECRET_KEY, algorithm="HS256"
        )
        return jsonify({"token": token, "user": user, "message": "Account created successfully!"}), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 409


@app.route("/api/login", methods=["POST"])
def login():
    data = request.json
    if not data or not all(k in data for k in ["username", "password"]):
        return jsonify({"error": "Username and password are required."}), 400
    user = get_user_by_username(data["username"])
    if not user or not bcrypt.checkpw(data["password"].encode(), user["password_hash"].encode()):
        return jsonify({"error": "Invalid username or password."}), 401
    token = jwt.encode(
        {"user_id": user["id"], "username": user["username"],
         "exp": datetime.utcnow() + timedelta(days=7)},
        SECRET_KEY, algorithm="HS256"
    )
    return jsonify({
        "token": token,
        "user": {"id": user["id"], "username": user["username"], "email": user["email"]},
        "message": "Login successful!"
    })


# ══════════════════════════════════════════
# PREDICTION ENDPOINT
# ══════════════════════════════════════════
FEATURE_DEFAULTS = {
    "gender": 0, "SeniorCitizen": 0, "Partner": 0, "Dependents": 0,
    "tenure": 12, "PhoneService": 1, "PaperlessBilling": 0,
    "MonthlyCharges": 50.0, "TotalCharges": 600.0,
    "InternetService": "DSL", "Contract": "One year",
    "PaymentMethod": "Credit card (automatic)",
    "MultipleLines": "No", "OnlineSecurity": "No",
    "OnlineBackup": "No", "DeviceProtection": "No",
    "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No"
}

REQUIRED_FIELDS = ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService"]


def validate_input(data):
    """Validate prediction input data. Returns (clean_data, error_msg)."""
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    if "tenure" in data:
        try:
            t = int(data["tenure"])
            if t < 0 or t > 120:
                errors.append("Tenure must be between 0 and 120 months.")
        except (ValueError, TypeError):
            errors.append("Tenure must be a number.")

    if "MonthlyCharges" in data:
        try:
            mc = float(data["MonthlyCharges"])
            if mc < 0 or mc > 500:
                errors.append("Monthly Charges must be between $0 and $500.")
        except (ValueError, TypeError):
            errors.append("Monthly Charges must be a number.")

    if "TotalCharges" in data:
        try:
            tc = float(data["TotalCharges"])
            if tc < 0:
                errors.append("Total Charges cannot be negative.")
        except (ValueError, TypeError):
            errors.append("Total Charges must be a number.")

    return errors


def prepare_input(data):
    """Convert user input to model-ready feature vector."""
    input_row = {}
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}

    for key, default in FEATURE_DEFAULTS.items():
        val = data.get(key, default)
        if isinstance(default, (int, float)) and not isinstance(val, str):
            input_row[key] = val
        elif val in binary_map:
            input_row[key] = binary_map[val]
        else:
            input_row[key] = val

    # Build feature vector matching training columns
    row = {}
    for feat in feature_names:
        if feat in input_row:
            row[feat] = input_row[feat]
        else:
            matched = False
            for orig_key in [
                "MultipleLines", "InternetService", "OnlineSecurity",
                "OnlineBackup", "DeviceProtection", "TechSupport",
                "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"
            ]:
                if feat.startswith(orig_key + "_"):
                    category = feat[len(orig_key) + 1:]
                    row[feat] = 1 if str(input_row.get(orig_key, "")) == category else 0
                    matched = True
                    break
            if not matched:
                row[feat] = 0

    df = pd.DataFrame([row])[feature_names]
    return scaler.transform(df)


def get_shap_explanation(X):
    """Get SHAP-based feature importance values for a prediction."""
    if shap_explainer is None:
        return None
    try:
        shap_values = shap_explainer.shap_values(X, nsamples=50)
        # shap_values[1] = SHAP values for class 1 (churn)
        vals = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        # Build sorted list of feature importances
        importance = []
        for i, name in enumerate(feature_names):
            importance.append({"feature": name, "value": round(float(vals[i]), 4)})
        importance.sort(key=lambda x: abs(x["value"]), reverse=True)
        return importance[:10]  # top 10 features
    except Exception as e:
        print(f"[WARN] SHAP explanation failed: {e}")
        return None


def get_feature_explanation(X, raw_input):
    """Generate human-readable feature importance explanations."""
    explanations = []
    if raw_input.get("Contract", "") == "Month-to-month":
        explanations.append({
            "feature": "Contract", "value": "Month-to-month",
            "impact": "high_risk",
            "reason": "Month-to-month contracts have 3x higher churn rate"
        })
    if float(raw_input.get("MonthlyCharges", 0)) > 70:
        explanations.append({
            "feature": "MonthlyCharges", "value": raw_input.get("MonthlyCharges"),
            "impact": "high_risk",
            "reason": "Above average monthly charges increase churn risk"
        })
    if int(raw_input.get("tenure", 0)) < 12:
        explanations.append({
            "feature": "Tenure", "value": raw_input.get("tenure"),
            "impact": "high_risk",
            "reason": "New customers (<12 months) churn more frequently"
        })
    if raw_input.get("InternetService", "") == "Fiber optic":
        explanations.append({
            "feature": "InternetService", "value": "Fiber optic",
            "impact": "medium_risk",
            "reason": "Fiber optic customers churn more, often due to pricing"
        })
    if raw_input.get("TechSupport", "No") == "No":
        explanations.append({
            "feature": "TechSupport", "value": "No",
            "impact": "medium_risk",
            "reason": "Lack of tech support increases churn probability"
        })
    if raw_input.get("OnlineSecurity", "No") == "No" and raw_input.get("InternetService", "") != "No":
        explanations.append({
            "feature": "OnlineSecurity", "value": "No",
            "impact": "medium_risk",
            "reason": "No online security increases vulnerability to churn"
        })
    if int(raw_input.get("tenure", 0)) > 36:
        explanations.append({
            "feature": "Tenure", "value": raw_input.get("tenure"),
            "impact": "low_risk",
            "reason": "Long-term customers are loyal and less likely to churn"
        })
    if raw_input.get("Contract", "") == "Two year":
        explanations.append({
            "feature": "Contract", "value": "Two year",
            "impact": "low_risk",
            "reason": "Long-term contracts significantly reduce churn"
        })
    if raw_input.get("PaymentMethod", "") == "Electronic check":
        explanations.append({
            "feature": "PaymentMethod", "value": "Electronic check",
            "impact": "medium_risk",
            "reason": "Electronic check users have a higher churn rate"
        })
    return explanations


def get_risk_score(proba):
    """Convert probability to a 0–100 risk score and risk level."""
    score = round(proba * 100, 1)
    if score >= 70:
        level = "High"
    elif score >= 40:
        level = "Medium"
    else:
        level = "Low"
    return score, level


def get_alerts(prediction, risk_score, risk_level, data):
    """Generate alerts for high-risk predictions."""
    alerts = []
    if risk_level == "High":
        alerts.append({
            "type": "critical",
            "title": "Immediate Attention Required",
            "message": f"This customer has a {risk_score}% churn risk. Immediate retention action is needed."
        })
        if data.get("Contract", "") == "Month-to-month":
            alerts.append({
                "type": "action",
                "title": "Offer Contract Upgrade",
                "message": "Upgrade to annual contract with 20% discount immediately."
            })
        if float(data.get("MonthlyCharges", 0)) > 80:
            alerts.append({
                "type": "action",
                "title": "Apply Emergency Discount",
                "message": "Apply a 15% loyalty discount to reduce monthly charges."
            })
    elif risk_level == "Medium":
        alerts.append({
            "type": "warning",
            "title": "Monitor This Customer",
            "message": f"Moderate churn risk ({risk_score}%). Schedule a follow-up within 7 days."
        })
    return alerts


def get_recommendations(data, prediction, probability):
    """Generate personalized retention recommendations."""
    recs = []
    if prediction == 1:
        if data.get("Contract", "") == "Month-to-month":
            recs.append({
                "action": "Offer Annual Contract",
                "impact": "High",
                "detail": "Customers on month-to-month contracts churn 3x more. Offer 20% discount for annual commitment."
            })
        if float(data.get("MonthlyCharges", 0)) > 70:
            recs.append({
                "action": "Provide Pricing Review",
                "impact": "High",
                "detail": "High monthly charges drive churn. Consider a loyalty discount or plan optimization."
            })
        if data.get("TechSupport", "No") == "No":
            recs.append({
                "action": "Add Free Tech Support",
                "impact": "Medium",
                "detail": "Customers without tech support are more likely to churn. Offer 3 months free."
            })
        if data.get("OnlineSecurity", "No") == "No":
            recs.append({
                "action": "Bundle Online Security",
                "impact": "Medium",
                "detail": "Add online security to the plan. It significantly reduces churn probability."
            })
        if int(data.get("tenure", 0)) < 12:
            recs.append({
                "action": "New Customer Engagement",
                "impact": "High",
                "detail": "New customers need special attention. Assign a dedicated account manager."
            })
        if data.get("PaymentMethod", "") == "Electronic check":
            recs.append({
                "action": "Switch Payment Method",
                "impact": "Low",
                "detail": "Electronic check users churn more. Offer incentive to switch to auto-pay."
            })
        if not recs:
            recs.append({
                "action": "Proactive Outreach",
                "impact": "Medium",
                "detail": "Schedule a customer satisfaction call to identify and address concerns."
            })
    else:
        recs.append({
            "action": "Maintain Service Quality",
            "impact": "Low",
            "detail": "Customer is likely to stay. Focus on upselling and loyalty rewards."
        })
        if int(data.get("tenure", 0)) > 24:
            recs.append({
                "action": "Offer Loyalty Reward",
                "impact": "Medium",
                "detail": "Reward long-term loyalty with exclusive perks or upgrade offers."
            })
    return recs


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    POST /api/predict
    Returns: prediction, risk_score, risk_level, SHAP values, alerts, recommendations.
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Run: python run_pipeline.py"}), 503

    data = request.json
    if not data:
        return jsonify({"error": "No input data provided. Send JSON body."}), 400

    validation_errors = validate_input(data)
    if validation_errors:
        return jsonify({"error": "Validation failed", "details": validation_errors}), 400

    try:
        X = prepare_input(data)
        prediction = int(model.predict(X)[0])
        proba = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else (0.8 if prediction else 0.2)

        # Risk scoring
        risk_score, risk_level = get_risk_score(proba)

        # SHAP feature importance
        shap_data = get_shap_explanation(X)

        # Rule-based explanations
        explanation = get_feature_explanation(X, data)
        recommendations = get_recommendations(data, prediction, proba)
        alerts = get_alerts(prediction, risk_score, risk_level, data)

        # Customer Segmentation
        segment = "Unknown Segment"
        if kmeans_model is not None:
            try:
                cluster_id = kmeans_model.predict(X)[0]
                # Map cluster IDs to descriptive segments (logic based on typical K-Means results)
                # We'll use tenure and charges to provide descriptive names
                tenure = int(data.get("tenure", 0))
                monthly = float(data.get("MonthlyCharges", 0))
                if tenure > 36 and monthly > 70:
                    segment = "High Value & Loyal"
                elif tenure > 36:
                    segment = "Steady & Reliable"
                elif tenure <= 12 and monthly > 70:
                    segment = "High Risk & New"
                elif tenure <= 12:
                    segment = "New & Budget"
                else:
                    segment = "Average User"
            except Exception as e:
                print(f"[WARN] Segmentation failed: {e}")

        # Save to database
        user_id = extract_user_id_optional()
        save_prediction(user_id, data, prediction, proba, explanation, recommendations)

        return jsonify({
            "prediction": prediction,
            "churn": "Yes" if prediction == 1 else "No",
            "probability": round(proba * 100, 1),
            "risk_score": risk_score,
            "risk_level": risk_level,
            "segment": segment,
            "shap_values": shap_data,
            "explanation": explanation,
            "recommendations": recommendations,
            "alerts": alerts,
            "message": "Prediction completed successfully!"
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/predict/bulk", methods=["POST"])
def bulk_predict():
    """
    POST /api/predict/bulk
    Accepts a CSV file, performs predictions, and returns a CSV of results.
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files are allowed"}), 400

    try:
        df_input = pd.read_csv(file)
        results = []

        for index, row in df_input.iterrows():
            data = row.to_dict()
            try:
                X = prepare_input(data)
                pred = int(model.predict(X)[0])
                prob = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else (0.8 if pred else 0.2)
                risk_score, risk_level = get_risk_score(prob)
                
                segment = "Average User"
                if kmeans_model is not None:
                    tenure = int(data.get("tenure", 0))
                    monthly = float(data.get("MonthlyCharges", 0))
                    if tenure > 36 and monthly > 70:
                        segment = "High Value & Loyal"
                    elif tenure > 36:
                        segment = "Steady & Reliable"
                    elif tenure <= 12 and monthly > 70:
                        segment = "High Risk & New"
                    elif tenure <= 12:
                        segment = "New & Budget"
                
                data["Prediction"] = "Churn" if pred == 1 else "Retained"
                data["Risk Score"] = risk_score
                data["Risk Level"] = risk_level
                data["Segment"] = segment
                results.append(data)
            except Exception as e:
                # If a row fails validation, append with error
                data["Prediction"] = f"Error: {str(e)}"
                results.append(data)

        return jsonify({"results": results, "message": "Bulk prediction completed successfully!"})

    except Exception as e:
        return jsonify({"error": f"Bulk prediction failed: {str(e)}"}), 500


# ══════════════════════════════════════════
# HISTORY ENDPOINTS (PUBLIC + AUTH)
# ══════════════════════════════════════════
@app.route("/api/history", methods=["GET"])
def history_public():
    """
    GET /api/history
    Returns ALL prediction history (no login required).
    Query params: ?filter=churn | ?filter=no_churn | ?limit=100
    """
    try:
        filter_type = request.args.get("filter", "all")
        limit = min(int(request.args.get("limit", 100)), 500)

        records = get_all_prediction_history(limit=limit)

        # Apply churn filter
        if filter_type == "churn":
            records = [r for r in records if r["prediction"] == 1]
        elif filter_type == "no_churn":
            records = [r for r in records if r["prediction"] == 0]

        return jsonify({
            "records": records,
            "total": len(records),
            "filter": filter_type
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/my-history", methods=["GET"])
@token_required
def my_history():
    """GET /api/my-history - Returns logged-in user's history only."""
    records = get_prediction_history(user_id=request.user_id)
    return jsonify({"records": records, "total": len(records)})


@app.route("/api/history", methods=["DELETE"])
@token_required
def delete_history():
    """DELETE /api/history - Clear history. ?scope=all (admin) or ?scope=mine (user)."""
    scope = request.args.get("scope", "mine")
    try:
        if scope == "all":
            count = clear_all_history()
            return jsonify({"message": f"Cleared all {count} predictions.", "count": count})
        else:
            count = clear_user_history(request.user_id)
            return jsonify({"message": f"Cleared {count} of your predictions.", "count": count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════
# DASHBOARD (ENHANCED)
# ══════════════════════════════════════════
@app.route("/api/dashboard", methods=["GET"])
def dashboard():
    try:
        stats = get_dashboard_stats()
        meta_path = os.path.join(MODEL_DIR, "best_model_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                stats["model_info"] = json.load(f)
        comparison_path = os.path.join(MODEL_DIR, "model_comparison.json")
        if os.path.exists(comparison_path):
            with open(comparison_path) as f:
                stats["model_comparison"] = json.load(f)

        # High-risk customers from recent predictions
        all_preds = get_all_prediction_history(limit=200)
        high_risk = []
        for r in all_preds:
            prob = r.get("probability", 0)
            if isinstance(prob, (int, float)) and (prob > 0.7 if prob <= 1 else prob > 70):
                inp = r.get("input_data", {})
                high_risk.append({
                    "id": r.get("id"),
                    "contract": inp.get("Contract", "-"),
                    "tenure": inp.get("tenure", "-"),
                    "monthly": inp.get("MonthlyCharges", "-"),
                    "probability": round(prob * 100, 1) if prob <= 1 else round(prob, 1),
                    "date": r.get("display_time", "-")
                })
        stats["high_risk_customers"] = high_risk[:10]
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════
# SMART CHATBOT (ENHANCED)
# ══════════════════════════════════════════
CHATBOT_RESPONSES = {
    "churn": "Customer churn occurs when customers stop using your service. Key factors include: contract type, monthly charges, tenure, tech support, and internet service type. In our dataset, ~27% of customers churned.",
    "reduce": "To reduce churn: 1) Offer annual contracts with discounts, 2) Provide free tech support, 3) Review pricing for high-bill customers, 4) Engage new customers proactively, 5) Bundle security services. These strategies can reduce churn by up to 30%.",
    "contract": "Month-to-month contracts have ~42% churn rate vs ~3% for two-year contracts. Offering a 15-20% discount for annual commitment can convert 25% of monthly users.",
    "price": "High monthly charges (>$70) significantly increase churn. The sweet spot is $50-65/month. Consider loyalty discounts, plan optimization, or value-added services to justify pricing.",
    "tenure": "Customers with <12 months tenure churn the most (~47%). The first 90 days are critical. Focus on: welcome calls, onboarding guides, dedicated support, and early engagement programs.",
    "support": "Adding tech support reduces churn by ~15%. Consider offering free support for the first 6 months. 24/7 chat support costs less than losing customers.",
    "payment": "Electronic check users churn ~33% more. Incentivize switching to automatic bank transfer or credit card payments with a $5/month discount.",
    "internet": "Fiber optic customers churn more despite faster speeds, often due to higher pricing ($10-20 more). Ensure competitive pricing and bundle value-added services.",
    "senior": "Senior citizens have ~41% higher churn rates. Offer simplified plans, dedicated helplines, senior discounts (10-15%), and easy-to-understand billing.",
    "help": "I can help you understand: churn factors, retention strategies, pricing impact, contract analysis, customer segmentation, risk scoring, SHAP explanations, and model performance. Just ask!",
    "shap": "SHAP (SHapley Additive exPlanations) shows which features contributed most to each prediction. Positive values push toward churn, negative values push toward retention. This makes our predictions interpretable and trustworthy.",
    "risk": "Risk scoring converts the churn probability to a 0-100 scale: Low (0-39), Medium (40-69), High (70-100). High-risk customers need immediate attention with personalized offers.",
    "model": "We trained 5 models: Logistic Regression, Decision Tree, Random Forest, SVM, and XGBoost. The best model is selected based on F1-score, which balances precision and recall for our imbalanced dataset.",
    "feature": "Top churn predictors: 1) Contract type (strongest), 2) Monthly charges, 3) Tenure, 4) Tech support, 5) Online security, 6) Internet service type, 7) Payment method.",
    "accuracy": "Our best model achieves ~79% accuracy with an F1-score of ~0.34. Precision is 58% (churn predictions correctness) and recall is 24% (catching actual churners). We optimize for F1 to balance both.",
    "dashboard": "The dashboard shows: total predictions, churn rate, churned vs retained counts, model comparison chart, and a list of high-risk customers needing immediate attention.",
    "export": "You can export prediction history as CSV or PDF from the History page. CSV is great for Excel analysis, PDF for formal reports and presentations.",
    "recommend": "Our AI recommendation system suggests personalized retention actions based on each customer's profile: contract upgrades, pricing reviews, tech support bundles, and payment method changes.",
}


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    msg = data.get("message", "").lower().strip() if data else ""
    if not msg:
        return jsonify({"error": "Message is required."}), 400

    response = None

    # Smart matching: check for multiple keywords
    matched_keys = [key for key in CHATBOT_RESPONSES if key in msg]
    if matched_keys:
        # If multiple matches, combine responses
        if len(matched_keys) > 1:
            parts = [CHATBOT_RESPONSES[k] for k in matched_keys[:2]]
            response = " | ".join(parts)
        else:
            response = CHATBOT_RESPONSES[matched_keys[0]]

    # Handle common question patterns
    if response is None:
        if any(w in msg for w in ["why", "reason", "cause", "factor"]):
            response = CHATBOT_RESPONSES["churn"]
        elif any(w in msg for w in ["how", "prevent", "stop", "retain", "save"]):
            response = CHATBOT_RESPONSES["reduce"]
        elif any(w in msg for w in ["what", "explain", "tell"]):
            response = CHATBOT_RESPONSES["help"]
        elif any(w in msg for w in ["hi", "hello", "hey"]):
            response = "Hello! I'm the ChurnGuard AI Advisor. I can help you understand churn patterns, retention strategies, and our ML models. What would you like to know?"
        elif any(w in msg for w in ["thank", "thanks", "bye"]):
            response = "You're welcome! Feel free to ask more questions anytime. Happy analyzing!"
        else:
            response = "I'm the Churn Advisor bot. I can help with: churn analysis, retention strategies, pricing, contracts, tenure, support, risk scoring, SHAP explanations, model comparison, and export features. Try asking about any of these topics!"

    user_id = extract_user_id_optional()
    save_chat(user_id, msg, response)
    return jsonify({"response": response})


# ══════════════════════════════════════════
# EDA IMAGES
# ══════════════════════════════════════════
@app.route("/api/eda-images", methods=["GET"])
def eda_images():
    eda_dir = os.path.join(PROJECT_DIR, "frontend", "static", "eda")
    if os.path.exists(eda_dir):
        images = sorted([f for f in os.listdir(eda_dir) if f.endswith(".png")])
        return jsonify({"images": [f"/static/eda/{img}" for img in images]})
    return jsonify({"images": []})


# ══════════════════════════════════════════
# EXPORT ENDPOINTS
# ══════════════════════════════════════════
@app.route("/api/export/csv", methods=["GET"])
def export_csv():
    """Export prediction history as CSV file."""
    try:
        records = get_all_prediction_history(limit=500)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["#", "Result", "Probability", "Contract", "Tenure",
                         "Monthly Charges", "Internet", "Payment", "Date"])
        for i, r in enumerate(records, 1):
            inp = r.get("input_data", {})
            prob = r.get("probability", 0)
            prob_pct = f"{prob * 100:.1f}%" if prob <= 1 else f"{prob:.1f}%"
            writer.writerow([
                i,
                "Churn" if r.get("prediction") == 1 else "Retained",
                prob_pct,
                inp.get("Contract", "-"),
                inp.get("tenure", "-"),
                inp.get("MonthlyCharges", "-"),
                inp.get("InternetService", "-"),
                inp.get("PaymentMethod", "-"),
                r.get("display_time", "-")
            ])
        resp = Response(output.getvalue(), mimetype="text/csv")
        resp.headers["Content-Disposition"] = "attachment; filename=churn_predictions.csv"
        return resp
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/export/pdf", methods=["GET"])
def export_pdf():
    """Export prediction history as PDF report."""
    if not HAS_FPDF:
        return jsonify({"error": "PDF export not available. Install: pip install fpdf2"}), 503
    try:
        records = get_all_prediction_history(limit=100)
        stats = get_dashboard_stats()

        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Title
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 12, "ChurnGuard AI - Prediction Report", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(8)

        # Summary Stats
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Summary Statistics", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"Total Predictions: {stats['total_predictions']}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 6, f"Churned: {stats['total_churn']}  |  Retained: {stats['total_no_churn']}  |  Churn Rate: {stats['churn_rate']}%", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(6)

        # Table Header
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Prediction History", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "B", 8)
        col_widths = [10, 18, 22, 35, 18, 22, 30, 35]
        headers = ["#", "Result", "Prob %", "Contract", "Tenure", "Monthly$", "Internet", "Date"]
        for w, h in zip(col_widths, headers):
            pdf.cell(w, 7, h, border=1)
        pdf.ln()

        # Table Rows
        pdf.set_font("Helvetica", "", 7)
        for i, r in enumerate(records[:50], 1):
            inp = r.get("input_data", {})
            prob = r.get("probability", 0)
            prob_str = f"{prob * 100:.1f}" if prob <= 1 else f"{prob:.1f}"
            row_data = [
                str(i),
                "Churn" if r.get("prediction") == 1 else "Retain",
                prob_str,
                str(inp.get("Contract", "-"))[:15],
                str(inp.get("tenure", "-")),
                str(inp.get("MonthlyCharges", "-")),
                str(inp.get("InternetService", "-"))[:12],
                str(r.get("display_time", "-"))[:16]
            ]
            for w, val in zip(col_widths, row_data):
                pdf.cell(w, 6, val, border=1)
            pdf.ln()

        pdf_bytes = pdf.output()
        resp = make_response(pdf_bytes)
        resp.headers["Content-Type"] = "application/pdf"
        resp.headers["Content-Disposition"] = "attachment; filename=churn_report.pdf"
        return resp
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════
# MODEL COMPARISON
# ══════════════════════════════════════════
@app.route("/api/models", methods=["GET"])
def model_comparison():
    """Return detailed model comparison data."""
    try:
        comparison_path = os.path.join(MODEL_DIR, "model_comparison.json")
        meta_path = os.path.join(MODEL_DIR, "best_model_meta.json")
        result = {"models": [], "best_model": None}
        if os.path.exists(comparison_path):
            with open(comparison_path) as f:
                result["models"] = json.load(f)
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                result["best_model"] = json.load(f)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# ══════════════════════════════════════════
# RUN SERVER
# ══════════════════════════════════════════
if __name__ == "__main__":
    print("\n  ChurnGuard AI - Advanced Customer Churn Prediction")
    print("  http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)

