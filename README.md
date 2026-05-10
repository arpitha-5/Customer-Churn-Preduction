# 🛡️ ChurnGuard AI — Customer Churn Prediction System

An **industry-level, full-stack Machine Learning application** that predicts customer churn with explainable AI, real-time analytics, and intelligent retention recommendations.

---

## 🔥 Features

| Feature | Description |
|---------|-------------|
| 🎯 **Churn Prediction** | ML-powered prediction with 5 trained models |
| 📊 **Analytics Dashboard** | Real-time churn statistics with Chart.js |
| 📈 **EDA Visualizations** | 5 types of data analysis charts |
| 🧠 **Explainable AI** | Feature importance explanations for every prediction |
| 🎯 **Retention Recommendations** | Personalized action items to reduce churn |
| 🤖 **AI Chatbot** | Churn advisor that answers retention questions |
| 🔐 **Authentication** | JWT-based login/signup system |
| 📋 **Prediction History** | Track all past predictions per user |
| 💾 **Database** | SQLite storage for users, predictions, chat |
| 📱 **Responsive UI** | Premium dark glassmorphism design |

---

## 🏗️ Tech Stack

```
Frontend  → HTML5, CSS3, JavaScript, Chart.js
Backend   → Flask (Python), Flask-CORS
ML        → scikit-learn, XGBoost
Database  → SQLite
Auth      → JWT (PyJWT), bcrypt
Model     → joblib (persistence)
```

---

## 📂 Project Structure

```
customer-churn-prediction/
├── app.py                 # Flask backend (API server)
├── run_pipeline.py        # ML training pipeline
├── requirements.txt       # Python dependencies
│
├── model/
│   ├── data_preprocessing.py  # Data loading & cleaning
│   ├── eda.py                 # Exploratory Data Analysis
│   ├── train_models.py        # Model training & optimization
│   └── saved_models/          # Trained model files (.pkl)
│
├── frontend/
│   ├── index.html         # Main UI
│   ├── style.css          # Premium dark theme
│   ├── script.js          # Frontend logic
│   └── static/eda/        # EDA chart images
│
├── database/
│   └── db.py              # SQLite operations
│
└── data/
    ├── telco_churn.csv    # Dataset (auto-generated)
    └── processed/         # Processed data
```

---

## 🚀 Quick Start

### 1. Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Train ML Models

```bash
python run_pipeline.py
```

This runs:
- Data preprocessing (cleaning, encoding, scaling)
- EDA (generates visualization charts)
- Model training (5 algorithms)
- Hyperparameter optimization (GridSearchCV)
- Model saving (best model + scaler)

### 3. Start Application

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 🤖 ML Models Trained

| Model | Algorithm | Use Case |
|-------|-----------|----------|
| Logistic Regression | Linear classifier | Baseline, interpretable |
| Decision Tree | Tree-based splits | Visual decision rules |
| Random Forest | Ensemble of trees | Best overall performance |
| SVM | Support Vector Machine | High-dimensional data |
| XGBoost | Gradient boosting | State-of-the-art accuracy |

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/predict` | Predict churn for a customer |
| POST | `/api/login` | User login |
| POST | `/api/signup` | User registration |
| GET | `/api/dashboard` | Analytics statistics |
| GET | `/api/history` | Prediction history (auth) |
| POST | `/api/chat` | AI chatbot |
| GET | `/api/eda-images` | EDA visualizations |

---

## 📄 Sample API Request

```json
POST /api/predict
{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 3,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.50,
    "TotalCharges": 286.50
}
```

---

## 👤 Author

Built as an industry-level ML project demonstrating:
- End-to-end ML pipeline development
- Full-stack web application architecture
- Explainable AI and model interpretability
- REST API design with authentication
- Database integration and data persistence

---

## 📝 License

MIT License
