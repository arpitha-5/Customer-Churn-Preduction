"""
=========================================================
PHASE 2: DATA PREPROCESSING
=========================================================
This module handles:
  - Loading the Telco Customer Churn dataset
  - Handling missing values
  - Removing duplicates
  - Encoding categorical variables (Label Encoding + One-Hot)
  - Feature scaling (StandardScaler)
  - Train-test split (80/20)

WHY EACH STEP?
  1. Missing values  → ML models can't handle NaN; causes errors or bad predictions
  2. Duplicates       → Biases the model toward repeated data points
  3. Encoding         → ML models need numeric input; strings aren't math-friendly
  4. Scaling          → Features on different scales (e.g., $20 vs 72 months) 
                         dominate gradient-based models unfairly
  5. Train/Test split → We need unseen data to evaluate how well the model generalizes

COMMON MISTAKES:
  - Scaling BEFORE splitting (causes data leakage — test info leaks into training)
  - Forgetting to convert 'TotalCharges' from string to float
  - Not dropping customerID (it's unique per row, no predictive value)
=========================================================
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(PROJECT_DIR, "model", "saved_models")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def generate_telco_dataset():
    """
    Generate a realistic synthetic Telco Customer Churn dataset.
    This mimics the real Kaggle Telco dataset structure.
    """
    np.random.seed(42)
    n_samples = 7043  # Same size as real Telco dataset

    # --- Generate features ---
    customer_ids = [f"CUST-{str(i).zfill(5)}" for i in range(1, n_samples + 1)]

    gender = np.random.choice(["Male", "Female"], n_samples)
    senior_citizen = np.random.choice([0, 1], n_samples, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], n_samples, p=[0.48, 0.52])
    dependents = np.random.choice(["Yes", "No"], n_samples, p=[0.30, 0.70])

    # Tenure in months (1-72)
    tenure = np.random.randint(1, 73, n_samples)

    # Phone service
    phone_service = np.random.choice(["Yes", "No"], n_samples, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No", "No phone service",
        np.random.choice(["Yes", "No"], n_samples, p=[0.42, 0.58])
    )

    # Internet service
    internet_service = np.random.choice(
        ["DSL", "Fiber optic", "No"], n_samples, p=[0.34, 0.44, 0.22]
    )

    # Internet-dependent features
    def internet_feature(internet_service, yes_prob=0.30):
        return np.where(
            internet_service == "No", "No internet service",
            np.random.choice(["Yes", "No"], n_samples, p=[yes_prob, 1 - yes_prob])
        )

    online_security = internet_feature(internet_service, 0.29)
    online_backup = internet_feature(internet_service, 0.34)
    device_protection = internet_feature(internet_service, 0.34)
    tech_support = internet_feature(internet_service, 0.29)
    streaming_tv = internet_feature(internet_service, 0.38)
    streaming_movies = internet_feature(internet_service, 0.39)

    # Contract
    contract = np.random.choice(
        ["Month-to-month", "One year", "Two year"], n_samples, p=[0.55, 0.21, 0.24]
    )

    # Billing
    paperless_billing = np.random.choice(["Yes", "No"], n_samples, p=[0.59, 0.41])
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)",
         "Credit card (automatic)"],
        n_samples, p=[0.34, 0.23, 0.22, 0.21]
    )

    # Monthly charges
    monthly_charges = np.round(np.random.uniform(18.25, 118.75, n_samples), 2)

    # Total charges correlated with tenure and monthly
    total_charges = np.round(monthly_charges * tenure * np.random.uniform(0.85, 1.15, n_samples), 2)

    # --- Generate Churn (target) with realistic correlations ---
    churn_prob = np.zeros(n_samples) + 0.15  # Base churn rate ~26.5%

    # Higher churn for month-to-month contracts
    churn_prob += np.where(contract == "Month-to-month", 0.25, 0)
    # Higher churn for fiber optic (often due to price)
    churn_prob += np.where(internet_service == "Fiber optic", 0.10, 0)
    # Higher churn for electronic check
    churn_prob += np.where(payment_method == "Electronic check", 0.08, 0)
    # Lower churn for longer tenure
    churn_prob -= tenure / 200
    # Higher churn for high monthly charges
    churn_prob += (monthly_charges - 50) / 400
    # Lower churn with tech support
    churn_prob -= np.where(tech_support == "Yes", 0.10, 0)
    # Lower churn with online security
    churn_prob -= np.where(online_security == "Yes", 0.08, 0)
    # Senior citizens churn more
    churn_prob += senior_citizen * 0.05
    # No dependents = higher churn
    churn_prob += np.where(dependents == "No", 0.03, 0)

    # Clip probabilities
    churn_prob = np.clip(churn_prob, 0.02, 0.95)

    churn = np.array(["Yes" if np.random.random() < p else "No" for p in churn_prob])

    # Add some missing values to TotalCharges (like real dataset)
    missing_indices = np.random.choice(n_samples, size=11, replace=False)

    total_charges_str = total_charges.astype(str)
    total_charges_str[missing_indices] = " "

    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges_str,
        "Churn": churn
    })

    return df


def load_data():
    """Load or generate the Telco Customer Churn dataset."""
    csv_path = os.path.join(DATA_DIR, "telco_churn.csv")
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(csv_path):
        print(f"[INFO] Loading dataset from {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print("[INFO] Generating synthetic Telco Customer Churn dataset...")
        df = generate_telco_dataset()
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Dataset saved to {csv_path}")

    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")
    return df


def handle_missing_values(df):
    """
    Handle missing / blank values.
    TotalCharges has blank strings in the real Telco dataset.
    """
    # Convert TotalCharges to numeric (blanks become NaN)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    missing_count = df.isnull().sum().sum()
    print(f"[INFO] Missing values found: {missing_count}")

    # Fill missing TotalCharges with median (robust to outliers)
    if df["TotalCharges"].isnull().any():
        median_val = df["TotalCharges"].median()
        df["TotalCharges"].fillna(median_val, inplace=True)
        print(f"[INFO] Filled TotalCharges NaN with median: {median_val:.2f}")

    return df


def remove_duplicates(df):
    """Remove duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    removed = before - after
    print(f"[INFO] Duplicates removed: {removed}")
    return df


def encode_features(df):
    """
    Encode categorical variables.
    - Binary columns → Label Encoding (0/1)
    - Multi-class columns → One-Hot Encoding
    """
    # Drop customerID — it's a unique identifier, NOT a feature
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # Encode target variable
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Binary columns
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    le = LabelEncoder()
    for col in binary_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    # Multi-class columns → One-Hot Encoding
    multi_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"
    ]
    existing_multi_cols = [c for c in multi_cols if c in df.columns]
    df = pd.get_dummies(df, columns=existing_multi_cols, drop_first=True)

    # Convert boolean columns to int
    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)

    print(f"[INFO] Encoded dataset shape: {df.shape}")
    return df


def scale_features(X_train, X_test):
    """
    Apply StandardScaler AFTER train-test split to avoid data leakage.
    StandardScaler: mean=0, std=1 → works well for most ML models.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Use SAME scaler from training

    # Save scaler for production use
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"[INFO] Scaler saved to {scaler_path}")

    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline():
    """
    Run the full preprocessing pipeline.
    Returns train/test data ready for model training.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: DATA PREPROCESSING")
    print("=" * 60)

    # Step 1: Load data
    df = load_data()

    # Step 2: Handle missing values
    df = handle_missing_values(df)

    # Step 3: Remove duplicates
    df = remove_duplicates(df)

    # Step 4: Encode features
    df = encode_features(df)

    # Step 5: Split features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    print(f"[INFO] Features shape: {X.shape}")
    print(f"[INFO] Target distribution:\n{y.value_counts(normalize=True).round(3)}")

    # Save feature names for later use
    feature_names = list(X.columns)
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))

    # Step 6: Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # Step 7: Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Save processed data
    processed_data = {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train.values,
        "y_test": y_test.values,
        "feature_names": feature_names,
        "X_train_df": X_train,
        "X_test_df": X_test,
    }
    joblib.dump(processed_data, os.path.join(PROCESSED_DIR, "processed_data.pkl"))
    print(f"[INFO] Processed data saved.")

    print("\n✅ Preprocessing complete!")
    return processed_data


if __name__ == "__main__":
    data = preprocess_pipeline()
