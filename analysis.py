# ============================================================
# PHISHING URL BEHAVIOUR CLASSIFICATION
# Complete Machine Learning Analysis
# ============================================================

import os
import re
import warnings
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. CREATE PROJECT DIRECTORIES
# ============================================================

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("=" * 70)
print("PHISHING URL BEHAVIOUR CLASSIFICATION")
print("=" * 70)


# ============================================================
# 2. LOAD UCI DATASET
# ============================================================

print("\n[1/12] Loading UCI PhiUSIIL Phishing URL Dataset...")

dataset = fetch_ucirepo(id=967)

X_original = dataset.data.features
y_original = dataset.data.targets

df = pd.concat(
    [X_original, y_original],
    axis=1
)

print("Dataset loaded successfully!")

print("Dataset shape:", df.shape)
print("Number of features:", X_original.shape[1])
print("Target column:", y_original.columns[0])


# ============================================================
# 3. BASIC DATA INSPECTION
# ============================================================

print("\n[2/12] Performing data inspection...")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
missing_values = df.isnull().sum()

total_missing = missing_values.sum()

print("Total missing values:", total_missing)

if total_missing == 0:
    print("No missing values found.")
else:
    print(missing_values[missing_values > 0])


# ============================================================
# 4. CREATE PROJECT TARGET
# ============================================================

print("\n[3/12] Creating phishing target...")

# UCI:
# 1 = Legitimate
# 0 = Phishing
#
# Project:
# 0 = Legitimate
# 1 = Phishing

target_column = y_original.columns[0]

df["is_phishing"] = (
    df[target_column] == 0
).astype(int)

print("\nTarget distribution:")
print(df["is_phishing"].value_counts())

print("\nTarget percentages:")
print(
    df["is_phishing"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 5. CLASS DISTRIBUTION PLOT
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="is_phishing"
)

plt.title("Legitimate vs Phishing URLs")
plt.xlabel("Class (0 = Legitimate, 1 = Phishing)")
plt.ylabel("Number of URLs")

plt.tight_layout()

plt.savefig(
    "plots/01_class_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 6. URL FEATURE EXTRACTION FUNCTION
# ============================================================

print("\n[4/12] Creating URL feature extractor...")


def extract_url_features(url):
    """
    Extract lexical and structural characteristics
    directly from a URL.

    These same features can later be used by
    the Streamlit application.
    """

    url = str(url)

    # Add scheme if missing
    if re.match(r"^[a-zA-Z]+://", url):
        parsed_url = urlparse(url)
    else:
        parsed_url = urlparse("http://" + url)

    hostname = parsed_url.hostname or ""
    path = parsed_url.path or ""

    # --------------------------------------------------------
    # Basic URL measurements
    # --------------------------------------------------------

    url_length = len(url)

    hostname_length = len(hostname)

    path_length = len(path)

    # --------------------------------------------------------
    # Character counts
    # --------------------------------------------------------

    digit_count = sum(
        char.isdigit()
        for char in url
    )

    special_char_count = sum(
        not char.isalnum()
        for char in url
    )

    dot_count = url.count(".")

    hyphen_count = url.count("-")

    slash_count = url.count("/")

    at_count = url.count("@")

    question_count = url.count("?")

    equal_count = url.count("=")

    ampersand_count = url.count("&")

    percent_count = url.count("%")

    # --------------------------------------------------------
    # Ratios
    # --------------------------------------------------------

    digit_ratio = (
        digit_count / url_length
        if url_length > 0
        else 0
    )

    special_char_ratio = (
        special_char_count / url_length
        if url_length > 0
        else 0
    )

    # --------------------------------------------------------
    # Subdomain count
    # --------------------------------------------------------

    hostname_parts = [
        part for part in hostname.split(".")
        if part
    ]

    if len(hostname_parts) > 2:
        subdomain_count = len(hostname_parts) - 2
    else:
        subdomain_count = 0

    # --------------------------------------------------------
    # HTTPS
    # --------------------------------------------------------

    has_https = int(
        parsed_url.scheme.lower() == "https"
    )

    # --------------------------------------------------------
    # IP address detection
    # --------------------------------------------------------

    ip_pattern = (
        r"^(?:\d{1,3}\.){3}\d{1,3}$"
    )

    has_ip_address = int(
        bool(re.match(ip_pattern, hostname))
    )

    # --------------------------------------------------------
    # Suspicious words
    # --------------------------------------------------------

    suspicious_words = [
        "login",
        "signin",
        "verify",
        "verification",
        "secure",
        "account",
        "update",
        "confirm",
        "password",
        "bank",
        "billing",
        "payment",
        "authenticate",
        "wallet",
        "credential"
    ]

    url_lower = url.lower()

    suspicious_word_count = sum(
        word in url_lower
        for word in suspicious_words
    )

    # --------------------------------------------------------
    # Punycode
    # --------------------------------------------------------

    has_punycode = int(
        "xn--" in hostname.lower()
    )

    # --------------------------------------------------------
    # Return feature dictionary
    # --------------------------------------------------------

    return {
        "url_length": url_length,
        "hostname_length": hostname_length,
        "path_length": path_length,
        "digit_count": digit_count,
        "digit_ratio": digit_ratio,
        "special_char_count": special_char_count,
        "special_char_ratio": special_char_ratio,
        "dot_count": dot_count,
        "hyphen_count": hyphen_count,
        "slash_count": slash_count,
        "at_count": at_count,
        "question_count": question_count,
        "equal_count": equal_count,
        "ampersand_count": ampersand_count,
        "percent_count": percent_count,
        "subdomain_count": subdomain_count,
        "has_https": has_https,
        "has_ip_address": has_ip_address,
        "suspicious_word_count": suspicious_word_count,
        "has_punycode": has_punycode
    }


# ============================================================
# 7. EXTRACT FEATURES FROM COMPLETE DATASET
# ============================================================

print("\n[5/12] Extracting URL features...")

url_features = df["URL"].apply(
    extract_url_features
)

url_features_df = pd.DataFrame(
    url_features.tolist()
)

print(
    "Extracted feature shape:",
    url_features_df.shape
)

print("\nExtracted features:")
print(url_features_df.columns.tolist())


# ============================================================
# 8. CREATE FINAL ML DATASET
# ============================================================

model_df = pd.concat(
    [
        url_features_df,
        df["is_phishing"].reset_index(drop=True)
    ],
    axis=1
)

print("\nFinal ML dataset:")
print(model_df.shape)

print("\nMissing values:")
print(
    model_df.isnull().sum().sum()
)


# Save extracted dataset

model_df.to_csv(
    "data/processed_url_features.csv",
    index=False
)

print(
    "\nProcessed feature dataset saved to:",
    "data/processed_url_features.csv"
)


# ============================================================
# 9. EDA
# ============================================================

print("\n[6/12] Performing Exploratory Data Analysis...")

feature_columns = [
    column
    for column in model_df.columns
    if column != "is_phishing"
]


# ------------------------------------------------------------
# Basic statistics
# ------------------------------------------------------------

statistics = model_df.describe().T

statistics.to_csv(
    "results/descriptive_statistics.csv"
)


# ------------------------------------------------------------
# URL length
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=model_df,
    x="is_phishing",
    y="url_length"
)

plt.title("URL Length by Class")
plt.xlabel("Class (0 = Legitimate, 1 = Phishing)")
plt.ylabel("URL Length")

plt.tight_layout()

plt.savefig(
    "plots/02_url_length_by_class.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Subdomain count
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=model_df,
    x="is_phishing",
    y="subdomain_count"
)

plt.title("Subdomain Count by Class")
plt.xlabel("Class (0 = Legitimate, 1 = Phishing)")
plt.ylabel("Subdomain Count")

plt.tight_layout()

plt.savefig(
    "plots/03_subdomain_by_class.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Digit ratio
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=model_df,
    x="is_phishing",
    y="digit_ratio"
)

plt.title("Digit Ratio by Class")
plt.xlabel("Class (0 = Legitimate, 1 = Phishing)")
plt.ylabel("Digit Ratio")

plt.tight_layout()

plt.savefig(
    "plots/04_digit_ratio_by_class.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Correlation heatmap
# ------------------------------------------------------------

correlation = model_df.corr(
    numeric_only=True
)

plt.figure(
    figsize=(16, 12)
)

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title(
    "Correlation Heatmap of URL Features"
)

plt.tight_layout()

plt.savefig(
    "plots/05_correlation_heatmap.png",
    dpi=300
)

plt.close()


# ============================================================
# 10. DUPLICATE FEATURE INVESTIGATION
# ============================================================

print("\n[7/12] Investigating duplicate feature representations...")

unique_feature_rows = (
    model_df[
        feature_columns
    ]
    .drop_duplicates()
)

original_rows = len(model_df)

unique_rows = len(
    unique_feature_rows
)

duplicate_rows = (
    original_rows - unique_rows
)

print(
    "Original rows:",
    original_rows
)

print(
    "Unique feature combinations:",
    unique_rows
)

print(
    "Duplicate feature rows:",
    duplicate_rows
)


# ------------------------------------------------------------
# Conflicting labels
# ------------------------------------------------------------

label_conflicts = (
    model_df
    .groupby(feature_columns)["is_phishing"]
    .nunique()
)

conflicting_groups = (
    label_conflicts[
        label_conflicts > 1
    ]
)

print(
    "Feature groups with conflicting labels:",
    len(conflicting_groups)
)


# Save duplicate investigation

duplicate_report = pd.DataFrame({
    "original_rows": [original_rows],
    "unique_feature_combinations": [unique_rows],
    "duplicate_feature_rows": [duplicate_rows],
    "conflicting_feature_groups": [
        len(conflicting_groups)
    ]
})

duplicate_report.to_csv(
    "results/duplicate_analysis.csv",
    index=False
)


# ============================================================
# 11. GROUP-AWARE TRAIN / TEST SPLIT
# ============================================================

print("\n[8/12] Creating leakage-safe train/test split...")

X = model_df[
    feature_columns
]

y = model_df[
    "is_phishing"
]


# ------------------------------------------------------------
# Create group ID for identical feature representations
# ------------------------------------------------------------

groups = pd.factorize(
    model_df[
        feature_columns
    ].apply(
        lambda row: tuple(row),
        axis=1
    )
)[0]


print(
    "Total feature groups:",
    len(np.unique(groups))
)


# ------------------------------------------------------------
# Group-aware stratified split
# ------------------------------------------------------------

sgkf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_idx, test_idx = next(
    sgkf.split(
        X,
        y,
        groups
    )
)


X_train = X.iloc[
    train_idx
]

X_test = X.iloc[
    test_idx
]

y_train = y.iloc[
    train_idx
]

y_test = y.iloc[
    test_idx
]


print(
    "\nTraining shape:",
    X_train.shape
)

print(
    "Testing shape:",
    X_test.shape
)


print(
    "\nTraining class distribution:"
)

print(
    y_train.value_counts(
        normalize=True
    ).round(4)
)


print(
    "\nTesting class distribution:"
)

print(
    y_test.value_counts(
        normalize=True
    ).round(4)
)


# ============================================================
# 12. VERIFY NO GROUP LEAKAGE
# ============================================================

train_groups = set(
    groups[train_idx]
)

test_groups = set(
    groups[test_idx]
)

group_overlap = (
    train_groups.intersection(
        test_groups
    )
)

print(
    "\nOverlapping feature groups:",
    len(group_overlap)
)

if len(group_overlap) == 0:
    print(
        "SUCCESS: No identical feature groups "
        "appear in both train and test sets."
    )
else:
    print(
        "WARNING: Group overlap detected!"
    )


# ============================================================
# 13. TRAIN MODELS
# ============================================================

print("\n[9/12] Training machine learning models...")


# ------------------------------------------------------------
# Logistic Regression
# ------------------------------------------------------------

print(
    "\nTraining Logistic Regression..."
)

logistic_model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

logistic_model.fit(
    X_train,
    y_train
)

print(
    "Logistic Regression completed."
)


# ------------------------------------------------------------
# Decision Tree
# ------------------------------------------------------------

print(
    "\nTraining Decision Tree..."
)

decision_tree_model = (
    DecisionTreeClassifier(
        max_depth=12,
        random_state=42,
        class_weight="balanced"
    )
)

decision_tree_model.fit(
    X_train,
    y_train
)

print(
    "Decision Tree completed."
)


# ------------------------------------------------------------
# KNN
# ------------------------------------------------------------

print(
    "\nTraining KNN..."
)

knn_model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "classifier",
        KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1
        )
    )
])

knn_model.fit(
    X_train,
    y_train
)

print(
    "KNN completed."
)


# ============================================================
# 14. MODEL EVALUATION
# ============================================================

print("\n[10/12] Evaluating models...")


models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_tree_model,
    "KNN": knn_model
}


results = []

predictions_dictionary = {}

probabilities_dictionary = {}


for name, model in models.items():

    print(
        f"\nEvaluating {name}..."
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions_dictionary[name] = (
        predictions
    )

    probabilities_dictionary[name] = (
        probabilities
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })


results_df = pd.DataFrame(
    results
)

results_df = (
    results_df
    .sort_values(
        by="F1 Score",
        ascending=False
    )
    .reset_index(drop=True)
)


print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# Save results

results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)


# ============================================================
# 15. CONFUSION MATRICES
# ============================================================

print(
    "\nCreating confusion matrices..."
)

for name, predictions in (
    predictions_dictionary.items()
):

    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(
        figsize=(7, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Legitimate",
            "Phishing"
        ],
        yticklabels=[
            "Legitimate",
            "Phishing"
        ]
    )

    plt.title(
        f"Confusion Matrix - {name}"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.tight_layout()

    filename = (
        name
        .lower()
        .replace(" ", "_")
        .replace("-", "")
    )

    plt.savefig(
        f"plots/confusion_matrix_{filename}.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 16. CLASSIFICATION REPORTS
# ============================================================

print(
    "\nGenerating classification reports..."
)

for name, predictions in (
    predictions_dictionary.items()
):

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "Legitimate",
            "Phishing"
        ],
        zero_division=0
    )

    filename = (
        name
        .lower()
        .replace(" ", "_")
        .replace("-", "")
    )

    with open(
        f"results/classification_report_{filename}.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            f"Classification Report - {name}\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(report)


# ============================================================
# 17. ROC CURVE
# ============================================================

print(
    "\nCreating ROC curve..."
)

plt.figure(
    figsize=(9, 7)
)

for name, probabilities in (
    probabilities_dictionary.items()
):

    RocCurveDisplay.from_predictions(
        y_test,
        probabilities,
        name=name
    )

plt.title(
    "ROC Curve - Model Comparison"
)

plt.tight_layout()

plt.savefig(
    "plots/06_roc_curve_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# 18. FEATURE IMPORTANCE - DECISION TREE
# ============================================================

print(
    "\nCalculating Decision Tree feature importance..."
)

importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": (
        decision_tree_model
        .feature_importances_
    )
})

importance_df = (
    importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
)

importance_df.to_csv(
    "results/feature_importance.csv",
    index=False
)


plt.figure(
    figsize=(10, 8)
)

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)

plt.title(
    "Decision Tree Feature Importance"
)

plt.tight_layout()

plt.savefig(
    "plots/07_feature_importance.png",
    dpi=300
)

plt.close()


# ============================================================
# 19. SELECT FINAL MODEL
# ============================================================

print(
    "\n[11/12] Selecting final model..."
)

best_model_name = results_df.iloc[0]["Model"]

print(
    "Selected model based on highest F1 Score:",
    best_model_name
)


best_model = models[
    best_model_name
]


# ============================================================
# 20. SAVE EVALUATED MODEL
# ============================================================

joblib.dump(
    best_model,
    "models/phishing_url_model_evaluated.pkl"
)

print(
    "Evaluation model saved."
)


# ============================================================
# 21. TRAIN FINAL DEPLOYMENT MODEL
# ============================================================

print(
    "\n[12/12] Training final deployment model..."
)

print(
    "The selected model will now be trained "
    "using the complete feature dataset."
)


if best_model_name == "Logistic Regression":

    final_model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

elif best_model_name == "Decision Tree":

    final_model = DecisionTreeClassifier(
        max_depth=12,
        random_state=42,
        class_weight="balanced"
    )

else:

    final_model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1
            )
        )
    ])


final_model.fit(
    X,
    y
)


# ============================================================
# 22. SAVE FINAL MODEL
# ============================================================

joblib.dump(
    final_model,
    "models/phishing_url_model.pkl"
)

print(
    "\nFinal model saved to:"
)

print(
    "models/phishing_url_model.pkl"
)


# ============================================================
# 23. SAVE FEATURE INFORMATION
# ============================================================

feature_information = pd.DataFrame({
    "feature_name": feature_columns,
    "data_type": [
        str(model_df[col].dtype)
        for col in feature_columns
    ]
})

feature_information.to_csv(
    "models/feature_columns.csv",
    index=False
)


# ============================================================
# 24. SAVE PROJECT METADATA
# ============================================================

metadata = {
    "dataset": "UCI PhiUSIIL Phishing URL Dataset",
    "dataset_id": 967,
    "original_rows": len(df),
    "original_features": X_original.shape[1],
    "extracted_features": len(feature_columns),
    "target": "is_phishing",
    "target_definition": "0 = Legitimate, 1 = Phishing",
    "duplicate_feature_rows": duplicate_rows,
    "conflicting_feature_groups": len(conflicting_groups),
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "group_overlap": len(group_overlap),
    "selected_model": best_model_name
}

metadata_df = pd.DataFrame(
    list(metadata.items()),
    columns=[
        "Property",
        "Value"
    ]
)

metadata_df.to_csv(
    "results/project_metadata.csv",
    index=False
)


# ============================================================
# 25. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    "\nDataset:",
    len(df),
    "URLs"
)

print(
    "Extracted features:",
    len(feature_columns)
)

print(
    "Duplicate feature rows:",
    duplicate_rows
)

print(
    "Conflicting feature groups:",
    len(conflicting_groups)
)

print(
    "Group overlap:",
    len(group_overlap)
)

print(
    "\nMODEL RESULTS:"
)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print(
    "\nFINAL MODEL:",
    best_model_name
)

print(
    "\nGenerated folders:"
)

print(
    "data/"
)

print(
    "models/"
)

print(
    "plots/"
)

print(
    "results/"
)

print(
    "\nMain model:"
)

print(
    "models/phishing_url_model.pkl"
)

print(
    "\nAnalysis finished!"
)

print("=" * 70)