

# ─────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# Make output folder for all plots
os.makedirs("outputs", exist_ok=True)

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
DATA_PATH   = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
TARGET      = "Attrition"
RANDOM_SEED = 42
TEST_SIZE   = 0.2

PALETTE     = {"Yes": "#E24B4A", "No": "#378ADD"}
PLOT_STYLE  = "white"


# ============================================================
#  STEP 1 — LOAD & FIRST LOOK
# ============================================================
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print("=" * 55)
    print("STEP 1 | Dataset loaded")
    print(f"  Shape      : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Target     : '{TARGET}'")
    print(f"  Attrition  : {df[TARGET].value_counts().to_dict()}")
    attrition_rate = (df[TARGET] == "Yes").mean() * 100
    print(f"  Rate       : {attrition_rate:.1f}%  ← imbalanced dataset!")
    print("=" * 55)
    return df


# ============================================================
#  STEP 2 — EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
def run_eda(df: pd.DataFrame) -> None:
    print("\nSTEP 2 | Exploratory Data Analysis")
    print("  Generating EDA plots...")
    sns.set_style(PLOT_STYLE)
    
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Employee Attrition — Exploratory Data Analysis",
                 fontsize=16, fontweight="bold", y=1.01)

    # ── 2a. Attrition distribution (donut chart) ─────────────
    ax1 = fig.add_subplot(3, 3, 1)
    counts = df[TARGET].value_counts()
    wedges, texts, autotexts = ax1.pie(
        counts, labels=counts.index,
        colors=[PALETTE["Yes"], PALETTE["No"]],
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(width=0.55)
    )
    for at in autotexts:
        at.set_fontsize(11)
    ax1.set_title("Overall Attrition Split")

    # ── 2b. Attrition by Department ───────────────────────────
    ax2 = fig.add_subplot(3, 3, 2)
    dept_attr = (
        df.groupby("Department")[TARGET]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .sort_values(ascending=False)
    )
    dept_attr.plot(kind="barh", ax=ax2, color="#E24B4A", edgecolor="white")
    ax2.set_xlabel("Attrition Rate (%)")
    ax2.set_title("Attrition Rate by Department")
    for i, v in enumerate(dept_attr):
        ax2.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)

    # ── 2c. Age distribution ──────────────────────────────────
    ax3 = fig.add_subplot(3, 3, 3)
    for label, color in PALETTE.items():
        df[df[TARGET] == label]["Age"].plot(
            kind="kde", ax=ax3, color=color, label=label, linewidth=2
        )
    ax3.set_title("Age Distribution by Attrition")
    ax3.set_xlabel("Age")
    ax3.legend(title="Attrition")

    # ── 2d. Monthly Income (box plot) ─────────────────────────
    ax4 = fig.add_subplot(3, 3, 4)
    df.boxplot(column="MonthlyIncome", by=TARGET, ax=ax4,
               patch_artist=True,
               boxprops=dict(facecolor="#B5D4F4"),
               medianprops=dict(color="#185FA5", linewidth=2))
    ax4.set_title("Monthly Income vs Attrition")
    ax4.set_xlabel("Attrition")
    plt.sca(ax4)
    plt.title("Monthly Income vs Attrition")
    plt.suptitle("")

    # ── 2e. Overtime ──────────────────────────────────────────
    ax5 = fig.add_subplot(3, 3, 5)
    ot_attr = df.groupby(["OverTime", TARGET]).size().unstack(fill_value=0)
    ot_pct  = ot_attr.div(ot_attr.sum(axis=1), axis=0) * 100
    ot_pct.plot(kind="bar", ax=ax5, color=[PALETTE["No"], PALETTE["Yes"]],
                edgecolor="white", rot=0)
    ax5.set_title("Attrition by Overtime")
    ax5.set_xlabel("Overtime")
    ax5.set_ylabel("Percentage (%)")
    ax5.legend(title="Attrition")

    # ── 2f. Job Satisfaction ──────────────────────────────────
    ax6 = fig.add_subplot(3, 3, 6)
    js_attr = (
        df.groupby("JobSatisfaction")[TARGET]
        .apply(lambda x: (x == "Yes").mean() * 100)
    )
    js_attr.plot(kind="bar", ax=ax6, color="#534AB7", edgecolor="white", rot=0)
    ax6.set_title("Attrition Rate by Job Satisfaction (1=Low)")
    ax6.set_xlabel("Job Satisfaction Level")
    ax6.set_ylabel("Attrition Rate (%)")

    # ── 2g. Years at Company ──────────────────────────────────
    ax7 = fig.add_subplot(3, 3, 7)
    for label, color in PALETTE.items():
        df[df[TARGET] == label]["YearsAtCompany"].plot(
            kind="kde", ax=ax7, color=color, label=label, linewidth=2
        )
    ax7.set_title("Tenure (Years at Company)")
    ax7.set_xlabel("Years")
    ax7.legend(title="Attrition")

    # ── 2h. Work-Life Balance ─────────────────────────────────
    ax8 = fig.add_subplot(3, 3, 8)
    wlb = (
        df.groupby("WorkLifeBalance")[TARGET]
        .apply(lambda x: (x == "Yes").mean() * 100)
    )
    wlb.plot(kind="bar", ax=ax8, color="#1D9E75", edgecolor="white", rot=0)
    ax8.set_title("Attrition Rate by Work-Life Balance (1=Bad)")
    ax8.set_xlabel("Work-Life Balance")
    ax8.set_ylabel("Attrition Rate (%)")

    # ── 2i. Correlation heatmap (numeric only) ───────────────
    ax9 = fig.add_subplot(3, 3, 9)
    df_temp = df.copy()
    df_temp[TARGET] = (df_temp[TARGET] == "Yes").astype(int)
    numeric_cols = df_temp.select_dtypes(include=[np.number]).columns.tolist()
    top_corr = (
        df_temp[numeric_cols]
        .corr()[TARGET]
        .drop(TARGET)
        .abs()
        .sort_values(ascending=False)
        .head(8)
    )
    top_corr.sort_values().plot(kind="barh", ax=ax9, color="#7F77DD", edgecolor="white")
    ax9.set_title("Top Features Correlated with Attrition")
    ax9.set_xlabel("|Correlation|")

    plt.tight_layout()
    plt.savefig("outputs/01_eda.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → Saved: outputs/01_eda.png")
    print("  EDA complete!")


# ============================================================
#  STEP 3 — PREPROCESSING
# ============================================================
def preprocess(df: pd.DataFrame):
    print("\nSTEP 3 | Preprocessing")

    df = df.copy()

    # Drop columns that add no signal (constant or leaking)
    useless_cols = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
    df.drop(columns=useless_cols, inplace=True)
    print(f"  Dropped     : {useless_cols}")

    # Encode target
    df[TARGET] = (df[TARGET] == "Yes").astype(int)   # 1 = left, 0 = stayed

    # Identify column types
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols.remove(TARGET)

    print(f"  Categorical : {cat_cols}")
    print(f"  Numeric     : {len(num_cols)} features")

    # Label-encode categoricals
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    # Features / target split
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Train / test split (stratified to preserve class ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # Handle class imbalance with SMOTE on training set only
    smote = SMOTE(random_state=RANDOM_SEED)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"  Before SMOTE: {dict(y_train.value_counts())}")
    print(f"  After  SMOTE: {dict(pd.Series(y_train_sm).value_counts())}")

    # Scale numeric features
    scaler = StandardScaler()
    X_train_sm[num_cols] = scaler.fit_transform(X_train_sm[num_cols])
    X_test[num_cols]     = scaler.transform(X_test[num_cols])

    print(f"  Train size  : {X_train_sm.shape[0]}  |  Test size: {X_test.shape[0]}")
    return X_train_sm, X_test, y_train_sm, y_test, X.columns.tolist(), scaler, le_dict, num_cols


# ============================================================
#  STEP 4 — TRAIN MODEL
# ============================================================
def train_model(X_train, y_train):
    print("\nSTEP 4 | Training Random Forest")

    # Hyperparameter grid
    param_grid = {
        "n_estimators"     : [100, 200],
        "max_depth"        : [8, 12, None],
        "min_samples_split": [2, 5],
        "class_weight"     : ["balanced"]
    }

    rf = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)

    grid_search = GridSearchCV(
        rf, param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"  Best params : {grid_search.best_params_}")
    print(f"  CV ROC-AUC  : {grid_search.best_score_:.4f}")

    # Save model
    joblib.dump(best_model, "outputs/attrition_model.pkl")
    print("  → Saved: outputs/attrition_model.pkl")

    return best_model


# ============================================================
#  STEP 5 — EVALUATE MODEL
# ============================================================
def evaluate_model(model, X_test, y_test) -> None:
    print("\nSTEP 5 | Evaluation")

    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Stayed", "Left"]))
    print(f"  ROC-AUC Score : {roc_auc_score(y_test, y_pred_prob):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Model Evaluation", fontsize=14, fontweight="bold")

    # ── Confusion Matrix ─────────────────────────────────────
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, y_pred),
        display_labels=["Stayed", "Left"]
    ).plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Confusion Matrix")

    # ── ROC Curve ────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc_score   = roc_auc_score(y_test, y_pred_prob)
    axes[1].plot(fpr, tpr, color="#378ADD", linewidth=2,
                 label=f"ROC Curve (AUC = {auc_score:.3f})")
    axes[1].plot([0, 1], [0, 1], "--", color="#888780", linewidth=1, label="Random baseline")
    axes[1].fill_between(fpr, tpr, alpha=0.1, color="#378ADD")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("outputs/02_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → Saved: outputs/02_evaluation.png")


# ============================================================
#  STEP 6 — FEATURE IMPORTANCE + SHAP EXPLAINABILITY
# ============================================================
def explain_model(model, X_test, feature_names: list) -> None:
    print("\nSTEP 6 | Feature Importance & SHAP Explainability")

    # ── 6a. Built-in Feature Importance ──────────────────────
    importance_df = pd.DataFrame({
        "Feature"   : feature_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).head(15)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Why Are Employees Leaving? — Model Insights",
                 fontsize=14, fontweight="bold")

    # Bar chart (Top 15 features)
    colors = plt.cm.RdYlGn_r(
        np.linspace(0.15, 0.85, len(importance_df))
    )
    axes[0].barh(importance_df["Feature"][::-1],
                 importance_df["Importance"][::-1],
                 color=colors[::-1], edgecolor="white")
    axes[0].set_title("Top 15 Feature Importances (Random Forest)")
    axes[0].set_xlabel("Importance Score")
    axes[0].axvline(x=importance_df["Importance"].mean(),
                    color="gray", linestyle="--", linewidth=1,
                    label="Mean importance")
    axes[0].legend()

    # ── 6b. SHAP Summary Plot ─────────────────────────────────
    print("  Computing SHAP values (this takes ~30 seconds)...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        # For binary classification, shap_values[1] is for class 1 (attrition)
        sv = shap_values[1]
    else:
        # For multi-class or newer versions
        sv = shap_values
    
    # Ensure sv is 2D
    if len(sv.shape) == 3:
        sv = sv[:, :, 1]  # Take the second class if 3D
    
    print(f"  SHAP values shape: {sv.shape}")

    # SHAP summary (beeswarm) in second subplot
    plt.sca(axes[1])
    shap.summary_plot(
        sv, X_test,
        feature_names=feature_names,
        show=False,
        plot_type="dot",
        max_display=15,
        color_bar=True
    )
    axes[1].set_title("SHAP Summary Plot\n(Red = increases attrition risk)", fontsize=11)

    plt.tight_layout()
    plt.savefig("outputs/03_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → Saved: outputs/03_shap_summary.png")

    # ── 6c. SHAP Bar Plot (mean absolute) ─────────────────────
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        sv, X_test,
        feature_names=feature_names,
        plot_type="bar",
        max_display=15,
        show=False
    )
    plt.title("Mean |SHAP| Value — Top Drivers of Attrition", fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/04_shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → Saved: outputs/04_shap_bar.png")

    # ── 6d. SHAP Force Plot for one at-risk employee ──────────
    # Find the employee with the highest predicted attrition probability
    probs         = model.predict_proba(X_test)[:, 1]
    high_risk_idx = np.argmax(probs)
    print(f"\n  Highest-risk employee (test index {high_risk_idx}):")
    print(f"  Predicted attrition probability : {probs[high_risk_idx]:.1%}")

    # Create force plot using the new SHAP API
    try:
        # Get the expected value for class 1
        expected_value = explainer.expected_value
        if isinstance(expected_value, list):
            expected_value = expected_value[1]
        
        # Get SHAP values for this single prediction
        shap_single = sv[high_risk_idx:high_risk_idx+1, :]
        
        # Get feature values for this single prediction
        X_single = X_test.iloc[high_risk_idx:high_risk_idx+1, :]
        
        # Use the new SHAP API for force plot
        force_plot = shap.force_plot(
            expected_value, 
            shap_single, 
            X_single,
            feature_names=feature_names,
            matplotlib=False,
            show=False
        )
        
        # Save as HTML
        shap.save_html("outputs/05_shap_force_plot.html", force_plot)
        print("  → Saved: outputs/05_shap_force_plot.html  (open in browser)")
        
    except Exception as e:
        print(f"  Note: Could not create interactive force plot: {e}")
        print("  Skipping force plot due to SHAP version compatibility.")
        print("  (The summary plots above still provide valuable insights)")

    # ── Print top 5 drivers correctly ────────────────────────
    # Calculate mean absolute SHAP values correctly
    mean_abs_shap = np.abs(sv).mean(axis=0)
    
    # Create dataframe correctly
    mean_shap_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean |SHAP|": mean_abs_shap
    }).sort_values("Mean |SHAP|", ascending=False)

    print("\n  ══════════════════════════════════════")
    print("  TOP 5 REASONS EMPLOYEES ARE LEAVING:")
    print("  ══════════════════════════════════════")
    for idx in range(min(5, len(mean_shap_df))):
        row = mean_shap_df.iloc[idx]
        print(f"  {idx+1}. {row['Feature']:<30}  SHAP={row['Mean |SHAP|']:.4f}")
    print()


# ============================================================
#  STEP 7 — PREDICT ON NEW EMPLOYEE (Inference Demo)
# ============================================================
def predict_new_employee(model, scaler, le_dict, feature_names: list, num_cols: list) -> None:
    """
    Shows how to use the trained model to predict attrition
    for a single new employee record.
    Customize the values below to test different scenarios.
    """
    print("\nSTEP 7 | Predict on a New Employee")

    # ── Example new employee profile ─────────────────────────
    #    Tweak these values to see how risk changes.
    new_employee = {
        "Age": 28,
        "BusinessTravel": "Travel_Frequently",
        "DailyRate": 800,
        "Department": "Sales",
        "DistanceFromHome": 20,
        "Education": 3,
        "EducationField": "Life Sciences",
        "EnvironmentSatisfaction": 2,
        "Gender": "Male",
        "HourlyRate": 65,
        "JobInvolvement": 2,
        "JobLevel": 1,
        "JobRole": "Sales Representative",
        "JobSatisfaction": 2,
        "MaritalStatus": "Single",
        "MonthlyIncome": 2500,
        "MonthlyRate": 15000,
        "NumCompaniesWorked": 4,
        "OverTime": "Yes",
        "PercentSalaryHike": 11,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 2,
        "StockOptionLevel": 0,
        "TotalWorkingYears": 4,
        "TrainingTimesLastYear": 2,
        "WorkLifeBalance": 1,
        "YearsAtCompany": 1,
        "YearsInCurrentRole": 0,
        "YearsSinceLastPromotion": 0,
        "YearsWithCurrManager": 0,
    }

    emp_df = pd.DataFrame([new_employee])
    
    print(f"  Original employee data shape: {emp_df.shape}")

    # Encode categoricals using saved label encoders
    for col, le in le_dict.items():
        if col in emp_df.columns:
            emp_df[col] = le.transform(emp_df[col])
    
    print(f"  After encoding shape: {emp_df.shape}")
    
    # Ensure all feature columns exist (add missing ones with default value 0)
    for col in feature_names:
        if col not in emp_df.columns:
            emp_df[col] = 0
    
    # Ensure column order matches training
    emp_df = emp_df[feature_names]
    
    print(f"  Final feature set shape: {emp_df.shape}")
    
    # Scale numeric columns
    emp_df[num_cols] = scaler.transform(emp_df[num_cols])

    prob = model.predict_proba(emp_df)[0][1]
    risk = "HIGH RISK" if prob > 0.5 else "LOW RISK"

    print(f"\n  Predicted attrition probability : {prob:.1%}")
    print(f"  Risk level                      : {risk}")
    if prob > 0.5:
        print("  Recommendation: Schedule 1:1, review compensation & overtime load.")
    else:
        print("  Recommendation: Employee appears stable. Continue engagement.")


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  EMPLOYEE ATTRITION PREDICTOR — FULL PIPELINE")
    print("=" * 55 + "\n")

    # 1. Load
    df = load_data(DATA_PATH)

    # 2. EDA
    run_eda(df)

    # 3. Preprocess
    X_train, X_test, y_train, y_test, feature_names, scaler, le_dict, num_cols = preprocess(df)

    # 4. Train
    model = train_model(X_train, y_train)

    # 5. Evaluate
    evaluate_model(model, X_test, y_test)

    # 6. Explain (Feature Importance + SHAP)
    explain_model(model, X_test, feature_names)

    # 7. Inference demo
    predict_new_employee(model, scaler, le_dict, feature_names, num_cols)

    print("\n" + "=" * 55)
    print("  ALL DONE! Check the 'outputs/' folder for plots.")
    print("=" * 55 + "\n")