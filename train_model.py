"""
Train Model — Complete ML Pipeline for IDS in MANET
=====================================================
Implements the full pipeline:
1. Data loading and preprocessing
2. Label encoding
3. Z-score normalization
4. SMOTE oversampling
5. Artificial Butterfly Optimizer feature selection
6. Training 7 classifiers
7. Adaptive Voting Mechanism ensemble
8. Evaluation and model persistence
"""

import os
import sys
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              roc_curve, auc)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from butterfly_optimizer import ArtificialButterflyOptimizer

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = 'Train_data.csv'
TEST_DATA_PATH = 'Test_data.csv'
MODELS_DIR = 'models'
STATIC_DIR = os.path.join('static', 'images')
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


def load_and_preprocess_data():
    """Load NSL-KDD dataset and perform preprocessing."""
    print("\n" + "=" * 70)
    print("STEP 1: Loading and Preprocessing Data")
    print("=" * 70)
    
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Class distribution:\n{df['class'].value_counts()}")
    
    # Drop columns with zero variance
    zero_var_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].nunique() <= 1:
            zero_var_cols.append(col)
    
    if zero_var_cols:
        print(f"\nDropping zero-variance columns: {zero_var_cols}")
        df = df.drop(columns=zero_var_cols)
    
    print(f"After preprocessing: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def encode_labels(df):
    """Label encode categorical columns."""
    print("\n" + "=" * 70)
    print("STEP 2: Label Encoding")
    print("=" * 70)
    
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    
    # Remove target column from categorical encoding list
    if 'class' in categorical_cols:
        categorical_cols.remove('class')
    
    print(f"Categorical columns to encode: {categorical_cols}")
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"  {col}: {len(le.classes_)} unique values encoded")
    
    # Encode target
    target_le = LabelEncoder()
    df['class'] = target_le.fit_transform(df['class'].astype(str))
    label_encoders['class'] = target_le
    print(f"  class: {list(target_le.classes_)}")
    
    # Save encoders
    with open(os.path.join(MODELS_DIR, 'label_encoders.pkl'), 'wb') as f:
        pickle.dump(label_encoders, f)
    
    return df, label_encoders


def zscore_normalization(X_train, X_test):
    """Apply Z-score normalization (standardization)."""
    print("\n" + "=" * 70)
    print("STEP 3: Z-score Normalization")
    print("=" * 70)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  Train mean (sample): {X_train_scaled.mean(axis=0)[:5].round(4)}")
    print(f"  Train std (sample):  {X_train_scaled.std(axis=0)[:5].round(4)}")
    
    # Save scaler
    with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    return X_train_scaled, X_test_scaled, scaler


def apply_smote(X_train, y_train):
    """Apply SMOTE oversampling to balance classes."""
    print("\n" + "=" * 70)
    print("STEP 4: SMOTE Oversampling")
    print("=" * 70)
    
    print(f"  Before SMOTE:")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    Class {u}: {c} samples")
    
    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"  After SMOTE:")
    unique, counts = np.unique(y_resampled, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    Class {u}: {c} samples")
    
    print(f"  Total samples: {len(y_train)} -> {len(y_resampled)}")
    
    return X_resampled, y_resampled


def feature_selection_abo(X_train, y_train, feature_names):
    """Apply Artificial Butterfly Optimizer for feature selection."""
    print("\n" + "=" * 70)
    print("STEP 5: Feature Selection - Artificial Butterfly Optimizer")
    print("=" * 70)
    
    # Use a fast classifier for fitness evaluation
    base_clf = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=8)
    
    abo = ArtificialButterflyOptimizer(
        n_butterflies=25,
        max_iter=40,
        p_sunspot=0.8,
        p_canopy=0.6,
        crossover_rate=0.5,
        mutation_rate=0.1,
        min_features=8,
        random_state=RANDOM_STATE
    )
    
    # Use a subset for faster optimization
    n_subset = min(5000, len(X_train))
    indices = np.random.RandomState(RANDOM_STATE).choice(len(X_train), n_subset, replace=False)
    X_subset = X_train[indices]
    y_subset = y_train[indices]
    
    selected_indices, best_fitness = abo.optimize(X_subset, y_subset, base_clf, verbose=True)
    
    selected_names = [feature_names[i] for i in selected_indices]
    print(f"\nSelected Features ({len(selected_names)}):")
    for i, name in enumerate(selected_names):
        print(f"  {i+1}. {name}")
    
    # Plot convergence
    convergence = abo.get_convergence_history()
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(convergence) + 1), convergence, 'b-o', markersize=3, linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best Fitness', fontsize=12)
    plt.title('ABO Convergence Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'abo_convergence.png'), dpi=150)
    plt.close()
    
    # Save selected features
    with open(os.path.join(MODELS_DIR, 'selected_features.pkl'), 'wb') as f:
        pickle.dump({'indices': selected_indices, 'names': selected_names}, f)
    
    return selected_indices, selected_names


def build_classifiers():
    """Initialize all 7 classifiers."""
    classifiers = {
        'Decision Tree': DecisionTreeClassifier(
            random_state=RANDOM_STATE, max_depth=15, min_samples_split=5
        ),
        'XGBoost': XGBClassifier(
            random_state=RANDOM_STATE, n_estimators=150, max_depth=8,
            learning_rate=0.1, use_label_encoder=False, eval_metric='logloss',
            verbosity=0
        ),
        'Random Forest': RandomForestClassifier(
            random_state=RANDOM_STATE, n_estimators=150, max_depth=15,
            min_samples_split=5, n_jobs=-1
        ),
        'SVM': SVC(
            random_state=RANDOM_STATE, kernel='rbf', C=10, gamma='scale',
            probability=True
        ),
        'Logistic Regression': LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000, C=1.0, solver='lbfgs'
        ),
        'MLP Classifier': MLPClassifier(
            random_state=RANDOM_STATE, hidden_layer_sizes=(128, 64, 32),
            max_iter=500, learning_rate='adaptive', early_stopping=True,
            validation_fraction=0.1
        ),
        'Boosted Regression Tree': GradientBoostingClassifier(
            random_state=RANDOM_STATE, n_estimators=150, max_depth=6,
            learning_rate=0.1, subsample=0.8
        )
    }
    return classifiers


def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names_selected):
    """Train all classifiers and evaluate performance."""
    print("\n" + "=" * 70)
    print("STEP 6: Training and Evaluating Classifiers")
    print("=" * 70)
    
    classifiers = build_classifiers()
    results = {}
    trained_models = {}
    predictions = {}
    
    for name, clf in classifiers.items():
        print(f"\n  Training {name}...")
        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = round(time.time() - t0, 3)
        
        t0 = time.time()
        y_pred = clf.predict(X_test)
        infer_time = round(time.time() - t0, 5)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # AUC score
        auc_score = 0.0
        if hasattr(clf, 'predict_proba'):
            y_proba = clf.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_score = round(auc(fpr, tpr), 4)
        
        # Classification report
        cls_report = classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly'], output_dict=True, zero_division=0)
        
        # Model size
        model_size = round(sys.getsizeof(pickle.dumps(clf)) / 1024, 1)
        
        results[name] = {
            'accuracy': round(acc * 100, 2),
            'precision': round(prec * 100, 2),
            'recall': round(rec * 100, 2),
            'f1_score': round(f1 * 100, 2),
            'auc': auc_score,
            'train_time': train_time,
            'infer_time': infer_time,
            'model_size_kb': model_size,
            'classification_report': {
                'Normal': {k: round(v, 4) for k, v in cls_report['Normal'].items() if k != 'support'},
                'Anomaly': {k: round(v, 4) for k, v in cls_report['Anomaly'].items() if k != 'support'}
            }
        }
        
        trained_models[name] = clf
        predictions[name] = y_pred
        
        print(f"    Accuracy:  {acc*100:.2f}%  |  AUC: {auc_score:.4f}")
        print(f"    Precision: {prec*100:.2f}%  |  Train: {train_time}s")
        print(f"    Recall:    {rec*100:.2f}%  |  Size: {model_size}KB")
        print(f"    F1 Score:  {f1*100:.2f}%")
    
    return results, trained_models, predictions


def adaptive_voting(trained_models, results, X_test, y_test):
    """
    Adaptive Voting Mechanism - Dynamically weighted ensemble.
    
    Weights are proportional to each model's accuracy, giving more
    influence to better-performing classifiers.
    """
    print("\n" + "=" * 70)
    print("STEP 7: Adaptive Voting Mechanism")
    print("=" * 70)
    
    # Calculate adaptive weights based on model performance
    accuracies = {name: res['accuracy'] / 100.0 for name, res in results.items()}
    total_acc = sum(accuracies.values())
    weights = {name: acc / total_acc for name, acc in accuracies.items()}
    
    print("\n  Adaptive Weights:")
    for name, w in weights.items():
        print(f"    {name}: {w:.4f} (accuracy: {accuracies[name]*100:.2f}%)")
    
    # Weighted voting
    n_samples = len(y_test)
    n_classes = len(np.unique(y_test))
    
    weighted_votes = np.zeros((n_samples, n_classes))
    
    for name, model in trained_models.items():
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test)
        else:
            # For models without predict_proba, use one-hot predictions
            pred = model.predict(X_test)
            proba = np.zeros((n_samples, n_classes))
            for i, p in enumerate(pred):
                proba[i, int(p)] = 1.0
        
        weighted_votes += weights[name] * proba
    
    # Final prediction
    ensemble_pred = np.argmax(weighted_votes, axis=1)
    
    # Evaluate ensemble
    acc = accuracy_score(y_test, ensemble_pred)
    prec = precision_score(y_test, ensemble_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, ensemble_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, ensemble_pred, average='weighted', zero_division=0)
    
    ensemble_results = {
        'accuracy': round(acc * 100, 2),
        'precision': round(prec * 100, 2),
        'recall': round(rec * 100, 2),
        'f1_score': round(f1 * 100, 2)
    }
    
    print(f"\n  Adaptive Voting Ensemble Results:")
    print(f"    Accuracy:  {acc*100:.2f}%")
    print(f"    Precision: {prec*100:.2f}%")
    print(f"    Recall:    {rec*100:.2f}%")
    print(f"    F1 Score:  {f1*100:.2f}%")
    
    return ensemble_pred, ensemble_results, weights


def generate_visualizations(results, ensemble_results, y_test, ensemble_pred,
                            trained_models, X_test):
    """Generate all comparison charts and confusion matrices."""
    print("\n" + "=" * 70)
    print("STEP 8: Generating Visualizations")
    print("=" * 70)
    
    # Combine results
    all_results = {**results, 'Adaptive Voting Ensemble': ensemble_results}
    
    # --- 1. Accuracy Comparison Bar Chart ---
    fig, ax = plt.subplots(figsize=(14, 7))
    models = list(all_results.keys())
    accuracies = [all_results[m]['accuracy'] for m in models]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', 
              '#1abc9c', '#e67e22', '#FF6B6B']
    
    bars = ax.bar(range(len(models)), accuracies, color=colors[:len(models)],
                  edgecolor='white', linewidth=1.5)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(min(accuracies) - 5, 102)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'accuracy_comparison.png'), dpi=150)
    plt.close()
    
    # --- 2. All Metrics Grouped Bar Chart ---
    fig, ax = plt.subplots(figsize=(16, 8))
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    x = np.arange(len(models))
    width = 0.2
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [all_results[m][metric] for m in models]
        ax.bar(x + i * width, values, width, label=label, alpha=0.85)
    
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'metrics_comparison.png'), dpi=150)
    plt.close()
    
    # --- 3. Confusion Matrix for Ensemble ---
    cm = confusion_matrix(y_test, ensemble_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Adaptive Voting Ensemble - Confusion Matrix',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()
    
    # --- 4. Individual model confusion matrices ---
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    model_names = list(trained_models.keys())
    for i, name in enumerate(model_names):
        y_pred = trained_models[name].predict(X_test)
        cm_i = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm_i, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Normal', 'Anomaly'],
                    yticklabels=['Normal', 'Anomaly'])
        axes[i].set_title(name, fontsize=11, fontweight='bold')
        axes[i].set_xlabel('Predicted', fontsize=9)
        axes[i].set_ylabel('Actual', fontsize=9)
    
    # Ensemble in last subplot
    cm_ens = confusion_matrix(y_test, ensemble_pred)
    sns.heatmap(cm_ens, annot=True, fmt='d', cmap='Reds', ax=axes[7],
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    axes[7].set_title('Adaptive Voting', fontsize=11, fontweight='bold')
    axes[7].set_xlabel('Predicted', fontsize=9)
    axes[7].set_ylabel('Actual', fontsize=9)
    
    fig.suptitle('Confusion Matrices - All Models', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'all_confusion_matrices.png'), dpi=150)
    plt.close()
    
    print("  Visualizations saved to static/images/")
    return all_results


def generate_roc_curves(trained_models, X_test, y_test):
    """Generate ROC curves for all models."""
    print("\n  Generating ROC curves...")
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#667eea', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']
    
    for (name, model), color in zip(trained_models.items(), colors):
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={roc_auc:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - All Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'roc_curves.png'), dpi=150)
    plt.close()


def generate_feature_importance(trained_models, selected_names):
    """Generate feature importance chart from tree-based models."""
    print("  Generating feature importance...")
    importances = {}
    
    for name in ['Random Forest', 'XGBoost', 'Decision Tree']:
        if name in trained_models and hasattr(trained_models[name], 'feature_importances_'):
            importances[name] = trained_models[name].feature_importances_
    
    if not importances:
        return {}
    
    # Average importances across tree models
    avg_importance = np.mean(list(importances.values()), axis=0)
    sorted_idx = np.argsort(avg_importance)
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(selected_names) * 0.4)))
    colors_bar = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_idx)))
    ax.barh(range(len(sorted_idx)), avg_importance[sorted_idx], color=colors_bar)
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([selected_names[i] for i in sorted_idx], fontsize=10)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('Feature Importance (Avg of Tree Models)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'feature_importance.png'), dpi=150)
    plt.close()
    
    return {selected_names[i]: round(float(avg_importance[i]), 4) for i in range(len(selected_names))}


def generate_dataset_eda(df_original):
    """Generate dataset exploration visualizations."""
    print("\n" + "=" * 70)
    print("STEP 8b: Dataset EDA Visualizations")
    print("=" * 70)
    
    # 1. Class distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    class_counts = df_original['class'].value_counts()
    colors_pie = ['#43e97b', '#f5576c']
    
    axes[0].pie(class_counts.values, labels=class_counts.index, colors=colors_pie,
                autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
    axes[0].set_title('Class Distribution', fontsize=13, fontweight='bold')
    
    axes[1].bar(class_counts.index, class_counts.values, color=colors_pie, edgecolor='white', linewidth=2)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Class Counts', fontsize=13, fontweight='bold')
    for i, v in enumerate(class_counts.values):
        axes[1].text(i, v + 100, str(v), ha='center', fontweight='bold', fontsize=11)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'class_distribution.png'), dpi=150)
    plt.close()
    
    # 2. Correlation heatmap (top 15 numeric features)
    numeric_df = df_original.select_dtypes(include=[np.number])
    if numeric_df.shape[1] > 15:
        top_cols = numeric_df.var().nlargest(15).index.tolist()
        numeric_df = numeric_df[top_cols]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0, ax=ax,
                square=True, linewidths=0.5, annot=False,
                cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Heatmap (Top 15)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'correlation_heatmap.png'), dpi=150)
    plt.close()
    
    # 3. Feature distributions (top 6 numeric features by variance)
    top6 = numeric_df.var().nlargest(6).index.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()
    for i, col in enumerate(top6):
        axes[i].hist(df_original[col], bins=50, color='#667eea', alpha=0.7, edgecolor='white')
        axes[i].set_title(col, fontsize=11, fontweight='bold')
        axes[i].grid(axis='y', alpha=0.3)
    fig.suptitle('Top Feature Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'feature_distributions.png'), dpi=150)
    plt.close()
    
    # Dataset stats
    stats = {
        'total_samples': int(df_original.shape[0]),
        'total_features': int(df_original.shape[1] - 1),
        'normal_count': int(class_counts.get('normal', class_counts.get(1, 0))),
        'anomaly_count': int(class_counts.get('anomaly', class_counts.get(0, 0))),
        'numeric_features': int(df_original.select_dtypes(include=[np.number]).shape[1]),
        'categorical_features': int(df_original.select_dtypes(include=['object']).shape[1])
    }
    print(f"  Dataset stats saved. {stats['total_samples']} samples, {stats['total_features']} features.")
    return stats


def generate_training_time_chart(results):
    """Generate training time comparison chart."""
    print("  Generating training time chart...")
    models = [n for n in results.keys() if n != 'Adaptive Voting Ensemble']
    times = [results[n].get('train_time', 0) for n in models]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#667eea', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']
    bars = ax.bar(range(len(models)), times, color=colors[:len(models)], edgecolor='white', linewidth=1.5)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=25, ha='right', fontsize=10)
    ax.set_ylabel('Training Time (seconds)', fontsize=12)
    ax.set_title('Model Training Time Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{t:.3f}s', ha='center', va='bottom', fontweight='bold', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'training_time.png'), dpi=150)
    plt.close()


def save_models_and_results(trained_models, weights, all_results, 
                             feature_names, selected_indices):
    """Save all trained models, weights, and results."""
    print("\n" + "=" * 70)
    print("STEP 9: Saving Models and Results")
    print("=" * 70)
    
    # Save individual models
    for name, model in trained_models.items():
        filename = name.lower().replace(' ', '_') + '.pkl'
        with open(os.path.join(MODELS_DIR, filename), 'wb') as f:
            pickle.dump(model, f)
        print(f"  Saved: {filename}")
    
    # Save ensemble weights
    with open(os.path.join(MODELS_DIR, 'ensemble_weights.pkl'), 'wb') as f:
        pickle.dump(weights, f)
    print(f"  Saved: ensemble_weights.pkl")
    
    # Save results
    with open(os.path.join(MODELS_DIR, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved: results.json")
    
    # Save feature info
    with open(os.path.join(MODELS_DIR, 'feature_info.pkl'), 'wb') as f:
        pickle.dump({
            'all_features': feature_names,
            'selected_indices': selected_indices,
            'selected_names': [feature_names[i] for i in selected_indices]
        }, f)
    print(f"  Saved: feature_info.pkl")
    
    print("\n  All models and artifacts saved successfully!")


def main():
    """Run the complete ML pipeline."""
    print("\n" + "#" * 70)
    print("#  Adaptive Voting Mechanism with Artificial Butterfly Algorithm")
    print("#  Feature Selection for IDS in MANET")
    print("#" * 70)
    
    # Step 1: Load and preprocess
    df = load_and_preprocess_data()
    
    # Generate dataset EDA BEFORE encoding (uses original labels)
    dataset_stats = generate_dataset_eda(df)
    
    # Step 2: Label encoding
    df, label_encoders = encode_labels(df)
    
    # Separate features and target
    X = df.drop('class', axis=1).values
    y = df['class'].values
    feature_names = list(df.drop('class', axis=1).columns)
    
    print(f"\nFeature matrix: {X.shape}")
    print(f"Target vector: {y.shape}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    
    # Step 3: Z-score normalization
    X_train_scaled, X_test_scaled, scaler = zscore_normalization(X_train, X_test)
    
    # Step 4: SMOTE
    X_train_balanced, y_train_balanced = apply_smote(X_train_scaled, y_train)
    
    # Step 5: Feature selection with ABO
    selected_indices, selected_names = feature_selection_abo(
        X_train_balanced, y_train_balanced, feature_names
    )
    
    # Apply feature selection
    X_train_selected = X_train_balanced[:, selected_indices]
    X_test_selected = X_test_scaled[:, selected_indices]
    
    print(f"\nFeatures after ABO selection: {X_train_selected.shape[1]}")
    
    # Step 6: Train and evaluate all classifiers
    results, trained_models, predictions = train_and_evaluate(
        X_train_selected, X_test_selected, y_train_balanced, y_test, selected_names
    )
    
    # Step 7: Adaptive Voting Mechanism
    ensemble_pred, ensemble_results, weights = adaptive_voting(
        trained_models, results, X_test_selected, y_test
    )
    
    # Step 8: Generate visualizations
    all_results = generate_visualizations(
        results, ensemble_results, y_test, ensemble_pred,
        trained_models, X_test_selected
    )
    
    # Step 8b: ROC curves
    generate_roc_curves(trained_models, X_test_selected, y_test)
    
    # Step 8c: Feature importance
    importance_data = generate_feature_importance(trained_models, selected_names)
    
    # Step 8d: Training time chart
    generate_training_time_chart(all_results)
    
    # Enrich results with extra data
    all_results['_meta'] = {
        'dataset_stats': dataset_stats,
        'feature_importance': importance_data
    }
    
    # Step 9: Save everything
    save_models_and_results(
        trained_models, weights, all_results, feature_names, selected_indices
    )
    
    # Final summary
    print("\n" + "#" * 70)
    print("#  FINAL RESULTS SUMMARY")
    print("#" * 70)
    print(f"\n{'Model':<30} {'Accuracy':>10} {'AUC':>8} {'Train(s)':>10}")
    print("-" * 70)
    for name, res in all_results.items():
        if name.startswith('_'):
            continue
        auc_v = res.get('auc', '-')
        tt = res.get('train_time', '-')
        auc_str = f"{auc_v:.4f}" if isinstance(auc_v, float) else str(auc_v)
        tt_str = f"{tt:.3f}" if isinstance(tt, float) else str(tt)
        print(f"{name:<30} {res['accuracy']:>9.2f}% {auc_str:>8} {tt_str:>10}")
    print("-" * 70)
    
    best_model = max(
        [(n, r) for n, r in all_results.items() if not n.startswith('_')],
        key=lambda x: x[1]['accuracy']
    )
    print(f"\n** Best Model: {best_model[0]} with {best_model[1]['accuracy']:.2f}% accuracy")
    print("\n>> Training complete! Run 'python app.py' to start the web application.")


if __name__ == "__main__":
    main()
