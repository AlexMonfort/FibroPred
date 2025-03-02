import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
def load_data(file_path):
    df = pd.read_excel(file_path, header=1)
    return df

# Preprocess data including categorical variables
def preprocess_data_with_categoricals(df):
    # Replace -9 with NaN for missing values
    df.replace(-9, np.nan, inplace=True)

    # Drop columns with >50% missing values
    missing_percentage = df.isnull().sum() / len(df) * 100
    df = df.drop(columns=missing_percentage[missing_percentage > 50].index)

    # Drop specific columns
    drop_columns = ['ProgressiveDisease', 'Final diagnosis', 'Transplantation date', 'Cause of death', 'Date of death', 'COD NUMBER']
    df = df.drop(columns=[col for col in drop_columns if col in df.columns])
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # Handle binary variables specifically
    if 'Binary diagnosis' in df.columns:
        df['Binary diagnosis'] = df['Binary diagnosis'].apply(
            lambda x: 1 if str(x).strip().lower() == "ipf" else 0
        )

    if 'Death' in df.columns:
        df['Death'] = df['Death'].apply(
            lambda x: 1 if str(x).strip().lower() == "yes" else 0
        )

    # Apply one-hot encoding to categorical variables
    df = apply_one_hot_encoding(df)

    # Separate categorical and numerical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    print("Categorical Variables:", categorical_cols.tolist())
    print("Numerical Variables:", numeric_cols.tolist())
    return df, numeric_cols, categorical_cols

# Apply one-hot encoding to categorical variables
def apply_one_hot_encoding(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

# Select predictors using feature importance
def select_important_features(X, y, threshold=0.03):
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    selector = SelectFromModel(model, threshold=threshold, prefit=True)
    X_reduced = selector.transform(X)
    selected_features = X.columns[selector.get_support()]
    return pd.DataFrame(X_reduced, columns=selected_features), selected_features

# Visualize feature importance
def plot_feature_importance(model, features, target):
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance[sorted_idx], y=np.array(features)[sorted_idx])
    plt.title(f'Feature Importance for {target}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

# Visualize overfitting and optimization results
def plot_model_performance(cv_scores, train_scores, test_scores, target ,metric_name="Accuracy"):
    plt.figure(figsize=(12, 6))

    # Cross-validation scores
    plt.subplot(1, 2, 1)
    plt.plot(cv_scores, label='Cross-validation scores', marker='o')
    plt.title(f'Cross-validation {metric_name} for {target}')
    plt.xlabel('Fold')
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.legend()

    # Train vs Test comparison
    plt.subplot(1, 2, 2)
    plt.bar(['Train', 'Test'], [train_scores.mean(), test_scores], color=['blue', 'orange'])
    plt.title(f'{metric_name}: Train vs Test')
    plt.ylabel(metric_name)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Plot ROC-AUC curve
def plot_roc_auc(model, X_test, y_test, target):
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilidades para la clase positiva
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC-AUC Curve for {target}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

# Save trained model
def save_model(model, target, selected_features):

    if not os.path.exists("models"):
        os.makedirs("models")
    file_name = f"models/{target}_random_forest_model.pkl"
    joblib.dump({'model': model, 'features': selected_features}, file_name)
    print(f"Model and features saved to {file_name}")


# Main pipeline
def main():
    file_path = 'FibroPredCODIFICADA.xlsx'
    df = load_data(file_path)

    # Target columns
    target_columns = ['Death', 'Progressive disease', 'Necessity of transplantation']

    # Preprocess data
    df, numeric_cols, categorical_cols = preprocess_data_with_categoricals(df)

    for target in target_columns:
        print(f"Processing target: {target}")
        X = df[numeric_cols].drop(columns=target_columns, errors='ignore')  # Ensure target variables are excluded
        y = df[target]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Select important features
        X_train_selected, selected_features = select_important_features(X_train, y_train)
        X_test_selected = X_test[selected_features]

        print(f"Selected predictors for training {target} ({len(selected_features)} predictors): {selected_features.tolist()}")

        # Train RandomForest model
        model = RandomForestClassifier(n_estimators=300,
            max_depth=4, 
            min_samples_split=10, 
            min_samples_leaf=10,
            class_weight='balanced',
            max_features='sqrt',
            random_state=42)
        model.fit(X_train_selected, y_train)

        # Cross-validation to check overfitting
        cv = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='accuracy')
        train_scores = cross_val_score(model, X_train_selected, y_train, cv=15, scoring='accuracy')
        y_pred_test = model.predict(X_test_selected)
        test_score = accuracy_score(y_test, y_pred_test)

        print(f"Cross-validation accuracy for {target}: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"Test accuracy for {target}: {test_score:.4f}")
        print(classification_report(y_test, y_pred_test))

        # Plot model performance
        plot_model_performance(cv_scores, train_scores, test_score, target, metric_name="Accuracy")

        # Plot feature importance
        print(f"Feature importance for {target}:")
        plot_feature_importance(model, selected_features, target)

        # Plot ROC-AUC Curve
        plot_roc_auc(model, X_test_selected, y_test, target)

        # Save trained model
        save_model(model, target, selected_features.tolist())

    print("Pipeline completed.")

if __name__ == "__main__":
    main()
