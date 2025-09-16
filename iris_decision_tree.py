#!/usr/bin/env python3
"""
Simple Decision Tree Classifier on Iris Dataset
This script demonstrates a basic decision tree implementation using scikit-learn
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_explore_data():
    """Load and explore the Iris dataset"""
    print("=" * 60)
    print("IRIS DATASET - DECISION TREE CLASSIFICATION")
    print("=" * 60)
    
    # Load the dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Classes: {target_names}")
    print(f"Number of samples per class: {np.bincount(y)}")
    
    return X, y, feature_names, target_names

def create_decision_tree(X, y, feature_names, target_names):
    """Create and train a decision tree classifier"""
    print("\n" + "=" * 60)
    print("TRAINING DECISION TREE")
    print("=" * 60)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Create and train the decision tree
    clf = DecisionTreeClassifier(
        random_state=42,
        max_depth=3,  # Limit depth for interpretability
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return clf, X_train, X_test, y_train, y_test, y_pred

def evaluate_model(y_test, y_pred, target_names):
    """Evaluate the model performance"""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix - Iris Decision Tree')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('iris_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nConfusion matrix saved as 'iris_confusion_matrix.png'")

def visualize_tree(clf, feature_names, target_names):
    """Visualize the decision tree"""
    print("\n" + "=" * 60)
    print("DECISION TREE VISUALIZATION")
    print("=" * 60)
    
    plt.figure(figsize=(15, 10))
    plot_tree(clf, 
              feature_names=feature_names,
              class_names=target_names,
              filled=True,
              rounded=True,
              fontsize=10)
    plt.title('Decision Tree for Iris Classification', fontsize=16)
    plt.tight_layout()
    plt.savefig('iris_decision_tree.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Decision tree saved as 'iris_decision_tree.png'")

def analyze_feature_importance(clf, feature_names):
    """Analyze feature importance"""
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    
    importance = clf.feature_importances_
    feature_importance = list(zip(feature_names, importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("Feature Importance (higher = more important):")
    for feature, imp in feature_importance:
        print(f"  {feature}: {imp:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    features = [f[0] for f in feature_importance]
    importances = [f[1] for f in feature_importance]
    
    plt.bar(features, importances)
    plt.title('Feature Importance in Decision Tree')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('iris_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Feature importance plot saved as 'iris_feature_importance.png'")

def make_predictions(clf, feature_names, target_names):
    """Make sample predictions"""
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)
    
    # Sample data points
    sample_data = [
        [5.1, 3.5, 1.4, 0.2],  # Setosa
        [6.2, 2.9, 4.3, 1.3],  # Versicolor
        [7.3, 2.9, 6.3, 1.8]   # Virginica
    ]
    
    sample_names = ["Setosa-like", "Versicolor-like", "Virginica-like"]
    
    for i, sample in enumerate(sample_data):
        prediction = clf.predict([sample])[0]
        probability = clf.predict_proba([sample])[0]
        
        print(f"\nSample {i+1} ({sample_names[i]}):")
        print(f"  Features: {dict(zip(feature_names, sample))}")
        print(f"  Prediction: {target_names[prediction]}")
        print(f"  Confidence: {probability[prediction]:.4f}")
        print(f"  All probabilities: {dict(zip(target_names, probability))}")

def main():
    """Main function to run the complete analysis"""
    try:
        # Load and explore data
        X, y, feature_names, target_names = load_and_explore_data()
        
        # Create and train decision tree
        clf, X_train, X_test, y_train, y_test, y_pred = create_decision_tree(
            X, y, feature_names, target_names
        )
        
        # Evaluate model
        evaluate_model(y_test, y_pred, target_names)
        
        # Visualize tree
        visualize_tree(clf, feature_names, target_names)
        
        # Analyze feature importance
        analyze_feature_importance(clf, feature_names)
        
        # Make sample predictions
        make_predictions(clf, feature_names, target_names)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        print("Generated files:")
        print("  - iris_confusion_matrix.png")
        print("  - iris_decision_tree.png")
        print("  - iris_feature_importance.png")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
