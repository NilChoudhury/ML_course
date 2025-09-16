#!/usr/bin/env python3
"""
K-Nearest Neighbors (KNN) Classifier on Iris Dataset
This script demonstrates KNN implementation with detailed explanations
"""

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_explore_data():
    """
    Load the Iris dataset and explore its basic properties
    
    The Iris dataset is a classic dataset in machine learning containing:
    - 150 samples of iris flowers
    - 4 features: sepal length, sepal width, petal length, petal width
    - 3 classes: setosa, versicolor, virginica (50 samples each)
    """
    print("=" * 70)
    print("IRIS DATASET - K-NEAREST NEIGHBORS (KNN) CLASSIFICATION")
    print("=" * 70)
    
    # Load the dataset from sklearn
    iris = load_iris()
    X, y = iris.data, iris.target  # X = features, y = labels
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Display basic information about the dataset
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Classes: {target_names}")
    print(f"Number of samples per class: {np.bincount(y)}")
    
    # Show first few samples
    print(f"\nFirst 5 samples:")
    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = [target_names[i] for i in y]
    print(df.head())
    
    return X, y, feature_names, target_names

def find_optimal_k(X, y, max_k=20):
    """
    Find the optimal number of neighbors (k) using cross-validation
    
    Why we need to find optimal k:
    - k=1: Very sensitive to noise, can overfit
    - k too large: May underfit, loses local patterns
    - We use cross-validation to find the best k
    """
    print("\n" + "=" * 70)
    print("FINDING OPTIMAL K VALUE")
    print("=" * 70)
    
    # Test different k values from 1 to max_k
    k_range = range(1, max_k + 1)
    scores = []
    
    print("Testing different k values...")
    for k in k_range:
        # Create KNN classifier with current k
        knn = KNeighborsClassifier(n_neighbors=k)
        
        # Use 5-fold cross-validation to get average score
        # This splits data into 5 parts, trains on 4, tests on 1, repeats 5 times
        cv_scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
        scores.append(cv_scores.mean())
        
        print(f"k={k:2d}: Accuracy = {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Find the best k
    best_k = k_range[np.argmax(scores)]
    best_score = max(scores)
    
    print(f"\nBest k: {best_k} with accuracy: {best_score:.4f}")
    
    # Plot k vs accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, scores, 'bo-', linewidth=2, markersize=8)
    plt.axvline(x=best_k, color='red', linestyle='--', linewidth=2, label=f'Best k={best_k}')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('KNN: Number of Neighbors vs Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('iris_knn_optimal_k.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("K vs Accuracy plot saved as 'iris_knn_optimal_k.png'")
    
    return best_k, scores

def create_and_train_knn(X, y, k, feature_names, target_names, use_scaling=True):
    """
    Create and train KNN classifier with optional feature scaling
    
    Why scaling might be important:
    - KNN uses distance calculations
    - Features with larger scales dominate distance calculations
    - Scaling ensures all features contribute equally
    """
    print("\n" + "=" * 70)
    print("TRAINING KNN CLASSIFIER")
    print("=" * 70)
    
    # Split data into training and testing sets
    # 70% for training, 30% for testing
    # stratify=y ensures equal representation of each class in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Optional: Scale features to have mean=0 and std=1
    if use_scaling:
        print("\nScaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print("Features scaled to have mean=0 and std=1")
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
        print("Using original features (no scaling)")
    
    # Create KNN classifier
    # n_neighbors=k: Number of neighbors to consider
    # weights='uniform': All neighbors have equal weight
    # metric='minkowski': Distance metric (Euclidean when p=2)
    knn = KNeighborsClassifier(
        n_neighbors=k,
        weights='uniform',
        metric='minkowski',
        p=2  # p=2 means Euclidean distance
    )
    
    # Train the model
    knn.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = knn.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return knn, X_train_scaled, X_test_scaled, y_train, y_test, y_pred, scaler if use_scaling else None

def evaluate_model(y_test, y_pred, target_names):
    """
    Evaluate the model performance using various metrics
    
    Classification metrics explained:
    - Precision: Of all predicted positive cases, how many were actually positive?
    - Recall: Of all actual positive cases, how many did we predict correctly?
    - F1-score: Harmonic mean of precision and recall
    - Support: Number of actual samples for each class
    """
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print("Rows = Actual, Columns = Predicted")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix - Iris KNN Classification')
    plt.xlabel('Predicted Species')
    plt.ylabel('Actual Species')
    plt.tight_layout()
    plt.savefig('iris_knn_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nConfusion matrix saved as 'iris_knn_confusion_matrix.png'")

def analyze_prediction_confidence(knn, X_test, y_test, target_names):
    """
    Analyze prediction confidence and show which samples were misclassified
    
    KNN provides prediction probabilities by looking at the proportion
    of neighbors belonging to each class
    """
    print("\n" + "=" * 70)
    print("PREDICTION CONFIDENCE ANALYSIS")
    print("=" * 70)
    
    # Get prediction probabilities
    y_proba = knn.predict_proba(X_test)
    
    # Find misclassified samples
    y_pred = knn.predict(X_test)
    misclassified = np.where(y_pred != y_test)[0]
    
    print(f"Number of misclassified samples: {len(misclassified)}")
    
    if len(misclassified) > 0:
        print("\nMisclassified samples:")
        for i, idx in enumerate(misclassified[:5]):  # Show first 5
            actual = target_names[y_test[idx]]
            predicted = target_names[y_pred[idx]]
            confidence = y_proba[idx][y_pred[idx]]
            print(f"  Sample {i+1}: Actual={actual}, Predicted={predicted}, Confidence={confidence:.3f}")
    
    # Show confidence distribution
    max_confidences = np.max(y_proba, axis=1)
    print(f"\nConfidence statistics:")
    print(f"  Mean confidence: {np.mean(max_confidences):.3f}")
    print(f"  Min confidence: {np.min(max_confidences):.3f}")
    print(f"  Max confidence: {np.max(max_confidences):.3f}")
    
    # Plot confidence distribution
    plt.figure(figsize=(10, 6))
    plt.hist(max_confidences, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(max_confidences), color='red', linestyle='--', 
                label=f'Mean: {np.mean(max_confidences):.3f}')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Confidences')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('iris_knn_confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Confidence distribution saved as 'iris_knn_confidence_distribution.png'")

def make_sample_predictions(knn, scaler, feature_names, target_names):
    """
    Make predictions on new sample data points
    
    This demonstrates how to use the trained model for new predictions
    """
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS ON NEW DATA")
    print("=" * 70)
    
    # Sample data points (new flowers to classify)
    sample_data = [
        [5.1, 3.5, 1.4, 0.2],  # Setosa-like
        [6.2, 2.9, 4.3, 1.3],  # Versicolor-like
        [7.3, 2.9, 6.3, 1.8],  # Virginica-like
        [5.8, 2.7, 4.1, 1.0],  # Borderline case
        [6.7, 3.0, 5.2, 2.3]   # Another borderline case
    ]
    
    sample_names = ["Setosa-like", "Versicolor-like", "Virginica-like", 
                   "Borderline 1", "Borderline 2"]
    
    print("Predicting species for new flower samples:")
    
    for i, sample in enumerate(sample_data):
        # Scale the sample if scaler was used
        if scaler is not None:
            sample_scaled = scaler.transform([sample])
        else:
            sample_scaled = [sample]
        
        # Make prediction
        prediction = knn.predict(sample_scaled)[0]
        probability = knn.predict_proba(sample_scaled)[0]
        
        print(f"\nSample {i+1} ({sample_names[i]}):")
        print(f"  Features: {dict(zip(feature_names, sample))}")
        print(f"  Prediction: {target_names[prediction]}")
        print(f"  Confidence: {probability[prediction]:.4f}")
        print(f"  All probabilities:")
        for j, prob in enumerate(probability):
            print(f"    {target_names[j]}: {prob:.4f}")

def compare_with_without_scaling(X, y, k, target_names):
    """
    Compare KNN performance with and without feature scaling
    
    This shows the importance of feature scaling in distance-based algorithms
    """
    print("\n" + "=" * 70)
    print("COMPARING WITH AND WITHOUT FEATURE SCALING")
    print("=" * 70)
    
    # Test without scaling
    knn_no_scale, _, X_test_no_scale, _, y_test, y_pred_no_scale, _ = create_and_train_knn(
        X, y, k, ['f1', 'f2', 'f3', 'f4'], target_names, use_scaling=False
    )
    
    # Test with scaling
    knn_scale, _, X_test_scale, _, _, y_pred_scale, _ = create_and_train_knn(
        X, y, k, ['f1', 'f2', 'f3', 'f4'], target_names, use_scaling=True
    )
    
    accuracy_no_scale = accuracy_score(y_test, y_pred_no_scale)
    accuracy_scale = accuracy_score(y_test, y_pred_scale)
    
    print(f"\nAccuracy without scaling: {accuracy_no_scale:.4f}")
    print(f"Accuracy with scaling: {accuracy_scale:.4f}")
    print(f"Improvement: {accuracy_scale - accuracy_no_scale:.4f}")
    
    if accuracy_scale > accuracy_no_scale:
        print("✓ Feature scaling improved performance!")
    else:
        print("⚠ Feature scaling did not improve performance (unusual for this dataset)")

def main():
    """
    Main function that orchestrates the entire KNN analysis
    
    This function demonstrates a complete machine learning workflow:
    1. Load and explore data
    2. Find optimal hyperparameters
    3. Train the model
    4. Evaluate performance
    5. Analyze results
    6. Make predictions on new data
    """
    try:
        print("Starting K-Nearest Neighbors analysis on Iris dataset...")
        
        # Step 1: Load and explore the dataset
        X, y, feature_names, target_names = load_and_explore_data()
        
        # Step 2: Find the optimal number of neighbors
        best_k, scores = find_optimal_k(X, y, max_k=15)
        
        # Step 3: Train KNN with optimal k and feature scaling
        knn, X_train, X_test, y_train, y_test, y_pred, scaler = create_and_train_knn(
            X, y, best_k, feature_names, target_names, use_scaling=True
        )
        
        # Step 4: Evaluate the model
        evaluate_model(y_test, y_pred, target_names)
        
        # Step 5: Analyze prediction confidence
        analyze_prediction_confidence(knn, X_test, y_test, target_names)
        
        # Step 6: Make sample predictions
        make_sample_predictions(knn, scaler, feature_names, target_names)
        
        # Step 7: Compare with and without scaling
        compare_with_without_scaling(X, y, best_k, target_names)
        
        # Final summary
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE!")
        print("=" * 70)
        print("Generated files:")
        print("  - iris_knn_optimal_k.png (K vs Accuracy plot)")
        print("  - iris_knn_confusion_matrix.png")
        print("  - iris_knn_confidence_distribution.png")
        print(f"\nBest k value: {best_k}")
        print(f"Final accuracy: {accuracy_score(y_test, y_pred):.4f}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
