# Classification Model Comparison: Image & Text Tasks

This repository contains two supervised classification projects:

1. Fashion-MNIST image classification  
2. SMS spam detection  

Both projects emphasize model comparison, feature representation, and cross-validated evaluation using multiple performance metrics.

---

## 1. Fashion-MNIST Classification

### Objective
Classify grayscale clothing images into 10 categories using linear and kernel-based models.

### Models Evaluated
- SGDClassifier
- LogisticRegression
- LinearSVC
- SVC (RBF kernel)

### Methodology
- 60,000 training samples split into equal training and validation sets
- Hyperparameter tuning using GridSearchCV (2-fold cross-validation)
- Evaluation on held-out test set

### Key Result
RBF SVC (C = 5) achieved approximately **89.9% test accuracy** when trained on 30,000 images.

This section demonstrates:
- Hyperparameter tuning
- Cross-validation design
- Regularization effects
- Kernel vs linear model comparison

---

## 2. SMS Spam Classification

### Objective
Classify SMS messages as spam or ham using text-based features.

### Part A: Manual Feature Engineering
Engineered structured features including:
- Message length
- Capitalization counts and proportions
- Digit counts and proportions
- Special character indicators

Models evaluated:
- LogisticRegression
- SGDClassifier
- LinearSVC

Manual features achieved F1 scores up to **0.907**, showing that simple structural signals are highly informative for spam detection.

---

### Part B: Text Vectorization

Used:
- CountVectorizer
- CountVectorizer with bigrams and filtering
- TfidfVectorizer

Models evaluated:
- LogisticRegression
- SGDClassifier
- LinearSVC
- MultinomialNB

Best performance:
- **MultinomialNB with count features (F1 ≈ 0.947)**
- Precision and recall both above 90%

Findings:
- Word-level features substantially outperform manual structural features
- MultinomialNB performs best with raw counts
- TF-IDF does not universally improve performance
- Feature representation interacts strongly with model assumptions

---

## Evaluation Metrics

Models were compared using:
- Accuracy
- Precision
- Recall
- F1 Score

Cross-validation was used throughout to ensure robust performance estimates.

---

## Skills Demonstrated

- Feature engineering
- Text vectorization (Bag-of-Words, TF-IDF)
- Hyperparameter tuning with GridSearchCV
- Model comparison across metrics
- Handling imbalanced classification problems
- Interpretation of precision/recall trade-offs

---

## Tools

- Python
- scikit-learn
- pandas
- NumPy

---

This repository serves as an applied comparison of linear models across image and text classification settings, highlighting how representation choices influence model performance.
