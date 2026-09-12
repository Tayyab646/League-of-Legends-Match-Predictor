# League of Legends Match Predictor

A logistic regression model built with **PyTorch** that predicts the outcome (win/loss) of League of Legends matches from in-game statistics. This was completed as a final project for an IBM Skills Network machine learning course.

## Overview

The notebook (`Final Project League of Legends Match Predictor.ipynb`) implements a complete binary classification workflow:

1. **Data loading & preprocessing**
   - Loads `league_of_legends_data_large.csv` (in-game match statistics with a `win` target column)
   - Splits into train/test sets (80/20, `random_state=42`)
   - Standardizes features with `StandardScaler`
   - Converts everything to PyTorch tensors
2. **Model**
   - A `LogisticRegressionModel` (`torch.nn.Module`) with a single `nn.Linear` layer followed by a sigmoid activation
   - Loss: `BCELoss` (Binary Cross-Entropy)
   - Optimizer: `SGD` (`lr=0.01`)
3. **Training**
   - 1000 epochs, with loss printed every 100 epochs
   - Reports training and test accuracy (threshold = 0.5)
4. **Regularization**
   - Retrains the model with L2 regularization via `weight_decay=0.01` on the optimizer, and compares accuracy against the unregularized model
5. **Evaluation & visualization**
   - Confusion matrix and classification report (`sklearn.metrics`)
   - ROC curve with AUC score
6. **Hyperparameter tuning / feature importance**
   - Scaffolding for testing multiple learning rates and inspecting the learned weights as feature-importance scores (see [Status](#status) below)

## Tech stack

- Python, PyTorch
- pandas, NumPy
- scikit-learn (`train_test_split`, `StandardScaler`, `confusion_matrix`, `classification_report`, `roc_curve`, `auc`)
- Matplotlib

## Getting started

```bash
git clone https://github.com/Tayyab646/League-of-Legends-Match-Predictor.git
cd League-of-Legends-Match-Predictor
pip install pandas scikit-learn matplotlib torch
jupyter notebook "Final Project League of Legends Match Predictor.ipynb"
```

The dataset is loaded directly from a public IBM Cloud Object Storage URL inside the notebook, so no manual download is required.

## Status

This notebook was completed as a guided course exercise. The core pipeline (data prep → model → training → L2-regularized retraining → evaluation/visualization) is fully implemented and working. The final two sections — model save/load (`torch.save`/`torch.load`) and the learning-rate hyperparameter sweep — are left as unstarted exercise stubs in the current notebook and are good next steps for anyone building on this project.

## License

See [LICENSE](LICENSE) for details.
