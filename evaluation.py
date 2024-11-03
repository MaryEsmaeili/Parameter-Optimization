import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import label_binarize
from itertools import cycle

class ModelEvaluator:
    def __init__(self, estimator, X_train, y_train, X_test, y_test):
        """
        Initialize the ModelEvaluator with a model and training/test data.
        
        Parameters:
            estimator: Trained model estimator
            X_train (pd.DataFrame or np.array): Training feature data
            y_train (pd.Series or np.array): Training target data
            X_test (pd.DataFrame or np.array): Testing feature data
            y_test (pd.Series or np.array): Testing target data
        """
        self.estimator = estimator
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def plot_learning_curve(self, train_sizes=np.linspace(0.1, 1.0, 5)):
        """
        Plot the learning curve for the estimator.
        
        Parameters:
            train_sizes (np.array): Proportion of training examples used to generate learning curves.
        """
        train_sizes, train_scores, test_scores = learning_curve(
            self.estimator, self.X_train, self.y_train, train_sizes=train_sizes, cv=5
        )
        train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
        test_mean, test_std = test_scores.mean(axis=1), test_scores.std(axis=1)

        plt.figure()
        plt.plot(train_sizes, train_mean, 'o-', label="Training score")
        plt.plot(train_sizes, test_mean, 'o-', label="Cross-validation score")
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.legend(loc="best")
        plt.title("Learning Curve")
        plt.show()

    def plot_multiclass_roc_curve(self):
        """
        Plot ROC curve for multiclass classification.
        """
        y_test_bin = label_binarize(self.y_test, classes=np.unique(self.y_test))
        n_classes = y_test_bin.shape[1]
        y_score = self.estimator.predict_proba(self.X_test)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure()
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'Class {i} (area = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve for Multi-Class Classification")
        plt.legend(loc="best")
        plt.show()

    def plot_confusion_matrix(self):
        """
        Plot the confusion matrix for the model's predictions.
        """
        y_pred = self.estimator.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(cm))
        plt.xticks(tick_marks, rotation=45)
        plt.yticks(tick_marks)

        fmt = 'd'
        thresh = cm.max() / 2
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.show()
        
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))

