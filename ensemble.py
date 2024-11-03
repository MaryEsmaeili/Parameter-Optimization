from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

class EnsembleModel:
    def __init__(self):
        """
        Initializes an EnsembleModel with a set of base models.
        """
        self.models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "Bagging": BaggingClassifier(estimator=LogisticRegression(), n_estimators=100, random_state=42),
            "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
            "LogisticRegression": LogisticRegression(),
            "GaussianNB": GaussianNB(),
            "SVM": SVC(probability=True, random_state=42)
        }
        self.voting_clf_hard = None
        self.voting_clf_soft = None

    def fit_base_models(self, X_train, y_train):
        """
        Fits each of the base models on the training data.
        """
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            print(f"{name} trained.")

    def vote_hard(self, X_train, y_train, X_test, y_test):
        """
        Performs hard voting using the base models and evaluates the result.
        """
        self.voting_clf_hard = VotingClassifier(
            estimators=[(name, model) for name, model in self.models.items()],
            voting='hard'
        )
        self.voting_clf_hard.fit(X_train, y_train)
        y_pred = self.voting_clf_hard.predict(X_test)
        print("\n--- Hard Voting ---")
        self.evaluate_metrics(y_test, y_pred)

    def vote_soft(self, X_train, y_train, X_test, y_test):
        """
        Performs soft voting using the base models and evaluates the result.
        """
        self.voting_clf_soft = VotingClassifier(
            estimators=[(name, model) for name, model in self.models.items()],
            voting='soft'
        )
        self.voting_clf_soft.fit(X_train, y_train)
        y_pred = self.voting_clf_soft.predict(X_test)
        print("\n--- Soft Voting ---")
        self.evaluate_metrics(y_test, y_pred, model=self.voting_clf_soft, X_test=X_test)

    def evaluate_metrics(self, y_test, y_pred, model=None, X_test=None):
        """
        Evaluates model performance and prints metrics including accuracy, F1 score, ROC AUC, and classification report.
        """
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")

        if model is not None and X_test is not None:
            try:
                print(f"ROC AUC Score: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.2f}")
            except ValueError:
                print("ROC AUC Score calculation failed. Ensure binary or multiclass format is correct.")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))