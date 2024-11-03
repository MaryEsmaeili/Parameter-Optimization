from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid

class CustomPipeline:
    def __init__(self, steps):
        self.steps = steps
        self.models = {}
    
    def fit(self, X, y):
        for name, estimator in self.steps:
            if name == "encoder":
                X = estimator.fit_transform(X)
            elif name == "scaler" or name == "pca":
                X = estimator.fit_transform(X)
            else:
                estimator.fit(X, y)
                self.models[name] = estimator
    
    def predict(self, X):
        for name, estimator in self.steps:
            if name == "encoder":
                X = estimator.transform(X)
            elif name == "scaler" or name == "pca":
                X = estimator.transform(X)
        return self.models["classifier"].predict(X)

    def set_params(self, params):
        for param, value in params.items():
            step_name, param_name = param.split("__")
            for i, (name, estimator) in enumerate(self.steps):
                if name == step_name:
                    setattr(estimator, param_name, value)
                    self.steps[i] = (name, estimator)

# Define grid search
def grid_search_cv(X_train, y_train, pipeline, param_grid):
    best_score = 0
    best_params = None
    for params in ParameterGrid(param_grid):
        pipeline.set_params(params)
        pipeline.fit(X_train, y_train)
        score = accuracy_score(y_train, pipeline.predict(X_train))
        if score > best_score:
            best_score = score
            best_params = params
    return best_score, best_params


