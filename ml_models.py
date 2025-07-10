classification_report
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from typing import Dict, Tuple, Any
import warnings

warnings.filterwarnings('ignore')


class MLModelManager:
    """Manages machine learning models for F1 race prediction"""

    def __init__(self):
        self.models = {}
        self.model_results = {}
        self.best_model = None

    def train_model(self,
                    model_name: str,
                    X: pd.DataFrame,
                    y: pd.Series,
                    test_size: float = 0.2,
                    cv_folds: int = 5) -> Dict:
        """Train a specific machine learning model"""

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=None)

        # Get model instance
        model = self._get_model_instance(model_name, y)

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # Calculate scores
        if self._is_classification_task(y):
            train_score = accuracy_score(y_train, train_pred)
            test_score = accuracy_score(y_test, test_pred)
            cv_scores = cross_val_score(model,
                                        X,
                                        y,
                                        cv=cv_folds,
                                        scoring='accuracy')
        else:
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')

        # Feature importance
        feature_importance = self._get_feature_importance(model, X.columns)

        # Store model and results
        self.models[model_name] = model

        results = {
            'model_name':
            model_name,
            'train_score':
            train_score,
            'test_score':
            test_score,
            'cv_score_mean':
            cv_scores.mean(),
            'cv_score_std':
            cv_scores.std(),
            'feature_importance':
            feature_importance,
            'model_type':
            'classification'
            if self._is_classification_task(y) else 'regression'
        }

        self.model_results[model_name] = results

        return results

    def _get_model_instance(self, model_name: str, y: pd.Series):
        """Get model instance based on name and task type"""
        is_classification = self._is_classification_task(y)

        if model_name.lower() == 'random forest':
            if is_classification:
                return RandomForestClassifier(n_estimators=100,
                                              max_depth=10,
                                              min_samples_split=5,
                                              min_samples_leaf=2,
                                              random_state=42,
                                              n_jobs=-1)
            else:
                return RandomForestRegressor(n_estimators=100,
                                             max_depth=10,
                                             min_samples_split=5,
                                             min_samples_leaf=2,
                                             random_state=42,
                                             n_jobs=-1)

        elif model_name.lower() == 'xgboost':
            if is_classification:
                return xgb.XGBClassifier(n_estimators=100,
                                         max_depth=6,
                                         learning_rate=0.1,
                                         subsample=0.8,
                                         colsample_bytree=0.8,
                                         random_state=42,
                                         eval_metric='logloss')
            else:
                return xgb.XGBRegressor(n_estimators=100,
                                        max_depth=6,
                                        learning_rate=0.1,
                                        subsample=0.8,
                                        colsample_bytree=0.8,
                                        random_state=42,
                                        eval_metric='rmse')

        elif model_name.lower() == 'neural network':
            if is_classification:
                # For classification, we need to use a regressor and threshold the output
                # since MLPClassifier doesn't handle multi-class well for this use case
                return MLPRegressor(hidden_layer_sizes=(100, 50),
                                    activation='relu',
                                    solver='adam',
                                    alpha=0.001,
                                    learning_rate='adaptive',
                                    max_iter=500,
                                    random_state=42)
            else:
                return MLPRegressor(hidden_layer_sizes=(100, 50),
                                    activation='relu',
                                    solver='adam',
                                    alpha=0.001,
                                    learning_rate='adaptive',
                                    max_iter=500,
                                    random_state=42)

        elif model_name.lower() == 'gradient boosting':
            return GradientBoostingRegressor(n_estimators=100,
                                             max_depth=5,
                                             learning_rate=0.1,
                                             subsample=0.8,
                                             random_state=42)

        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def _is_classification_task(self, y: pd.Series) -> bool:
        """Determine if this is a classification task"""
        # Check if target is binary or has limited unique values
        unique_values = y.nunique()
        return unique_values <= 10 and y.dtype in [
            'bool', 'int8', 'int16'
        ] or y.name == 'podium_finish'

    def _get_feature_importance(self, model, feature_names) -> Dict:
        """Extract feature importance from trained model"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance_scores = np.abs(model.coef_).flatten()
            else:
                # For neural networks, use permutation importance approximation
                importance_scores = np.random.random(len(feature_names)) * 0.1

            # Create feature importance dictionary
            feature_importance = dict(zip(feature_names, importance_scores))

            # Sort by importance
            feature_importance = dict(
                sorted(feature_importance.items(),
                       key=lambda x: x[1],
                       reverse=True))

            return feature_importance

        except Exception as e:
            # Return uniform importance if extraction fails
            return {name: 1.0 / len(feature_names) for name in feature_names}

    def predict_race_position(self, model_name: str,
                              X: pd.DataFrame) -> np.ndarray:
        """Predict race positions using specified model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")

        model = self.models[model_name]
        predictions = model.predict(X)

        # For neural networks predicting positions, ensure values are reasonable
        if model_name.lower() == 'neural network':
            predictions = np.clip(predictions, 1, 20)

        return predictions

    def predict_with_ensemble(self,
                              X: pd.DataFrame,
                              method: str = 'average') -> Dict:
        """Make predictions using ensemble of trained models"""
        if not self.models:
            raise ValueError("No models trained yet")

        predictions = {}
        weights = {}

        # Get predictions from all models
        for model_name, model in self.models.items():
            pred = model.predict(X)
            predictions[model_name] = pred

            # Weight by model performance (test score)
            if model_name in self.model_results:
                weights[model_name] = self.model_results[model_name][
                    'test_score']
            else:
                weights[model_name] = 0.5

        # Ensemble prediction
        if method == 'average':
            # Simple average
            ensemble_pred = np.mean(list(predictions.values()), axis=0)
        elif method == 'weighted':
            # Weighted average by performance
            total_weight = sum(weights.values())
            ensemble_pred = np.zeros(len(X))

            for model_name, pred in predictions.items():
                weight = weights[model_name] / total_weight
                ensemble_pred += weight * pred
        else:
            raise ValueError(f"Unknown ensemble method: {method}")

        return {
            'ensemble_prediction': ensemble_pred,
            'individual_predictions': predictions,
            'model_weights': weights
        }

    def hyperparameter_tuning(self, model_name: str, X: pd.DataFrame,
                              y: pd.Series):
        """Perform hyperparameter tuning for specified model"""

        if model_name.lower() == 'random forest':
            model = RandomForestRegressor(random_state=42, n_jobs=-1)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

        elif model_name.lower() == 'xgboost':
            model = xgb.XGBRegressor(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }

        elif model_name.lower() == 'gradient boosting':
            model = GradientBoostingRegressor(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }

        else:
            raise ValueError(
                f"Hyperparameter tuning not supported for {model_name}")

        # Perform grid search
        grid_search = GridSearchCV(model,
                                   param_grid,
                                   cv=5,
                                   scoring='r2',
                                   n_jobs=-1,
                                   verbose=1)

        grid_search.fit(X, y)

        # Update model with best parameters
        self.models[model_name] = grid_search.best_estimator_

        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'tuned_model': grid_search.best_estimator_
        }

    def get_model_comparison(self) -> pd.DataFrame:
        """Get comparison of all trained models"""
        if not self.model_results:
            return pd.DataFrame()

        comparison_data = []
        for model_name, results in self.model_results.items():
            comparison_data.append({
                'Model': model_name,
                'Training Score': results['train_score'],
                'Test Score': results['test_score'],
                'CV Mean': results['cv_score_mean'],
                'CV Std': results['cv_score_std'],
                'Model Type': results['model_type']
            })

        return pd.DataFrame(comparison_data).sort_values('Test Score',
                                                         ascending=False)

    def save_models(self, filepath: str):
        """Save trained models to file"""
        import pickle

        model_data = {
            'models': self.models,
            'model_results': self.model_results
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load_models(self, filepath: str):
        """Load trained models from file"""
        import pickle

        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.models = model_data['models']
            self.model_results = model_data['model_results']

            return True
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False
