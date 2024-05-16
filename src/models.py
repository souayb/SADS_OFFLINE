from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import pickle
import os

def train_model(data, model_type: str = "ifor", **kwargs):
    """Train and return an anomaly detection model.

    Args:
        data (pd.DataFrame): The training data.
        model_type (str): The type of model to use. Default is 'ifor'.
        **kwargs: Additional arguments for the selected model.

    Returns:
        An instance of the selected anomaly detection model.
    """

    # Select the appropriate model
    if model_type == "ifor":
        model = IsolationForest(
            n_estimators=kwargs.get("n_estimators", 100),
            max_samples=kwargs.get("max_samples", "auto"),
            contamination=kwargs.get("contamination", 0.01),
            random_state=0,
        )

    elif model_type == "gmm":
        model = GaussianMixture(
            n_components=kwargs.get("n_components", 2),
            covariance_type=kwargs.get("covariance_type", "full"),
        )

    elif model_type == "bgmm":
        model = BayesianGaussianMixture(
            n_components=kwargs.get("n_components", 2),
            covariance_type=kwargs.get("covariance_type", "full"),
        )

    elif model_type == "lof":
        model = LocalOutlierFactor(
            n_neighbors=kwargs.get("n_neighbors", 2), novelty=True
        )

    elif model_type == "svm":
        model = OneClassSVM(gamma="auto")

    else:
        raise ValueError("Invalid model type.")

    # Fit the model
 
    data_train = data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1',]].values
    model_out = Pipeline([
            ('scaler',  StandardScaler()),
            ('clf', model)
        ])
    model_out.fit(data_train)
         
    return model_out
def load_model(model_path: str = "src/data"):
    """Load a trained model from disk.

    Args:
        model_path (str): The path to the model file.

    Returns:
        The trained model.
    """
    model_path = os.path.join(model_path, 'model.pkl')
    if not os.path.exists( model_path):
        raise FileNotFoundError("Model file not found.")
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def save_model(model, model_path: str = "src/data"):
    """Save a trained model to disk.

    Args:
        model: The trained model.
        model_path (str): The path to save the model file.
    """
    print("Saving model to disk...")
    with open(os.path.join(model_path, 'model.pkl'), 'wb') as file:

        pickle.dump(model, file)
    return