from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


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
