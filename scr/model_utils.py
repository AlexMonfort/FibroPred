import os
import joblib

def load_all_models(models_dir="models"):
    """
    Load all models and their features from the given directory.
    """
    models = {}
    features = {}
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory '{models_dir}' not found.")

    for model_file in os.listdir(models_dir):
        if model_file.endswith(".pkl"):
            model_name = os.path.splitext(model_file)[0]
            data = joblib.load(os.path.join(models_dir, model_file))
            models[model_name] = data['model']
            features[model_name] = data['features']
            print(f"Model '{model_name}' loaded successfully with features: {features[model_name]}")
    return models, features

def predict_with_model(model, input_data):
    """
    Predict using a loaded model.

    Parameters:
    - model: The loaded model.
    - input_data: A dictionary or Pandas DataFrame row containing input features.

    Returns:
    - prediction: Model prediction.
    """
    prediction = model.predict([input_data])
    return int(prediction[0])