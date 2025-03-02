import gradio as gr
from model_utils import load_all_models, predict_with_model

# Load all models
models, model_features = load_all_models()

# Mapeo de nombres amigables a nombres reales
MODEL_MAPPING = {
    "Death": "Death_random_forest_model",
    "Binary diagnosis": "Binary diagnosis_random_forest_model",
    "Necessity of transplantation": "Necessity of transplantation_random_forest_model",
    "Progressive disease": "Progressive disease_random_forest_model"
}

# Invertir el mapeo (opcional para facilidad)
INVERSE_MODEL_MAPPING = {v: k for k, v in MODEL_MAPPING.items()}

# Feature sets for each target variable
FEATURES = {
    "Death": [
        'ProgressiveDisease', 'DLCO (%) at diagnosis', 'FVC (%) at diagnosis',
        'FVC (L) at diagnosis', 'RadioWorsening2y', 'Age at diagnosis',
        'Pedigree', 'Severity of telomere shortening - Transform 4'
    ],
    "Binary diagnosis": [
        'Age at diagnosis', 'Pedigree', 'DLCO (%) at diagnosis', 'TOBACCO',
        'FVC (%) at diagnosis', 'FVC (L) at diagnosis',
        'Multidisciplinary committee', 'Severity of telomere shortening - Transform 4',
        'Severity of telomere shortening', 'Biopsy'
    ],
    "Necessity of transplantation": [
        'RadioWorsening2y', 'FVC (%) 1 year after diagnosis',
        'Genetic mutation studied in patient', 'Comorbidities',
        'DLCO (%) 1 year after diagnosis', 'FVC (L) at diagnosis',
        'DLCO (%) at diagnosis', 'Biopsy',
        'Severity of telomere shortening - Transform 4', 'FVC (%) at diagnosis',
        'FVC (L) 1 year after diagnosis'
    ],
    "Progressive disease": [
        'DLCO (%) at diagnosis', 'Age at diagnosis', 'Pedigree',
        '1st degree relative', 'FVC (L) at diagnosis', 'FVC (%) at diagnosis',
        'Biopsy', 'Genetic mutation studied in patient',
        'Severity of telomere shortening - Transform 4', 'Severity of telomere shortening'
    ]
}

FEATURE_RANGES = {
    'Pedigree': (0, 67),
    'Age at diagnosis': (0, 200),
    'FVC (L) at diagnosis': (0.0, 5.0),
    'FVC (%) at diagnosis': (0.0, 200.0),
    'DLCO (%) at diagnosis': (0.0, 200.0),
    'RadioWorsening2y': (0, 3),
    'Severity of telomere shortening - Transform 4': (1, 6),
    'ProgressiveDisease': (0, 1),
    'TOBACCO': (0, 1),
    'Multidisciplinary committee': (0, 1),
    'Biopsy': (0, 1),
    'Genetic mutation studied in patient': (0, 1),
    'Comorbidities': (0, 1),
    'FVC (L) 1 year after diagnosis': (0.0, 5.0),
    'FVC (%) 1 year after diagnosis': (0.0, 200.0),
    'DLCO (%) 1 year after diagnosis': (0.0, 200.0),
    '1st degree relative': (0, 1),
    'Severity of telomere shortening': (1, 6)
}

# Define prediction function
def make_prediction(input_features, friendly_model_name):
    """
    Predict using the selected model and input features.
    """
    # Map the friendly model name to the real model name
    target_model = MODEL_MAPPING.get(friendly_model_name)
    if target_model not in models:
        return f"Model '{friendly_model_name}' not found. Please select a valid model."

    model = models[target_model]
    features = model_features[target_model]

    if len(input_features) != len(features):
        return f"Invalid input. Expected features: {features}"

    input_array = [float(x) for x in input_features]
    prediction = predict_with_model(model, input_array)
    return f"Prediction for {friendly_model_name}: {prediction}"

# Define Gradio interface
def gradio_interface():
    def create_inputs_for_features(features):
        inputs = []
        for feature in features:
            min_val, max_val = FEATURE_RANGES.get(feature, (None, None))
            inputs.append(gr.Number(label=f"{feature} (Range: {min_val} - {max_val})", minimum=min_val, maximum=max_val))
        return inputs

    # Create a separate interface for each target variable
    interfaces = []
    for target, features in FEATURES.items():
        inputs = create_inputs_for_features(features)
        interface = gr.Interface(
            fn=lambda *args, target=target: make_prediction(args, target),
            inputs=inputs,
            outputs=gr.Text(label="Prediction Result"),
            title=f"Prediction for {target}",
            description=f"Provide values for features relevant to {target}"
        )
        interfaces.append(interface)

    # Combine all interfaces into a tabbed layout
    tabbed_interface = gr.TabbedInterface(
        interface_list=interfaces,
        tab_names=list(FEATURES.keys())
    )
    return tabbed_interface

# Launch Gradio app
if __name__ == "__main__":
    interface = gradio_interface()
    interface.launch()
