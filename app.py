import numpy as np
import pickle
from flask import Flask, request, render_template
import warnings  
warnings.filterwarnings("ignore", category=UserWarning)

# Create Flask app
app = Flask(__name__)

# Load models, PCA, scaler, and vectorizer
log_reg_model = pickle.load(open('models/log_reg_model.pkl', 'rb'))
svm_model = pickle.load(open('models/svm_model.pkl', 'rb'))
rf_model = pickle.load(open('models/rf_model.pkl', 'rb'))
pca = pickle.load(open('models/pca_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

# Function to extract features from URL
def extract_features(url):
    # Extract various features from the URL
    url_length = len(url)
    num_dots = url.count('.')
    num_special_chars = sum(c in "!@#$%^&*()" for c in url)
    has_ip = 1 if any(char.isdigit() for char in url) else 0
    is_https = 1 if url.startswith('https') else 0
    path_length = len(url.split('/')) - 1
    num_subdomains = url.count('.') - 1
    has_suspicious_words = 1 if any(word in url for word in ['phish', 'deface', 'suspicious']) else 0
    url_entropy = -sum((url.count(c) / len(url)) * np.log2(url.count(c) / len(url)) for c in set(url))
    
    # Vectorize the URL using the loaded vectorizer
    url_vector = vectorizer.transform([url]).toarray()  # Shape (1, n_features)

    # Combine the URL vector features with the other numerical features
    numerical_features = np.array([url_length, num_dots, num_special_chars, has_ip, is_https,
                                   path_length, num_subdomains, has_suspicious_words, url_entropy])
    
    # Reshape the numerical features to be a 2D array with 1 row and n columns
    numerical_features = numerical_features.reshape(1, -1)

    # Stack the URL vector (2D) and numerical features (1D reshaped to 2D)
    combined_features = np.hstack((url_vector, numerical_features))  # Now both have the same number of dimensions

    # Ensure the combined features have 2D shape (1, n_features) 
    # combined_features should have the shape (1, n_features)
    return combined_features



# Route the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction and classification report
@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url']  # Get the URL from user input

    # Extract features from the URL
    features = extract_features(url)

    # Apply the scaling step to the features
    features_scaled = scaler.transform(features)  # Apply the same scaling as the training data

    # Apply PCA to the scaled features
    features_pca = pca.transform(features_scaled)  # Apply PCA

    # Make predictions with each model
    log_reg_pred = log_reg_model.predict(features_pca)
    svm_pred = svm_model.predict(features_pca)
    rf_pred = rf_model.predict(features_pca)

    # Get the prediction results
    prediction_text = f"URL: '{url}' - "  # Display the URL along with the prediction

    # Set prediction result based on model output
    if log_reg_pred == 'phishing' or svm_pred == 'phishing' or rf_pred == 'phishing':
        prediction_text += "The URL is likely a Phishing URL!"
    elif log_reg_pred == 'defacement' or svm_pred == 'defacement' or rf_pred == 'defacement':
        prediction_text += "The URL is likely a Defacement URL!"
    else:
        prediction_text += "The URL seems safe (Benign)."

    # Load classification reports
    with open('classification_reports/log_reg_report.txt', 'r') as f:
        log_reg_report = f.read()
    with open('classification_reports/svm_report.txt', 'r') as f:
        svm_report = f.read()
    with open('classification_reports/rf_report.txt', 'r') as f:
        rf_report = f.read()

    # Render results in the frontend
    return render_template('index.html', prediction_text=prediction_text, 
                           log_reg_report=log_reg_report, svm_report=svm_report, rf_report=rf_report)

# Run the application
if __name__ == '__main__':
    app.run(debug=True)
