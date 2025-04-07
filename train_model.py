import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import numpy as np

# Load dataset
df = pd.read_csv('/content/enhanced_malware_detection.csv')  # Update with your dataset path

# Preprocessing
X = df.drop(columns=['URL', 'Label'])  # Drop URL and Label columns for features
y = df['Label']  # The target variable

# Apply TfidfVectorizer to the 'URL' column for text-based features
vectorizer = TfidfVectorizer(max_features=500)  # Limit to top 500 features
X_url = vectorizer.fit_transform(df['URL']).toarray()  # Vectorize the URLs

# Combine the URL vectorized features with the existing numerical features
X_combined = np.hstack((X_url, X))  # Combine the vectorized features with numerical features

# Standardize the data (apply scaling) 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=5)  # Reduce to 5 components for example
X_pca = pca.fit_transform(X_scaled)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
log_reg_model = LogisticRegression(max_iter=1000)
log_reg_model.fit(X_train, y_train)

# Train SVM Model
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)

# Train Random Forest Model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Evaluate models
log_reg_report = classification_report(y_test, log_reg_model.predict(X_test))
svm_report = classification_report(y_test, svm_model.predict(X_test))
rf_report = classification_report(y_test, rf_model.predict(X_test))

# Save models, PCA, scaler, and vectorizer
pickle.dump(log_reg_model, open('log_reg_model.pkl', 'wb'))
pickle.dump(svm_model, open('svm_model.pkl', 'wb'))
pickle.dump(rf_model, open('rf_model.pkl', 'wb'))
pickle.dump(pca, open('pca_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

# Save classification reports
with open('log_reg_report.txt', 'w') as f:
    f.write(log_reg_report)
with open('svm_report.txt', 'w') as f:
    f.write(svm_report)
with open('rf_report.txt', 'w') as f:
    f.write(rf_report)

print("Models and reports saved successfully!")
