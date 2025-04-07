# 🔒 URL Shield

**URL Shield** is a machine learning-based tool designed to detect and classify URLs as **benign**, **phishing**, or **malicious**. It helps enhance user security by identifying suspicious URLs using pre-trained models.

---

## 🧠 Features

- Accepts a URL as input.
- Extracts features from the URL.
- Applies multiple trained ML models (Logistic Regression, SVM, Random Forest).
- Predicts and displays whether the URL is safe or malicious.
- Easy-to-use frontend interface.
- Backend built with Flask for seamless integration.

---

## 🛠️ Tech Stack

- **Frontend**: HTML, CSS, JavaScript  
- **Backend**: Python (Flask)  
- **Machine Learning Models**: Trained using `scikit-learn`  
- **Storage**: Models stored as `.pkl` (pickle) files  
- **Other**: Pandas, NumPy, Regex, TfidfVectorizer, StandardScaler

---

## ✅ Steps Followed (Development Workflow)

1. **Collected Dataset**  
   - Used a labeled dataset of URLs with their classes (benign/phishing/malicious).

2. **Preprocessed the Data**  
   - Cleaned URLs, extracted features using regex, and converted them to a numerical form using TF-IDF and StandardScaler.

3. **Trained ML Models**  
   - Trained several models including:
     - Logistic Regression  
     - SVM   
     - Random Forest  

4. **Saved the Models**  
   - Stored models as `.pkl` files using `pickle`.

5. **Created a Flask Backend**  
   - A Flask route (/predict) accepts the URL via a POST request from the form. It loads the .pkl models and returns predictions rendered using Jinja templating.

6. **Built a Simple Frontend**  
   - HTML form to accept URL input and display prediction output.

7. **Integrated Frontend & Backend**  
   - The form uses a standard HTML POST.
     The Flask backend handles submission and renders prediction results directly into the HTML.

8. **Tested the Application Locally**  
   - Verified predictions with multiple inputs.
   - Ensured models were correctly loaded.

9. **Ensemble-Based Prediction Logic**

   - The system uses predictions from three models: Logistic Regression, SVM, and Random Forest.
   - If any one of the models predicts the URL as malicious or defacement, the final output is considered not safe.
   - A URL is labeled benign (safe) only if all three models agree that it is benign.

---

## ⚠️ Version Warning

> 🟡 **Note:** The `.pkl` models were saved using `scikit-learn==1.6.0`.  
> If you are using `scikit-learn==1.6.1` or newer, you may encounter `InconsistentVersionWarning`.  
> While the models should still load, **for best compatibility**, install the recommended version:

```bash
pip install scikit-learn==1.6.0


## Setup Instructions
1. Clone the repo.
2. Install dependencies using `pip install -r requirements.txt`.
3. Run using `python app.py`.

```




