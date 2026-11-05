# Football Match Outcome Prediction Platform

This project is a web-based application for predicting football match outcomes  
(**Home Win / Draw / Away Win**) using machine learning.  
It combines a Flask backend, a JavaScript frontend, and a modular ML pipeline with
cross-validated Random Forest models.

---

## 📌 Features

- Interactive web interface displaying football fixtures and standings
- Probabilistic match outcome prediction (win / draw / lose)
- Machine learning pipeline with:
  - feature engineering
  - data normalization
  - stratified cross-validation
  - model registry and versioning
- Secure API key handling via environment variables
- Easily extensible to support additional models (e.g., Logistic Regression, XGBoost)

---

## 🏗️ Project Architecture

You will find it here : Arch.html


---

## 🔁 Prediction Flow (End-to-End)

1. User clicks **Predict** on a match in the frontend
2. Frontend sends a POST request to:
/api/predict

3. Backend:
- validates the request
- builds numerical features
- selects the active model using `metadata.json`
4. ML model returns probabilities
5. A draw-adjustment is applied for close matches
6. Final probabilities are returned to the frontend and displayed

---

## 🤖 Machine Learning Approach

### Model
- **Random Forest Classifier** (scikit-learn)
- Chosen for:
- non-linear decision boundaries
- better probability realism compared to linear models

### Training
- Dataset split into:
- Train
- Validation
- Test
- Hyperparameters tuned using **5-fold Stratified Cross-Validation**
- Selection metric:
- **Macro F1-score** (balanced across Home / Draw / Away)

### Draw Handling
Predicting draws is inherently difficult.  
To improve realism, a small post-processing adjustment is applied when team strength
(Elo rating difference) is small.  
This adjustment:
- only affects close matches
- preserves probability normalization
- avoids bias in mismatched games

---

## 🔐 API Key Management (Important)

This project uses the **Football-Data API**.

⚠️ API keys are **not included** in the repository.

Instead:
- `.env` holds the real key (not shared)
- `.env.example` provides a template

---

## ⚙️ Setup Instructions (Windows)

### 1️⃣ Install Python
- Python **3.10 or newer** (recommended: 3.11)
- Ensure **“Add Python to PATH”** is checked

Verify:
```bat
python --version

2️⃣ Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Configure environment variables
Create .env from the template:

copy .env.example .env
Edit .env and add your Football-Data API key:

FOOTBALL_DATA_API_KEY =YOUR_API_KEY_HERE

5️⃣ Run the application
python backend/app.py

Open in your browser:
http://127.0.0.1:5000

🧪 Training a New Model (Optional)
To train a Random Forest model with cross-validation:

cd backend/app/ml/predictors
python train_rf.py
This will:

perform cross-validation

train the final model

save the model artifact

update metadata.json

The application will automatically use the latest model.

📊 Reproducibility
Dependencies are listed in requirements.txt

Models are versioned in metadata.json

Feature order is stored and reused for inference

API keys are externalized via environment variables

🚀 Future Improvements
Add expected goals (xG) or betting odds as features

Support league-specific models

Experiment with Gradient Boosting / XGBoost

Add confidence intervals for predictions

📄 License
This project is for academic and educational purposes.