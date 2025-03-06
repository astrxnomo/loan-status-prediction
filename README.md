# Loan Status Prediction with Machine Learning  
This project develops a **machine learning model** to predict whether a loan will be **fully paid** or **charged off**, helping financial institutions assess **credit risk** and improve decision-making.  

## 📊 Best Model: XGBRFClassifier  
✔ **Accuracy:** 82.7%  
✔ **F1 Score:** 0.89  
✔ **AUC-ROC:** 0.64  
✔ **Inference Time:** 32.3 ms  

## 🔍 Key Insights  
- **Most Influential Factors:**  
  1️⃣ Credit Score  
  2️⃣ Annual Income  
  3️⃣ Years of Credit History  
- **XGBRFClassifier** provided the best balance between precision and efficiency.  
- **Deployed via Streamlit Cloud** for real-time predictions.  

## 🚀 How to Run Locally  
1️⃣ **Clone the repository:**  
```bash
git clone [REPO_URL]
cd [PROJECT_NAME]
```

2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

3️⃣ Run the Streamlit app:
```bash
streamlit run app.py
```

4️⃣ Open in browser: http://localhost:8501

## 📌 Technologies Used
- Python (pandas, numpy, scikit-learn, xgboost, joblib)
- Machine Learning Models: XGBRFClassifier, Random Forest, Logistic Regression, Gradient Boosting
- Deployment: Streamlit Cloud
