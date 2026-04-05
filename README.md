# 📉 Telecom Customer Churn Prediction 

## 📌 Project Overview
This project predicts customer churn for a telecommunications company using Machine Learning (XGBoost). By identifying customers at risk of leaving, the business can take proactive retention measures.

## 🚀 Live Demo
**Try the interactive web app here:** [👉 Click Here to View the App](https://customer-churn-prediction-chset6utrubffhc2kwfjed.streamlit.app/)

## 🛠️ Architecture
- **Model Hosting:** Hosted on [Hugging Face](https://huggingface.co/khalidv5/churn-model) due to file size constraints.
- **Frontend:** Interactive dashboard built with **Streamlit**.
- **Deployment:** **Streamlit Cloud** linked to this GitHub repository.

## 🛠️ Tech Stack
* **Language:** Python 3.8+
* **Data Analysis:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn, XGBoost
* **Deployment:** Streamlit, Hugging Face (Model Hub)

## 📊 Workflow & Methodology
1. **EDA:** Analyzed features like tenure, contract type, and monthly charges to find churn drivers.
2. **Preprocessing:** Handled missing values, applied One-Hot Encoding, and managed class imbalance.
3. **Modeling:** Trained an **XGBoost** model to prioritize high **Recall** for churners.
4. **Integration:** The Streamlit app dynamically fetches the model from Hugging Face for real-time inference.

## 📈 Key Insights
* **Contract Type:** Month-to-month contracts are the highest indicator of churn.
* **Internet Service:** Fiber Optic users tend to churn more than DSL users.
* **Tenure:** New customers (1-6 months) require the most attention.

## 💻 How to Run Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
