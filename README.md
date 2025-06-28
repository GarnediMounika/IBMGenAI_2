# IBMGenAI_2
# NLP Projects: Sentiment Analysis & Review Classification using GenAI

This project showcases two complete NLP pipelines using synthetic datasets generated with GenAI tools. It aims to solve two practical business problems—**sentiment analysis of airline passenger feedback** and **automatic categorization of e-commerce product reviews**.

These tasks demonstrate how artificial intelligence can be applied to understand customer feedback, enhance user experience, and support business intelligence using supervised learning, feature engineering, and transformer-based methods.

---

## Project Summary

| Use Case | Goal | Techniques |
|----------|------|------------|
| Airline Sentiment Analysis | Identify customer sentiment (positive, neutral, or negative) | Logistic Regression, Transformers |
| E-Commerce Review Categorization | Predict product category based on review content | Supervised Classification, Feature Engineering |

Both datasets were **synthetically generated using GenAI tools** like ChatGPT and structured to mirror real-world customer data.

---

## Tools & Technologies Used

- 🐍 **Python 3.8+**
- 📘 **Pandas, NumPy** – Data handling
- 📊 **Scikit-learn** – Classification models
- 🤗 **Hugging Face Transformers** – For fine-tuning transformer models (optional)
- 🧪 **Google Colab / IBM Cloud / VSCode** – Code execution environments
- 📄 **Synthetic Data** – Generated using GenAI tools

---

## Directory Structure
GenAI-Task2/
├── task2_sentiment_analysis.ipynb # Sentiment analysis implementation
├── task2_review_classification.ipynb # Product categorization implementation
├── data/
│ ├── airline_feedback_train.csv
│ ├── airline_feedback_test.csv
│ ├── ecommerce_reviews_train.csv
│ └── ecommerce_reviews_test.csv
├── outputs/
│ ├── confusion_matrices/
│ ├── classification_reports/
│ └── model_predictions/
└── README.md


---

##  Use Case 1: Sentiment Analysis of Airline Passenger Feedback

### 🔽 Objective
To automatically analyze airline service feedback and classify it into:
- **Positive**
- **Neutral**
- **Negative**

### 📊 Dataset Overview
- **Training File:** `airline_feedback_train.csv`
- **Testing File:** `airline_feedback_test.csv`
- **Columns:** `feedback_text`, `label` (0: Negative, 1: Neutral, 2: Positive)

### ⚙️ Workflow
1. Data cleaning and preprocessing
2. Text vectorization using TF-IDF
3. Model training using **Logistic Regression**
4. Performance evaluation using accuracy, F1-score, and confusion matrix

### 📈 Sample Output
Test Accuracy: 0.84
F1 Score: 0.82


---

## Use Case 2: E-Commerce Review Classification

###  Objective
To categorize customer reviews into predefined **product categories**:
- Electronics
- Clothing
- Home Appliances

###  Dataset Overview
- **Training File:** `ecommerce_reviews_train.csv`
- **Testing File:** `ecommerce_reviews_test.csv`
- **Columns:** `review_text`, `category` (as labels)

### Workflow
1. Clean and normalize the text (stopword removal, stemming)
2. Convert text to features using:
   - TF-IDF Vectorization
   - Optionally: Embeddings from transformer models
3. Train classifiers such as:
   - Logistic Regression
   - SVM
   - Random Forest (optional)
4. Evaluate with classification reports

### Sample Output
Test Accuracy: 0.91
Class-wise F1 Scores:

Electronics: 0.90

Clothing: 0.92

Home Appliances: 0.89
