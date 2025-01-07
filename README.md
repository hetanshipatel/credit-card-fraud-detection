# Credit Card Fraud Detection

This project implements a machine learning pipeline to detect fraudulent credit card transactions using a dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data). The pipeline includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, and comparison of multiple algorithms.

---

## **Dataset**

The dataset contains transactions made with credit cards in September 2013 by European cardholders. It includes a total of 284,807 transactions, with 492 labeled as fraudulent. The dataset is highly imbalanced, with the positive class (frauds) accounting for 0.172% of all transactions.

---

## **Project Structure**

The project consists of the following files:

- **`fraud_detection_notebook.ipynb`**: Jupyter notebook containing the entire pipeline: data preprocessing, EDA, model training, and evaluation.
- **`fraud_detection_best_model.pkl`**: Saved model file for the best-performing machine learning model.
- **`results.txt`**: A detailed comparison of the evaluation metrics for all models.
- **`creditcard.csv`**: Dataset used for training and evaluation (to be downloaded separately).
- **`requirements.txt`**: List of Python dependencies for the project.

---

## **Models Used**

We implemented and compared the following machine learning algorithms:

1. **Random Forest**  
2. **XGBoost**  
3. **Logistic Regression**

**Evaluation Metrics**:
- Accuracy
- Precision
- Recall
- F1 Score

---

## **Results**

The evaluation metrics for all the models are documented in the `results.txt` file, which includes:

- Accuracy
- Precision
- Recall
- F1 Score

### **Comparison Table:**
| Model               | Accuracy  | Precision | Recall  | F1 Score |
|---------------------|-----------|-----------|---------|----------|
| Random Forest       | 0.999899  | 0.999799  | 1.0000  | 0.999900 |
| XGBoost             | 0.999749  | 0.999498  | 1.0000  | 0.999749 |
| Logistic Regression | 0.948774  | 0.974147  | 0.9221  | 0.947434 |

### **Best Model**
The **Random Forest** model is the best-performing model based on the F1 Score.

---

## **Steps to Run Locally**

### **Step 1: Clone the Repository**

1. Open your terminal or command prompt.
2. Clone the repository using the following command:
   ```bash
   git clone https://github.com/hetanshipatel/credit-card-fraud-detection.git
   ```
    
### **Step 2: Install Dependencies**

1. Ensure you have Python and `pip` installed on your system.
2. Use the provided `requirements.txt` file to install the necessary Python libraries by running the following command:
   ```bash
   pip install -r requirements.txt
   ```

### **Step 3: Run the Notebook**

1. Open Jupyter Notebook by executing the following command:
   ```bash
   jupyter notebook
   ```

### **Step 4: Save and Load the Model**

1. If you want to save the trained model, use the following code in your notebook:
   ```python
   import joblib
   joblib.dump(best_model, 'fraud_detection_best_model.pkl')
   ```


