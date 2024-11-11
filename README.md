# Coursera_Data-Science-Challenge
This project establishes an effective customer churn prediction model by analyzing customer behavior data, helping businesses identify at-risk customers and develop corresponding retention strategies.
## Solution
To address the issue of customer churn, this project employs multiple machine learning models to build a customer churn prediction system, with a focus on **data preprocessing**, **feature engineering**, and **model optimization**. First, the data is cleaned and missing values and categorical variables are handled. Then, several machine learning models, including Random Forest and XGBoost, are trained based on different feature combinations. The best-performing model is ultimately selected for testing to accurately identify potential churn customers. Through this prediction system, businesses can assess the likelihood of customer churn and take proactive measures for high-risk customers, thereby improving customer loyalty and satisfaction."
## Method
1. **Data Preprocessing**: Handle missing values and outliers in the raw data, and encode categorical variables for model use. Data normalization is also performed to improve model stability.
2. **Model Selection and Training**: Train multiple models (e.g., **Random Forest** and **XGBoost**), and use cross-validation to select the best-performing model. Additionally, apply grid search to tune model hyperparameters and enhance prediction accuracy.
3. **Feature Engineering**: Analyze the impact of different variables on customer churn to select important features that strengthen model performance, and use feature scaling techniques to standardize feature ranges.
4. **Model Evaluation**: Evaluate model performance using metrics such as the confusion matrix, accuracy, recall, and F1 score, and select the best-performing model for the prediction task.
## Results
**Random Forest**

<img width="269" alt="截圖 2024-11-10 下午11 06 42" src="https://github.com/user-attachments/assets/48c5ae36-46f4-4c8d-af9a-5a533236f015">

**XGBoost**

<img width="264" alt="截圖 2024-11-10 下午11 07 38" src="https://github.com/user-attachments/assets/63b7131d-87db-466e-bd63-9c0492fba437">

## Conclusion
In this project, after comparing multiple models,**XGBoost** was chosen for customer churn prediction due to its excellent accuracy and computational efficiency. XGBoost effectively identifies high-risk churn customers, enabling businesses to take proactive retention measures. This project successfully established a stable churn prediction model, providing strong support for improving customer loyalty, **through the Coursera course**.
