# Customer_Churn-Prediction
  This repository includes a machine learning solution that uses past customer data to forecast client attrition for a telecom company. Several categorization algorithms and data pretreatment methods are used by the solution to precisely forecast if a client would churn, or discontinue using the service.
  
  ## Important Features
**Data Preprocessing**: - The separation of features and the target variable (`Churn`) is part of the preparation and loading of the dataset.
   "-" For better model performance, `StandardScaler` is used to normalize numerical features, particularly for gradient-based algorithms like Neural Networks and Logistic Regression.

2. **Class Imbalance Handling**: the target variable (`Churn`) is subjected to class imbalance through the application of **SMOTE (Synthetic Minority Oversampling Technique)**. In order to improve model performance and avoid bias towards the majority class, this technique creates synthetic samples for the minority class.

3. **Model Building**: The dataset is used to build and assess the following models:
   A linear classifier with L1 regularization is called a logistic regression.
   In order to determine which characteristics have the most impact on churn, the Random Forest ensemble model gives feature importance.
   Gradient Boosting is a sophisticated ensemble model that has hyperparameter utilizing `RandomizedSearchCV` for tweaking.
   A multilayer perceptron classifier with two hidden layers for churn prediction is called a neural network (MLP).

4. **Model Comparison**: - Each model's accuracy is computed and compared, making it simple to determine which model performs the best.
   - To illustrate the model contrast, a bar chart is created.

5. **Feature Importance Visualization**: - To determine which features have the greatest influence on the churn forecast, feature importance is displayed for the Random Forest model.


