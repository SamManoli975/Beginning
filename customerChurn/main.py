# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset (replace with your actual dataset path)
data = pd.read_csv('C:/VS CODE/ML/MachineLearning/customerChurn/customer_churn_dataset-training-master.csv')

# Step 1: Data Exploration
# Display basic information about the dataset
print(data.info())
print(data.describe())

# Step 2: Preprocessing
# Identify categorical columns and apply Label Encoding
string_columns = data.select_dtypes(include=['object']).columns

# Initialize LabelEncoder
label_encoders = {}
for column in string_columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Check for missing values
print(data.isnull().sum())

# Handle missing values (optional: can drop rows or fill them with appropriate values)
data = data.dropna()  # For simplicity, drop rows with missing values

# Step 3: Split data into features (X) and target (y)
X = data.drop(['Churn'], axis=1)
y = data['Churn']

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Training and Evaluation

# 1. Decision Tree Classifier
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)
y_pred_dt = decision_tree.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))

# 2. Random Forest Classifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# 3. XGBoost Classifier (Boosting Model)
xgboost_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgboost_model.fit(X_train, y_train)
y_pred_xgb = xgboost_model.predict(X_test)

print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))



new_data = pd.read_csv('C:/VS CODE/ML/MachineLearning/customerChurn/customer_churn_dataset-testing-master.csv')

# Preprocess the new data
# Ensure the new data has the same features and is encoded in the same way
for column in string_columns:
    if column in new_data.columns:
        new_data[column] = label_encoders[column].transform(new_data[column])

# Handle missing values in new data (if any)
new_data = new_data.dropna()  # Drop rows with missing values for simplicity

# Ensure the new data has the same columns as the training data
new_data = new_data[X.columns]

# Step 6: Make predictions using each of the trained models

# 1. Decision Tree Predictions
predictions_dt = decision_tree.predict(new_data)
print("Decision Tree Predictions:")
print(predictions_dt)

# 2. Random Forest Predictions
predictions_rf = random_forest.predict(new_data)
print("Random Forest Predictions:")
print(predictions_rf)

# 3. XGBoost Predictions
predictions_xgb = xgboost_model.predict(new_data)
print("XGBoost Predictions:")
print(predictions_xgb)

# Optionally, you can save the predictions to a CSV file
output = pd.DataFrame({
    'Decision_Tree_Predictions': predictions_dt,
    'Random_Forest_Predictions': predictions_rf,
    'XGBoost_Predictions': predictions_xgb
})

output.to_csv('customer_churn_predictions.csv', index=False)
print("Predictions saved to customer_churn_predictions.csv")
