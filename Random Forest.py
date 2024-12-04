import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("COMP1801_Coursework_Dataset.csv")

# Display the first few rows of the dataset
data.head()

# Define classes based on Lifespan thresholds
# Low: < 1000, Medium: 1000-1500, High: > 1500
bins = [0, 1000, 1500, float('inf')]
labels = ['Low', 'Medium', 'High']
data['Lifespan_Class'] = pd.cut(data['Lifespan'], bins=bins, labels=labels)

# Encode classes as numerical labels
data['Lifespan_Class_Encoded'] = data['Lifespan_Class'].cat.codes

# Drop the original Lifespan column since we're classifying
data = data.drop(columns=['Lifespan'])

# Display the updated dataset structure
data.head()

# Encode categorical features using LabelEncoder
categorical_columns = ['partType', 'microstructure', 'seedLocation', 'castType']
label_encoders = {col: LabelEncoder() for col in categorical_columns}

for col, encoder in label_encoders.items():
    data[col] = encoder.fit_transform(data[col])

# Separate features and target variable
X = data.drop(columns=['Lifespan_Class', 'Lifespan_Class_Encoded'])  # Features
y = data['Lifespan_Class_Encoded']  # Target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

accuracy, report, conf_matrix

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Plot feature importance
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feature_importances.plot(kind="bar", color="skyblue")
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.show()
