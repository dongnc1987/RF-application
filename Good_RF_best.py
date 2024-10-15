import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import joblib

# Load dataset
path_file = r'data.csv'
dataframe = pd.read_csv(path_file, sep=',', header=None)

# Split into input (X) and output (y) variables
X = dataframe.iloc[:, :5]  # Features
y = dataframe.iloc[:, 5:9]  # Target variables

# Initialize variables to store the maximum R2 value and its corresponding model
max_r2_value = -np.inf
best_model = None

# Repeat the process 50 times
for _ in range(50):
    # Split the data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Random Forest Regression
    RegModel = RandomForestRegressor(n_estimators=100, criterion='squared_error')  # Using 'squared_error' instead of 'mse'
    RF = RegModel.fit(X_train, y_train)
    prediction = RF.predict(X_test)

    # Model evaluation
    r2_value = metrics.r2_score(y_train, RF.predict(X_train))
    
    # Update the maximum R2 value and the corresponding model if a higher value is found
    if r2_value > max_r2_value:
        max_r2_value = r2_value
        best_model = RF
        
# Save the trained model
model_filename = 'RF_model.joblib'
joblib.dump(best_model, model_filename)
print(f"Model saved as {model_filename}")

# Print the maximum R2 value
print('Maximum R2 Value:', max_r2_value)


# Set global font size
plt.rcParams.update({'font.size': 20})


# Plotting the feature importance for Top 10 most important columns
feature_importances = pd.Series(best_model.feature_importances_, index=X.columns)
feature_importances.nlargest(5).plot(kind='barh')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Top 5 Most Important Features')
plt.show()

# Scatter plots for each output column using the best model
plt.figure(figsize=(10, 8))
for i, column in enumerate(y.columns):
    plt.subplot(2, 2, i + 1)
    plt.scatter(y_test[column], best_model.predict(X_test)[:, i], alpha=0.7)
    min_val = min(np.min(y_test[column]), np.min(best_model.predict(X_test)[:, i]))
    max_val = max(np.max(y_test[column]), np.max(best_model.predict(X_test)[:, i]))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')  # Diagonal line for reference
    
    # Labeling x-axis and y-axis based on the column
    if i == 0:
        plt.xlabel('Actual Jsc')
        plt.ylabel('Predicted Jsc')
    elif i == 1:
        plt.xlabel('Actual Voc')
        plt.ylabel('Predicted Voc')
    elif i == 2:
        plt.xlabel('Actual FF')
        plt.ylabel('Predicted FF')
    elif i == 3:
        plt.xlabel('Actual PCE')
        plt.ylabel('Predicted PCE')
    
    #plt.title(f'Actual vs Predicted for {column}')  # Title for each subplot
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
plt.tight_layout()
plt.show()