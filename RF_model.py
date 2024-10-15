import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import joblib
import pickle

# Load dataset
path_file = r'C:\Users\dongn\OneDrive\Desktop\Postdoc\HZB streamlit\Streamlit tutorials\data.csv'
dataframe = pd.read_csv(path_file, sep=',', header=None)

# Remove rows where all values are less than 0 and column 8 (FF) values are less than 1
dataframe = dataframe[(dataframe.iloc[:, :9] >= 0).all(axis=1) & (dataframe.iloc[:, 7] < 1)]
st.dataframe(dataframe)

# Calculate min and max for columns 1 to 6
min_max = {
    "time_min": dataframe.iloc[:, 0].min(), "time_max": dataframe.iloc[:, 0].max(),
    "irrad_min": dataframe.iloc[:, 1].min(), "irrad_max": dataframe.iloc[:, 1].max(),
    "cell_temp_min": dataframe.iloc[:, 2].min(), "cell_temp_max": dataframe.iloc[:, 2].max(),
    "amb_temp_min": dataframe.iloc[:, 3].min(), "amb_temp_max": dataframe.iloc[:, 3].max(),
    "humid_min": dataframe.iloc[:, 4].min(), "humid_max": dataframe.iloc[:, 4].max(),
}

# Save min and max values to a file
with open('min_max.pkl', 'wb') as f:
    pickle.dump(min_max, f)

# Split into input (X) and output (y) variables
X = dataframe.iloc[:, :5]  # Features
y = dataframe.iloc[:, 5:9]  # Target variables
#%%

# Initialize variables to store the maximum R2 value and its corresponding model
max_r2_value = -np.inf
best_model = None

# Repeat the process 1000 times
for _ in range(1):
    # Split the data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Decision Tree Regression
    with st.spinner('Training the model, please wait...'):
        RegModel = RandomForestRegressor(n_estimators=100, criterion='squared_error')  # Using 'squared_error' instead of 'mse'
        RF = RegModel.fit(X_train, y_train)
        st.success('Model training complete!')
        prediction = RF.predict(X_test)
        

    # Model evaluation
    r2_value = metrics.r2_score(y_train, RF.predict(X_train))
    
    # Update the maximum R2 value and the corresponding model if a higher value is found
    if r2_value > max_r2_value:
        max_r2_value = r2_value
        best_model = RF

# Save the trained model
model1_filename = 'RF_model.joblib'
joblib.dump(best_model, model1_filename)
st.write(f"Model saved as {model1_filename}")

# Or saving file
model2_filename = 'RF_model.sav'
joblib.dump(best_model, model2_filename)
st.write(f"Model saved as {model2_filename}")

# Print the maximum R2 value
st.write(f"Maximum R2 Value: {max_r2_value}")

# Set global font size
plt.rcParams.update({'font.size': 20})

# Plotting the feature importance for Top most important columns
st.title("Top 5 Most Important Features")
feature_importances = pd.Series(best_model.feature_importances_, index=X.columns)
fig, ax = plt.subplots()
feature_importances.nlargest(5).plot(kind='barh', ax=ax)
ax.set_xlabel('Feature Importance')
ax.set_ylabel('Features')
#ax.set_title('Top 5 Most Important Features')

# Customizing y-axis labels
yticks_labels = ['Time', 'Irrad', 'Cell_temp', 'Amb_temp', 'Humid']
ax.set_yticklabels(yticks_labels)

plt.tight_layout()
st.pyplot(fig)  # Display the plot in Streamlit



# Scatter plots for each output column using the best model
st.title("Regression plots")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
for i, column in enumerate(y_test.columns):
    ax = axes[i // 2, i % 2]
    ax.scatter(y_test[column], prediction[:, i], alpha=0.7)
    min_val = min(np.min(y_test[column]), np.min(prediction[:, i]))
    max_val = max(np.max(y_test[column]), np.max(prediction[:, i]))
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')  # Diagonal line for reference

    # Labeling x-axis and y-axis based on the column index
    if i == 0:
        ax.set_xlabel('Actual Jsc')
        ax.set_ylabel('Predicted Jsc')
    elif i == 1:
        ax.set_xlabel('Actual Voc')
        ax.set_ylabel('Predicted Voc')
    elif i == 2:
        ax.set_xlabel('Actual FF')
        ax.set_ylabel('Predicted FF')
    elif i == 3:
        ax.set_xlabel('Actual PCE')
        ax.set_ylabel('Predicted PCE')

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

plt.tight_layout()
st.pyplot(fig)  # Display the plot in Streamlit