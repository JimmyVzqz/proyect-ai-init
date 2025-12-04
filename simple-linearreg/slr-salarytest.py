import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values  # Years of Experience
y = dataset.iloc[:, -1].values    # Salary

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Predicting the Test set results
y_pred = regressor.predict(X_test)


# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# Predicting a single value (e.g., 10.3 years of experience)
years_exp = 10.3
predicted_salary = regressor.predict([[years_exp]])

print(f"The predicted salary for {years_exp} years of experience is: {predicted_salary[0]}")


# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red', label='Actual Data')
plt.plot(X_train, regressor.predict(X_train), color = 'blue', label='Regression Line')
# --- NEW: Plotting the custom point ---
plt.scatter(years_exp, predicted_salary, color='orange', s=50, zorder=5, label='My Prediction (10.3 yrs)')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# Visualising the specific math "formula" the ML model created
print(f"Coefficient (Slope): {regressor.coef_[0]}")
print(f"Intercept: {regressor.intercept_}")


# Measuring the Error (Residual)
# Create a DataFrame to compare Actual vs Predicted side-by-side
df_comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Calculate the Error (Residual)
df_comparison['Error'] = df_comparison['Actual'] - df_comparison['Predicted']

print("\n--- Detailed Comparison ---")
print(df_comparison)

# Calculate the global metrics
from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Absolute Error: ${mae:.2f}")
print(f"R2 Score (Accuracy): {r2:.2f}")