import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("cleaned_energy_consumption.csv")

label_enc = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = label_enc.fit_transform(df[col])


plt.figure(figsize=(8,5))
sns.histplot(df['EnergyConsumption'], kde=True)
plt.title("Energy Consumption Distribution")
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

y = df['EnergyConsumption']
X = df.drop(columns=['EnergyConsumption'])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n MODEL PREDICTION: ")
print(f"MAE   : {mae}")
print(f"RMSE  : {rmse}")
print(f"R²    : {r2}")
print(f"MAE  : {mae:.2f} kWh  (Average error in energy consumption)")
print(f"RMSE : {rmse:.2f} kWh  (Penalizes larger prediction errors)")
print(f"R²   : {r2:.3f}  (model accuracy)")

importance = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(10,6))
importance.nlargest(10).plot(kind='barh')
plt.title("Top 10 Feature Importance")
plt.show()

comparison = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
comparison = comparison.reset_index(drop=True)

plt.figure(figsize=(10,6))
plt.plot(comparison['Actual'][:50], label='Actual')
plt.plot(comparison['Predicted'][:50], label='Predicted')
plt.title("Actual vs Predicted Energy Consumption")
plt.xlabel("Samples")
plt.ylabel("Energy Consumption")
plt.legend()
plt.show()

plt.figure(figsize=(6,6))
sns.scatterplot(x=comparison['Actual'], y=comparison['Predicted'])
plt.title("Actual vs Predicted Scatter Plot")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

print("\n Prediction Complete")
