import pandas as pd
df = pd.read_csv("Energy_consumption.csv")

df['Timestamp'] = pd.to_datetime(df['Timestamp'])

df['Year'] = df['Timestamp'].dt.year
df['Month'] = df['Timestamp'].dt.month
df['Day'] = df['Timestamp'].dt.day
df['Hour'] = df['Timestamp'].dt.hour

df = df.drop(columns=['Timestamp'])

numeric_cols = [
    'Temperature','Humidity','SquareFootage','Occupancy',
    'HVACUsage','LightingUsage','RenewableEnergy','EnergyConsumption'
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.fillna(df.mean(numeric_only=True))

print("\nColumns after cleaning:\n", df.columns.tolist())
print("\nPreview of cleaned data:\n", df.head())
print("\nMissing values check:\n", df.isnull().sum())

df.to_csv("cleaned_energy_consumption.csv", index=False)
print("\n✔ Cleaned dataset saved as 'cleaned_energy_consumption.csv'")
