#%% 🚀 Imports and Initial Setup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Display settings
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

#%% 📥 Load Data
data_path = os.path.join("data", "CO2 Emissions_Canada.csv")
df = pd.read_csv(data_path)

print("✅ Data loaded successfully!")
print(f"Shape: {df.shape}")
df.head()

#%% 🔍 Basic Info
print("🔧 Data Types and Non-Null Counts:")
print(df.info())

print("\n📊 Summary Stats:")
print(df.describe())

#%% 🧼 Missing Values
print("\n🕳️ Missing Values:")
print(df.isnull().sum())

#%% 🧪 Unique Values per Column
print("\n🔁 Unique Values per Column:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()}")

#%% 🎯 Target Variable Distribution
sns.histplot(df["CO2 Emissions(g/km)"], kde=True, bins=30, color='coral')
plt.title("Distribution of CO2 Emissions")
plt.xlabel("CO2 Emissions (g/km)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

#%% 🔗 Correlation Matrix
numeric_cols = df.select_dtypes(include=['number']).columns
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("📈 Correlation Matrix of Numeric Features")
plt.tight_layout()
plt.show()

#%% 📊 Scatterplots for Top Correlates
top_features = corr_matrix["CO2 Emissions(g/km)"].abs().sort_values(ascending=False)[1:4].index.tolist()
for feature in top_features:
    sns.scatterplot(data=df, x=feature, y="CO2 Emissions(g/km)", hue="Fuel Type", palette="viridis", alpha=0.7)
    plt.title(f"🔁 {feature} vs CO2 Emissions")
    plt.tight_layout()
    plt.show()

#%% 🔤 Categorical Features Overview
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    print(f"\n📦 {col} value counts:")
    print(df[col].value_counts())

#%% 🛠️ Save Cleaned Version for Modeling (Optional)
cleaned_data_path = os.path.join("data", "CO2_Emissions_cleaned.csv")
df.to_csv(cleaned_data_path, index=False)
print(f"💾 Cleaned data saved to: {cleaned_data_path}")

# %%
