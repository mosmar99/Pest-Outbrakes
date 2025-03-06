import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xrf import XRandomForestRegressor

class RandomForestModel:
    def __init__(self, data):
        self.data = data
        self.model = XRandomForestRegressor(n_jobs=-1)
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.preds = None

    def prepare_data(self):
        df_resampled = self.data.set_index('graderingsdatum').resample('D').interpolate()
        X = np.array(df_resampled.drop(['varde'], axis=1))
        y = np.array(df_resampled['varde'])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)

    def fit_and_predict(self):
        self.model.fit(self.X_train, self.y_train)
        self.preds = self.model.predict(self.X_test)
        return self.preds

    def evaluate(self):
        mae = mean_absolute_error(self.y_test, self.preds)
        mse = mean_squared_error(self.y_test, self.preds)
        r2 = r2_score(self.y_test, self.preds)
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"R² Score: {r2:.4f}")

def plot_correlation_heatmaps(df, top_groups, x_c=2, y_c=2):
    fig, axes = plt.subplots(x_c, y_c, figsize=(20, 8))
    axes = axes.flatten()

    for i, name in enumerate(top_groups):
        group_df = df[df['geometry'] == name].drop(columns=['geometry', 'graderingsdatum', 'Year', 'GrowthZone'], errors='ignore')
        corr = group_df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True, ax=axes[i], cbar_kws={"shrink": .8})
        axes[i].set_title(f'Group: {i+1}')

    plt.tight_layout()
    plt.show()

def plot_actual_vs_index(data):
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(data['varde'])), data['varde'], color='blue', alpha=0.7, edgecolors='black')
    plt.ylim(0, 100)
    plt.xlabel("Index")
    plt.ylabel("Actual Value (Varde)")
    plt.title("Actual Values Over Index")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_actual_vs_pred(y_test, preds):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, preds, color='red', alpha=0.7, edgecolors='black')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs. Predicted Values")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add diagonal reference line
    min_val = min(min(y_test), min(preds))
    max_val = max(max(y_test), max(preds))
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='dashed')

    plt.show()

# Load and preprocess data
df = pd.read_csv('data.csv')
df = df.drop(['groda', 'skadegorare'], axis=1)

# Assign growth zones
def assign_growth_zone(geometry_str):
    coords = geometry_str.replace('POINT (', '').replace(')', '').split()
    latitude = float(coords[1]) 
    if latitude < 60.0:
        return 0  # Southern Sweden
    elif 60.0 <= latitude <= 64.0:
        return 1  # Central Sweden
    else:
        return 2  # Northern Sweden

df['GrowthZone'] = df['geometry'].apply(assign_growth_zone)

# Filter by specific years
df['graderingsdatum'] = pd.to_datetime(df['graderingsdatum'], errors='coerce')
df['Year'] = df['graderingsdatum'].dt.year
years_to_keep = [2019]  
df = df[df['Year'].isin(years_to_keep)]

# Select the most frequent geometries
groups = df.groupby('geometry')
top_groups = groups.size().nlargest(4).index  # Top 4 groups
group_df = df[df['geometry'] == top_groups[0]].drop(columns=['geometry', 'Year', 'GrowthZone'], errors='ignore')

# Train and evaluate model
rf_model = RandomForestModel(group_df)
rf_model.prepare_data()
rf_model.fit_and_predict()
rf_model.evaluate()

# Generate plots
# plot_correlation_heatmaps(df, top_groups)
plot_actual_vs_index(group_df)
plot_actual_vs_pred(rf_model.y_test, rf_model.preds)
