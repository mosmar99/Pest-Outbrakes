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
        self.r2 = None

    def prepare_data(self):
        df_resampled = self.data.set_index('graderingsdatum').resample('D').interpolate()
        X = np.array(df_resampled.drop(['varde'], axis=1))
        y = np.array(df_resampled['varde'])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def fit_and_predict(self):
        self.model.fit(self.X_train, self.y_train)
        self.preds = self.model.predict(self.X_test)
        return self.preds

    def evaluate(self):
        self.r2 = r2_score(self.y_test, self.preds)
        mae = mean_absolute_error(self.y_test, self.preds)
        mse = mean_squared_error(self.y_test, self.preds)
        print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {self.r2:.4f}")
        return mae, mse, self.r2

def plot_actual_vs_pred(y_test, preds, title="Actual vs. Predicted Values"):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, preds, color='red', alpha=0.7, edgecolors='black')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add diagonal reference line
    min_val = min(min(y_test), min(preds))
    max_val = max(max(y_test), max(preds))
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='dashed')

    plt.show()

df = pd.read_csv('data.csv')
df = df.drop(['groda', 'skadegorare'], axis=1)

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

df['graderingsdatum'] = pd.to_datetime(df['graderingsdatum'], errors='coerce')
df['Year'] = df['graderingsdatum'].dt.year
years_to_keep = [2019]
df = df[df['Year'].isin(years_to_keep)]

groups = df.groupby('geometry')
top_groups = groups.size().nlargest(4).index  

models = {}
predictions = {}
test_sets = {}
r2_scores = {}

for i, group in enumerate(top_groups):
    print(f"\nTraining model for Group {i+1} ({group})...")
    group_df = df[df['geometry'] == group].drop(columns=['geometry', 'Year', 'GrowthZone'], errors='ignore')

    model = RandomForestModel(group_df)
    model.prepare_data()
    model.fit_and_predict()
    
    models[group] = model
    predictions[group] = model.preds
    test_sets[group] = model.y_test
    
    _, _, r2 = model.evaluate()
    r2_scores[group] = max(r2 if r2 is not None else 0.01, 0.01)  

print("\nCombining models into a strong classifier using weighted averaging...")

total_weight = sum(r2_scores.values())  
weights = {group: r2_scores[group] / total_weight for group in top_groups}

combined_y_test = np.concatenate(list(test_sets.values()))
combined_preds = np.zeros_like(combined_y_test)

for group in top_groups:
    y_test_group = test_sets[group]
    preds_group = predictions[group]

    interpolated_preds = np.interp(combined_y_test, np.sort(y_test_group), np.sort(preds_group))

    combined_preds += weights[group] * interpolated_preds

print("\nStrong Classifier Evaluation:")
mae, mse, r2 = mean_absolute_error(combined_y_test, combined_preds), mean_squared_error(combined_y_test, combined_preds), r2_score(combined_y_test, combined_preds)
print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}")

plot_actual_vs_pred(combined_y_test, combined_preds, title="Strong Classifier: Actual vs Predicted")
