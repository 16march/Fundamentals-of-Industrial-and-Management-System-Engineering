import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
import seaborn as sns

DATA_FOLDER = 'f1_data'
try:
    standings = pd.read_csv(os.path.join(DATA_FOLDER, 'constructor_standings.csv'))
    constructors = pd.read_csv(os.path.join(DATA_FOLDER, 'constructors.csv'))
    races = pd.read_csv(os.path.join(DATA_FOLDER, 'races.csv'))
except FileNotFoundError as e:
    exit(f"Error: Could not find file {e.filename}. Please check the 'f1_data' folder.")

constructors = constructors[['constructorId', 'name']]
races = races[['raceId', 'year']]
df = pd.merge(standings, races, on='raceId')
df = pd.merge(df, constructors, on='constructorId')
df.rename(columns={'name': 'team_name', 'position': 'rank'}, inplace=True)

# processed final standings for each team per year.
final_standings = df.loc[df.groupby(['year', 'constructorId'])['raceId'].idxmax()]
final_standings = final_standings[['year', 'team_name', 'rank']].copy()


TEAM_BUDGETS = {
    2014: {'Mercedes': 450, 'Red Bull': 460, 'Ferrari': 455, 'McLaren': 218, 'Williams': 170}, 
    2015: {'Mercedes': 467, 'Red Bull': 470, 'Ferrari': 510, 'McLaren': 260, 'Williams': 186}, 
    2016: {'Mercedes': 450, 'Red Bull': 468, 'Ferrari': 490, 'McLaren': 250, 'Williams': 180, 'Renault': 200}, 
    2017: {'Mercedes': 460, 'Red Bull': 475, 'Ferrari': 520, 'McLaren': 240, 'Williams': 175, 'Renault': 210}, 
    2018: {'Mercedes': 480, 'Red Bull': 480, 'Ferrari': 550, 'McLaren': 220, 'Williams': 160, 'Renault': 230, 'Sauber': 135}, 
    2019: {'Mercedes': 484, 'Red Bull': 485, 'Ferrari': 570, 'McLaren': 269, 'Williams': 140, 'Racing Point': 180, 'Renault': 272}, 
    2020: {'Mercedes': 442, 'Red Bull': 445, 'Ferrari': 463, 'McLaren': 240, 'Williams': 135, 'Racing Point': 188, 'AlphaTauri': 170}, 
    2021: {'Mercedes': 145, 'Red Bull': 145, 'Ferrari': 145, 'McLaren': 145, 'Williams': 140, 'Aston Martin': 145, 'AlphaTauri': 142}, 
    2022: {'Mercedes': 140, 'Red Bull': 140, 'Ferrari': 140, 'McLaren': 140, 'Williams': 138, 'Aston Martin': 140, 'AlphaTauri': 138},
}
def get_budget(year, name):
    try: return TEAM_BUDGETS[year][name]
    except: return np.mean(list(TEAM_BUDGETS.get(year, {}).values())) if TEAM_BUDGETS.get(year) else 150

ENGINE_SUPPLIERS = ['Mercedes', 'Ferrari', 'Renault', 'Red Bull']

prediction_data = []
for year_to_predict in range(2015, 2023):
    current_year_data = final_standings[final_standings['year'] == year_to_predict]
    previous_year_data = final_standings[final_standings['year'] == year_to_predict - 1]
    
    for _, team_row in current_year_data.iterrows():
        team_name = team_row['team_name']
        prev_rank_series = previous_year_data[previous_year_data['team_name'] == team_name]['rank']
        if prev_rank_series.empty: continue
            
        prediction_data.append({
            'year': year_to_predict,
            'team_name': team_name,
            'rank_current_year': team_row['rank'],
            'rank_previous_year': prev_rank_series.iloc[0],
            'budget_current_year': get_budget(year_to_predict, team_name),
            'is_engine_supplier': 1 if team_name in ENGINE_SUPPLIERS else 0
        })

model_df = pd.DataFrame(prediction_data)
model_df.dropna(inplace=True)
print("Predictive dataset built successfully.")

# Building the season rank prediction model
features = ['rank_previous_year', 'budget_current_year', 'is_engine_supplier']
X = model_df[features]
y = model_df['rank_current_year']

scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
model = LinearRegression(); model.fit(X_scaled, y)
print("Model training complete.")

# result visualizations
model_df['predicted_rank'] = model.predict(X_scaled)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 8))
scatter_plot = sns.scatterplot(
    x='rank_current_year', 
    y='predicted_rank', 
    data=model_df, 
    hue='year', 
    palette='viridis', 
    s=100,
    legend='full'
)

max_rank = model_df['rank_current_year'].max()
plt.plot([1, max_rank], [1, max_rank], '--', color='red', label='Perfect Prediction Line')
plt.title('Model Performance: Actual vs. Predicted Rank', fontsize=16)
plt.xlabel('Actual Final Standing', fontsize=12)
plt.ylabel('Predicted Final Standing', fontsize=12)
scatter_plot.legend(title='Season Year', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.grid(True)
plt.gca().invert_xaxis() 
plt.gca().invert_yaxis() 
plt.tight_layout()
plt.savefig("f1_model_performance.png", dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*50); print("      Season Rank Prediction Model - Results"); print("="*50)
r_squared = model.score(X_scaled, y)
results_df = pd.DataFrame({'Predictor': features, 'Coefficient': model.coef_})
print(f"\nR-squared: {r_squared:.4f}")
print("\nCoefficients for each variable:"); print(results_df)