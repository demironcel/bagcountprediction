from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, ParameterGrid
from joblib import Parallel, delayed
from tqdm import tqdm

df = pd.read_csv(r'C:\Users\trdonc\OneDrive - Vanderlande\Desktop\dailybagnew.csv')
df['Value'] = df['Value'].astype(float)
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

df.rename(columns={'Date': 'ds', 'Value': 'y'}, inplace=True)

# Calculate IQR, determine outlier bounds, and filter outliers
Q1 = df['y'].quantile(0.25)
Q3 = df['y'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['y'] >= lower_bound) & (df['y'] <= upper_bound)]

train_df, test_df = train_test_split(df, test_size=0.25, shuffle=False)

param_grid = {
    'changepoint_prior_scale': [0.01, 0.1, 0.5, 1.0, 2.0],
    'seasonality_prior_scale': [1.0, 10.0, 50.0, 100.0, 200.0],
    'holidays_prior_scale': [0.01, 0.1, 0.5, 1.0, 2.0],
    'seasonality_mode': ['additive', 'multiplicative'],
    'changepoint_range': [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
    # Add more hyperparameters and values as needed
}

best_mape = float('inf')
best_params = None

# Generate all parameter combinations
grid = ParameterGrid(param_grid)

# Function to fit Prophet model and calculate MAPE for a single parameter combination
def fit_model(params):
    model = Prophet(**params)
    model.fit(train_df)
    df_cv = cross_validation(model, initial='370 days', period='180 days', horizon='30 days')
    df_p = performance_metrics(df_cv)
    mape_values = df_p['mape']
    mean_mape = mape_values.mean()
    return params, mean_mape, df_cv

# Execute parameter combinations in parallel
results = Parallel(n_jobs=-1)(delayed(fit_model)(params) for params in tqdm(grid))

# Iterate through results to find the best parameters
for params, mean_mape, df_cv in results:
    if mean_mape < best_mape:
        best_mape = mean_mape
        best_params = params

# Train the best model with the best parameters found
if best_params:
    best_model = Prophet(**best_params)
    best_model.fit(train_df)

    # Plotting the training and testing intervals using the best model
    plt.figure(figsize=(10, 5))
    plt.plot(train_df['ds'], train_df['y'], label='Training Data', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Prophet Cross-Validation: Training and Testing Intervals')

    # Plotting the period intervals as dashed lines
    for idx, row in df_cv.iterrows():
        plt.axvline(x=row['cutoff'], color='black', linestyle='--')

    # Plotting the forecast horizon as shaded areas
    for idx, row in df_cv.iterrows():
        plt.axvspan(row['cutoff'], row['cutoff'] + pd.Timedelta('30 days'), alpha=0.2, color='red')

    # Plotting the cross-validation horizons with dotted lines
    for idx, row in df_cv.iterrows():
        plt.axvline(x=row['ds'], color='green', linestyle=':', alpha=0.5)

    plt.legend()
    plt.show()

    # Displaying the extracted MAPE values
    print("Best Parameters:")
    print(best_params)
    print("Best MAPE:", best_mape)
else:
    print("No better parameters found. Using default parameters.")
    best_model = Prophet()  # Use default parameters

