import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load your data
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

# Train the model
model = Prophet(
    changepoint_prior_scale=1.0,
    seasonality_prior_scale=100.0,
    holidays_prior_scale=0.01,
    seasonality_mode='additive',
    changepoint_range=0.9
)
model.fit(df)

# Measure performance using test set (you can use a separate test set or part of the original dataset)
# For demonstration, let's use the last portion of the data as a test set
test_size = int(0.25 * len(df))
train_df = df[:-test_size]
test_df = df[-test_size:]

# Make predictions for the test set
forecast = model.predict(test_df)
actual_values = df[df['ds'] >= test_df['ds'].min()]['y'].reset_index(drop=True)


# Calculate performance metrics (e.g., MAE, MSE, RMSE, MAPE)
mape = np.mean(np.abs((test_df['y'] - forecast['yhat']) / test_df['y'])) * 100
mape = np.mean(np.abs((actual_values - forecast['yhat']) / actual_values.replace(0, np.nan))) * 100
mae = mean_absolute_error(test_df['y'], forecast['yhat'])
mse = mean_squared_error(test_df['y'], forecast['yhat'])
rmse = np.sqrt(mse)

# Print the performance metrics
print(f"Mean Absolute Error (MAE) for test period: {mae:.2f}")
print(f"Mean Squared Error (MSE) for test period: {mse:.2f}")
print(f"Root Mean Squared Error (RMSE) for test period: {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE) for test period: {mape:.2f}%")

# Plot actual vs predicted for the test set
fig, ax = plt.subplots()
ax.plot(test_df['ds'], test_df['y'], label='Actual Values')
ax.plot(test_df['ds'], forecast['yhat'], label='Forecasted Values', linestyle='dashed')
ax.fill_between(test_df['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.2)
ax.legend()
plt.show()

# Generate future dates until September 1 for strategic planning
future_periods = (pd.to_datetime('2024-09-01') - df['ds'].max()).days
future_df = model.make_future_dataframe(periods=future_periods, include_history=False)

# Make predictions for the distant future until September 1
future_forecast = model.predict(future_df)

print(future_forecast)

max_value = future_forecast['yhat'].max()

print("Maximum value in the 'yhat' column:", max_value)

# Plot the forecast for the distant future
fig, ax = plt.subplots()
ax.plot(df['ds'], df['y'], label='Historical Data')
ax.plot(future_forecast['ds'], future_forecast['yhat'], label='Future Forecast', linestyle='dashed')
ax.fill_between(future_forecast['ds'], future_forecast['yhat_lower'], future_forecast['yhat_upper'], color='gray', alpha=0.2)
ax.legend()
plt.show()
