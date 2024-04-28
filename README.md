# Bag Count Prediction
This repository contains Python code for time series analysis using the Prophet library to predict daily bag count at an airport until September. The analysis involves two main components:

1. Cross Validation:
Filename: cross_validation.py
This script utilizes the ParameterGrid function from the sklearn module to perform cross-validation and identify the best parameter values for the Prophet model.
For enhanced performance, the Parallel function from the tqdm library is used to speed up the runtime.
2. Modeling and Testing:
Filename: modeling_testing.py
After obtaining the best parameter values through cross-validation, this script inserts these values into the forecasting code.
The script generates forecasts and evaluates model performance using metrics such as Mean Absolute Percentage Error (MAPE).
Currently, the MAPE for the test model is around 4%.
Key Insights:

The analysis suggests that the airport will experience its highest daily baggage count during the summer tariff, particularly on August 18th, with over 200,000 bags expected.

Additional Notes:
------------------------------------------------------
-This code assumes that you have access to the relevant dataset containing historical baggage count data.
-Feel free to modify the code and parameters as needed for your specific use case.
