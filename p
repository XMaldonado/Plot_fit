import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import curve_fit

# Function to eliminate low-frequency tickers
def eliminate_low_frequency_tickers(df, column_name='TICKER', min_count=5):
    ticker_counts = df[column_name].value_counts()
    return df[df[column_name].isin(ticker_counts[ticker_counts >= min_count].index)]

# Apply function to filter tickers
df_selected = eliminate_low_frequency_tickers(df, column_name='TICKER', min_count=5)

# Filter based on bond quality ratings
df_selected = df_selected[df_selected['QUALITYE'].isin(
    ['AAA', 'AA1', 'AA2', 'AA3', 'A1', 'A2', 'A3', 'BAA1', 'BAA2', 'BAA3', 'BA1']
)]

# Select necessary columns
df_selected = df_selected[['CUSIP', 'DURADJMOD', 'OAS_BP', 'TICKER']]

# Convert numeric columns to proper format
df_selected['DURADJMOD'] = pd.to_numeric(df_selected['DURADJMOD'], errors='coerce')
df_selected['OAS_BP'] = pd.to_numeric(df_selected['OAS_BP'], errors='coerce')

# Drop missing values and duplicates
df_selected.dropna(subset=['DURADJMOD', 'OAS_BP'], inplace=True)
df_selected = df_selected.drop_duplicates(subset=['CUSIP'])

# Filter for OAS_BP ≥ 30 and TICKER == 'TMUS'
df_filtered = df_selected[(df_selected['OAS_BP'] >= 30) & (df_selected['TICKER'] == 'TMUS')]
df_filtered = df_filtered.drop_duplicates(subset=['DURADJMOD', 'OAS_BP'])

# Remove zero or negative values before applying log
df_filtered = df_filtered[df_filtered['DURADJMOD'] > 0]

# Extract x and y values
x = df_filtered['DURADJMOD']
y = df_filtered['OAS_BP']

# Define logarithmic function
def log_func(x, a, b):
    return a * np.log(x) + b

# Fit curve
params, _ = curve_fit(log_func, x, y)
log_fit_y = log_func(x, *params)

# Create scatter plot
fig = px.scatter(df_filtered, x='DURADJMOD', y='OAS_BP', title='Modified Duration vs OAS',
                 labels={'DURADJMOD': 'Modified Duration', 'OAS_BP': 'OAS'},
                 hover_data=['CUSIP', 'TICKER'])

# Add logarithmic fit line
fig.add_scatter(x=x, y=log_fit_y, mode='lines', name='Logarithmic Fit', line=dict(color='green'))

# Show plot
fig.show()





import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Example dataset (assuming 37 data points)
x = df_filtered['DURADJMOD']
y = df_filtered['OAS_BP']

# Ensure x > 0
x = x[x > 0]
y = y.loc[x.index]  # Match y values to filtered x

# Define the logarithmic function
def log_func(x, a, b):
    return a * np.log(x) + b

# Fit the log function
params, covariance = curve_fit(log_func, x, y)

# Extract fitted parameters
a, b = params

# Generate fitted values
y_fit = log_func(x, a, b)

# Plot data and fit
plt.figure(figsize=(8,5))
plt.scatter(x, y, label="Data", color='blue')
plt.plot(x, y_fit, label=f"Log Fit: y = {a:.3f} * log(x) + {b:.3f}", color='red')
plt.xlabel("DURADJMOD")
plt.ylabel("OAS_BP")
plt.title("Logarithmic Fit to 37 Data Points")
plt.legend()
plt.show()



import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit

# Ensure DURADJMOD is positive
df_filtered = df_filtered[df_filtered['DURADJMOD'] > 0]

# Extract x and y values
x = df_filtered['DURADJMOD']
y = df_filtered['OAS_BP']

# Define log function
def log_func(x, a, b):
    return a * np.log(x) + b

# Fit the log function
params, _ = curve_fit(log_func, x, y)
y_fit = log_func(x, *params)

# Sort x-values for a smooth line
sorted_indices = np.argsort(x)
x_sorted = x.iloc[sorted_indices]
y_fit_sorted = y_fit[sorted_indices]

# Create scatter plot with Plotly Express
fig = px.scatter(df_filtered, x='DURADJMOD', y='OAS_BP', title='Modified Duration vs OAS',
                 labels={'DURADJMOD': 'Modified Duration', 'OAS_BP': 'OAS'},
                 hover_data=['CUSIP', 'TICKER'])

# Add log-fit line using Plotly Graph Objects
fig.add_trace(go.Scatter(x=x_sorted, y=y_fit_sorted, mode='lines', 
                         name='Logarithmic Fit', line=dict(color='red')))

# Show plot
fig.show()





# Sample data
data = {
    'CUSIP': ['87264AAT', '87264AAX', '87264AAZ', '87264ABL', '87264ABN', '87264ABT', '87264ABU', '87264ABV', '87264ABX', '87264ABZ', '87264ACA', '87264ACW', '87264ADA', '87264ADB', '87264ABS', '87264ACB', '87264ACV', '87264ACZ', '87264ADD', '87264ADE', '87264ACS', '87264ADL', '87264ABD', '87264ABF', '87264ABR', '87264ABW', '87264ACQ', '87264ACX', '87264ADC', '87264ADM', '87264ADN', '87264AAV', '87264ABY', '87264ACT', '87264ADF', '87264ACY'],
    'DURADJMOD': [0.195957, 10.508028, 13.801842, 11.957162, 15.319429, 4.974973, 1.061923, 3.231687, 6.015830, 0.941978, 2.808230, 13.842199, 3.036167, 13.968807, 3.432350, 5.381027, 6.308612, 2.702008, 13.866229, 3.449895, 3.717738, 4.026430, 1.945059, 4.454522, 0.935959, 4.718884, 6.146110, 14.833652, 6.841497, 7.690534, 14.432515, 1.268688, 16.384743, 15.332133, 7.029721, 6.662572],
    'OAS_BP': [3.832233, 102.280628, 104.621630, 101.307740, 101.824732, 74.215465, 30.733083, 56.212321, 71.652176, 39.353237, 50.847827, 110.812124, 54.580535, 112.762990, 58.196295, 75.143776, 82.904543, 54.520434, 112.938248, 56.228657, 68.383500, 58.303346, 48.456803, 70.115085, 47.216601, 65.801122, 83.037342, 124.388226, 84.652055, 89.711784, 111.981874, -12.203598, 112.368863, 102.907868, 83.561249, 84.914456],
    'TICKER': ['TMUS'] * 36
}

df_selected = pd.DataFrame(data)
