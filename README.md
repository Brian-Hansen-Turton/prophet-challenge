
# Forecasting Net Prophet: A Data Science Challenge

**Project Overview**

This project focuses on analyzing and forecasting search traffic data for MercadoLibre, the largest e-commerce platform in Latin America. We aim to understand how search traffic correlates with stock prices and whether we can predict trends using time-series forecasting models. The key challenge is to analyze the company's financial and user data in creative ways to drive growth.

**Objective:** 
To determine if patterns in Google search traffic can predict stock price movements for MercadoLibre and create a time-series model using Facebook Prophet to forecast future search trends.

## Steps to Complete

The project consists of the following four steps:

### Step 1: Analyze Unusual Patterns in Google Search Traffic
We explore hourly Google search traffic data to find patterns or anomalies related to corporate financial events.

1. **Load the search traffic data** for MercadoLibre.
2. **Identify patterns** during significant financial events (e.g., quarterly financial results in May 2020).
3. **Compare monthly search traffic** to the overall median to identify spikes or trends.

### Step 2: Mine the Search Data for Seasonality
We investigate if search traffic follows predictable seasonal patterns by analyzing traffic by hour, day of the week, and week of the year.

1. **Group and analyze search traffic by hour** of the day to identify peak hours.
2. **Analyze traffic by day of the week** to find the most popular days.
3. **Group by week of the year** to see if there's an increase in traffic around specific holidays or events.

### Step 3: Relate Search Traffic to Stock Price Patterns
We investigate whether a relationship exists between search traffic and MercadoLibre's stock prices.

1. **Load stock price data** and compare it to search trends.
2. **Analyze 2020 trends** during the first half of the year.
3. **Add lagged search trends** and calculate stock volatility and returns to find correlations between them.

### Step 4: Time Series Forecasting with Prophet
We build a time-series forecasting model using Facebook's Prophet to predict future search traffic based on past trends.

1. **Prepare the data** for Prophet by formatting it into appropriate columns.
2. **Fit the Prophet model** and generate predictions for the next 2000 hours (~80 days).
3. **Analyze the forecast** to identify near-term trends and visualize the results.

---

## Setup Instructions

### Prerequisites

- **Python 3.8+**
- **Required Libraries:** Prophet, Pandas, NumPy, Matplotlib, and other dependencies.

### Installation

To install the required libraries, run:

```bash
pip install prophet pandas matplotlib numpy
```

### Data

Download the datasets used in this project:

1. **Google Hourly Search Trends**: 
   Download from [link](https://static.bc-edx.com/ai/ail-v-1-0/m8/lms/datasets/google_hourly_search_trends.csv)
   
2. **Mercado Stock Price Data**: 
   Download from [link](https://static.bc-edx.com/ai/ail-v-1-0/m8/lms/datasets/mercado_stock_price.csv)

### Running the Code

1. Load the search traffic and stock price data into separate Pandas DataFrames.
2. Complete each of the steps as outlined in the notebook.
3. Visualize key results using Matplotlib and plot the forecasted search trends using Prophet.

```python
import pandas as pd
from prophet import Prophet
import numpy as np
import matplotlib.pyplot as plt

# Load datasets
df_trends = pd.read_csv('google_hourly_search_trends.csv', index_col='Date', parse_dates=True).dropna()
df_stock = pd.read_csv('mercado_stock_price.csv', index_col='date', parse_dates=True).dropna()

# Prepare data for Prophet
df_trends_reset = df_trends.reset_index()
df_trends_reset.columns = ['ds', 'y']

# Fit Prophet model
model = Prophet()
model.fit(df_trends_reset)

# Forecast future traffic for 2000 hours
future = model.make_future_dataframe(periods=2000, freq='H')
forecast = model.predict(future)

# Plot forecast
model.plot(forecast)
plt.show()

# Plot forecast components (trends, seasonality, etc.)
model.plot_components(forecast)
plt.show()
```

---

## Analysis & Insights

- **Unusual Patterns:** We identified increased search traffic in May 2020 when MercadoLibre released its quarterly financial results.
- **Seasonality:** Search traffic peaks during late-night hours and on Tuesdays, with an annual dip around late September.
- **Stock Price Correlation:** No strong predictable relationship was found between lagged search traffic and stock volatility or price returns.
- **Forecasting with Prophet:** The model predicts a downward trend in search traffic for the next 80 days, peaking around midnight each day.

---

## Conclusion

This analysis provides insights into how search traffic data can be leveraged to understand stock price movements. Though a direct correlation is not always evident, seasonal trends in search traffic offer valuable insights that could inform marketing and financial strategies for MercadoLibre.

---

Let me know if you need further modifications or clarification!
