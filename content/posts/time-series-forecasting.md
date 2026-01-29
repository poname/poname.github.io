---
title: "Optimizing Time Series Forecasting with Prophet"
date: 2023-10-27
draft: false
tags: ["Python", "Machine Learning", "Time Series"]
categories: ["Technical"]
summary: "A deep dive into hyperparameter tuning for seasonality detection."
cover:
    image: "https://via.placeholder.com/800x400" # Replace with your chart image
    alt: "Time Series Chart"
    relative: false # Use absolute URL or put image in static folder
---

## The Challenge

In retail demand planning, standard ARIMA models often fail to capture holiday effects. In this project, I explored using Meta's **Prophet** library to handle non-linear trends.

## The Mathematical Model

Prophet works as an additive regression model:

$$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$

Where $g(t)$ is the trend, $s(t)$ is seasonality, and $h(t)$ represents holiday effects.

## Implementation

We started by initializing the model with a custom seasonality prior scale to prevent overfitting on weekly fluctuations.

```python
import pandas as pd
from prophet import Prophet

# Load data
df = pd.read_csv('sales_data.csv')
df.columns = ['ds', 'y']

# Configure model with relaxed seasonality
m = Prophet(seasonality_prior_scale=0.1)
m.fit(df)

future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
