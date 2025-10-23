import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import streamlit as st
import plotly.express as px

def _perform_forecasting_methods(self, feeder_row, method, forecast_months):
    """Performs the selected seasonality-aware forecasting method for a single feeder."""
    
    # 1. Prepare Data and Check Validity
    consumption_series = feeder_row[self.date_columns]
    values = consumption_series.values.astype(np.float64)
    
    # Filter for valid data (consumption > 0)
    mask = (values > 0) & (~np.isnan(values))
    clean_values = values[mask]
    
    if len(clean_values) < 12:
        return None, "Insufficient data: Need at least 12 valid months for seasonality-aware forecasting."
    
    historical_labels = [self.date_columns[i] for i, val in enumerate(values) if mask[i]]
    historical_dates = pd.to_datetime(historical_labels)

    # 2. Decompose Time Series (Additive Seasonality)
    historical_df = pd.DataFrame({'date': historical_dates, 'consumption': clean_values, 'index': np.arange(len(clean_values))})
    historical_df['month'] = historical_df['date'].dt.month
    
    # Calculate monthly factors (Seasonality)
    overall_mean = historical_df['consumption'].mean()
    if overall_mean == 0:
        return None, "Cannot forecast: Historical mean consumption is zero."
        
    historical_df['detrended'] = historical_df['consumption'] - overall_mean
    monthly_seasonality = historical_df.groupby('month')['detrended'].mean()
    
    # Detrended Series (Level/Trend Component)
    historical_df['trend_component'] = historical_df.apply(
        lambda row: row['consumption'] - monthly_seasonality.get(row['month'], 0), axis=1
    )
    trend_values = historical_df['trend_component'].values
    
    # 3. Forecast the Trend Component
    x_historic = historical_df['index'].values
    x_future = np.arange(len(clean_values), len(clean_values) + forecast_months)
    
    trend_forecast = np.array([0.0] * forecast_months)
    
    if method == 'Linear Regression':
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_historic, trend_values)
        trend_forecast = intercept + slope * x_future
        status_suffix = "Linear Trend."

    elif method == 'Polynomial (2nd Order)':
        try:
            coeffs = np.polyfit(x_historic, trend_values, 2)
            poly = np.poly1d(coeffs)
            trend_forecast = poly(x_future)
        except Exception:
            # Fallback to linear if polyfit fails (e.g., singular matrix)
            slope, intercept, _, _, _ = stats.linregress(x_historic, trend_values)
            trend_forecast = intercept + slope * x_future
        status_suffix = "Polynomial Trend."

    elif method == 'Exponential Smoothing':
        # Exponential Smoothing on the DETRENDED series (level component)
        train_size = max(12, len(trend_values) - 6)
        train_data = trend_values[:train_size]
        
        def exp_smoothing(data, alpha):
            data = np.asarray(data, dtype=float)
            result = [data[0] if not np.isnan(data[0]) else 0]
            for i in range(1, len(data)):
                current_value = data[i] if not np.isnan(data[i]) else result[-1]
                result.append(alpha * current_value + (1 - alpha) * result[-1])
            return np.array(result)

        def mse(alpha):
            smoothed = exp_smoothing(train_data, alpha[0])
            return np.mean((train_data - smoothed) ** 2)

        try:
            result = minimize(mse, [0.3], bounds=[(0.01, 0.99)], method='L-BFGS-B')
            optimal_alpha = result.x[0]
        except Exception:
            optimal_alpha = 0.3
        
        smoothed_series = exp_smoothing(trend_values, optimal_alpha)
        last_smoothed_value = smoothed_series[-1] if len(smoothed_series) > 0 else 0
        trend_forecast = np.array([last_smoothed_value] * forecast_months)
        status_suffix = f"Exp. Smoothing (α={optimal_alpha:.3f})."

    elif method == 'Seasonal Average':
        # This method does not need decomposition, but we keep the same decomposition
        # framework to project the detrended level (using the overall mean as level)
        trend_forecast = np.array([overall_mean] * forecast_months)
        status_suffix = "Historical Seasonal Average."
        
    else:
        return None, f"Unknown forecasting method: {method}"

    # 4. Reintroduce Seasonality
    last_historical_month = historical_dates[-1]
    forecast_dates_dt = [(last_historical_month + pd.DateOffset(months=i+1)) for i in range(forecast_months)]
    forecast_dates_label = [dt.strftime('%Y-%m') for dt in forecast_dates_dt]

    final_forecast = []
    for i, trend_val in enumerate(trend_forecast):
        target_month = forecast_dates_dt[i].month
        seasonal_factor = monthly_seasonality.get(target_month, 0) # Default to 0 if no factor found
        
        # Combine trend projection and seasonal factor
        predicted_value = trend_val + seasonal_factor
        final_forecast.append(max(0, predicted_value)) # Ensure non-negative
    
    return np.array(final_forecast), f"Forecast successful ({status_suffix})."

def forecast_feeder(self, feeder_name, method, forecast_months):
    """Performs single feeder forecast and generates plot/data."""
    feeder_row = self.df[self.df['feeder_name'] == feeder_name].iloc[0]
    
    forecast, status_msg = _perform_forecasting_methods(self, feeder_row, method, forecast_months)
    
    if forecast is None:
        return None, status_msg
        
    # --- Plotting logic ---
    values = feeder_row[self.date_columns].values.astype(np.float64)
    mask = (values > 0) & (~np.isnan(values))
    clean_values = values[mask]
    historical_labels = [self.date_columns[i] for i, val in enumerate(values) if mask[i]]
    last_historical_month = pd.to_datetime(historical_labels[-1])
    forecast_months_labels = [(last_historical_month + pd.DateOffset(months=i+1)).strftime('%Y-%m') for i in range(forecast_months)]

    forecast_series = pd.Series(forecast, index=forecast_months_labels)
    
    history_df = pd.DataFrame({'Month': historical_labels, 'Consumption': clean_values, 'Type': 'Historical'})
    forecast_df = pd.DataFrame({'Month': forecast_months_labels, 'Consumption': forecast, 'Type': 'Forecast'})
    combined_df = pd.concat([history_df, forecast_df])
    
    fig = px.line(combined_df, x='Month', y='Consumption', color='Type',
                  line_dash='Type', markers=True,
                  title=f'Consumption Forecast for {feeder_name} ({method})',
                  labels={'Consumption': 'Consumption (units)', 'Month': 'Month'},
                  color_discrete_map={'Historical': '#1f77b4', 'Forecast': '#d62728'},
                  template="plotly_white")
                  
    fig.update_layout(yaxis_tickformat='~s', xaxis_tickangle=-45)
    
    return fig, forecast_series.to_frame(name='Forecast Consumption')

def bulk_forecasting(self, method, forecast_months):
    """Performs the selected forecasting method on all feeders and generates an Excel report."""
    
    forecast_data = []
    
    # Use st.progress for visual feedback
    progress_bar = st.progress(0, text="Forecasting in progress...")
    
    # Determine forecast month labels only once for consistent column naming
    last_historical_month = pd.to_datetime(self.date_columns[-1])
    forecast_month_labels = [(last_historical_month + pd.DateOffset(months=i+1)).strftime('%Y-%m') for i in range(forecast_months)]

    for i, (idx, feeder_row) in enumerate(self.df.iterrows()):
        feeder_name = feeder_row['feeder_name']
        
        # Use the core forecasting method
        forecast, status_msg = _perform_forecasting_methods(self, feeder_row, method, forecast_months)
        
        if forecast is not None:
            row_data = {
                'feeder_name': feeder_name,
                'substation_name': feeder_row['substation_name'],
                'nocs': feeder_row['nocs'],
                'total_consumption_forecast': np.sum(forecast)
            }
            # FIX: Correctly iterate over forecast and forecast_month_labels using zip
            forecast_dict = {f'Forecast_{i+1}_{date}': val for i, (date, val) in enumerate(zip(forecast_month_labels, forecast))}
            row_data.update(forecast_dict)
            forecast_data.append(row_data)
        
        progress_bar.progress((i + 1) / len(self.df), text=f"Feeder: {feeder_name} | {i+1}/{len(self.df)} completed.")

    progress_bar.empty()
    
    if not forecast_data:
        return None, "No feeders had sufficient data for bulk forecasting."

    forecast_df = pd.DataFrame(forecast_data)
    
    # Consistent list of forecast column names
    forecast_cols = [f'Forecast_{i+1}_{date}' for i, date in enumerate(forecast_month_labels)]

    # Summarize by NOCS and Substation
    nocs_summary = forecast_df.groupby('nocs').agg(
        Total_Feeder_Count=('feeder_name', 'count'),
        **{f'Forecast {date} (Sum)': (col, 'sum') for col, date in zip(forecast_cols, forecast_month_labels)},
        Total_Forecast_Sum=('total_consumption_forecast', 'sum')
    ).reset_index().rename(columns={'Total_Forecast_Sum': f'Total_Forecast_({forecast_months}_Months)'})
    
    substation_summary = forecast_df.groupby('substation_name').agg(
        Total_Feeder_Count=('feeder_name', 'count'),
        **{f'Forecast {date} (Sum)': (col, 'sum') for col, date in zip(forecast_cols, forecast_month_labels)},
        Total_Forecast_Sum=('total_consumption_forecast', 'sum')
    ).reset_index().rename(columns={'Total_Forecast_Sum': f'Total_Forecast_({forecast_months}_Months)'})
    
    # Generate Excel Report
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        forecast_df.to_excel(writer, sheet_name='Feederwise Forecast', index=False)
        nocs_summary.to_excel(writer, sheet_name='NOCS Forecast Summary', index=False)
        substation_summary.to_excel(writer, sheet_name='Substation Forecast Summary', index=False)
    
    output.seek(0)
    
    return output, forecast_df.shape[0]
