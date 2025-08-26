import streamlit as st # Streamlit for building web apps
import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

# Streamlit page setup 
st.set_page_config(page_title="Sales Forecast", layout="wide")
st.title("SARIMAX Forecast App (Separate History & Future Excel Upload)")

# File upload section
history_file = st.file_uploader("Upload historical Excel file (must include 'Sales')", type=["xlsx"])
future_file = st.file_uploader("Upload future Excel file (without 'Sales')", type=["xlsx"])
param_file = st.file_uploader("Optional: Upload SARIMAX parameter Excel file", type=["xlsx"])

if history_file and future_file:
    try:
        # Read uploaded Excel files
        df_hist = pd.read_excel(history_file)
        df_future = pd.read_excel(future_file)
        df_hist['date'] = pd.to_datetime(df_hist['date'])
        df_future['date'] = pd.to_datetime(df_future['date'])
    except Exception as e:
        st.error(f"Failed to read Excel file: {e}")
        st.stop()

    # Manually define hourly percentage table
    hourly_pct_table = pd.DataFrame([
        {'weekday': 'Monday'   , 'hour':  0, 'pct':  0.8217},
        {'weekday': 'Monday'   , 'hour': 10, 'pct':  1.1163},
        {'weekday': 'Monday'   , 'hour': 11, 'pct':  3.9205},
        {'weekday': 'Monday'   , 'hour': 12, 'pct':  6.2146},
        {'weekday': 'Monday'   , 'hour': 13, 'pct':  6.1320},
        {'weekday': 'Monday'   , 'hour': 14, 'pct':  9.8118},
        {'weekday': 'Monday'   , 'hour': 15, 'pct': 14.4240},
        {'weekday': 'Monday'   , 'hour': 16, 'pct': 14.3342},
        {'weekday': 'Monday'   , 'hour': 17, 'pct': 12.2613},
        {'weekday': 'Monday'   , 'hour': 18, 'pct':  9.0240},
        {'weekday': 'Monday'   , 'hour': 19, 'pct':  5.4747},
        {'weekday': 'Monday'   , 'hour': 20, 'pct':  3.5286},
        {'weekday': 'Monday'   , 'hour': 21, 'pct':  5.2027},
        {'weekday': 'Monday'   , 'hour': 22, 'pct':  5.6337},
        {'weekday': 'Monday'   , 'hour': 23, 'pct':  3.6789},
        {'weekday': 'Tuesday'  , 'hour':  0, 'pct':  1.6432},
        {'weekday': 'Tuesday'  , 'hour':  9, 'pct':  0.1089},
        {'weekday': 'Tuesday'  , 'hour': 10, 'pct':  1.6844},
        {'weekday': 'Tuesday'  , 'hour': 11, 'pct':  4.3496},
        {'weekday': 'Tuesday'  , 'hour': 12, 'pct':  5.6373},
        {'weekday': 'Tuesday'  , 'hour': 13, 'pct':  6.0121},
        {'weekday': 'Tuesday'  , 'hour': 14, 'pct':  9.9924},
        {'weekday': 'Tuesday'  , 'hour': 15, 'pct': 13.6871},
        {'weekday': 'Tuesday'  , 'hour': 16, 'pct': 14.0856},
        {'weekday': 'Tuesday'  , 'hour': 17, 'pct': 12.1938},
        {'weekday': 'Tuesday'  , 'hour': 18, 'pct':  8.4243},
        {'weekday': 'Tuesday'  , 'hour': 19, 'pct':  5.1254},
        {'weekday': 'Tuesday'  , 'hour': 20, 'pct':  4.5047},
        {'weekday': 'Tuesday'  , 'hour': 21, 'pct':  5.9784},
        {'weekday': 'Tuesday'  , 'hour': 22, 'pct':  5.5029},
        {'weekday': 'Tuesday'  , 'hour': 23, 'pct':  3.5821},
        {'weekday': 'Wednesday', 'hour':  0, 'pct':  0.7063},
        {'weekday': 'Wednesday', 'hour':  8, 'pct':  0.1461},
        {'weekday': 'Wednesday', 'hour':  9, 'pct':  0.1004},
        {'weekday': 'Wednesday', 'hour': 10, 'pct':  2.1683},
        {'weekday': 'Wednesday', 'hour': 11, 'pct':  3.9858},
        {'weekday': 'Wednesday', 'hour': 12, 'pct':  6.0009},
        {'weekday': 'Wednesday', 'hour': 13, 'pct':  7.2354},
        {'weekday': 'Wednesday', 'hour': 14, 'pct':  9.6753},
        {'weekday': 'Wednesday', 'hour': 15, 'pct': 12.8243},
        {'weekday': 'Wednesday', 'hour': 16, 'pct': 13.2680},
        {'weekday': 'Wednesday', 'hour': 17, 'pct': 11.5005},
        {'weekday': 'Wednesday', 'hour': 18, 'pct':  8.6733},
        {'weekday': 'Wednesday', 'hour': 19, 'pct':  5.5665},
        {'weekday': 'Wednesday', 'hour': 20, 'pct':  4.2607},
        {'weekday': 'Wednesday', 'hour': 21, 'pct':  6.3898},
        {'weekday': 'Wednesday', 'hour': 22, 'pct':  5.8136},
        {'weekday': 'Wednesday', 'hour': 23, 'pct':  3.5611},
        {'weekday': 'Thursday' , 'hour':  0, 'pct':  1.3566},
        {'weekday': 'Thursday' , 'hour':  1, 'pct':  0.0936},
        {'weekday': 'Thursday' , 'hour':  7, 'pct':  0.0733},
        {'weekday': 'Thursday' , 'hour':  8, 'pct':  0.2753},
        {'weekday': 'Thursday' , 'hour': 10, 'pct':  1.5299},
        {'weekday': 'Thursday' , 'hour': 11, 'pct':  4.6718},
        {'weekday': 'Thursday' , 'hour': 12, 'pct':  5.0097},
        {'weekday': 'Thursday' , 'hour': 13, 'pct':  5.6916},
        {'weekday': 'Thursday' , 'hour': 14, 'pct':  9.3588},
        {'weekday': 'Thursday' , 'hour': 15, 'pct': 13.6138},
        {'weekday': 'Thursday' , 'hour': 16, 'pct': 13.4083},
        {'weekday': 'Thursday' , 'hour': 17, 'pct': 11.9871},
        {'weekday': 'Thursday' , 'hour': 18, 'pct':  8.6683},
        {'weekday': 'Thursday' , 'hour': 19, 'pct':  6.2902},
        {'weekday': 'Thursday' , 'hour': 20, 'pct':  4.2991},
        {'weekday': 'Thursday' , 'hour': 21, 'pct':  5.2933},
        {'weekday': 'Thursday' , 'hour': 22, 'pct':  6.5979},
        {'weekday': 'Thursday' , 'hour': 23, 'pct':  4.1326},
        {'weekday': 'Friday'   , 'hour':  0, 'pct':  3.0171},
        {'weekday': 'Friday'   , 'hour':  1, 'pct':  0.2611},
        {'weekday': 'Friday'   , 'hour': 10, 'pct':  1.1176},
        {'weekday': 'Friday'   , 'hour': 11, 'pct':  3.4583},
        {'weekday': 'Friday'   , 'hour': 12, 'pct':  3.8167},
        {'weekday': 'Friday'   , 'hour': 13, 'pct':  4.7360},
        {'weekday': 'Friday'   , 'hour': 14, 'pct':  8.0680},
        {'weekday': 'Friday'   , 'hour': 15, 'pct': 12.3407},
        {'weekday': 'Friday'   , 'hour': 16, 'pct': 13.6161},
        {'weekday': 'Friday'   , 'hour': 17, 'pct': 12.3480},
        {'weekday': 'Friday'   , 'hour': 18, 'pct':  8.7785},
        {'weekday': 'Friday'   , 'hour': 19, 'pct':  6.2695},
        {'weekday': 'Friday'   , 'hour': 20, 'pct':  4.9406},
        {'weekday': 'Friday'   , 'hour': 21, 'pct':  5.2641},
        {'weekday': 'Friday'   , 'hour': 22, 'pct':  7.4941},
        {'weekday': 'Friday'   , 'hour': 23, 'pct':  6.1373},
        {'weekday': 'Saturday' , 'hour':  0, 'pct':  2.2974},
        {'weekday': 'Saturday' , 'hour':  1, 'pct':  0.4611},
        {'weekday': 'Saturday' , 'hour':  9, 'pct':  0.2650},
        {'weekday': 'Saturday' , 'hour': 10, 'pct':  1.4649},
        {'weekday': 'Saturday' , 'hour': 11, 'pct':  3.1850},
        {'weekday': 'Saturday' , 'hour': 12, 'pct':  3.2666},
        {'weekday': 'Saturday' , 'hour': 13, 'pct':  4.6632},
        {'weekday': 'Saturday' , 'hour': 14, 'pct':  8.4139},
        {'weekday': 'Saturday' , 'hour': 15, 'pct': 14.2941},
        {'weekday': 'Saturday' , 'hour': 16, 'pct': 15.0762},
        {'weekday': 'Saturday' , 'hour': 17, 'pct': 12.6801},
        {'weekday': 'Saturday' , 'hour': 18, 'pct': 10.7997},
        {'weekday': 'Saturday' , 'hour': 19, 'pct':  6.9441},
        {'weekday': 'Saturday' , 'hour': 20, 'pct':  3.9891},
        {'weekday': 'Saturday' , 'hour': 21, 'pct':  4.5549},
        {'weekday': 'Saturday' , 'hour': 22, 'pct':  5.8209},
        {'weekday': 'Saturday' , 'hour': 23, 'pct':  4.5980},
        {'weekday': 'Sunday'   , 'hour':  0, 'pct':  0.6381},
        {'weekday': 'Sunday'   , 'hour':  9, 'pct':  0.0575},
        {'weekday': 'Sunday'   , 'hour': 10, 'pct':  1.1614},
        {'weekday': 'Sunday'   , 'hour': 11, 'pct':  4.6414},
        {'weekday': 'Sunday'   , 'hour': 12, 'pct':  4.6417},
        {'weekday': 'Sunday'   , 'hour': 13, 'pct':  6.1632},
        {'weekday': 'Sunday'   , 'hour': 14, 'pct':  9.2333},
        {'weekday': 'Sunday'   , 'hour': 15, 'pct': 12.7658},
        {'weekday': 'Sunday'   , 'hour': 16, 'pct': 15.3479},
        {'weekday': 'Sunday'   , 'hour': 17, 'pct': 13.7818},
        {'weekday': 'Sunday'   , 'hour': 18, 'pct':  9.9990},
        {'weekday': 'Sunday'   , 'hour': 19, 'pct':  6.3048},
        {'weekday': 'Sunday'   , 'hour': 20, 'pct':  4.1489},
        {'weekday': 'Sunday'   , 'hour': 21, 'pct':  4.2741},
        {'weekday': 'Sunday'   , 'hour': 22, 'pct':  8.0625},
        {'weekday': 'Sunday'   , 'hour': 23, 'pct':  5.5539}
    ])

    weekday_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
                   4: 'Friday', 5: 'Saturday', 6: 'Sunday'}

    # Ensure 'Sales' column exists in historical data 
    if 'Sales' not in df_hist.columns:
        st.error("Historical data must include a 'Sales' column.")
        st.stop()

    for df in [df_hist, df_future]:
        df['is_weekend'] = df['date'].dt.weekday >= 5
        df['day_of_week'] = df['date'].dt.weekday
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        bool_cols = df.select_dtypes(include='bool').columns
        df[bool_cols] = df[bool_cols].astype(int)

    # Extract exogenous variables
    exog_cols = [col for col in df_hist.columns if col not in ['date', 'Sales']]

    best_rmse = float('inf')
    best_model = None
    best_result = None

    if param_file:
        try:
            param_df = pd.read_excel(param_file)
        except Exception as e:
            st.error(f"Failed to read parameter file: {e}")
            st.stop()
    else:
        param_df = pd.DataFrame([{
            'order_p': 1, 'order_d': 0, 'order_q': 1,
            'seasonal_p': 1, 'seasonal_d': 0, 'seasonal_q': 1, 'seasonal_s': 7
        }])

    # Model training loop
    with st.spinner("Training models..."):
        for idx, row in param_df.iterrows():
            try:
                model = SARIMAX(
                    df_hist['Sales'],
                    exog=df_hist[exog_cols],
                    order=(row['order_p'], row['order_d'], row['order_q']),
                    seasonal_order=(row['seasonal_p'], row['seasonal_d'], row['seasonal_q'], row['seasonal_s']),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                result = model.fit(disp=False)
                pred = result.get_prediction(
                    start=len(df_hist),
                    end=len(df_hist) + len(df_future) - 1,
                    exog=df_future[exog_cols],
                    dynamic=True
                )
                train_pred = result.fittedvalues
                train_rmse = sqrt(mean_squared_error(df_hist['Sales'], train_pred))
                # Save model with lowest RMSE
                if train_rmse < best_rmse:
                    best_rmse = train_rmse
                    best_model = result
                    best_result = pred
            except Exception as e:
                st.warning(f"Failed on parameter set {row.to_dict()} due to error: {e}")
                continue

    if best_model is None:
        st.error("All parameter sets failed. Please check your data and parameters.")
        st.stop()

    forecast = best_result.predicted_mean
    df_future['Predicted_Sales'] = forecast.values

    st.success(f"Prediction complete! Best RMSE on training set: {best_rmse:.2f}")
    st.dataframe(df_future[['date', 'Predicted_Sales']]) # Show forecasted sales

    df_future['day_of_week'] = df_future['date'].dt.weekday
    df_future['weekday_name'] = df_future['day_of_week'].map(weekday_map)

    hourly_preds = []
    for _, row in df_future.iterrows():
        weekday = row['weekday_name']
        daily_sales = row['Predicted_Sales']
        date = row['date']
        daily_pct = hourly_pct_table[hourly_pct_table['weekday'] == weekday]
        for _, pct_row in daily_pct.iterrows():
            hour = int(pct_row['hour'])
            pct = pct_row['pct']
            hourly_sales = daily_sales * pct / 100 # Calculate hourly sales
            hourly_preds.append({
                'date': date,
                'hour': hour,
                'Predicted_Hourly_Sales': round(hourly_sales, 2)
            })

    hourly_df = pd.DataFrame(hourly_preds)
    hourly_df = hourly_df.sort_values(['date', 'hour'])

    # Show hourly forecast table 
    st.subheader("Hourly Forecast Table")
    st.dataframe(hourly_df)

    # Plot sales forecast
    st.subheader("Sales Forecast Plot")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_hist['date'], df_hist['Sales'], label="Historical Sales", color="blue")
    ax.plot(df_future['date'], df_future['Predicted_Sales'], label="Forecast", color="orange", linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.set_title("SARIMAX Sales Forecast")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

else:
    # Prompt user to upload files
    st.info("Please upload both historical and future Excel files to proceed.")
