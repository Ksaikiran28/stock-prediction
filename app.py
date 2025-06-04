import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
from keras.models import load_model
from flask import Flask, render_template, request, send_file
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

plt.style.use("fivethirtyeight")

app = Flask(__name__)

# Load the model
model = load_model('stock_dl_model.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock = request.form.get('stock', 'POWERGRID.NS')
        
        # Define the start and end dates for stock data
        start = dt.datetime(2000, 1, 1)
        end = dt.datetime(2025, 6, 4)
        
        try:
            df = yf.download(stock, start=start, end=end)
            if df.empty:
                raise ValueError("No data fetched for the given stock symbol.")
        except Exception as e:
            return f"Error: {str(e)}"
        
        # Descriptive Data
        data_desc = df.describe()

        # Calculate EMAs
        ema20 = df.Close.ewm(span=20, adjust=False).mean()
        ema50 = df.Close.ewm(span=50, adjust=False).mean()
        ema100 = df.Close.ewm(span=100, adjust=False).mean()
        ema200 = df.Close.ewm(span=200, adjust=False).mean()

        # Data splitting
        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])

        # Scaling data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)

        # Prepare data for prediction
        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.fit_transform(final_df)

        x_test, y_test = [], []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i - 100:i])
            y_test.append(input_data[i, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)

        # Make predictions
        y_predicted = model.predict(x_test)

        # Inverse scaling for predictions
        scaler = scaler.scale_
        scale_factor = 1 / scaler[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

        # Ensure directories exist
        os.makedirs('static', exist_ok=True)

        # Plot 1: EMA 20 & 50
        ema_chart_path = "static/ema_20_50.png"
        if os.path.exists(ema_chart_path):
            os.remove(ema_chart_path)
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df.Close, 'y', label='Closing Price')
        ax1.plot(ema20, 'g', label='EMA 20')
        ax1.plot(ema50, 'r', label='EMA 50')
        ax1.set_title("Closing Price vs Time (20 & 50 Days EMA)")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Price")
        ax1.legend()
        fig1.savefig(ema_chart_path)
        plt.close(fig1)

        # Plot 2: EMA 100 & 200
        ema_chart_path_100_200 = "static/ema_100_200.png"
        if os.path.exists(ema_chart_path_100_200):
            os.remove(ema_chart_path_100_200)
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(df.Close, 'y', label='Closing Price')
        ax2.plot(ema100, 'g', label='EMA 100')
        ax2.plot(ema200, 'r', label='EMA 200')
        ax2.set_title("Closing Price vs Time (100 & 200 Days EMA)")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Price")
        ax2.legend()
        fig2.savefig(ema_chart_path_100_200)
        plt.close(fig2)

        # Plot 3: Prediction vs Original Trend
        prediction_chart_path = "static/stock_prediction.png"
        if os.path.exists(prediction_chart_path):
            os.remove(prediction_chart_path)
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(y_test, 'g', label="Original Price", linewidth=1)
        ax3.plot(y_predicted, 'r', label="Predicted Price", linewidth=1)
        ax3.set_title("Prediction vs Original Trend")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Price")
        ax3.legend()
        fig3.savefig(prediction_chart_path)
        plt.close(fig3)

        # Save dataset as CSV
        csv_file_path = f"static/{stock}_dataset.csv"
        df.to_csv(csv_file_path)

        return render_template('index.html', **{
            'plot_path_ema_20_50': ema_chart_path, 
            'plot_path_ema_100_200': ema_chart_path_100_200, 
            'plot_path_prediction': prediction_chart_path, 
            'data_desc': data_desc.to_html(classes='table table-bordered'),
            'dataset_link': csv_file_path
        })

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    file_path = f"static/{filename}"
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return f"File {filename} not found!", 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)