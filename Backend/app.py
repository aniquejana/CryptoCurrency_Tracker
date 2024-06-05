from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import io
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the CSV data from the request
        csv_data = request.data.decode('utf-8')
        print("Received CSV data:", csv_data)

        # Convert the CSV data to a pandas DataFrame
        df = pd.read_csv(io.StringIO(csv_data))

        # Preprocess the data
        data = df['Price (INR)'].values

        # Perform a grid search for the best ARIMA parameters
        best_aic = np.inf
        best_order = None
        best_model = None

        # Define the range of parameters for p, d, q
        p_range = range(0, 6)
        d_range = range(0, 3)
        q_range = range(0, 6)

        for p in p_range:
            for d in d_range:
                for q in q_range:
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        fitted_model = model.fit()
                        aic = fitted_model.aic
                        if aic < best_aic:
                            best_aic = aic
                            best_order = (p, d, q)
                            best_model = fitted_model
                    except Exception as e:
                        continue

        # Use the best model for predictions
        prediction_steps = 1
        predictions = best_model.forecast(steps=prediction_steps)
        future_price = predictions[0]

        # Calculate the future price range
        future_price_range = (future_price * 0.95, future_price * 1.05)
        avg_price = np.mean([future_price * 0.95, future_price * 1.05])

        # Calculate the accuracy (mean absolute percentage error)
        forecast_values = best_model.forecast(steps=len(data))
        actual_values = data[-len(forecast_values):]
        mape = np.mean(np.abs((actual_values - forecast_values) / actual_values)) * 100

        # Determine the recommendation
        def get_recommendation(current_price, avg_price):
            if current_price < avg_price:
                return "Buy"
            elif current_price > avg_price:
                return "Sell"
            else:
                return "Hold"

        last_price = data[-1]
        recommendation = get_recommendation(last_price, avg_price)

        # Return the prediction result
        prediction_result = {
            'future_price_range': future_price_range,
            'average_future_price': avg_price,
            'accuracy_percentage': 100 - mape,
            'recommendation': recommendation
        }

        print("Prediction Result:", prediction_result)
        return jsonify(prediction_result)

    except Exception as e:
        print("Error processing request:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()
