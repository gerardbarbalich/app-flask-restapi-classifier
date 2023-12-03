from flask import Flask, jsonify, request
from sklearn.preprocessing import MinMaxScaler
import pickle

app = Flask(__name__)
model_filename = 'data/random_forest_model.pkl'
url = 'http://127.0.0.1:5000/predict'  # Update this URL with your server's IP and port
host = '0.0.0.0'
port = 8080

class RandomForestClassifierAPI:
    def __init__(self, model_filename):
        with open(model_filename, 'rb') as file:
            self.model = pickle.load(file)
        self.scaler = MinMaxScaler(feature_range=(0, 1))  # Scaling between 0 and 1

    def preprocess_input(self, data):
        try:
            # Extracting features from the JSON data
            amount = int(data['Amount'])
            refund = float(data['Refund'])
            is_male = bool(data['Is_male'])
            days_since_incident = int(data['Days_since_incident'])

            # Scaling the features to desired ranges
            scaled_amount = self.scaler.fit_transform([[amount]])[0][0] * 1200
            scaled_refund = self.scaler.fit_transform([[refund]])[0][0] * 3500
            scaled_days_since_incident = self.scaler.fit_transform([[days_since_incident]])[0][0] * 90

            # Check if scaled values are within the desired ranges
            if not (0 <= scaled_amount <= 1200 and 0 <= scaled_refund <= 3500 and 0 <= scaled_days_since_incident <= 90):
                raise ValueError("Scaled values are outside the specified ranges")

            return [scaled_amount, scaled_refund, is_male, scaled_days_since_incident]

        except (KeyError, ValueError, TypeError) as e:
            return {'error': f'Invalid input data: {e}'}

    def predict(self, data):
        try:
            processed_input = self.preprocess_input(data)
            if 'error' in processed_input:
                return processed_input

            prediction_input = [processed_input]

            prediction = self.model.predict(prediction_input)

            return {'prediction': str(prediction[0])}

        except (KeyError, ValueError, TypeError) as e:
            return {'error': f'Invalid input data: {e}'}

classifier = RandomForestClassifierAPI(model_filename)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if data is None:
        return jsonify({'error': 'No JSON data received'})

    result = classifier.predict(data)
    return jsonify(result)

def test_app():
    test_data = {
        'Amount': 76,
        'Refund': 12.12,
        'Is_male': True,
        'Days_since_incident': 12
    }

    response = requests.post(url, json=test_data)

    if response.status_code == 200:
        result = response.json()
        if 'prediction' in result:
            print("Test successful!")
            print("Prediction:", result['prediction'])
        else:
            print("Test failed. Unexpected response.")
    else:
        print("Test failed. Status code:", response.status_code)


if __name__ == '__main__':
    app.run(host=host, port=port, debug=True)
    
    # Run the test after the app starts running
    test_app()
