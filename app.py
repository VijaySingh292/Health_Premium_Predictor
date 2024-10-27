from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('health_insurance_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        age = int(request.form['age'])
        sex = 1 if request.form['sex'] == 'male' else 0
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = 1 if request.form['smoker'] == 'yes' else 0
        region = request.form['region']

        # Map region to dummy values (assuming the model expects this)
        region_values = {'northeast': [1, 0, 0], 'northwest': [0, 1, 0],
                         'southeast': [0, 0, 1], 'southwest': [0, 0, 0]}
        region_features = region_values[region]

        # Create input array for prediction
        input_data = np.array([[age, sex, bmi, children, smoker] + region_features])

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Redirect to result page with the prediction value
        return redirect(url_for('result', value=round(prediction, 2)))

    return render_template('index.html')

@app.route('/result')
def result():
    value = request.args.get('value', None)
    return render_template('result.html', prediction=value)

if __name__ == '__main__':
    app.run(debug=True)
