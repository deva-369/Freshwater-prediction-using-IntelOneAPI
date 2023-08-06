import daal4py as d4p
import pandas as pd
from flask import Flask, request, render_template

# Load the pre-trained PCA model
pca_model = d4p.load('pca_model.pickle')

# Load the scaler used to normalize the data for the PCA model
scaler = pd.read_pickle('scaler.pickle')

# Load the classification model to predict water suitability for human consumption
clf = pd.read_pickle('clf.pickle')

# Create a Flask app
app = Flask(__name__)

# Define a function to preprocess user input and make predictions
def predict_water_suitability(temp, do, bod, ph):
    # Create a DataFrame from user input
    user_input = pd.DataFrame({'temperature': [temp], 'DO': [do], 'BOD': [bod], 'pH': [ph]})

    # Scale the user input data using the scaler used for the PCA model
    scaled_input = scaler.transform(user_input)

    # Convert the scaled data to a oneDAL NumericTable
    input_table = d4p.NumericTable(scaled_input, fptype='float32')

    # Compute PCA scores for the user input data
    pca_scores = pca_model.compute(input_table).scores

    # Predict water suitability using the classification model
    prediction = clf.predict(pca_scores)

    # Map the prediction to a human-readable string
    if prediction == 0:
        return 'Not suitable for human consumption'
    else:
        return 'Suitable for human consumption'

# Define a route to handle user input and display results
@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        temp = float(request.form['temp'])
        do = float(request.form['do'])
        bod = float(request.form['bod'])
        ph = float(request.form['ph'])
        result = predict_water_suitability(temp, do, bod, ph)
        return render_template('result.html', result=result)
    else:
        return render_template('index.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
