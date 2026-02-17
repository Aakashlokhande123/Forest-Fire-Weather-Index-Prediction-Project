import pickle
from flask import Flask, request, render_template

web = Flask(__name__)


ridge_model = pickle.load(open(
    'C:/Users/lokha/OneDrive/Desktop/Forest Fire Weather Index Prediction Project/Models/ridge.pkl', 'rb'))
scaler_model = pickle.load(open(
    'C:/Users/lokha/OneDrive/Desktop/Forest Fire Weather Index Prediction Project/Models/scaler.pkl', 'rb'))

@web.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        Temprature = float(request.form['Temprature'])
        RH = float(request.form['RH'])
        Ws = float(request.form['Ws'])
        Rain = float(request.form['Rain'])
        FFMC = float(request.form['FFMC'])
        DMC = float(request.form['DMC'])
        ISI = float(request.form['ISI'])
        Classes = float(request.form['Classes'])
        Region = float(request.form['Region'])

        data = [[Temprature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
        scaled_data = scaler_model.transform(data)
        result = ridge_model.predict(scaled_data)[0]

    return render_template('index.html', result=result)

if __name__ == "__main__":
    web.run(debug=True)
