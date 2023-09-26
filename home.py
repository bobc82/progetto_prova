from flask import Flask, render_template, request
from regression_predict import linear_housing, ransac_housing

app = Flask(__name__)

@app.route('/')
def index():
   return render_template('form_predict.html')

@app.route('/predict', methods = ['POST'])
def predict():
	if request.method == 'POST':
		room_med = float(request.form['rm'])
		print(type(room_med))
		lin_pred = linear_housing([[room_med]])
		ransac_pred = ransac_housing([[room_med]])
		return render_template("result.html", pred = lin_pred, rpred = ransac_pred)

if __name__ == '__main__':
   app.run(debug=True)