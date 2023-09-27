from flask import Flask, render_template, request
from regression_predict import linear_housing, ransac_housing
from housing import visualizza_dataset_html

app = Flask(__name__)

#nella home visualizzo il form contenente un input numerico che richiede il valore RM e un pulsante che fa redirect sulla route predict con una POST
@app.route('/')
def index():
	return render_template('form_predict.html')

#visualizzo le prime cinque righe del dataset come una tabella html
@app.route('/data')
def visualizza():
	html_head = visualizza_dataset_html()
	return render_template("dataset.html", html_h = html_head)

#visualizzo grafici di entrambi i modelli: LinearRegressor e RansacRegressor prodotti in precedenza da Jupyter
@app.route('/graphics')
def visualizza_plot():
	return render_template("graphics.html")

#Questa è l'url di "atterraggio" a seguito del submit all'interno del form precedente. La funzione predict legge il valore numerico rm dal form e poi fornisce due predizioni:
#la prima è data dalla funzione linear_housing che fa uso del modello LinearRegressor e la seconda, ransac_housing, fa uso del modello RANSACRegressor. Le funzioni sono state
#importate dal file regression_predict.py. Infine stampo a video i risultati visualizzando il template result.html a cui passo pred ed rpred. 
@app.route('/predict', methods = ['POST'])
def predict():
	if request.method == 'POST':
		room_med = float(request.form['rm'])
		lin_pred = linear_housing([[room_med]])
		ransac_pred = ransac_housing([[room_med]])
		return render_template("result.html", pred = lin_pred, rpred = ransac_pred)

if __name__ == '__main__':
   app.run(debug=True)