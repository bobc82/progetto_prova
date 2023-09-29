from flask import Flask, render_template, request
from regression_predict import linear_housing, ransac_housing, errore_quadratico, punteggio_r2, quadratic_housing, punteggio_r2_quad
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

#visualizzo grafici dei modelli LinearRegressor e RansacRegressor prodotti in precedenza da Jupyter
@app.route('/graphics')
def visualizza_plot():
	return render_template("graphics.html")

#visualizzo grafico regressione quadratica prodotto in precedenza da Jupyter
@app.route('/graphicsquad')
def visualizza_plot_quad():
	return render_template("graphicsquad.html")

#stampa errore quadratico medio e punteggio r2
@app.route('/val')
def valutazione_modelli():
	errq_str = errore_quadratico()
	r2_str = punteggio_r2()
	return render_template("valutazione.html", errq = errq_str, r2str= r2_str)

#stampa errore punteggio r2 regressione quadratica
@app.route('/valquad')
def valutazione_modello_quad():
	r2_str = punteggio_r2_quad()
	return render_template("valutazionequad.html", r2str= r2_str)

#Url di "atterraggio" a seguito del submit all'interno del form precedente. La funzione predict legge il valore numerico rm dal form e poi fornisce due predizioni:
#la prima è data dalla funzione linear_housing che fa uso del modello LinearRegressor e la seconda, ransac_housing, fa uso del modello RANSACRegressor. Le funzioni sono state
#importate dal file regression_predict.py. Infine stampo a video i risultati visualizzando il template result.html a cui passo pred ed rpred. 
@app.route('/predict', methods = ['POST'])
def predict():
	if request.method == 'POST':
		room_med = float(request.form['rm'])
		lin_pred = linear_housing([[room_med]])
		ransac_pred = ransac_housing([[room_med]])
		return render_template("result.html", pred = lin_pred, rpred = ransac_pred)

#predizione regressione polinomiale LSTAT - MEDV
@app.route('/predictquad', methods = ['POST'])
def predictquad():
	if request.method == 'POST':
		lstat = float(request.form['lstat'])
		quad_pred = quadratic_housing([[lstat]])
		return render_template("resultquad.html", pred = quad_pred)

if __name__ == '__main__':
   app.run(debug=True)