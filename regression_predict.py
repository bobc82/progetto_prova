import pickle
import os
from housing import minimo_medv, carica_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

# carico modello LinearRegressor serializzato e stampo la predizione da un nuovo valore in input
def linear_housing(rm):
	rlin = pickle.load(open(os.path.join('predictor', 'pkl_objects', 'lin_regressor.pkl'),'rb'))
	y_pred = rlin.predict(rm)
	y_pred_f = 0
	#Inserisco controllo nel caso in cui predict restituisce un valore negativo. Al posto di esso metto il prezzo minimo presente all'interno del dataset
	if y_pred[0] > 0:
		y_pred_f = y_pred[0]
	else:
		y_pred_f = minimo_medv()
	return round(y_pred_f, 3)

# carico modello RANSACRegressor serializzato e stampo la predizione da un nuovo valore in input
def ransac_housing(rm):
	ran = pickle.load(open(os.path.join('predictor', 'pkl_objects', 'ransac_regressor.pkl'),'rb'))
	y_pred_ransac = ran.predict(rm)
	y_pred_ransac_f = 0
	if y_pred_ransac[0] > 0:
		y_pred_ransac_f = y_pred_ransac[0]
	else:
		y_pred_ransac_f = minimo_medv()
	return round(y_pred_ransac_f, 3)

# carico modello LinearRegressor serializzato con regressione quadratica e stampo la predizione da un nuovo valore in input
def quadratic_housing(lstat):
	rquad = pickle.load(open(os.path.join('predictor', 'pkl_objects', 'quad_regressor.pkl'),'rb'))
	quadratic = PolynomialFeatures(degree=2)
	y_pred = rquad.predict(quadratic.fit_transform(lstat))
	y_pred_f = 0
	if y_pred[0] > 0:
		y_pred_f = y_pred[0]
	else:
		y_pred_f = minimo_medv()
	return round(y_pred_f, 3)

#valutazione prestazioni: stampo punteggio r2 della regressione lineare semplice (l'errore quadratico medio non fornisce una predizione standard, poichè i valori possono essere su scala diversa)
def punteggio_r2():
	rlin = pickle.load(open(os.path.join('predictor', 'pkl_objects', 'lin_regressor.pkl'),'rb'))
	df = carica_dataset()
	X=df[['RM']].values
	y=df['MEDV'].values
	y_pred = rlin.predict(X)
	rdue = r2_score(y, y_pred)
	return "Punteggio R2 regressione lineare " + str(round(rdue, 3))

#valutazione prestazioni: stampo punteggio r2 della regressione polinomiale
def punteggio_r2_quad():
	rquadmul = pickle.load(open(os.path.join('predictor', 'pkl_objects', 'quad_regressor.pkl'),'rb'))
	df = carica_dataset()
	X=df[['LSTAT']].values
	y=df['MEDV'].values
	quadratic = PolynomialFeatures(degree=2)
	X_quad = quadratic.fit_transform(X)
	y_pred_q = rquadmul.predict(X_quad)
	rduequad = r2_score(y, y_pred_q)
	return "Punteggio R2 regressione polinomiale " + str(round(rduequad, 3))




