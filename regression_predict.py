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
	if y_pred[0] > 0:
		y_pred_f = y_pred[0]
	else:
		y_pred_f = minimo_medv()
	return y_pred_f

# carico modello RANSACRegressor serializzato e stampo la predizione da un nuovo valore in input
def ransac_housing(rm):
	ran = pickle.load(open(os.path.join('predictor', 'pkl_objects', 'ransac_regressor.pkl'),'rb'))
	y_pred_ransac = ran.predict(rm)
	y_pred_ransac_f = 0
	if y_pred_ransac[0] > 0:
		y_pred_ransac_f = y_pred_ransac[0]
	else:
		y_pred_ransac_f = minimo_medv()
	return y_pred_ransac_f

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
	return y_pred_f

#valutazione prestazioni: stampo errore quadratico medio
# def errore_quadratico():
# 	rlinmul = pickle.load(open(os.path.join('predictor', 'pkl_objects', 'linmul_regressor.pkl'),'rb'))
# 	df = carica_dataset()
# 	X=df.iloc[:,:-1].values
# 	y=df['MEDV'].values
# 	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)
# 	y_train_pred=rlinmul.predict(X_train)
# 	y_test_pred=rlinmul.predict(X_test)
# 	msetrain = mean_squared_error(y_train,y_train_pred)
# 	msetest= mean_squared_error(y_test, y_test_pred)
# 	return "MSE train " + str(msetrain) + " test " + str(msetest)

#valutazione prestazioni: stampo punteggio r2 (l'errore quadratico medio non fornisce una predizione standard, poichè i valori possono essere su scala diversa)
# def punteggio_r2():
# 	rlinmul = pickle.load(open(os.path.join('predictor', 'pkl_objects', 'linmul_regressor.pkl'),'rb'))
# 	df = carica_dataset()
# 	X=df.iloc[:,:-1].values
# 	y=df['MEDV'].values
# 	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)
# 	y_train_pred=rlinmul.predict(X_train)
# 	y_test_pred=rlinmul.predict(X_test)
# 	rtrain = r2_score(y_train,y_train_pred)
# 	rtest= r2_score(y_test, y_test_pred)
# 	return "R2 train " + str(rtrain) + " test " + str(rtest)

#valutazione prestazioni: stampo punteggio r2 (l'errore quadratico medio non fornisce una predizione standard, poichè i valori possono essere su scala diversa)
def punteggio_r2():
	rlin = pickle.load(open(os.path.join('predictor', 'pkl_objects', 'lin_regressor.pkl'),'rb'))
	df = carica_dataset()
	X=df[['RM']].values
	y=df['MEDV'].values
	y_pred = rlin.predict(X)
	rdue = r2_score(y, y_pred)
	return "Punteggio R2 regressione lineare " + str(rdue)

def punteggio_r2_quad():
	rquadmul = pickle.load(open(os.path.join('predictor', 'pkl_objects', 'quad_regressor.pkl'),'rb'))
	df = carica_dataset()
	X=df[['LSTAT']].values
	y=df['MEDV'].values
	quadratic = PolynomialFeatures(degree=2)
	X_quad = quadratic.fit_transform(X)
	y_pred_q = rquadmul.predict(X_quad)
	rduequad = r2_score(y, y_pred_q)
	return "Punteggio R2 regressione polinomiale " + str(rduequad)




