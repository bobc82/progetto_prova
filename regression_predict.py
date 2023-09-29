import pickle
import os
from housing import minimo_medv, carica_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

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

#valutazione prestazioni: stampo errore quadratico medio
def errore_quadratico():
	rlinmul = pickle.load(open(os.path.join('predictor', 'pkl_objects', 'linmul_regressor.pkl'),'rb'))
	df = carica_dataset()
	X=df.iloc[:,:-1].values
	y=df['MEDV'].values
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)
	y_train_pred=rlinmul.predict(X_train)
	y_test_pred=rlinmul.predict(X_test)
	msetrain = mean_squared_error(y_train,y_train_pred)
	msetest= mean_squared_error(y_test, y_test_pred)
	return "MSE train " + str(msetrain) + " test " + str(msetest)

#valutazione prestazioni: stampo punteggio r2 (l'errore quadratico medio non fornisce una predizione standard, poichè i valori possono essere su scala diversa)
def punteggio_r2():
	rlinmul = pickle.load(open(os.path.join('predictor', 'pkl_objects', 'linmul_regressor.pkl'),'rb'))
	df = carica_dataset()
	X=df.iloc[:,:-1].values
	y=df['MEDV'].values
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)
	y_train_pred=rlinmul.predict(X_train)
	y_test_pred=rlinmul.predict(X_test)
	rtrain = r2_score(y_train,y_train_pred)
	rtest= r2_score(y_test, y_test_pred)
	return "R2 train " + str(rtrain) + " test " + str(rtest)



