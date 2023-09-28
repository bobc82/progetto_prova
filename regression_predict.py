import pickle
import os
from housing import minimo_medv

# carico modello LinearRegressor serielizzato e stampo la predizione da un nuovo valore in input
def linear_housing(rm):
	rlin = pickle.load(open(os.path.join('predictor', 'pkl_objects', 'lin_regressor.pkl'),'rb'))
	y_pred = rlin.predict(rm)
	y_pred_f = 0
	if y_pred[0] > 0:
		y_pred_f = y_pred[0]
	else:
		y_pred_f = minimo_medv()
	return y_pred_f

# carico modello RANSACRegressor serielizzato e stampo la predizione da un nuovo valore in input
def ransac_housing(rm):
	ran = pickle.load(open(os.path.join('predictor', 'pkl_objects', 'ransac_regressor.pkl'),'rb'))
	y_pred_ransac = ran.predict(rm)
	y_pred_ransac_f = 0
	if y_pred_ransac[0] > 0:
		y_pred_ransac_f = y_pred_ransac[0]
	else:
		y_pred_ransac_f = minimo_medv()
	return y_pred_ransac_f


