import pickle
import re
import os
import numpy as np

# carico modello LinearRegressor serielizzato e stampo la predizione da un nuovo valore in input
def linear_housing(rm):
	rlin = pickle.load(open(os.path.join('predictor', 'pkl_objects', 'lin_regressor.pkl'),'rb'))
	y_pred = rlin.predict(rm)
	return y_pred

# carico modello RANSACRegressor serielizzato e stampo la predizione da un nuovo valore in input
def ransac_housing(rm):
	ran = pickle.load(open(os.path.join('predictor', 'pkl_objects', 'ransac_regressor.pkl'),'rb'))
	y_pred_ransac = ran.predict(rm)
	return y_pred_ransac

