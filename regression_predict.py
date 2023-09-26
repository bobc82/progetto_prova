import pickle
import re
import os
import numpy as np

def linear_housing(rm):
	rlin = pickle.load(open(os.path.join('predictor', 'pkl_objects', 'lin_regressor.pkl'),'rb'))
	y_pred = rlin.predict(rm)
	return y_pred

def ransac_housing(rm):
	ran = pickle.load(open(os.path.join('predictor', 'pkl_objects', 'ransac_regressor.pkl'),'rb'))
	y_pred_ransac = ran.predict(rm)
	return y_pred_ransac

print(linear_housing([[6]]))
print(ransac_housing([[7]]))

