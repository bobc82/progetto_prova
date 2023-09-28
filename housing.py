import pandas as pd

def carica_dataset():
	df = pd.read_csv('housing.data.txt', sep='\s+')
	df.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
	return df

def visualizza_dataset_html():
	df = carica_dataset()
	return df.head().to_html()

def minimo_medv():
	df = carica_dataset()
	y=df['MEDV'].values
	return min(y)