import pandas as pd

def visualizza_dataset():
	df = pd.read_csv('housing.data.txt', sep='\s+')
	df.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
	return df.head()

def visualizza_dataset_html():
	df = pd.read_csv('housing.data.txt', sep='\s+')
	df.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
	return df.head().to_html()