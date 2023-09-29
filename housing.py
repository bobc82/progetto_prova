import pandas as pd

#caricamento dataset housing dal file di testo presente nella stessa directory del progetto
def carica_dataset():
	df = pd.read_csv('housing.data.txt', sep='\s+')
	df.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
	return df

#visualizzazione prime righe del dataset come html
def visualizza_dataset_html():
	df = carica_dataset()
	return df.head().to_html()

#minimo valore di MEDV (per valori bassi di RM il modello a regressione lineare restituiva variabili negatiche per MEDV, ma non avevano senso)
def minimo_medv():
	df = carica_dataset()
	y=df['MEDV'].values
	return min(y)