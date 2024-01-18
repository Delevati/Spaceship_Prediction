import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

# Carregar os dados
train_data = pd.read_csv('/Users/luryan/Documents/persona_project/spaceship-titanic/data/test.csv')

# Visualizar as primeiras linhas dos dados
print(train_data.head())

# # Verificar informações sobre os dados, como tipos de dados e valores nulos
# print(train_data.info())

# # Verificar estatísticas descritivas das variáveis numéricas
# print(train_data.describe())