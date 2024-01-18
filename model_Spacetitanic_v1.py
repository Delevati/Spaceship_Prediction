import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Carregar dados do CSV
df = pd.read_csv('/Users/luryan/Documents/persona_project/spaceship-titanic/data/train.csv')

# Crie colunas binárias para os valores específicos da coluna 'HomePlanet'
df['Homeplanet_Mars'] = df['HomePlanet'].apply(lambda x: 1 if x == 'Mars' else 0)
df['Homeplanet_Earth'] = df['HomePlanet'].apply(lambda x: 1 if x == 'Earth' else 0)
df['Homeplanet_Europa'] = df['HomePlanet'].apply(lambda x: 1 if x == 'Europa' else 0)

# Verifique se a coluna 'HomePlanet' existe no DataFrame
if 'HomePlanet' in df.columns:
    # Exclua a coluna original 'HomePlanet'
    df = df.drop('HomePlanet', axis=1)

# Remover as colunas selecionadas
colunas_para_dropar = ['PassengerId', 'Destination', 'Cabin', 'Name']
df = df.drop(colunas_para_dropar, axis=1)

df = df.fillna(df.mean())

coluna_alvo = 'Transported'

X = df.drop(coluna_alvo, axis=1)
y = df[coluna_alvo]

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar um modelo de RandomForest (você pode escolher outro algoritmo)
modelo = RandomForestClassifier(n_estimators=100, random_state=42)

# Treinar o modelo
modelo.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
previsoes = modelo.predict(X_test)

# Avaliar a acurácia do modelo
acuracia = accuracy_score(y_test, previsoes)
print(f'Acurácia do modelo: {acuracia}')
