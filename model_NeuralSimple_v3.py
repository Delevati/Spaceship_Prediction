import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Carregar dados do CSV
df = pd.read_csv('/Users/luryan/Documents/persona_project/spaceship-titanic/data/train.csv')

# Remover as colunas selecionadas
colunas_para_dropar = ['PassengerId', 'Destination', 'HomePlanet', 'Cabin', 'Name']
df = df.drop(colunas_para_dropar, axis=1)

df = df.fillna(df.mean())

coluna_alvo = 'Transported'

X = df.drop(coluna_alvo, axis=1)
y = df[coluna_alvo]

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Criar um modelo de Rede Neural
modelo = Sequential()
modelo.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
modelo.add(Dense(32, activation='relu'))
modelo.add(Dense(1, activation='sigmoid'))

# Compilar o modelo
modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinar o modelo
modelo.fit(X_train, y_train, epochs=30, batch_size=27, validation_data=(X_test, y_test))

# Fazer previsões no conjunto de teste
previsoes = (modelo.predict(X_test) > 0.5).astype(int)



# Avaliar a acurácia do modelo
acuracia = accuracy_score(y_test, previsoes)
print(f'Acurácia do modelo: {acuracia}')
