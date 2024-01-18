import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

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

# Preencha os valores nulos na coluna 'Cabin' com a média dos valores não nulos
df['Cabin_S'] = df['Cabin'].apply(lambda x: 1 if str(x).split('/')[-1] == 'S' else 0)
df['Cabin_P'] = df['Cabin'].apply(lambda x: 1 if str(x).split('/')[-1] == 'P' else 0)

# Verifique se a coluna original 'Cabin' existe no DataFrame
if 'Cabin' in df.columns:
    # Exclua a coluna original 'Cabin'
    df = df.drop('Cabin', axis=1)

# Remover as colunas selecionadas
drop = ['PassengerId', 'Destination', 'Name', 'FoodCourt', 'ShoppingMall']
df = df.drop(drop, axis=1)

df = df.fillna(df.mean())

coluna_alvo = 'Transported'

X = df.drop(coluna_alvo, axis=1)
y = df[coluna_alvo]

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Padronizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

modelo = Sequential()
modelo.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
modelo.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.1)))
modelo.add(Dropout(0.3))
modelo.add(Dense(64, activation='relu'))
modelo.add(Dense(1, activation='sigmoid'))

modelo.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['accuracy'])

# Treinar o modelo
history = modelo.fit(X_train, y_train, epochs=15, batch_size=20, validation_data=(X_test, y_test))

# Avaliar a acurácia do modelo
_, acuracia = modelo.evaluate(X_test, y_test)
print(f'Acurácia do modelo: {acuracia}')

# Gerar previsões para o conjunto de teste
predictions = modelo.predict(X_test)

# Criar DataFrame de submissão
submission_df = pd.DataFrame({'PassengerId': X_test['PassengerId'], 'Transported': predictions.flatten().round().astype(int)})

# Salvar o DataFrame de submissão em um arquivo CSV
submission_df.to_csv('/kaggle/input/spaceship-titanic/sample_submission.csv', index=False)
