import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow_decision_forests as tfdf

# Your existing code
df = pd.read_csv('/Users/luryan/Documents/persona_project/spaceship-titanic/data/train.csv')

df['Homeplanet_Mars'] = df['HomePlanet'].apply(lambda x: 1 if x == 'Mars' else 0)
df['Homeplanet_Earth'] = df['HomePlanet'].apply(lambda x: 1 if x == 'Earth' else 0)
df['Homeplanet_Europa'] = df['HomePlanet'].apply(lambda x: 1 if x == 'Europa' else 0)

if 'HomePlanet' in df.columns:
    df = df.drop('HomePlanet', axis=1)

df['Cabin_S'] = df['Cabin'].apply(lambda x: 1 if str(x).split('/')[-1] == 'S' else 0)
df['Cabin_P'] = df['Cabin'].apply(lambda x: 1 if str(x).split('/')[-1] == 'P' else 0)

if 'Cabin' in df.columns:
    df = df.drop('Cabin', axis=1)

drop = ['PassengerId', 'Destination', 'Name']
df = df.drop(drop, axis=1)

df = df.fillna(df.mean())

coluna_alvo = 'Transported'

X = df.drop(coluna_alvo, axis=1)
y = df[coluna_alvo]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train = X.iloc[:int(4*len(X)/5)]
X_test = X[int(4*len(X)/5):]

y_train = X_train.Transported
y_test = X_test.Transported

modelo = Sequential()
modelo.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
modelo.add(Dropout(0.3))
modelo.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.2), input_dim=100))
modelo.add(Dense(64, activation='relu'))
modelo.add(Dense(1, activation='sigmoid'))

modelo.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['accuracy'])

modelo.fit(X_train, y_train, epochs=10, batch_size=22, validation_data=(X_test, y_test))

predictions = modelo.predict(X_test)

_, acuracia = modelo.evaluate(X_test, y_test)
print(f'Acurácia do modelo: {acuracia}')

print(f'Length of index: {len(df.index)}')
print(f'Length of predictions: {len(predictions)}')

final = pd.DataFrame({'PassengerId': X_test.index, 'Transported': predictions.flatten() > 0.5})
final.to_csv('/Users/luryan/Documents/persona_project/spaceship-titanic/output_submission.csv', index=False)

# Load the test dataset for submission
test_df_submission = pd.read_csv('/Users/luryan/Documents/persona_project/spaceship-titanic/data/test.csv')

test_df_submission['Homeplanet_Mars'] = test_df_submission['HomePlanet'].apply(lambda x: 1 if x == 'Mars' else 0)
test_df_submission['Homeplanet_Earth'] = test_df_submission['HomePlanet'].apply(lambda x: 1 if x == 'Earth' else 0)
test_df_submission['Homeplanet_Europa'] = test_df_submission['HomePlanet'].apply(lambda x: 1 if x == 'Europa' else 0)

if 'HomePlanet' in test_df_submission.columns:
    test_df_submission = test_df_submission.drop('HomePlanet', axis=1)

test_df_submission['Cabin_S'] = test_df_submission['Cabin'].apply(lambda x: 1 if str(x).split('/')[-1] == 'S' else 0)
test_df_submission['Cabin_P'] = test_df_submission['Cabin'].apply(lambda x: 1 if str(x).split('/')[-1] == 'P' else 0)

if 'Cabin' in test_df_submission.columns:
    test_df_submission = test_df_submission.drop('Cabin', axis=1)

test_df_submission = test_df_submission.fillna(test_df_submission.mean())

X_submission = scaler.transform(test_df_submission.drop(['PassengerId'], axis=1))

predictions_submission = modelo.predict(X_submission)

n_predictions_submission = (predictions_submission > 0.5).astype(bool)

output_submission = pd.DataFrame({'PassengerId': test_df_submission['PassengerId'], 'Transported': n_predictions_submission.squeeze()})
output_submission.to_csv('/Users/luryan/Documents/persona_project/spaceship-titanic/output_submission.csv', index=False)
