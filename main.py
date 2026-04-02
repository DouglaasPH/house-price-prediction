# imports
import os
import ssl

import ssl

import os
from dotenv import load_dotenv

import pandas as pd
import numpy as np
from pyngrok import ngrok
from pyngrok import ngrok
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import urllib
import urllib
from sklearn.ensemble import RandomForestRegressor


# carregamento de dados
LINK = 'https://ocw.mit.edu/courses/15-071-the-analytics-edge-spring-2017/d4332a3056f44e1a1dec9600a31f21c8_boston.csv'
FILE_PATH = 'data.csv'

def carregar_dados():
    if os.path.exists(FILE_PATH):
        print("Carregando dados localmente...")
    else:
        try:
            # ⚠️ contorna erro de SSL
            ssl._create_default_https_context = ssl._create_unverified_context
            urllib.request.urlretrieve(LINK, FILE_PATH)

            print("Download concluído com sucesso!")

        except Exception as e:
            print("Erro ao baixar o dataset:")
            print(e)
            print("\nBaixe manualmente e coloque em: data/data.csv")

    data = pd.read_csv(FILE_PATH)

    print(data.head())
    print(data.describe())
    
    return data


data = carregar_dados()

print(data.head())
print(data.describe())


# análise explorátoria
def analise_exploratoria(df):
    correlacoes = df.drop(['TOWN', 'TRACT'], axis=1).corr()

    plt.figure(figsize=(16,6))
    sns.heatmap(correlacoes, annot=True)
    plt.show()

    px.scatter(df, x='RM', y='MEDV').show()
    px.scatter(df, x='PTRATIO', y='MEDV').show()

    ff.create_distplot([df.RM], ['RM'], bin_size=.2).show()
    px.box(df, y='RM').show()

    ff.create_distplot([df.MEDV], ['MEDV'], bin_size=.2).show()
    px.histogram(df, x='MEDV').show()
    px.box(df, y='MEDV').show()

analise_exploratoria(data)


# limpeza dos dados
def limpar_dados(df):
    df = df.copy()

    # Remove outliers
    top = df.nlargest(20, 'MEDV').index
    df.drop(top, inplace=True)

    # Ajuste tipo
    df['RM'] = df['RM'].astype(int)

    # Categorias
    def categorizar(valor):
        if valor <= 4:
            return 'Pequeno'
        elif valor < 7:
            return 'Medio'
        else:
            return 'Grande'

    df['categorias'] = df['RM'].apply(categorizar)

    return df

data = limpar_dados(data)


# baseline
medias = data.groupby('categorias')['MEDV'].mean()

dic_baseline = {
    'Pequeno': medias['Pequeno'],
    'Medio': medias['Medio'],
    'Grande': medias['Grande']
}

def baseline_pred(rm):
    if rm <= 4:
        return dic_baseline['Pequeno']
    elif rm < 7:
        return dic_baseline['Medio']
    else:
        return dic_baseline['Grande']


# preparação dos dados para modelagem
y = data['MEDV']
X = data.drop(['TOWN','TRACT','LAT','LON','RAD','TAX','MEDV','DIS','AGE','ZN','categorias'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=45
)


# avaliação dos modelos
def avaliar_modelo(nome, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{nome} RMSE: {rmse:.4f}")
    return rmse


# modelos

# Baseline
pred_baseline = [baseline_pred(rm) for rm in X_test['RM']]
avaliar_modelo("Baseline", y_test, pred_baseline)

# Regressão Linear
lin = LinearRegression()
lin.fit(X_train, y_train)
pred_lin = lin.predict(X_test)
avaliar_modelo("Regressão Linear", y_test, pred_lin)

# Árvore
tree = DecisionTreeRegressor()
tree.fit(X_train, y_train)
pred_tree = tree.predict(X_test)
avaliar_modelo("Árvore", y_test, pred_tree)

# Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
avaliar_modelo("Random Forest", y_test, pred_rf)


# visualização dos resultados
df_results = pd.DataFrame({
    'Real': y_test.values,
    'Baseline': pred_baseline,
    'Linear': pred_lin,
    'Arvore': pred_tree,
    'RandomForest': pred_rf
})

fig = go.Figure()

for col in df_results.columns:
    fig.add_trace(go.Scatter(
        x=df_results.index,
        y=df_results[col],
        mode='lines+markers',
        name=col
    ))

fig.show()


# exportação dos dados e treino final do modelo
X['MEDV'] = y
X.to_csv('data.csv', index=False)

data = pd.read_csv('data.csv')

X = data.drop('MEDV', axis=1)
y = data['MEDV']

model = RandomForestRegressor()
model.fit(X, y)

print("Modelo final treinado!")
