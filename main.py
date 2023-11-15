import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

# Cria um modelo de regressão linear
regLin = LinearRegression()

#atribuindo os arquivos a variáveis dentro do codigo
dfIbovespa = pd.read_csv('Dados/^BVSP.csv')
dfDolar = pd.read_csv('Dados/BRL=X.csv')
dfOuro = pd.read_csv('Dados/Ouro Tratado.csv')
dfPetroleo = pd.read_csv('Dados/Petroleo Tratado.csv')
dfGol = pd.read_csv('Dados/GOLL4.SA.csv')

dfOuro.rename(columns={'Data': 'Date'}, inplace=True) #alterando o nome da coluna para padonização
dfPetroleo.rename(columns={'Data': 'Date'}, inplace=True) #alterando o nome da coluna para padonização

dfOuro = dfOuro.iloc[::-1].reset_index(drop=True) #iverter as linhas sem alterar o index
dfPetroleo = dfPetroleo.iloc[::-1].reset_index(drop=True) #iverter as linhas sem alterar o index

dfOuro['Date'] = dfOuro['Date'].astype(str) #Transformando toda a coluna 'Date' em string
dfPetroleo['Date'] = dfPetroleo['Date'].astype(str) #Transformando toda a coluna 'Date' em string

for index, row in dfOuro.iterrows(): #Mudadando o formato da data para ficar igual a data base dfGol
    if len(row['Date']) == 7:
        dataO = row['Date']
        novaDataO = dataO[3:] + '-' + dataO[1:3] + '-' + '0' + dataO[0]
        dfOuro.at[index, 'Date'] = novaDataO
    else:
        dataO1 = row['Date']
        novaDataO1 = dataO[3:] + '-' + dataO1[2:4] + '-' + dataO1[0:2]
        dfOuro.at[index, 'Date'] = novaDataO1

for index, row in dfPetroleo.iterrows(): #Mudadando o formato da data para ficar igual a data base dfGol
    if len(row['Date']) == 7:
        dataP = row['Date']
        novaDataP = dataP[3:] + '-' + dataP[1:3] + '-' + '0' + dataP[0]
        dfPetroleo.at[index, 'Date'] = novaDataP
    else:
        dataP1 = row['Date']
        novaDataP1 = dataP[3:] + '-' + dataP1[2:4] + '-' + dataP1[0:2]
        dfPetroleo.at[index, 'Date'] = novaDataP1

dfOuro = dfOuro.merge(dfGol, on='Date', how='inner') #removendo as linhas diferentes da database dfGol para ficar igual
dfOuro = dfOuro.drop('Open', axis=1)
dfOuro = dfOuro.drop('High', axis=1)
dfOuro = dfOuro.drop('Low', axis=1)
dfOuro = dfOuro.drop('Close', axis=1)
dfOuro = dfOuro.drop('Adj Close', axis=1)
dfOuro = dfOuro.drop('Volume', axis=1)

dfPetroleo = dfPetroleo.merge(dfGol, on='Date', how='inner') #removendo as linhas diferentes da database dfGol para ficar igual
dfPetroleo = dfPetroleo.drop('Open', axis=1)
dfPetroleo = dfPetroleo.drop('High', axis=1)
dfPetroleo = dfPetroleo.drop('Low', axis=1)
dfPetroleo = dfPetroleo.drop('Close', axis=1)
dfPetroleo = dfPetroleo.drop('Adj Close', axis=1)
dfPetroleo = dfPetroleo.drop('Volume', axis=1)

dfDolar = dfDolar.merge(dfGol, on='Date', how='inner') #removendo as linhas diferentes da database dfGol para ficar igual
dfDolar = dfDolar.drop('Open_y', axis=1)
dfDolar = dfDolar.drop('High_y', axis=1)
dfDolar = dfDolar.drop('Low_y', axis=1)
dfDolar = dfDolar.drop('Close_y', axis=1)
dfDolar = dfDolar.drop('Adj Close_y', axis=1)
dfDolar = dfDolar.drop('Volume_y', axis=1)
dfDolar.columns = ['Date','Open','High','Low','Close','Adj Close','Volume'] #corrigindo o nome das colunas

close_Gol = dfGol['Close'] #Atribuindo a variaveis as colunas close/ultimo de cada database 
close_Dolar = dfDolar['Close']
close_Ouro = dfOuro['Ultimo']
close_Petroleo = dfPetroleo['Ultimo']
close_Ibovespa = dfIbovespa['Close']

# Cria a matriz de características (X) e a variável alvo (y)
X = pd.DataFrame(list(zip(close_Dolar, close_Ouro, close_Petroleo, close_Ibovespa)),
                 columns=['Dolar', 'Ouro', 'Petroleo', 'Ibovespa'])
y = close_Gol

# Divide os dados em conjuntos de treinamento e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.25)

# Treina o modelo de regressão linear
regLin.fit(X_treino, y_treino)

# Faz previsões no conjunto de teste
acertos = regLin.predict(X_teste)

# Avalia o modelo
mae = metrics.mean_absolute_error(y_teste, acertos)
rmse = np.sqrt(metrics.mean_squared_error(y_teste, acertos))

# Imprime as métricas de avaliação
print("Erro Médio Absoluto:", mae)
print("Raiz do Erro Quadrático Médio:", rmse)

# Fornece uma amostra para previsão durante a apresentação
dolar_value = close_Dolar.iloc[-1]
ouro_value = close_Ouro.iloc[-1]
petroleo_value = close_Petroleo.iloc[-1]
ibovespa_value = close_Ibovespa.iloc[-1]

sample_for_prediction = [[dolar_value, ouro_value, petroleo_value, ibovespa_value]]

# Faz previsões para a amostra fornecida
predicted_price = regLin.predict(sample_for_prediction)
print("Preço Previsto da GOL:", predicted_price)

# Plota os valores reais e as previsões
plt.figure(figsize=(10, 6))
plt.plot(y_teste.index, y_teste, label='Valores Reais', marker='o')
plt.plot(y_teste.index, acertos, label='Previsões', marker='o')
plt.title('Valores Reais vs. Previsões para GOL')
plt.xlabel('Índice do Teste')
plt.ylabel('Valor de Fechamento')
plt.legend()
plt.show()

#print(dfDolar)
#print(dfOuro)
#print(dfPetroleo)
#print(dfIbovespa)
#print(dfGol)