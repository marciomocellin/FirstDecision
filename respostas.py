# Exercício 1
# Suponha que você possui uma base de dados rotulada com 10 classes não balanceadas, essa base é formada por 40 features de metadados e mais 3 de dados textuais abertos. 
# Para todos os itens:  Informe as bibliotecas usadas, se necessário, o motivo de cada decisão, explore as possibilidades.
# a.	Descreva como faria a modelagem dessas classes.
# b.	Ao finalizar essa modelagem, como iria apresentar essa modelagem para a área contratante?
# c.	Como faria a validação desse modelo?
# d.	Supondo que esses dados são recebidos diariamente, como iria trabalhar com esse desafio?
# e.	Como levaria esse projeto para um ambiente produtivo? 
# f.	EXTRA - Existe mais algo que gostaria de relatar sobre esse caso?

#%%
#a. Modelagem das classes:
#Pré-processamento: verificaria a integridade dos dados buscando erros de encoder e, se necessário normalizando os dados com pandas.
#Balanceamento de classes: Poderia fazer um undersampling para equilibrar as classes e verificaria se as classes continuaram tendo representatividade

# Equilibramos o número de amostras em cada cluster
import pandas as pd
value_counts = dataclear2['DBSCAN'].value_counts()
print("Número de amostras em cada cluster antes do balanceamento:")
print(value_counts)
value_counts = min(value_counts)
dataclear3 = pd.DataFrame()
for i in dataclear2['DBSCAN'].unique():
    dataclear3 = pd.concat([dataclear3, dataclear2.loc[dataclear2['DBSCAN'] == i].sample(value_counts)])
print("Número de amostras em cada cluster após o balanceamento:")
print(dataclear3['DBSCAN'].value_counts())

#Seleção de modelo: Usaria deferentes métricas para comparar os modelos
from sklearn import metrics
print("Acurácia:", metrics.accuracy_score(y_test, y_pred))
print("Precisão (macro):", metrics.precision_score(y_test, y_pred, average='macro'))

    # para comparar modelos como:
import sklearn
sklearn.tree.DecisionTreeClassifier
sklearn.naive_bayes.BernoulliNB
sklearn.svm.LinearSVC
sklearn.linear_model.LogisticRegression

#b. Apresentação do modelo: Mostraria as métricas do modelo e o significado delas em uma apresentação de PowerPoint
#c. Validação do modelo:
# Dividiria os dados em conjuntos de treino e teste.
from sklearn.model_selection import train_test_split
train_test_split
# Usaria deferentes métricas para a validação do modelo
sklearn.metrics.recall_score
sklearn.metrics.f1_score
sklearn.metrics.roc_auc_score

#d. Vigiaria as métricas, apresentadas anteriormente, para saber se há uma perda de qualidade do modelo.

#e. Salvaria o modelo num pickle e encapsularia o projeto em um contêiner docker.

#f. A base de treino deverá ter 3.180 observações para que seja minimamente viável a modelagem.