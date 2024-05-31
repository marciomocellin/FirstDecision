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

#Suponha que você tenha uma base de dados contendo textos jurídicos, como decisões judiciais, petições e documentos legais. A base de dados inclui informações sobre o conteúdo do texto, data, jurisdição e outras informações relevantes. Seu objetivo é criar um sistema de recomendação que sugira textos jurídicos semelhantes a um texto de referência.
#Para todos os itens: Informe as bibliotecas usadas, se necessário, o motivo de cada decisão, explore as possibilidades.
#a. Descreva como você desenvolveria o sistema de recomendação que recebe um texto de referência e sugere os textos mais semelhantes a ele na base de dados.
#b. Como você avaliaria esse sistema de recomendação?
#c. Suponha que novos textos jurídicos sejam adicionados diariamente. Como você manteria o sistema de recomendação atualizado e garantiria que ele continue a fornece recomendações relevantes?

# a. Primeiramente usaria alguma lista de stopwords para remover palavras ruídos. Usaria um algoritmo Word2vec para para transportar as palavras do texto para um espaço vetorial e analisaria a semelhança comparando a distancia entre os vetores.
# b. Usaria métricas como acurácia e F1-score na avaliação, intrínseca e extrínseca, seu desempenho.
# c. Periodicamente teria que recalcular as representações vetoriais dos textos e atualizar o índice de similaridade.

##%
# TESTE 2 - 
# 1.	O que é um desvio padrão e qual é o seu papel na medição da dispersão dos dados?
# R. O desvio padrão é uma medida erro ou de dispersão dos dados em relação à média. Sendo a raiz quadrada de variancia estatistica.
# Ele indica o quanto os valores individuais se afastam da média, isso é, Quanto maior o desvio padrão maior, maior será a dispersão dos dados.

# 2.	Como funciona o teste de hipóteses e qual é a sua finalidade na análise estatística?
# R. Dado duas hipóteses, hipótese nula e hipótese alternativa. Teste de hipóteses usam a ponderacao através das estatísticas de teste (por exemplo, teste t, teste z) para definir se rejeita a hipótese nula em favor da hipótese alternativa.

# 3.	O que é aprendizado supervisionado e como ele difere do aprendizado não supervisionado?
# R. Nas técnicas supervisionadas se usa bases de treino e teste ou validação cruzada para se averiguar a eficácia do modelo, já nas não supervisionadas isso não é possível.

# 4.	O que é transfer learning e como ele é usado em deep learning?
# R. Transfer learning é uma técnica em deep learning onde um modelo pré-treinado é usado como ponto de partida para resolver uma tarefa diferente.

# 5.	Você está conduzindo um experimento A/B em um site de comércio eletrônico para determinar a eficácia de uma nova página de destino na conversão de visitantes em clientes. Como você projetaria o experimento, escolheria as métricas apropriadas para avaliação e realizaria a análise estatística para tirar conclusões significativas?
# R. Um experimento A/B é, em seu amago, é um teste t de Student da estatistica. Nesse caso teste t para proporcao, uma vez que a métrica de sucesso é a taxa de conversão. Então dividiria, aleatoriamente, os visitantes em dois grupos: controle (versão atual) e tratamento (nova página de destino) e aplicaria o teste t para proporcao.
# Sendo a hipótese nula que não há diferenca entre as taxas de conversão e uma hipótese alternativa que há diferenca.

# 6.	Após realizar um teste ANOVA e obter um valor de F significativo, como você determinaria quais grupos são estatisticamente diferentes entre si? 
# R. Aplicaria, dois a dois, o teste t com a correção bonferroni. A correção bonferroni evitaria o inflacionamento do erro do teste.

# 7.	Suponha que você tenha um conjunto de dados com três ou mais grupos para comparar e deseja determinar se há diferenças significativas entre eles. Descreva como você escolheria entre o teste ou outras técnicas estatísticas
# R. A rigor escolheria ANOVA par dados com distribuiçao normal e Kruskal-Wallis para os demais casos.

# 8.	Qual é a importância do pré-processamento de texto em tarefas de NLP? Quais são as etapas comuns no pré-processamento de texto?
# R. O pré-processamento serve para remove ruídos e padronizar texto. Remoção de palavras comuns (como “a”, “de”, “em”), Stemming/Lemmatização, remoção de pontuação e caracteres especiais, normalização de caixa, tokenização.

# 9.	Descreva o processo de vetorização de texto e como modelos de linguagem como o Word2Vec ou o TF-IDF podem ser usados para representar palavras e documentos.
# R. A vetorização de texto transporta as palavras para um espaço vetorial onde as caractesristicas da palavra são transcrita em numeros e formam as codernadas do vetor

# 10.	Como você lidaria com problemas de desequilíbrio de classe em tarefas de classificação de texto em NLP? Quais estratégias seriam eficazes?
# R. Se a classe menos abundante tiver um número significativo de amostra, prefiro alinhar por ela, mais é possível reamostrar para expandir classes pequenas.