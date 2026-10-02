# Projeto de Ciência de Dados — PCA, LDA, Ensemble e AutoML

Este projeto acadêmico reúne três experimentos de Ciência de Dados envolvendo técnicas de redução de dimensionalidade, métodos Ensemble e AutoML.

O objetivo é aplicar, comparar e analisar diferentes técnicas de aprendizado de máquina em problemas de classificação, observando seus resultados, vantagens e limitações.

## Tecnologias utilizadas

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- TPOT

## Conteúdo do projeto

O trabalho está dividido em três partes:

1. **PCA e LDA:** comparação de técnicas de redução de dimensionalidade aplicadas à classificação de vinhos.
2. **Voting e Stacking:** comparação de métodos Ensemble em um problema de classificação de crédito.
3. **AutoML com TPOT:** utilização de programação genética para busca automática de pipelines e hiperparâmetros.

---

# Questão 1 — Classificação de vinhos com LDA e PCA

O objetivo desta etapa é comparar PCA e LDA como técnicas de redução de dimensionalidade antes da aplicação de um classificador KNN (`k=5`) aos três tipos de vinho do dataset `load_wine`.

## Análise dos métodos

### Padronização

Cada característica química é colocada em uma escala comparável. Isso evita que uma variável com valores numericamente maiores exerça influência desproporcional sobre as demais.

### PCA

O PCA (Principal Component Analysis) procura direções que preservem a maior quantidade possível da variação total dos dados.

É uma técnica não supervisionada, ou seja, não utiliza os rótulos das classes durante a redução de dimensionalidade.

### LDA

O LDA (Linear Discriminant Analysis) procura uma projeção que deixe as classes conhecidas o mais separadas possível e, ao mesmo tempo, mantenha próximos os elementos pertencentes à mesma classe.

Diferentemente do PCA, o LDA utiliza os rótulos das classes durante o processo.

## Resultados

Após a execução do experimento, o LDA obteve **94,4% de acurácia**, enquanto o PCA obteve **91,7%**.

Neste cenário, portanto, o LDA apresentou maior acurácia.

## Discussão

O LDA tende a ser adequado para problemas de classificação supervisionada porque utiliza os rótulos durante a redução de dimensionalidade, buscando aumentar a distância entre as classes e reduzir a dispersão dentro de cada uma delas.

O PCA pode apresentar bons resultados quando a informação relevante para classificação está presente nas direções de maior variância dos dados. Também é útil quando se deseja uma representação exploratória independente das classes.

Além disso, o LDA possui uma limitação relacionada ao número de classes, já que pode gerar no máximo `n_classes - 1` componentes discriminantes.

## Conclusão

Após a padronização das variáveis, PCA e LDA foram utilizados para reduzir a dimensionalidade dos dados antes da classificação com KNN.

O PCA preserva a variação geral dos dados, enquanto o LDA utiliza os rótulos das classes para buscar uma representação que favoreça sua separação.

Neste experimento, o LDA apresentou maior acurácia no conjunto de teste, embora esse resultado seja específico às condições e ao conjunto de dados utilizados.

---

# Questão 2 — Stacking e Voting para classificação de crédito

Nesta etapa são comparadas diferentes estratégias de Ensemble aplicadas a um problema de classificação de crédito.

São utilizados modelos individuais e duas técnicas de combinação: Voting e Stacking.

## Métodos utilizados

### Voting Classifier

O Voting Classifier combina as decisões de diferentes modelos.

No Hard Voting, cada modelo realiza uma previsão e a classe que recebe a maioria dos votos é escolhida como resultado final.

Essa abordagem permite combinar modelos diferentes de maneira relativamente simples.

### Stacking Classifier

O Stacking utiliza diversos modelos-base e adiciona um metamodelo responsável por aprender como combinar suas previsões.

Nesse experimento, a regressão logística é utilizada como metamodelo.

## Resultados

A comparação dos modelos considera especialmente o **F1-Score**, métrica que equilibra Precision e Recall.

Neste experimento, o **Stacking superou os três modelos individuais e o Voting**, apresentando:

- **F1-Score: 89,3%**
- **Acurácia: 92,1%**

O melhor modelo individual foi o SVM, com **F1-Score de 87,2%**, enquanto o Voting alcançou **86,2%**.

Esses resultados dependem da amostra utilizada, do pré-processamento dos dados e dos hiperparâmetros escolhidos.

## Discussão: Voting e Stacking

O Voting pode ser uma alternativa interessante quando se busca simplicidade, transparência e menor custo computacional.

Ele também pode ser utilizado quando os modelos individuais apresentam resultados satisfatórios e possuem comportamentos complementares.

O Stacking, por outro lado, adiciona um metamodelo capaz de aprender como combinar as previsões dos modelos-base. Isso pode permitir relações mais sofisticadas, mas aumenta a complexidade do processo.

## Estimadores homogêneos e heterogêneos

Estimadores fracos **homogêneos** utilizam a mesma família de algoritmo, como várias árvores de decisão.

Já os estimadores **heterogêneos** utilizam diferentes famílias de algoritmos, como regressão logística, árvore de decisão e SVM.

A utilização de modelos heterogêneos pode ser interessante em métodos como Voting e Stacking porque diferentes algoritmos podem analisar os dados de maneiras distintas e produzir erros complementares.

## Conclusão

Os métodos Ensemble permitem combinar diferentes modelos com o objetivo de tornar a classificação mais robusta.

No Voting, a decisão final é obtida pela combinação direta dos votos dos estimadores.

No Stacking, um metamodelo aprende a combinar as previsões dos modelos-base.

Neste experimento, o Stacking apresentou o maior F1-Score entre as abordagens avaliadas, embora exija maior complexidade e validação cuidadosa.

---

# Questão 3 — AutoML com TPOT

A terceira etapa do projeto utiliza AutoML por meio da biblioteca TPOT.

O experimento utiliza o dataset `load_breast_cancer` e aplica programação genética para explorar diferentes pipelines e configurações de modelos.

## AutoML

AutoML (Automated Machine Learning) busca automatizar diferentes etapas relacionadas à criação e otimização de modelos de aprendizado de máquina.

Em vez de testar manualmente todas as combinações possíveis, ferramentas de AutoML podem explorar automaticamente diferentes modelos, transformações e hiperparâmetros.

## Programação genética

A programação genética utiliza conceitos inspirados no processo de evolução.

Cada solução candidata pode ser considerada um indivíduo dentro de uma população de pipelines.

As soluções são avaliadas e as mais promissoras são utilizadas para gerar novas combinações por meio de operações como mutação e recombinação.

Esse processo ocorre durante várias gerações.

## Resultados

O TPOT selecionou um pipeline composto por **PolynomialFeatures** seguido de **ExtraTreesClassifier**.

O pipeline utilizou:

- critério de entropia;
- `max_features = 0.8`;
- `min_samples_leaf = 19`;
- `min_samples_split = 5`;
- 100 árvores.

O TPOT obteve **93,9% de acurácia**, enquanto o modelo utilizado como baseline obteve **98,2%**.

Portanto, nas condições utilizadas neste experimento, o TPOT não superou o baseline.

Isso não significa que o AutoML tenha falhado. O resultado indica que, dentro das configurações e do espaço de busca utilizados, não foi encontrada uma solução superior à referência adotada.

## Vantagens e limitações do AutoML

O AutoML pode acelerar a criação de protótipos, aumentar a produtividade e auxiliar na descoberta de combinações de modelos e configurações.

No ambiente acadêmico, também pode facilitar comparações sistemáticas entre diferentes soluções.

Entre suas limitações estão:

- custo computacional;
- maior dificuldade de interpretação de determinadas escolhas;
- possibilidade de sobreajuste durante o processo de busca;
- dependência da qualidade dos dados;
- necessidade de validação adequada dos resultados.

O AutoML não substitui a definição correta do problema, a análise dos dados ou a avaliação humana dos resultados.

## Programação genética e otimização de hiperparâmetros

Na busca por hiperparâmetros, a programação genética pode representar cada pipeline como uma combinação de diferentes etapas e configurações.

Essas combinações podem incluir normalização, seleção de variáveis, escolha do modelo e seus respectivos parâmetros.

Durante a evolução, essas estruturas são modificadas e avaliadas de acordo com uma determinada métrica.

Dessa forma, a programação genética permite otimizar tanto a estrutura do pipeline quanto seus hiperparâmetros.

## Conclusão

O experimento com TPOT demonstrou como uma ferramenta de AutoML pode automatizar a busca por pipelines e hiperparâmetros.

Nas configurações utilizadas, o pipeline selecionado pelo TPOT alcançou **93,9% de acurácia**, enquanto o baseline obteve **98,2%**.

O resultado reforça que AutoML deve ser tratado como uma ferramenta de apoio à experimentação e otimização, e não como garantia de desempenho superior a modelos definidos manualmente.