# Questão 1 — Classificação de vinhos com LDA e PCA

Projeto acadêmico de visualização e apresentação executiva. O objetivo é comparar PCA e LDA como técnicas de redução de dimensionalidade antes de um classificador KNN (`k=5`) para os três tipos de vinho do dataset `load_wine`.

## Parte A — preparação do ambiente

### Ferramentas e bibliotecas

No terminal, dentro da pasta do projeto, execute:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Em linguagem simples:

- **Python**: linguagem usada para executar a análise.
- **NumPy**: operações numéricas e matrizes.
- **Pandas**: organização e leitura de dados tabulares.
- **Scikit-learn**: dataset, padronização, PCA, LDA, KNN e métricas.
- **Matplotlib**: criação e salvamento dos gráficos.
- **Seaborn**: acabamento visual e cores dos gráficos.

### Estrutura recomendada

```text
Entrega-Data-Sciense/
├── README.md
├── requirements.txt
├── src/
│   └── questao_1_vinhos.py
├── assets/
│   ├── q1_pca_scatter.png
│   ├── q1_lda_scatter.png
│   └── q1_accuracy_comparison.png
└── .gitignore
```

A pasta `assets/` concentra as imagens prontas para inserir nos slides.

## Execução

```bash
python src/questao_1_vinhos.py
```

O script cria ou atualiza automaticamente os três arquivos PNG em `assets/` e imprime as acurácias no terminal.

## Parte C — roteiro executivo para os slides

### 1. Intuição das técnicas

**Padronização:** cada característica química é colocada em uma escala comparável. É como converter todas as medidas para uma régua comum: uma variável com números muito grandes não domina artificialmente outra com números menores.

**PCA:** procura as direções que preservam a maior quantidade possível da variação total dos dados. A analogia é fotografar uma nuvem de pontos de um ângulo que mostre melhor seu formato. O PCA não recebe as respostas das classes; ele observa apenas a estrutura dos dados.

**LDA:** procura uma projeção que deixe as classes conhecidas o mais separadas possível e, ao mesmo tempo, mantenha os pontos da mesma classe próximos. É como ajustar a iluminação e o ângulo de uma fotografia sabendo quais pessoas pertencem a cada grupo, para que os grupos fiquem mais fáceis de distinguir.

### 2. Qual método proporcionou melhor acurácia?

Após a execução do experimento, o LDA obteve **94,4%** de acurácia, enquanto o PCA obteve **91,7%**. Portanto, neste cenário, o melhor método foi o **LDA**.

Essa forma de reportar evita apresentar um número sem executar exatamente o experimento definido no trabalho.

### 3. Por que o LDA tende a ser mais adequado? Quando o PCA pode superar?

O LDA tende a ser mais adequado para classificação supervisionada porque usa os rótulos durante a redução: ele busca diretamente aumentar a distância entre classes e reduzir a dispersão dentro de cada classe. Assim, a representação já nasce orientada para separar os grupos.

O PCA pode superar o LDA quando a maior parte da informação útil para classificar está em direções de alta variância, quando os rótulos são pouco confiáveis, quando há muito ruído ou quando se deseja uma representação exploratória independente das classes. Também pode ser mais conveniente quando existem muitas classes, pois o LDA possui limite de no máximo `n_classes - 1` componentes discriminantes.

### Mensagem final sugerida

> “Depois de colocar todas as variáveis na mesma escala, comparamos duas formas de resumir os dados. O PCA preserva a variação geral, enquanto o LDA usa os grupos conhecidos para maximizar a separação entre classes. No experimento, o método com maior acurácia no teste foi o indicado no gráfico comparativo; isso mostra qual representação foi mais útil para o KNN neste conjunto de dados, sem afirmar que será sempre superior em outros cenários.”

## Questão 2 — Stacking e Voting para classificação de crédito

### Execução

```bash
python src/questao_2_credito.py
```

O script gera um dataset sintético reproduzível com variáveis numéricas e categóricas, aplica normalização e one-hot encoding, treina os três modelos individuais, Voting por maioria e Stacking com regressão logística como metamodelo. Os resultados são salvos em `assets/q2_metrics_comparison.png` e `assets/q2_performance_table.png`.

### Roteiro executivo

**Voting Classifier:** imagine três analistas de crédito independentes. Cada um avalia o cliente e vota “bom pagador” ou “mau pagador”. O banco escolhe a decisão que recebeu mais votos. No *Hard Voting*, todos os modelos têm o mesmo peso na decisão final.

**Stacking Classifier:** agora, além dos três analistas, existe um gerente. O gerente observa as recomendações dos analistas em vários casos e aprende em quais situações deve confiar mais em cada um. Ele então combina essas recomendações para produzir a decisão final.

**Pergunta 1 — O Stacking superou os modelos individuais e o Voting?**

Consulte a tabela `q2_performance_table.png`, especialmente a coluna F1-Score, que equilibra Precision e Recall. Neste experimento, o **Stacking superou os três modelos individuais e o Voting**, com F1-Score de **89,3%** e acurácia de **92,1%**. O melhor modelo individual foi o SVM, com F1-Score de 87,2%; o Voting alcançou 86,2%. O resultado não deve ser generalizado para qualquer carteira de crédito: ele depende da amostra, do pré-processamento e dos hiperparâmetros escolhidos.

**Pergunta 2 — Quando o Voting é preferível ao Stacking?**

O Voting é preferível quando se busca simplicidade, transparência, menor custo computacional e menor risco de sobreajuste. Ele é especialmente útil quando os modelos já são razoavelmente bons e suas decisões são complementares, mas não há dados suficientes para treinar com segurança um metamodelo. Também é mais fácil de explicar para áreas de negócio: a decisão veio da maioria.

**Pergunta 3 — Estimadores fracos homogêneos e heterogêneos**

Estimadores fracos **homogêneos** usam a mesma família de algoritmo, como várias árvores rasas. Eles podem reduzir erros por meio da diversidade de amostras ou configurações, mas tendem a cometer erros parecidos. Estimadores **heterogêneos** usam famílias diferentes, como regressão logística, árvore e SVM. Como enxergam os dados de formas distintas, podem produzir erros complementares; por isso, são particularmente interessantes para Voting e Stacking.

### Mensagem final sugerida

> “O ensemble combina perspectivas diferentes para tornar a decisão de crédito mais robusta. No Voting, a decisão é democrática e simples de auditar. No Stacking, um metamodelo aprende como combinar as recomendações, podendo capturar relações mais sofisticadas, ao custo de maior complexidade e necessidade de validação cuidadosa.”

## Questão 3 — AutoML com TPOT

### Execução

```bash
python src/questao_3_tpot.py
```

O script usa o dataset `load_breast_cancer`, separa os dados em treino e teste, executa o TPOT com 5 gerações, população de 20 indivíduos e validação cruzada de 5 partes, exporta o pipeline vencedor para `outputs/q3_tpot_best_pipeline.py` e salva os dois visuais em `assets/`.

### Roteiro executivo

**AutoML:** é como contratar um assistente especializado para testar muitas combinações de preparação de dados, modelos e configurações. Em vez de escolher manualmente cada alternativa, o AutoML explora o espaço de possibilidades e apresenta uma solução competitiva.

**Programação Genética:** a analogia é a evolução das espécies. Cada solução candidata é um “indivíduo”; uma população inicial contém diferentes pipelines. Os melhores indivíduos, medidos pela validação cruzada, são selecionados para gerar novas combinações. Depois de várias gerações, sobrevivem as configurações com melhor desempenho.

**Pergunta 1 — Qual pipeline o TPOT selecionou? Ele superou o Baseline?**

O TPOT selecionou **PolynomialFeatures** seguido de **ExtraTreesClassifier**, com critério de entropia, `max_features=0.8`, `min_samples_leaf=19`, `min_samples_split=5` e 100 árvores. O TPOT obteve **93,9%** de acurácia, enquanto o Baseline obteve **98,2%**. Portanto, **o TPOT não superou o Baseline neste experimento**. Isso não significa que o AutoML falhou: o resultado mostra que a busca, sob essas configurações e nesse split, não encontrou uma solução melhor que a referência simples.

**Pergunta 2 — Vantagens e desvantagens do AutoML**

No mercado, o AutoML acelera protótipos, amplia a produtividade de equipes e ajuda a encontrar combinações que poderiam passar despercebidas. Na academia, facilita comparações sistemáticas e torna a busca mais reprodutível quando sementes e configurações são documentadas.

As desvantagens são custo computacional, maior dificuldade de explicar algumas escolhas, risco de sobreajuste ao processo de busca e dependência da qualidade dos dados. O AutoML não substitui a definição correta do problema, a análise de vieses, a validação externa nem a responsabilidade humana sobre a decisão.

**Pergunta 3 — Relação com hiperparâmetros e otimização**

Na busca de hiperparâmetros, a programação genética pode representar cada pipeline como uma estrutura com blocos e configurações: normalização, seleção de variáveis, modelo e parâmetros. A evolução altera esses blocos por mutação e recombinação, avalia cada candidato por uma métrica e mantém os mais promissores. Assim, ela otimiza simultaneamente a arquitetura do pipeline e seus hiperparâmetros, em vez de ajustar apenas um modelo isolado.

## Publicação no GitHub Pages

Site interativo: **https://SEU-USUARIO.github.io/SEU-REPOSITORIO/**

Substitua o endereço acima pelo link real após ativar o GitHub Pages. O `index.html` usa gráficos SVG inline e não depende dos arquivos PNG para funcionar no navegador.

### Clonar e executar

```bash
git clone https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git
cd SEU-REPOSITORIO
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

O comando executa as três questões, atualiza os gráficos em `assets/` e exporta o pipeline TPOT em `outputs/`.
