#  Projeto de Ciência de Dados

# Predição de Idade de Possums com Machine Learning

## Visão Geral

Este projeto realiza análise exploratória de dados e machine learning utilizando o dataset de possums.

O principal objetivo foi analisar características biológicas dos animais e desenvolver um modelo capaz de prever a idade de um possum com base em medidas corporais.

O projeto inclui:

* Limpeza e tratamento de dados
* Análise exploratória de dados (EDA)
* Visualização de dados
* Machine learning com Decision Tree Regressor
* Interpretação dos resultados

---

# Dataset

Dataset utilizado:

* openintro-possum

Principais variáveis analisadas:

* `totlngth` → Comprimento total do corpo
* `skullw` → Largura do crânio
* `hdlngth` → Comprimento da cabeça
* `taill` → Comprimento da cauda
* `footlgth` → Comprimento do pé
* `sex` → Sexo biológico
* `age` → Idade do possum
* `Pop` → População/região

---

# Limpeza e Tratamento dos Dados

## Valores Faltantes

O dataset apresentava valores faltantes nas colunas:

* `age`
* `footlgth`

Para evitar perda de dados importantes, os valores ausentes foram preenchidos utilizando a média da coluna.

```python
print(df.isnull().sum())

df['age'] = df['age'].fillna(df['age'].mean())
df['footlgth'] = df['footlgth'].fillna(df['footlgth'].mean())
```

## Justificativa

A imputação pela média foi escolhida porque:

* o número de valores faltantes era pequeno;
* evita remover linhas do dataset;
* preserva a quantidade de dados disponíveis.

---

# Análise Exploratória de Dados

## Média da Largura do Crânio

```python
average_skull = np.mean(df['skullw'])
```

Essa análise calcula a média da largura do crânio dos possums presentes no dataset.

---

## Comparação por Sexo

```python
sex_comparison = df.groupby('sex').mean(numeric_only=True)
```

Essa análise permite identificar possíveis diferenças biológicas entre machos e fêmeas.

Exemplos:

* diferenças no tamanho do crânio;
* diferenças no comprimento corporal;
* diferenças estruturais entre os sexos.

---

## Comparação por População

```python
pop_comparison = df.groupby('Pop').mean(numeric_only=True)
```

Essa análise compara características físicas entre diferentes populações de possums.

O objetivo é verificar se animais de regiões diferentes apresentam padrões biológicos distintos.

---

# Visualização de Dados

## Scatter Plot

O scatter plot compara o comprimento total do corpo com a largura do crânio entre diferentes populações.

```python
for pop in df['Pop'].unique():

    data = df[df['Pop'] == pop]

    plt.scatter(
        data['totlngth'],
        data['skullw'],
        label=pop
    )
```

## Interpretação

Esse gráfico permite identificar:

* possível correlação positiva entre tamanho corporal e largura do crânio;
* diferenças entre populações;
* padrões de distribuição;
* possíveis outliers.

---

## Histograma

```python
plt.hist(df['age'])
```

## Interpretação

O histograma mostra a distribuição das idades dos possums no dataset.

Essa visualização ajuda a identificar:

* concentração de idades;
* distribuição dos dados;
* possíveis desequilíbrios.

---

## Violin Plot

```python
sb.violinplot(
    x='sex',
    y='skullw',
    data=df
)
```

## Interpretação

O violin plot compara a distribuição da largura do crânio entre machos e fêmeas.

Essa visualização ajuda a analisar:

* densidade dos dados;
* dispersão;
* concentração das medidas;
* diferenças biológicas entre os sexos.

---

# Modelo de Machine Learning

## Modelo Escolhido

O modelo preditivo utilizado foi:

* Decision Tree Regressor

```python
from sklearn.tree import DecisionTreeRegressor
```

---

## Justificativa da Escolha

O Decision Tree Regressor foi escolhido porque:

* é intuitivo e fácil de interpretar;
* funciona bem com relações não lineares;
* apresenta bom desempenho em datasets pequenos;
* permite visualizar decisões do modelo.

---

# Engenharia de Features

A variável categórica `sex` foi convertida para valores numéricos.

```python
df['sex'] = df['sex'].map({
    'm': 0,
    'f': 1
})
```

Essa conversão permite que o modelo utilize a variável matematicamente.

---

# Variáveis Utilizadas

```python
X = df[['totlngth', 'skullw', 'sex', 'hdlngth', 'taill', 'footlgth']]
Y = df['age']
```

As variáveis foram escolhidas por representarem características biológicas potencialmente relacionadas à idade do animal.

---

# Treinamento do Modelo

```python
model = DecisionTreeRegressor()
model.fit(X, Y)
```

O modelo foi treinado para aprender relações entre medidas corporais e idade.

---

# Exemplo de Predição

```python
new_possum = pd.DataFrame([[
    90,
    58,
    0,
    95,
    36,
    75
]], columns=[
    'totlngth',
    'skullw',
    'sex',
    'hdlngth',
    'taill',
    'footlgth'
])

pred = model.predict(new_possum)

print(pred[0])
```

O modelo prevê a idade aproximada de um novo possum utilizando suas características físicas.

---

# Métricas de Validação

Para melhorias futuras, o projeto pode incluir métricas como:

```python
from sklearn.metrics import mean_absolute_error
```

Essa métrica mede o erro médio das previsões realizadas pelo modelo.

---

# Conclusão

O projeto demonstrou como técnicas de análise exploratória e machine learning podem ser aplicadas em datasets biológicos.

A análise exploratória revelou padrões entre medidas corporais, sexo e populações. Os gráficos ajudaram a visualizar diferenças biológicas importantes e padrões de distribuição dos dados.

O modelo Decision Tree Regressor conseguiu prever a idade dos possums com base em características físicas, mostrando como algoritmos de machine learning podem identificar relações entre variáveis biológicas.

Além disso, o projeto reforçou conceitos importantes de ciência de dados, como:

* limpeza de dados;
* pré-processamento;
* análise exploratória;
* engenharia de features;
* modelagem preditiva;
* visualização de dados.

---

# Tecnologias Utilizadas

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* KaggleHub

---
