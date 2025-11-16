# Cálculo de Métricas de Classificação

Este projeto faz parte da atividade do curso **Formação Machine Learning Specialist** da **DIO (Digital Innovation One)**, focando na compreensão e cálculo de métricas de avaliação para modelos de classificação.

## Descrição do Projeto

O objetivo deste projeto é demonstrar a geração de dados fictícios para um problema de classificação binária, treinar um modelo simples de Regressão Logística e, em seguida, calcular e visualizar as principais métricas de avaliação de modelos de classificação.

Foi desenvolvida uma implementação para calcular as métricas manualmente, diretamente da matriz de confusão, e compará-las com as implementações fornecidas pela biblioteca `scikit-learn`.

## Conteúdo do Projeto

1.  **Geração de Dados Fictícios**: Criação de um conjunto de dados sintético com 100 amostras e 2 características (`X`), e rótulos binários (`y`).
2.  **Treinamento do Modelo**: Utilização de um modelo de Regressão Logística para aprender o padrão nos dados de treinamento.
3.  **Previsões**: Geração de previsões sobre o conjunto de dados de teste.
4.  **Matriz de Confusão**: Cálculo e visualização da matriz de confusão, que é a base para a maioria das métricas de classificação.
5.  **Cálculo de Métricas**: Implementação e exibição das seguintes métricas:
    *   **Acurácia (Accuracy)**
    *   **Precisão (Precision)**
    *   **Sensibilidade (Recall)**
    *   **Especificidade (Specificity)**
    
    Todas as métricas são calculadas tanto manualmente (derivadas da matriz de confusão) quanto usando as funções equivalentes do Scikit-learn para validação.

## Tecnologias Utilizadas

*   **Python**
*   **NumPy**: Para manipulação de arrays e dados numéricos.
*   **Scikit-learn**: Para modelos de Machine Learning (Regressão Logística), divisão de dados e funções de métricas de avaliação.
*   **Matplotlib**: Para visualização da matriz de confusão.

## Como Executar o Código

Para executar o código, siga os passos abaixo:

1.  Certifique-se de ter Python instalado em seu ambiente.
2.  Instale as bibliotecas necessárias:
    ```bash
    pip install numpy scikit-learn matplotlib
    ```
3.  Copie o código Python fornecido no seu ambiente (por exemplo, em um arquivo `.py` ou em um notebook Jupyter/Colab).
4.  Execute o script Python. A saída incluirá a matriz de confusão e os valores das métricas calculadas, além de um gráfico da matriz de confusão.

```python
# Exemplo de como o código se parece (trecho)
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ... (restante do código para geração de dados, treinamento e cálculo de métricas)
```

## Referências

*   **Projeto Original**: `Proje4.pdf` (documento que serviu como base para a atividade).
*   **Curso**: Formação Machine Learning Specialist - DIO (Digital Innovation One).

---