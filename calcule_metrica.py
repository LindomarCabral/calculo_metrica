# Importando as bibliotecas necessárias
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1. Gerar dados fictícios para o exemplo
# Criaremos um conjunto de dados simples com 100 amostras
np.random.seed(42)
X = np.random.rand(100, 2)
# Criando rótulos binários (0 ou 1)
y = (X[:, 0] + X[:, 1] > 1.0).astype(int)

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Treinar um modelo simples (Regressão Logística)
model = LogisticRegression()
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# 3. Gerar a Matriz de Confusão
# A matriz de confusão é crucial para as métricas subsequentes
# y_test são os valores reais, y_pred são os valores previstos
cm = confusion_matrix(y_test, y_pred)

print("Matriz de Confusão:")
print(cm)

# Visualizar a matriz de confusão (opcional, mas recomendado)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Classe 0', 'Classe 1'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Matriz de Confusão Binária')
plt.show()

# 4. Calcular e exibir as Métricas

# Extrair valores da matriz de confusão
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
TP = cm[1, 1]

# Cálculo das métricas usando as fórmulas

# Acurácia (Accuracy) = (TP + TN) / (TP + TN + FP + FN)
# Proporção de predições corretas no total de casos
acuracia_manual = (TP + TN) / (TP + TN + FP + FN)
print(f"\nAcurácia (Accuracy) - Manual: {acuracia_manual:.4f}")
acuracia_sklearn = accuracy_score(y_test, y_pred)
print(f"Acurácia (Accuracy) - Scikit-learn: {acuracia_sklearn:.4f}")

# Precisão (Precision) = TP / (TP + FP)
# Dos casos positivos previstos, quantos estavam corretos
precisao_manual = TP / (TP + FP) if (TP + FP) != 0 else 0
print(f"Precisão (Precision) - Manual: {precisao_manual:.4f}")
precisao_sklearn = precision_score(y_test, y_pred, average='binary', zero_division=0)
print(f"Precisão (Precision) - Scikit-learn: {precisao_sklearn:.4f}")

# Sensibilidade (Recall / Revocação) = TP / (TP + FN)
# Dos casos positivos reais, quantos foram encontrados
sensibilidade_manual = TP / (TP + FN) if (TP + FN) != 0 else 0
print(f"Sensibilidade (Recall) - Manual: {sensibilidade_manual:.4f}")
sensibilidade_sklearn = recall_score(y_test, y_pred, average='binary', zero_division=0)
print(f"Sensibilidade (Recall) - Scikit-learn: {sensibilidade_sklearn:.4f}")

# Especificidade (Specificity) = TN / (TN + FP)
# Dos casos negativos reais, quantos foram encontrados
especificidade_manual = TN / (TN + FP) if (TN + FP) != 0 else 0
print(f"Especificidade (Specificity) - Manual: {especificidade_manual:.4f}")
# O Scikit-learn não tem uma função direta para especificidade, mas podemos calcular usando recall para a classe 0
especificidade_sklearn_check = recall_score(y_test, y_pred, pos_label=0, average='binary', zero_division=0)
print(f"Especificidade (Specificity) - Scikit-learn (via recall da classe 0): {especificidade_sklearn_check:.4f}")
