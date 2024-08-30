from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Dados
vendas_marmitas = {
  "clients": [
    {"id": 1, "name": "Carlos Silva", "weekly_marmitas": 5},
    {"id": 2, "name": "Maria Oliveira", "weekly_marmitas": 7},
    {"id": 3, "name": "João Santos", "weekly_marmitas": 4},
    {"id": 4, "name": "Ana Costa", "weekly_marmitas": 6},
    {"id": 5, "name": "Fernanda Lima", "weekly_marmitas": 3},
    {"id": 6, "name": "Roberto Souza", "weekly_marmitas": 8},
    {"id": 7, "name": "Paula Ferreira", "weekly_marmitas": 7},
    {"id": 8, "name": "Bruno Almeida", "weekly_marmitas": 5},
    {"id": 9, "name": "Camila Rocha", "weekly_marmitas": 4},
    {"id": 10, "name": "Lucas Mendes", "weekly_marmitas": 6}
  ]
}

# Extração dos dados
def extrair_marmitas():
    return [cliente["weekly_marmitas"] for cliente in vendas_marmitas["clients"]]

def extrair_nomes():
    return [cliente["name"] for cliente in vendas_marmitas["clients"]]

clientes_marmitas = np.array(extrair_marmitas()).reshape(-1, 1)
clientes_nomes = extrair_nomes()

# Separar treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(clientes_marmitas, clientes_nomes, test_size=0.3, random_state=42)

# Treinar o modelo
tree = DecisionTreeClassifier()
tree.fit(X_treino, y_treino)

# Testar o modelo
predicoes = tree.predict(X_teste)

# Exibir as predições e os valores reais
print(clientes_nomes)
print(clientes_marmitas)
print("Predições:", predicoes)
print("Real:", y_teste)
