
# Importar bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage

# Carregar o dataset iris
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
iris = load_iris()
X = iris.data
y = iris.target

# Padronizar os dados (média 0, desvio padrão 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Dados carregados e padronizados!")
y = iris.target  # Rótulos (apenas para comparação, não usaremos no clustering)

# Aplicar K-means com k=3
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
labels_kmeans = kmeans.labels_

# Método do cotovelo para escolher k
inertias = []
for k in range(1, 6):
    kmeans_test = KMeans(n_clusters=k, random_state=42)
    kmeans_test.fit(X_scaled)
    inertias.append(kmeans_test.inertia_)

# Plotar o gráfico do cotovelo, que  mostra a inércia (soma das distâncias aos centroides) para diferentes valores de k. O "cotovelo" em k=3 sugere que 3 clusters é uma boa escolha.
plt.plot(range(1, 6), inertias, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia')
plt.show()

print("Explicação: O gráfico mostra a inércia (soma das distâncias aos centroides) para diferentes valores de k. O 'cotovelo' em k=3 sugere que 3 clusters é uma boa escolha.")

# Calcular a matriz de ligação para clustering hierárquico (método Ward)
Z = linkage(X_scaled, method='ward')

# Plotar o dendrograma, que mostra como os pontos são agrupados. Cortando em uma altura específica (ex.: 10), podemos obter 3 clusters, similar ao k-means.
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title('Dendrograma - Clustering Hierárquico')
plt.xlabel('Amostras')
plt.ylabel('Distância')
plt.show()

print("Explicação: O dendrograma mostra como os pontos são agrupados. Cortando em uma altura específica (ex.: 10), podemos obter 3 clusters, similar ao k-means.")

# Aplicar PCA (Principal Component Analysis - Análise de Componentes Principais)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualizar os clusters do k-means com PCA, que reduz os dados para 2D, mantendo a maior parte da variância (geralmente 50-60% neste caso). Os clusters são visíveis em cores diferentes.
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans, cmap='viridis')
plt.title('Clusters K-means Visualizados com PCA')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()

# Variância explicada pelos componentes
print(f"Variância explicada pelos 2 componentes: {sum(pca.explained_variance_ratio_):.2%}")

print("Explicação: PCA reduz os dados para 2D, mantendo a maior parte da variância (geralmente 50-60% neste caso). Os clusters são visíveis em cores diferentes.")

# Aplicar t-SNE (t-Distributed Stochastic Neighbor Embedding)
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Visualizar os clusters do k-means com t-SNE
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_kmeans, cmap='viridis')
plt.title('Clusters K-means Visualizados com t-SNE')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.show()


print("Explicação: t-SNE foca em preservar vizinhanças locais, geralmente mostrando clusters mais separados que o PCA, mas sem garantir variância total explicada.")

"""
OBSERVAÇÕES: 
Cluster 1: Provavelmente representa flores com pétalas menores e estreitas, possivelmente correspondendo à espécie Setosa. Esse cluster geralmente se separa bem dos outros nas visualizações.
Cluster 2: Pode representar flores com pétalas de tamanho intermediário, com características entre as outras duas espécies (Versicolor e Virginica).
Cluster 3: Representa flores com pétalas maiores e largas, possivelmente correspondendo à espécie Virginica.

O método do Cotovelo: Sugere que 3 clusters é uma boa escolha para o dataset, pois o "cotovelo" no gráfico indica um ponto de inflexão na redução da inércia.
Dendrograma: Permite visualizar a hierarquia dos clusters e a distância entre eles. Cortando o dendrograma em uma altura específica, podemos obter o número desejado de clusters.
PCA e t-SNE: São técnicas de redução de dimensionalidade que ajudam a visualizar os clusters em 2D. O PCA busca maximizar a variância explicada, enquanto o t-SNE foca em preservar as relações de vizinhança entre os pontos.

PCA (Principal Component Analysis)
Visual geral: Temos três clusters visíveis, mas com sobreposição considerável entre os dois da esquerda (amarelo e verde).

Separação entre clusters: A separação entre o cluster da direita (roxo) e os da esquerda é clara, mas os clusters verde e amarelo estão muito próximos, o que indica que o PCA não conseguiu separar bem esses dois grupos.

Limitação: PCA é uma técnica linear. Ele tenta projetar os dados em direções que maximizam a variância, mas não captura relações complexas ou não-lineares entre os dados.

t-SNE (t-distributed Stochastic Neighbor Embedding)
Visual geral: Três clusters bem separados visualmente, com pouquíssima ou nenhuma sobreposição.

Destaque: O t-SNE "expande" os clusters visualmente para facilitar a leitura da separação local entre os dados. Mesmo os grupos amarelo e verde, que estavam misturados no PCA, aqui aparecem claramente divididos.

Vantagem clara: Mostra de forma muito mais precisa a estrutura local dos dados. É isso que queremos quando usamos K-means: saber se os grupos realmente estão isolados.

Veredito final: t-SNE foi melhor!
Por que?

Separação visual dos clusters foi mais clara.

Reduziu drasticamente a sobreposição entre grupos.

Preservou a estrutura local, que é super importante em clustering.
"""
