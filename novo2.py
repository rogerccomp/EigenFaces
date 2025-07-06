import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# novo2 e teste.py são os arquivos principais
# Função para ler e redimensionar a imagem para um tamanho específico
def ler_e_redimensionar_imagem(caminho_imagem, tamanho=(200, 200)):
    imagem = Image.open(caminho_imagem).convert('L')  # 'L' para escala de cinza
    imagem_redimensionada = imagem.resize(tamanho)  # Redimensiona a imagem
    return np.array(imagem_redimensionada)

# Caminho das imagens
imagem1_path = 'im11.png'
imagem2_path = 'im12.png'
imagem3_path = 'im13.png'
imagem4_path = 'im21.png'

# Novo tamanho para as imagens (exemplo 200x200)
novo_tamanho = (200, 200)

# Lendo as imagens e redimensionando
imagem1 = ler_e_redimensionar_imagem(imagem1_path, novo_tamanho)
imagem2 = ler_e_redimensionar_imagem(imagem2_path, novo_tamanho)
imagem3 = ler_e_redimensionar_imagem(imagem3_path, novo_tamanho)
imagem4 = ler_e_redimensionar_imagem(imagem4_path, novo_tamanho)  # Lendo a quarta imagem

# Se necessário, podemos cortar uma região central das imagens para centralizar rostos
def cortar_central(imagem, tamanho_corte=(150, 150)):
    altura, largura = imagem.shape
    corte_altura = (altura - tamanho_corte[0]) // 2
    corte_largura = (largura - tamanho_corte[1]) // 2
    return imagem[corte_altura:corte_altura+tamanho_corte[0], corte_largura:corte_largura+tamanho_corte[1]]

# Cortando as imagens para centralizar (opcional)
imagem1 = cortar_central(imagem1, (150, 150))
imagem2 = cortar_central(imagem2, (150, 150))
imagem3 = cortar_central(imagem3, (150, 150))
imagem4 = cortar_central(imagem4, (150, 150))  # Cortando a quarta imagem

# Esticando as imagens em vetores coluna
imagem1_vetor = imagem1.flatten()
imagem2_vetor = imagem2.flatten()
imagem3_vetor = imagem3.flatten()
imagem4_vetor = imagem4.flatten()  # Esticando a quarta imagem

# Montando a matriz A (onde cada coluna é uma imagem esticada)
A = np.column_stack((imagem1_vetor, imagem2_vetor, imagem3_vetor, imagem4_vetor))  # Matriz com 4 colunas

# Calculando a imagem média
imagem_media = np.mean(A, axis=1)

# Obtendo a matriz A' (matriz A menos a imagem média)
A_ = A - imagem_media[:, np.newaxis]

# Decomposição SVD
U, S, Vt = np.linalg.svd(A_, full_matrices=False)

# Exibindo os resultados da decomposição SVD
print("\nMatriz U (autovetores das colunas de A'):")  
print(U)

print("\nValores singulares (S):")
print(S)

print("\nMatriz V^T (autovetores das linhas de A'):")
print(Vt)

# Número de componentes principais (k) que queremos usar
k = 3  # Agora pedindo 3 eigenfaces (ou menos, dependendo do número de componentes disponíveis)

# Visualizando as k primeiras eigenfaces (as primeiras colunas de U)
fig, axs = plt.subplots(1, k, figsize=(16, 4))

for i in range(k):
    eigenface = U[:, i].reshape(imagem1.shape)  # Reshape para o formato de imagem
    axs[i].imshow(eigenface, cmap='gray')
    axs[i].set_title(f'Eigenface {i+1}')
    axs[i].axis('off')

plt.tight_layout()
plt.show()

# Visualizando as 4 imagens e a imagem média
fig, axs = plt.subplots(1, 5, figsize=(20, 4))

# Exibindo as imagens
axs[0].imshow(imagem1, cmap='gray')
axs[0].set_title('Imagem 1')
axs[0].axis('off')

axs[1].imshow(imagem2, cmap='gray')
axs[1].set_title('Imagem 2')
axs[1].axis('off')

axs[2].imshow(imagem3, cmap='gray')
axs[2].set_title('Imagem 3')
axs[2].axis('off')

axs[3].imshow(imagem4, cmap='gray')
axs[3].set_title('Imagem 4')
axs[3].axis('off')

# Exibindo a face média
axs[4].imshow(imagem_media.reshape((150, 150)), cmap='gray')
axs[4].set_title('Face Média')
axs[4].axis('off')

plt.tight_layout()
plt.show()

# Projetando as imagens nas k primeiras eigenfaces
projecao_imagem1 = np.dot(U[:, :k].T, imagem1_vetor - imagem_media)
projecao_imagem2 = np.dot(U[:, :k].T, imagem2_vetor - imagem_media)
projecao_imagem3 = np.dot(U[:, :k].T, imagem3_vetor - imagem_media)
projecao_imagem4 = np.dot(U[:, :k].T, imagem4_vetor - imagem_media)

# Exibindo os resultados da projeção
print("\nProjeção da Imagem 1 nas primeiras k eigenfaces:")
print(projecao_imagem1)

print("\nProjeção da Imagem 2 nas primeiras k eigenfaces:")
print(projecao_imagem2)

print("\nProjeção da Imagem 3 nas primeiras k eigenfaces:")
print(projecao_imagem3)

print("\nProjeção da Imagem 4 nas primeiras k eigenfaces:")
print(projecao_imagem4)
