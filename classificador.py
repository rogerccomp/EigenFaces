import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Função para ler e redimensionar a imagem para um tamanho específico
def ler_e_redimensionar_imagem(caminho_imagem, tamanho=(200, 200)):
    imagem = Image.open(caminho_imagem).convert('L')  # 'L' para escala de cinza
    imagem_redimensionada = imagem.resize(tamanho)  # Redimensiona a imagem
    return np.array(imagem_redimensionada)

# Caminho das imagens de treinamento (substitua pelos caminhos reais das imagens)
imagem1_path = 'im11.png'
imagem2_path = 'im12.png'
imagem3_path = 'im13.png'
imagem4_path = 'im22.png'

# Novo tamanho para as imagens (exemplo 200x200)
novo_tamanho = (150, 150)  # Ajustar para o tamanho das imagens de treinamento

# Lendo as imagens de treinamento e redimensionando
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

# Número de componentes principais (k) que queremos usar
k = 3  # Agora pedindo 3 eigenfaces (ou menos, dependendo do número de componentes disponíveis)

# Função para calcular a projeção de uma imagem no espaço das eigenfaces
def projetar_imagem(imagem_vetor, U, imagem_media, k):
    # Subtrai a imagem média e projeta a imagem no espaço das eigenfaces
    imagem_centrada = imagem_vetor - imagem_media  # Garantir que a imagem tenha o mesmo tamanho
    return np.dot(U[:, :k].T, imagem_centrada)  # Projeção no espaço das k primeiras eigenfaces

# Projetando as imagens de treinamento nas k primeiras eigenfaces
projecao_imagem1 = projetar_imagem(imagem1_vetor, U, imagem_media, k)
projecao_imagem2 = projetar_imagem(imagem2_vetor, U, imagem_media, k)
projecao_imagem3 = projetar_imagem(imagem3_vetor, U, imagem_media, k)
projecao_imagem4 = projetar_imagem(imagem4_vetor, U, imagem_media, k)

# Função para calcular a distância euclidiana entre a projeção de uma imagem de teste e as imagens de treinamento
def calcular_distancia(projecao_teste, projecoes_treinamento):
    # Calcular distância euclidiana entre a projeção da imagem de teste e as projeções das imagens de treinamento
    distancias = np.linalg.norm(projecao_teste[:, np.newaxis] - projecoes_treinamento, axis=0)
    return distancias

# Imagem de teste (substitua pelo caminho real da imagem de teste)
imagem_teste_path = 'im22.png'
imagem_teste = ler_e_redimensionar_imagem(imagem_teste_path, novo_tamanho)  # Certifique-se de redimensionar para o mesmo tamanho
imagem_teste_vetor = imagem_teste.flatten()

# Projeção da imagem de teste
projecao_teste = projetar_imagem(imagem_teste_vetor, U, imagem_media, k)

# Calculando a distância euclidiana entre a projeção da imagem de teste e as projeções das imagens de treinamento
distancias = calcular_distancia(projecao_teste, np.column_stack([projecao_imagem1, projecao_imagem2, projecao_imagem3, projecao_imagem4]))

# Definindo o threshold
threshold = 10  # Defina um valor adequado para o seu conjunto de dados

# Verificando se a distância é menor que o threshold (rosto identificado)
if np.min(distancias) < threshold:
    print("\033[1;32mRosto identificado!\033[m")
else:
    print("\033[1;31mRosto não identificado!\033[m")
