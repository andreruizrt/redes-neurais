import numpy as np


INICIO_INTERVALO_VALORES_ALEATORIOS = 0
FIM_INTERVALO_VALORES_ALEATORIOS = 1


class Perceptron:
    def __init__(self, taxa_aprendizado=0.01, epocas=5):
        # taxa de aprendizado (entre 0 e 1)
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas  # número de épocas

    def inicializar_pesos(self, n_features):
        # self.w_ = np.zeros(1 + n_features)  # inicializar com zeros
        # Inicializar os pesos com valores aleatórios entre -1 e 1
        self.w_ = np.random.uniform(
            INICIO_INTERVALO_VALORES_ALEATORIOS, FIM_INTERVALO_VALORES_ALEATORIOS, size=n_features + 1)
        # Mostrar valores aleatorios gerados
        print("Valores iniciais Pesos = " + str(self.w_))

        # função para treinar a rede
    def treinar(self, X, y):
        self.inicializar_pesos(X.shape[1])
        self.errors_ = []  # vetor de erros
        self.epoca_atual_ = 0

        # adiciona -1 para cada uma das amostras, a gente inicia com o menor
        # valor possível, para a rede ir aprendendo
        for epoca_atual in range(self.epocas):
            self.epoca_atual_ = epoca_atual
            self.amostra_atual_ = 0
            errors = 0

            for xi, target in zip(X, y):  # Mescla valor de x e o seu rótulo
                self.amostra_atual_ += 1

                self.print_epoca_message("Caracteristicas: " + str(xi) +
                                         " | " + "^y: " + str(target))

                erro = target - self.predict(xi)
                self.print_epoca_message("Erro: " + str(erro))

                atualizacao = self.taxa_aprendizado * erro
                self.print_epoca_message("Atualizacao: " + str(atualizacao))

                self.print_epoca_message(
                    "Pesos Nao Atualizados: " + str(self.w_[1:]))

                self.w_[1:] += atualizacao * xi

                self.print_epoca_message(
                    "Pesos Atualizados: " + str(self.w_[1:]))

                self.w_[0] += atualizacao

                self.print_epoca_message("Bias Atualizado: " + str(self.w_[0]))

                errors += int(atualizacao != 0.0)

            self.print_epoca_message("Quantidade de erros: " + str(errors))

            self.errors_.append(errors)
            self.amostra_atual_ = 0

        print("Valores finais Pesos: " + str(self.w_))

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def print_epoca_message(self, msg):
        print("[EPOCA #" + str(self.epoca_atual_) + "][AMOSTRA #" +
              str(self.amostra_atual_) + "] " + str(msg))
