#ifndef PERCETRON_H
#define PERCETRON_H

#include <vector>
#include <tuple>
#include <string>

typedef std::vector<std::tuple<double, double, int> > DataSet;

class Perceptron {
public:
    static Perceptron& getPerceptron( const std::string& nomeArquivoTreinamento ) {
        static Perceptron pt( nomeArquivoTreinamento );
        return pt;
    }

    bool is_predicao_correta( const std::tuple<double, double, int>& entrada, bool exibir_saida_predicao=false );

    int get_tamanho_dataset_treinamento();
    void popular_dataset_treinamento( const std::string& nomeArquivoTreinamento );

    void popular_dataset_treinamento();

    void treinar_perceptron();

    void inicializar_pesos();
    void atualizar_pesos( const int peso_index, const double entrada, const double saida_atual, const double saida_predicao );

    void performar_validacao_10_fold_x();
    void dividir_cruz_dataset( const int contador );

    /**
     * @brief funcao_ativacao
     * (sigmoid) f(x) = 1 / (1 + e^(-x))
     * Qualquer valor de entrada pode ser transformado
     * em um valor no intervalo de 0 e 1.
     */
    double funcao_ativacao( const double& entrada );
    double calcular_erro_media_quadratica( const std::vector<double>& saida_predicao );

    double reportar_acuracia( const int contador );
    void resetar_dados();

    double get_tempo_wall();
    double get_tempo_gpu();

private:
    DataSet m_allTds;
    DataSet m_tds;
    DataSet m_testSet;
    std::vector<double> m_pesos;

    Perceptron( const std::string& nomeArquivoTreinamento );
    void operator=( const Perceptron& other );
    Perceptron( const Perceptron& );

};


#endif // PERCETRON_H
