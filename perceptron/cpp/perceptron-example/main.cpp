#include <iostream>
#include <cstdlib>
#include <ctime>

#include "perceptron/percetron.h"

std::string ler( const std::string& msg ) {

    std::cout << msg;
    std::string dado;
    std::cin >> dado;

    return dado;

}

bool ler_confirmacao( const std::string& msg ) {

    std::string msgConfirmacao = msg + " [S/N]: ";
    std::string resposta = ler( msgConfirmacao );
    return resposta.compare( "N" ) == 0;

}

int main( int /*argc*/, char* /*argv*/[] ) {

    // std::string nomeArquivoTreinamento = ler( "Informar nome do arquivo de dataset: " );
    std::string nomeArquivoTreinamento = "./../resources/data.txt";

    srand( time( NULL ) );

    Perceptron& perceptron = Perceptron::getPerceptron( nomeArquivoTreinamento );

    if ( ler_confirmacao( "Pular validacao cruzada 10-fold" ) ) {

        if ( perceptron.get_tamanho_dataset_treinamento() >= 10 ) {

            double wallInicial = perceptron.get_tempo_wall();
            double cpuInicial = perceptron.get_tempo_gpu();

            perceptron.performar_validacao_10_fold_x();

            double wallFinal = perceptron.get_tempo_wall();
            double cpuFinal = perceptron.get_tempo_gpu();

            std::cout << "\nTempo(s) : 10-fold Cross-Validation : "
                      << "Tempo Wall: " << wallFinal - wallInicial
                      << ", Tempo CPU: " << cpuFinal - cpuInicial << "\n\n";
        }

    } else {

        std::cout << "Nao e possivel performar validacao cruzadas 10-fold, numero de instancias de treinamento e menor que 10..." << std::endl;

    }


    if ( ler_confirmacao( "Voce deseja pular o teste manual" ) ) {

        std::cout << "Treinando o Perceptron usando todas as instancias..." << std::endl;

        perceptron.popular_dataset_treinamento();
        perceptron.treinar_perceptron();

        while ( true ) {

            double entrada1 = 0.0, entrada2 = 0.0;

            std::cout << "Inserir entrada de teste 1: ";
            std::cin >> entrada1;

            std::cout << "Inserir entrada de teste 2: ";
            std::cin >> entrada2;

            std::tuple<double, double, int> entrada = std::make_tuple( entrada1, entrada2, 9999 );

            perceptron.is_predicao_correta( entrada, true );

            if ( ler_confirmacao( "Voce deseja performar mais teste manual" ) ) {
                break;
            }

        }

    }

    return 0;
}
