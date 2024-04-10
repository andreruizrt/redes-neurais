#include "percetron.h"

#include <sys/time.h>

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cassert>

namespace {
constexpr uint LIMITE_INTERACOES = 100000;
constexpr uint EPOCAS = 1000;
constexpr double TAXA_APREDIZAGEM = 0.001;
constexpr double TOLERANCIA_ERRO = 0.0001;
}

Perceptron::Perceptron( const std::string& nomeArquivoTreinamento ) :
    m_allTds( {} ),
    m_tds( {} ),
    m_testSet( {} ),
    m_pesos( {} ) {

    if ( m_allTds.size() == 0 && m_pesos.size() == 0 ) {
        popular_dataset_treinamento( nomeArquivoTreinamento );
        inicializar_pesos();
    }

}

void Perceptron::operator=( const Perceptron& other ) {
    m_allTds = other.m_allTds;
    m_tds = other.m_tds;
    m_pesos = other.m_pesos;
    m_testSet = other.m_testSet;
}

Perceptron::Perceptron( const Perceptron& ) :
    m_allTds( {} ),
    m_tds( {} ),
    m_testSet( {} ),
    m_pesos( {} ) {}


void Perceptron::popular_dataset_treinamento( const std::string& nomeArquivoTreinamento ) {

    std::ifstream arquivo;
    arquivo.open( nomeArquivoTreinamento.c_str() );

    if ( arquivo.is_open() ) {

        while ( !arquivo.eof() ) {

            char buffer[100];
            arquivo.getline( buffer, 100 );

            const char* linha[3] = {};

            // Pegando valor do X1, primeira entrada
            linha[0] = strtok( buffer, " " );

            if ( !linha[0] ) {
                std::cout << "A formatacao dos dados do arquivo esta incorreta. Primeiro parametro de entrada com valor inconsistente" << std::endl;
                continue;
            }

            double entrada1 = atof( linha[0] );

            // Pegando valor do X2, segunda entrada
            linha[1] = strtok( buffer, " " );

            if ( !linha[1] ) {
                std::cout << "A formatacao dos dados do arquivo esta incorreta. Segundo parametro de entrada com valor inconsistente" << std::endl;
                exit( 0 );
            }

            double entrada2 = atof( linha[1] );

            // Pegando valor do target, terceira entrada
            linha[2] = strtok( buffer, " " );

            if ( !linha[2] ) {
                std::cout << "A formatacao dos dados do arquivo esta incorreta. Terceiro parametro de entrada com valor inconsistente" << std::endl;
                exit( 0 );
            }

            int saida = atoi( linha[2] );

            m_allTds.push_back( std::make_tuple( entrada1, entrada2, saida ) );
        }

        arquivo.close();
        return;
    }

    std::cout << "Nao foi possivel ler arquivo. finalizando..." << std::endl;
    exit( 0 );

}

bool Perceptron::is_predicao_correta( const std::tuple<double, double, int>& entrada, bool exibir_saida_predicao ) {

    double entradaParaFuncaoAtivacao = m_pesos[0] * std::get<0>( entrada )
                                       + m_pesos[1] * std::get<1>( entrada )
                                       + m_pesos[2];

    double saidaPredicao = funcao_ativacao( entradaParaFuncaoAtivacao );
    saidaPredicao = saidaPredicao <= 0.5 ? 0 : 1;

    if ( exibir_saida_predicao ) {
        std::cout << "Saida predicao: " << saidaPredicao << std::endl;
    }

    return saidaPredicao == std::get<2>( entrada );

}

int Perceptron::get_tamanho_dataset_treinamento() {

    return m_allTds.size();

}

void Perceptron::popular_dataset_treinamento() {

    assert( m_allTds.size() > 0 );
    m_tds = m_allTds;

}

void Perceptron::treinar_perceptron() {

    double mediaTotalErro = 0.0;
    double erro = 1000.0;
    uint numInteracoes = 1;

    std::vector<double> saidasPredicao = {};

    while ( erro >= TOLERANCIA_ERRO && numInteracoes < LIMITE_INTERACOES ) {

        saidasPredicao.clear();

        for ( size_t pos = 0; pos < m_tds.size(); pos++ ) {

            double entradaParaFuncaoAtivacao = m_pesos[0] * std::get<0>( m_tds[pos] )
                                               + m_pesos[1] * std::get<1>( m_tds[pos] )
                                               + m_pesos[2];
            double saidaPredicao = funcao_ativacao( entradaParaFuncaoAtivacao );

            saidasPredicao.push_back( saidaPredicao );

            double saidaAtual = std::get<2>( m_tds[pos] );

            atualizar_pesos( 0, std::get<0>( m_tds[pos] ), saidaAtual, saidaPredicao );
            atualizar_pesos( 1, std::get<1>( m_tds[pos] ), saidaAtual, saidaPredicao );
            atualizar_pesos( 2, 1, saidaAtual, saidaPredicao );

        }

        erro = calcular_erro_media_quadratica( saidasPredicao );
        mediaTotalErro += erro;

        numInteracoes++;

    }

    mediaTotalErro /= numInteracoes;
    std::cout << "[Treinamento] Media total de erro: " << mediaTotalErro << std::endl;

}


void Perceptron::inicializar_pesos() {

    for ( int pos = 0; pos < 3; pos++ ) {

        int val = rand() % 5000 - 2000;
        m_pesos.push_back( (double)val / 10000 );

    }

}

void Perceptron::atualizar_pesos( const int peso_index, const double entrada, const double saida_atual, const double saida_predicao ) {

    double deltaPeso = -1 * TAXA_APREDIZAGEM
                       * ( saida_predicao - saida_atual ) // Diferenca entre o esperado e o resultado atual
                       * saida_predicao * ( 1 - saida_predicao ) // Derivada da função de ativação sigmoid
                       * entrada;

    m_pesos[peso_index] += deltaPeso;

}

double Perceptron::funcao_ativacao( const double& entrada ) {
    const double saida = 1 / ( 1 + exp( -1 * entrada ) );
    return saida;
}

double Perceptron::calcular_erro_media_quadratica( const std::vector<double>& saida_predicao ) {

    assert( saida_predicao.size() == m_tds.size() );

    double erroMediaQuadratica = 0.0;

    for ( size_t pos = 0; pos < m_tds.size(); pos++ ) {

        const double saidaPredicaoModificada = saida_predicao[pos] <= 0.5 ? 0 : 1;
        // double saidaPredicaoModificada = saida_predicao[pos];

        double error = saidaPredicaoModificada - std::get<2>( m_tds[pos] );
        double sqError = error * error;
        erroMediaQuadratica += sqError;
    }

    erroMediaQuadratica /= m_tds.size();

    return erroMediaQuadratica;

}

void Perceptron::performar_validacao_10_fold_x() {

    std::cout << "Performando 10-fold validacao cruzada..." << std::endl;

    constexpr int NUM_DIVISAO = 10;

    std::cout << "Seguintes resultados para " << NUM_DIVISAO << " divisoes..." << std::endl;
    std::cout << "Teste rodando  |  Acuracia na amostra de teste (%)" << std::endl;

    double acuraciaMedia = 0.0;

    int contador = 1;

    while ( contador <= NUM_DIVISAO ) {

        resetar_dados();
        dividir_cruzamento_dataset( contador );
        inicializar_pesos();
        treinar_perceptron();

        double acuracia = reportar_acuracia( contador );

        acuraciaMedia += acuracia;

        std::string formatacao = ( contador == 10 ) ? "         |   " : "          |   ";

        std::cout << contador << formatacao << acuracia << std::endl;

        contador++;
    }

    acuraciaMedia /= NUM_DIVISAO;

    std::cout << "\nAcuracia media(%): " << acuraciaMedia << std::endl;

}

void Perceptron::dividir_cruzamento_dataset( const int contador ) {

    assert( m_tds.size() == 0 && m_testSet.size() ==0 );

    int tamanhoTotalDataSet = m_allTds.size();
    int posInicial = ( contador - 1 ) * ( tamanhoTotalDataSet / 10 );
    int posFinal = ( contador == 10 ) ? ( posInicial + floor( float(tamanhoTotalDataSet) / 10 ) - 1 + ( tamanhoTotalDataSet % 10 ) )
        : ( posInicial + floor( float(tamanhoTotalDataSet) / 10 ) - 1 );


    for ( int pos = posInicial; pos <= posFinal; pos++ ) {
        m_testSet.push_back( m_allTds[pos] );
    }

    m_tds = m_allTds;
    for ( int pos = posInicial; pos <= posFinal; pos++ ) {
        m_tds.erase( m_allTds.begin() + pos );
    }

}

void Perceptron::resetar_dados() {

    m_tds.clear();
    m_testSet.clear();
    m_pesos.clear();

}

double Perceptron::reportar_acuracia( const int contador ) {

    assert( m_testSet.size() > 0 );

    int numPredicoesCorretas = 0;

    for ( size_t i = 0; i < m_testSet.size(); i++ ) {

        if ( !is_predicao_correta( m_testSet[i] ) ) {
            continue;
        }

        numPredicoesCorretas++;

    }

    return (double)numPredicoesCorretas * 100 / m_testSet.size();

}

double Perceptron::get_tempo_wall() {

    struct timeval tempo;

    if ( gettimeofday( &tempo, nullptr ) ) {
        return 0;
    }

    return (double)tempo.tv_sec + (double)tempo.tv_usec * 0.000001;

}

double Perceptron::get_tempo_gpu() {

    return (double)clock() / CLOCKS_PER_SEC;

}






























