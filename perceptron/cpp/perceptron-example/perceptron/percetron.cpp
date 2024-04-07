#include "percetron.h"

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

            const char* linha[4] = {};

            // Pegando valor do X1, primeira entrada
            linha[0] = strtok( buffer, " " );

            if ( !linha[0] ) {
                std::cout << "A formatacao dos dados do arquivo esta incorreta. Primeiro parametro de entrada com valor inconsistente" << std::endl;
                continue;
            }

            double entrada1 = atof( linha[0] );

            // Pegando valor do X2, segunda entrada
            linha[1] = strtok( buffer, " " );

            if ( linha[1] ) {
                std::cout << "A formatacao dos dados do arquivo esta incorreta. Segundo parametro de entrada com valor inconsistente" << std::endl;
                exit( 0 );
            }

            double entrada2 = atof( linha[1] );

            // Pegando valor do target, terceira entrada
            linha[2] = strtok( buffer, " " );

            if ( linha[2] ) {
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

int Perceptron::get_tamanho_dataset_treinamento() {

    return m_allTds.size();

}

void Perceptron::popular_dataset_treinamento() {

    assert( m_allTds.size() > 0 );
    m_tds = m_allTds;

}

void Perceptron::treinar_perceptron() {

    double erroMedioTotal = 0.0;
    double erro = 1000.0;
    uint numInteracoes = 1;

    std::vector<double> saidasPredicao = {};

    while ( erro >= TOLERANCIA_ERRO && numInteracoes < LIMITE_INTERACOES ) {

        saidasPredicao.clear();

        for ( size_t pos = 0; pos < m_tds.size(); pos++ ) {

            const double entradaParaFuncaoAtivacao = m_pesos[0] * std::get<0>( m_tds[pos] ) + m_pesos[1] * std::get<1>( m_tds[pos] ) + m_pesos[2];
            double saidaPredicao = funcao_ativacao( entradaParaFuncaoAtivacao );

            saidasPredicao.push_back( saidaPredicao );

            double saidaAtual = std::get<2>( m_tds[pos] );

            atualizar_pesos( 0, std::get<0>( m_tds[pos] ), saidaAtual, saidaPredicao );
            atualizar_pesos( 1, std::get<1>( m_tds[pos] ), saidaAtual, saidaPredicao );
            atualizar_pesos( 2, 1, saidaAtual, saidaPredicao );

        }

        erro = calcular_erro_media_quadratica( saidasPredicao );
        erroMedioTotal += erro;

        numInteracoes++;

    }

    erroMedioTotal /= numInteracoes;
    std::cout << "erroMedioTotal: " << erroMedioTotal << std::endl;

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
                       * saida_predicao * ( 1 - saida_predicao ) // Derivada da função de ativadação sigmoid
                       * entrada;

    m_pesos[peso_index] += deltaPeso;

}

double Perceptron::funcao_ativacao( const double& entrada ) {
    const double saida = 1 / ( 1 + exp( -1 * entrada ) );
    return saida;
}

double Perceptron::calcular_erro_media_quadratica( const std::vector<double>& saida_predicao ) {
    return 0.0;
}


































