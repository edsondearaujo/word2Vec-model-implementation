# Word2Vec Model Implementation
Autor: [Edson Soares](https://github.com/edsondearaujo)
<br><br>
Este projeto implementa o modelo **Word2Vec**, amplamente utilizado em **Processamento de Linguagem Natural (PLN)** para criar vetores de palavras (word embeddings) que capturam relações semânticas entre palavras. A implementação inclui tanto o modelo **Skip-gram** quanto funções de custo usando **softmax ingênuo** e **amostragem negativa** (negative sampling).

## Funcionalidades Principais

1. **Funções de Ativação**:
   - `sigmoid(x)`: Função sigmoide.
   - `softmax(x)`: Função softmax otimizada para uso frequente no treinamento do modelo.

2. **Modelos e Funções de Custo**:
   - **naiveSoftmaxLossAndGradient()**: Implementa o custo e os gradientes da função softmax ingênua para o modelo word2vec.
   - **negSamplingLossAndGradient()**: Implementa o custo e gradientes utilizando amostragem negativa, o que acelera o treinamento.

3. **Modelo Skip-gram**:
   - `skipgram()`: Implementa o modelo skip-gram para treinar vetores de palavras a partir de um corpus de texto, utilizando as funções de custo descritas acima.

4. **Verificação de Gradientes**:
   - `gradcheck_naive()`: Função que realiza uma verificação numérica para garantir a consistência entre a função de custo e os gradientes.
   - `grad_tests_softmax()` e `grad_tests_negsamp()`: Testes unitários para as funções softmax e amostragem negativa.

5. **Normalização**:
   - `normalizeRows(x)`: Função que normaliza as linhas de uma matriz para garantir que cada vetor tenha comprimento unitário.

## Testes Implementados

O projeto inclui vários testes para verificar a implementação do word2vec:
- **Teste de Gradientes**: Assegura que os gradientes calculados estão corretos para as funções softmax e amostragem negativa.
- **Testes de Perda**: Testa a função de custo do modelo para garantir que os valores de perda retornados estejam de acordo com as expectativas.

## Como Executar

Para rodar os testes e verificar a implementação, basta executar o arquivo `word2vec.py`:

```bash
python word2vec.py
