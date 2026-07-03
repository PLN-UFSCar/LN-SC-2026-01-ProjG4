# Detecção de Fake News em Português Brasileiro

Este repositório contém um pipeline completo de Processamento de Linguagem Natural (PLN) e Aprendizado de Máquina para a classificação de notícias em **Verdadeiras** ou **Falsas (Fake News)**. 

O projeto é estruturado de forma progressiva em dois notebooks Jupyter: a extração e o pré-processamento pesado dos dados linguísticos (`sanity.ipynb`) e, em seguida, a modelagem, avaliação e auditoria de múltiplos algoritmos (`Fake_News.ipynb`).

---

## 📊 Onde Encontrar os Dados?

O projeto utiliza o **Fake.Br Corpus**, que é a base de dados de referência mais conhecida e utilizada para o estudo de desinformação no cenário do português brasileiro.

1. **Link para Download:** Você deve descarregar o dataset diretamente do repositório oficial da Nilc (Núcleo Interinstitucional de Linguística Computacional):
   👉 **[(https://github.com/roneisilva/Fake.br-Corpus)](https://github.com/roneysco/Fake.br-Corpus)**
2. **Estrutura de Pastas Necessária:** Após clonar ou descarregar o arquivo ZIP do corpus, certifique-se de localizar a pasta `full_texts`. O notebook `sanity.ipynb` espera encontrar e ler as duas subpastas originais neste exato formato:
   * `full_texts/fake` (Contendo as 3.600 notícias falsas em formato `.txt`)
   * `full_texts/true` (Contendo as 3.600 notícias verdadeiras em formato `.txt`)

---

## 📂 Organização e Ordem de Execução

Para reproduzir os resultados perfeitamente, os notebooks devem ser executados na ordem cronológica abaixo:

### 1. `sanity.ipynb` (Saneamento e Engenharia de Atributos)
Este notebook realiza a varredura e a consolidação dos arquivos brutos de texto em uma estrutura tabular unificada (`pandas.DataFrame`).
* **Leitura Inteligente:** Lê os 7.200 ficheiros individuais de texto e separa de forma automatizada o Título do Corpo da notícia.
* **Engenharia de Atributos (Features Heurísticas):** Extrai métricas superficiais frequentemente associadas ao sensacionalismo, tais como:
  * Proporção de letras maiúsculas (`pct_maiusculas`).
  * Contagem de pontuação expressiva (`qtd_pontuacao_expressiva` como `!` e `?`).
  * Comprimento do texto e a razão de proporção entre o tamanho do título e do corpo.
* **Pipelines Diferenciados de Limpeza:**
  * **Limpeza para ML Clássico:** Conversão para minúsculas, remoção profunda de URLs, e-mails, números isolados, *stopwords* e aplicação de **Lematização** utilizando a biblioteca `spaCy` (`pt_core_news_sm`).
  * **Limpeza para Transformers:** Limpeza leve que remove ruídos de formatação, mas preserva a caixa das letras e a pontuação para manter o contexto semântico rico exigido pelo BERT.
* **Output:** Salva o arquivo compilado e limpo em `dataset_pre_processado.csv`.

### 2. `Fake_News.ipynb` (Modelagem, Aprendizado de Máquina e Auditoria)
A partir do arquivo gerado no passo anterior, este notebook desenvolve e compara três estratégias progressivas de classificação:
* **Estratégia 1 (Baseada em Heurísticas / Regras):** Um sistema especialista transparente que utiliza regras manuais e expressões regulares (Regex) de palavras de efeito (*clickbaits*) para pontuar o quão "suspeito" o texto é. Funciona como um baseline explicável.
* **Estratégia 2 (Machine Learning Clássico & Híbrido):** * *Abordagem de Texto:* Vetorização de texto com **TF-IDF** (incluindo unigramas, bigramas e trigramas) avaliado com algoritmos clássicos (*Naive Bayes, Regressão Logística e SVM*).
  * *Abordagem Híbrida:* União das matrizes de texto geradas pelo TF-IDF com os atributos numéricos de heurística extraídos no primeiro notebook, utilizando modelos poderosos como o *XGBoost*.
* **Estratégia 3 (Deep Learning - BERTimbau):** Ajuste fino (*Fine-Tuning*) do modelo de linguagem de última geração **`neuralmind/bert-base-portuguese-cased`** com suporte a parada antecipada (*Early Stopping*), alcançando o estado da arte em desempenho.
* **Auditoria Humana e Análise de Erros:** Inspeção estatística e qualitativa através de matrizes de confusão, curvas AUC-ROC e cálculo do **Coeficiente Kappa de Cohen** para medir e discutir a divergência entre o modelo Black-box (BERT) e o sistema interpretável de regras.

---

## 🛠️ Pré-requisitos e Instalação

Para rodar o pipeline completo localmente, você precisará instalar as seguintes bibliotecas do ecossistema de Ciência de Dados e PLN em Python:

```bash
# Instalação das bibliotecas base e algoritmos de ML
pip install pandas numpy scikit-learn matplotlib seaborn xgboost torch transformers

# Instalação do spaCy e descarregamento do modelo em português (utilizado no sanity.ipynb)
pip install spacy
python -m spacy download pt_core_news_sm
