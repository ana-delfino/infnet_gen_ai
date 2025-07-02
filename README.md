# Projeto Final IA generativa para linguagem
Aqui temos o readme do projeto e o link este mesmo readme no github: https://github.com/ana-delfino/infnet_gen_ai

### Índice
- [Parte 1 - Fundamentos das LLMs ](#parte1)
- [Parte 2 - Quizzes do Curso de NLP da Hugging Face](#parte2)
- [Parte 3 - Análise de Dados com NER](#parte3)
- [Parte 4 - Engenharia de Prompts ](#parte4)
- [Parte 5 - Projeto Prático com Streamlit, LLM e LangChain](#parte5)
  
<h2 id="parte1">Parte 1 - Fundamentos das LLMs</h2>

 - Questão 1) Explique os seguintes conceitos fundamentais dos LLMs, fornecendo exemplos práticos e diagramas onde for relevante:

    - `Pre-training`: Todos os modelos de LLM são treinados em grandes volumes de dados. Durante o pré-treino os LLMs tentam aprender e generalizar as relações entre as palavras do corpus, o desempenho dos LLMs nas diversas tarefas dependem desse dataset usado no pré-treino.
  
    - `Transfer Learning`: Transfer learning é uma técnica usada para aproveitar o conhecimento  de um modelo pré-treinado, ganho em uma tarefa, para melhorar a performance em outra tarefa similar. O transfer learning permite que a rede use o conhecimento adquirido na tarefa original para realizar outra tarefa usando muito menos dados do que seria necessário para treinar um modelo específico do zero e também permite utilizar o conhecimento já adquirido para gerar embeddings. Em termos de arquitetura, isso envolve separar o modelo em body e head, onde a head é a tarefa especifica da rede. Durante o treinamento, os pesos do body aprendem características gerais do domínio de origem, e esses pesos são usados ​​para inicializar um novo modelo para a nova tarefa ou para . A figura abaixo mostra representa essa relação:
    ![alt text](image-3.png)

    - `Embeddings`: Antes de explicar o que são embedding primeiro é necessário enternder o que é um token. O token é a representação de um sentença, palavra ou subpalavra. O processo de tokenização envolve a geração dos tokens. A representação numerica dos tokens são os vetores. Esses vetores são multidimensional (matrizes) de números que capturam a relações sintéticas (função gramatical dentro da linguagem) e semânticas (significado dentro da linguagem) entre as palavras do corpus. Esse vetores de palavras são chamados de embeddings. Durante o treinamento os modelos são  desenhados para identificar e aprender esses padrões, assegurando que as palavras com significados similares sejam mapeadas próximas no espaço de alta dimensão.Além disso, há vários tipos de embeddings são possíveis, incluindo embeddings de posição, que codificam a posição de um token em uma frase, e embeddings de token, que codificam o significado semântico de um token. Exemplos na imagem abaixo: 
    ![alt text](image-1.png)

    - `Transformers`: É uma arquitetura de rede neural profunda. Foi criado para resolver o problema que as redes neurais até então existentes não conseguiram, já os modelos de Redes Neurais Recorrentes e suas variantes processam as palavras sequencialmente e tem dificuldade de capturar as relações de longa distância (contexto). Os modelos com arquitetura tipo transformers são capazes de aprender contexto e tem 2 componentes principais:
      - *Encoder (codificador)*: tem a tarefa de receber texto bruto, dividi-lo em seus componentes principais, convertendo esses componentes em vetores e usar o mecanismo de atenção para entender o contexto do texto. 
      - *Decoder (decodificador)*: modifica a atenção para gerar a sequência de saída do modelo considerando o saída do codificador.
    Um dos diferenciais do modelo tranformers é uso um tipo especial de attention chamado self attention para permitir que cada palavra em uma sequência “dê atenção” (procure o contexto) todas as outras palavras na sequência, permitindo capturar dependências de longo alcance e relacionamentos contextuais entre palavras.

    - `Attention`: Attention é um dos componentes principais da arquitetura das redes neurais do tipo transformer. A ideia por trás da attention é que ao invés de produzir uma único hidden state para um sequência de input o encoder fornece um hidden state para cada step que o decoder pode acessar. Ao usar todos os steps ao mesmo tempo a atention permite criar um input grande para o decoder, então o mecanismo de attention é usado para priorizar quais states serão usados. Isso acontece pois a attention faz com que o decoder receba diferentes pesos ou atenção para cada encoder state em cada step do decoding. O processo é ilustrado pela figura abaixo
    ![alt text](image.png)

    - `Fine-Tunning`: Uma vez que o LLM foi treinado, podemos realizar o fine tunning que consiste em treinar o LLM em um dataset menor que o dataset usado no treino do LLM para o o modelo faça uma atividade especifica baseada nesse novos dados. Isso permite que o LLM utilize o conhecimento do pre-treino para melhorar a performance na tarefa especifica
  
    Fontes: 
    Lewis Tunstall, Leandro von Werra, Thomas Wolf - Natural Language Processing with Transformers
    James Phoenix, Mike Taylor - Prompt Engineering for Generative AI
    Sinan Ozdemir - Quick Start Guide to Large Language Models: Strategies and Best Practices for Using ChatGPT and Other LLMs


<h2 id="parte2"> Parte 2 - Quizzes do Curso de NLP da Hugging Face</h2>

- Questão 2) Acesse os quizzes dos capítulos 1, 2 e 3 do curso de NLP da Hugging Face através do link: Curso de NLP.

  - 2.1) Resolva os quizzes e capture screenshots dos resultados.
  - 2.2) Anexe as screenshots a esta avaliação e explique brevemente os conceitos abordados em cada quiz.
  
  `Resposta`: O **capítulo 1** fala sobre transformers NLP e LLM, explicando a diferença entre os dois tipos de modelos, as principais tarefas realizadas pelos modelos transformers, como eles funcionam, as arquiteturas, encode, decode e seq2seq. 
   prints do capítulo estão abaixo: 

    capitulo_1/Captura-de-Tela-2025-06-05-às-18.45.45.png
    ![alt text](capitulo_1/Captura-de-Tela-2025-06-05-às-18.45.45.png) 

    capitulo_1/Captura-de-Tela-2025-06-05-às-18.50.21.png
    ![alt text](capitulo_1/Captura-de-Tela-2025-06-05-às-18.50.21.png) 

    capitulo_1/Captura-de-Tela-2025-06-05-às-18.51.45.png
    ![alt text](capitulo_1/Captura-de-Tela-2025-06-05-às-18.51.45.png)

    capitulo_1/Captura-de-Tela-2025-06-05-às-19.02.52.png
    ![alt text](capitulo_1/Captura-de-Tela-2025-06-05-às-19.02.52.png)

    capitulo_1/Captura-de-Tela-2025-06-05-às-19.06.53.png
    ![alt text](capitulo_1/Captura-de-Tela-2025-06-05-às-19.06.53.png)
 
  O **capítulo 2** é mais prático e explica como usar os modelos transformers destacando que os modelos não preocessam textos diretamento, então o primeiro passo é converter os textos em números utilizando o tokenizar - cujo papel é separar os inputs em palavas, ca racters,  pontuação e subpalavras gerando os tokens, cada token é mapeado apra um inteiro.
  prints do capítulo estão abaixo: 

  capitulo_2/Captura-de-Tela-2025-06-05-às-20.43.23.png
  ![alt text](capitulo_2/Captura-de-Tela-2025-06-05-às-20.43.23.png)

  capitulo_2/Captura-de-Tela-2025-06-05-às-20.43.56.png
  ![alt text](capitulo_2/Captura-de-Tela-2025-06-05-às-20.43.56.png)

  capitulo_2/Captura-de-Tela-2025-06-05-às-20.44.58.png
  ![alt text](capitulo_2/Captura-de-Tela-2025-06-05-às-20.44.58.png)

  capitulo_2/Captura-de-Tela-2025-06-05-às-20.45.17.png
  ![alt text](capitulo_2/Captura-de-Tela-2025-06-05-às-20.45.17.png)

  O **capítulo 3** tem foco no fine tunning de modelos pré-treinados, no capítulo há explições de como usar seu dados para fazer o fine tunning de modelos. 
  prints do capítulo estão abaixo: 

  capitulo_3/Captura-de-Tela-2025-06-05-às-21.00.38.png
  ![alt text](capitulo_3/Captura-de-Tela-2025-06-05-às-21.00.38.png)

  capitulo_3/Captura-de-Tela-2025-06-05-às-21.00.57.png
  ![alt text](capitulo_3/Captura-de-Tela-2025-06-05-às-21.00.57.png)

  capitulo_3/Captura-de-Tela-2025-06-05-às-21.01.13.png
  ![alt text](capitulo_3/Captura-de-Tela-2025-06-05-às-21.01.13.png)


<h2 id="parte3"> Parte 3 - Análise de Dados com NER </h2>

- Questão 3) Baixe o conjunto de dados de notícias disponível em:Folha UOL News Dataset.

  - 3.1) Utilize o modelo 'monilouise/ner_pt_br' para identificar e extrair entidades mencionadas nas notícias.
  
    `Resposta`: O código foi desenvolvido no notebook: ana_delfino_gen_ai_pd.ipynb

  - 3.2) Crie um ranking das organizações que mais apareceram na seção "Mercado" no primeiro trimestre de 2015.

    ![alt text](image-4.png)
 
  - 3.3) Apresente os resultados em um relatório detalhado, incluindo a metodologia utilizada e visualizações para apoiar a análise.
  
    `Resposta`: O dataset tem 2111 textos de mercado no primeiro trimestre de 2015. A  estratégia de agregação dos tokens foi a max, outras como simples foram testadas mas produziam muitas palavras cortadas, dificultando a idenficação da organização. Além disso, para entender as organizações que estão nos textos considerei apenas as palavras com mais de 1 caracter para evitar empresas que não conseguiriamos interpretar.
    
    Conforme imagem abaixo é posível ver que a organização mais citada a folha, pois as notícias são do folha news em seguida vemos a organização "Brasil" temos Brasil sendo muito citado dado as notícias são do Brasil e a palavra representa tanto o país como o banco do brasil e como uma instituição que faz ações que afetam o mercado. Em seguida vemos um grupo de bancos Bradesco, HSBC, Itaú . Também observamos a palavra 'sete' como organização isso acontece porquê o dataset parece ser um erro na coluna de data de publicação, notícias de setembro aparecem como se fossem do primeiro trimestre de 2015. Isso não foi corrigido devido a ser um execício,se fosse na vida real teriamos que verificar e tratar. No geral as organizações identificadas pelo modelo fazem sentido para a seção selecionada.

<h2 id="parte4"> Parte 4 - Engenharia de Prompts</h2>

- Questão 4) Analise os seguintes prompts e identifique por que eles poderiam gerar respostas insatisfatórias ou irrelevantes:

    Exemplo 1: "Escreva sobre cachorros."
    Exemplo 2: "Explique física."
    
    `Subquestões`:

  - 4.1) Reformule cada prompt utilizando técnicas de engenharia de prompts para torná-los mais específicos e direcionados.
   Exemplo 1 - reformulado: 
  Aja como um veterinário atendendo uma família com idosos e crianças
  Dê sugestões de 2 raça de cachorros que seja adequada para a família.
  Explica 2 motivos para escolha das raças.
  Dê dicas de como cuidar do cachorro 

   Exemplo 2 - reformulado: 
  Aja como um professor de fisica para alunos crianças de 12 anos.
  Use 2 a 3 frases para explicar o que é física estuda.
  Dê 2 exemplos curtos da aplicação física no dia a dia.

  - 4.2) Explique as melhorias feitas em cada caso e os motivos por trás das reformulações.
  `Resposta`: As reformulações foram necessárias para tornar o prompt mais especifico e com objetivo claro, os prompts originais estavão muito genérico.  Para melhorar o prompt foi aplicado Chain of Thoughts (CoT),  explicitando os usuários da informação e a persona do LLM.
  
- Questão 5) O prompt "Descreva a história da internet." foi mal formulado. Aplique técnicas de engenharia de prompts para melhorá-lo. Reformule o prompt para melhorar a especificidade e a qualidade da resposta. Justifique as mudanças feitas e explique como elas contribuem para obter uma resposta mais eficaz e relevante.
  `Resposta`: O prompt abaixo direciona o modelo a refinar a quantidade de informações que devem ser trazidas ao usuário, que tipo de persona deve adotar na escrita e fornecendo o tipo de estrutura que a informação deve ser escrita.
  PROMPT:
  Haja como um aluno do ensino médio.
  Escreva um texto de no máximo 400 palavras sobre a história da internet e seu impacto da vida do ser humano.
  O texto deve ser extruturado ter Introdução, desenvolvimento, conclusão.
  Escolha um título adequado.
  Não cometa erros gramaticais.  

- Questão 6) Aplique a técnica de Chain of Thought (CoT) para melhorar o prompt "Explique como funciona a energia solar.", detalhando o raciocínio necessário para que o modelo forneça uma resposta completa e coerente. Explique como a aplicação da técnica CoT melhora a resposta do modelo.
 `Resposta`: A técnica de Chain of Thought (CoT) ajuda o prompt a fornecer uma resposta em subtarefas fazendo com que o modelo seja mais assertivo na resposta.
  PROMPT:
   Haja como vendedor de placas solares e explique como funciona a energia solar, abordando os seguintes pontos:
    Defina o que é energia solar.
    Cite os princípios básicos de funcionamento dos painéis solares, incluindo o efeito fotovoltaico.
    Explique como os painéis solares convertem a luz solar em eletricidade.
    Dê exemplos de uso da energia solar nas residências e empresas
    Descreva os benefícios e as limitações da energia solar em comparação com outras fontes de energia.

<h3 id="part5"> Projeto Prático com Streamlit, LLM e LangChain</h3>

- Questão 7) Escolha uma aplicação para desenvolver utilizando Streamlit, LLM e LangChain. Crie um aplicativo interativo que demonstre o uso de LLMs para resolver um problema específico.

Exemplos de Aplicação:
 - Sumarizador de Artigos:
Desenvolva um aplicativo que permita ao usuário inserir o texto de um artigo e obter um resumo conciso do conteúdo.
- Sistema de Perguntas e Respostas:
Crie um sistema que permita ao usuário fazer perguntas sobre um tópico específico e receba respostas precisas e relevantes.
- Agente de Viagem:
Desenvolva um agente virtual que possa ajudar usuários a planejar suas viagens, fornecendo informações sobre destinos, itinerários, e dicas de viagem.
- App de Auxílio em Aprendizagem:
Crie um aplicativo que auxilie estudantes a aprender um novo assunto, fornecendo explicações, exemplos e quizzes interativos.

Subquestões:
  - 7.1) Descreva a aplicação escolhida e os objetivos principais do projeto. Explique a arquitetura do aplicativo, incluindo como o Streamlit, LLM e LangChain são utilizados.
  - 7.2) Implemente o aplicativo e forneça o código-fonte, junto com instruções para execução.
  - 7.3) Apresente evidências e exemplos de uso do aplicativo e discuta os resultados obtidos.

Implemente um dashboard de monitoramento da operação usando Streamlit.
Para executar a aplicação rode o comando abaixo: 

```
cd streamlit/
streamlit run src/app.py
```

![Métricas do dataset](data/08_reporting/streamlit.png)