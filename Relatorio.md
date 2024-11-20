# Relatório do Projeto: Chatbot para Responder Dúvidas sobre o Vestibular da Unicamp 2025

## Visão Geral do Projeto

Este projeto foi desenvolvido com o objetivo de criar um chatbot baseado em RAG (Retrieval Augmented Generation) para responder dúvidas sobre o Vestibular da Unicamp 2025. O chatbot foi construído utilizando [edital do vestibular Unicamp 2025](https://www.pg.unicamp.br/norma/31879/0) como base de informação.

## Descrição da Implementação

### 1. Carregamento e Processamento do Documento

O projeto utilizou a biblioteca `langchain_community` para carregar e processar o documento “norma\_convest2025.md”, que continha as regras e informações do vestibular.

- **Carregamento do Documento**:

```python
from langchain_community.document_loaders import UnstructuredMarkdownLoader

markdown_path = "norma_convest2025.md"
loader = UnstructuredMarkdownLoader(markdown_path, mode="elements") 
documents = loader.load()
```

O arquivo foi carregado usando o `UnstructuredMarkdownLoader`, que permite a leitura de arquivos em formato Markdown e os carrega como documentos estruturados, o que facilita posteriormente a divisão e a leitura dos dados presentes nesse arquivo.

**Observação:** O arquivo HTML do edital do vestibular Unicamp 2025 foi convertido para o formato Markdown usando a ferramenta [Pandoc](https://pandoc.org/).

- **Divisão do Texto**:
  Para otimizar a busca de informações, o documento foi dividido em blocos menores de 1000 caracteres, com uma sobreposição de 100 caracteres, utilizando a classe `RecursiveCharacterTextSplitter`.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

split_documents = text_splitter.split_documents(documents)
```

Essa técnica divide o extenso documento do edital do vestibular em pedaços menores e mais gerenciáveis para processamento mais eficiente, facilitando o manuseio de textos extensos. Isso garante que as respostas sejam mais completas e evitem fragmentações de informações importantes.

- **Filtragem de Metadados**:  Quando um documento é carregado e divido  usando bibliotecas como o LangChain, cada segmento pode ter metadados associados que precisam ser filtrados para evitar informações desnecessárias ou complexas que podem atrapalhar o desempenho ou a relevância do modelo na hora de responder às perguntas.

```python
from langchain_community.vectorstores.utils import filter_complex_metadata

split_documents = filter_complex_metadata(split_documents)
```

A função `filter_complex_metadata()` foi usada para remover metadados que são considerados complexos ou que possam interferir na relevância da busca ou na simplicidade do armazenamento. Isso garante que os documentos processados e indexados no vetor de busca sejam mais leves e focados nos conteúdos essenciais.

### 2. Criação da Base de Dados Vetorial

- **Embeddings e Armazenamento Vetorial**:

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(split_documents, embedding=embeddings)
```

Os embeddings foram gerados com `OpenAIEmbeddings`, e os vetores resultantes foram armazenados usando o `Chroma`, um repositório vetorial eficiente para armazenamento e recuperação de documentos, com mecanismos de busca baseados em similaridade semântica. O `vector_store` posteriormente será transformado em um objeto de recuperação que pode ser usado pelo rag_chain para buscar documentos relevantes com base em consultas de entrada.

### 3. Modelo de Linguagem e Cadeia de Recuperação

- **Modelo de Linguagem**:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
```

O modelo `gpt-4o-mini` foi utilizado como o agente inteligente para a geração das respostas.&#x20;

- **Template de Prompt**:

```python
from langchain_core.prompts import ChatPromptTemplate

system_prompt = "Responda dúvidas sobre o vestibular da Unicamp 2025. Context: {context}"
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])
```

O template do prompt foi montado usando `ChatPromptTemplate`, no qual o `system_prompt` orienta o modelo a responder de forma objetiva, utilizando o contexto recuperado da base vetorial, ou seja, o sistema esta sendo orientado para responder dúvidas sobre o vestibular da Unicamp 2025. Já o "human" indica o input que conterá a dúvida do usuário.

### 4. RAG Chain e Geração de Respostas

A  RAG chain foi criada combinando a busca vetorial (criada anteriormente) e a geração de respostas:

```python
from langchain.chains import create_retrieval_chain

combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template)

rag_chain = create_retrieval_chain(retriever=vector_store.as_retriever(), combine_docs_chain=combine_docs_chain)
```

- `create_stuff_documents_chain:` Cria uma cadeia de combinação que utiliza o modelo de linguagem para gerar respostas baseadas em documentos recuperados. O template de prompt é passado para definir como o modelo deve processar o contexto e as entradas do usuário.

- `create_retrieval_chain:` Configura a cadeia de busca e geração de respostas. O `retriever` é responsável por recuperar os trechos mais relevantes do vector store, enquanto o `combine_docs_chain` organiza esses trechos e os apresenta ao modelo de linguagem para que ele gere uma resposta completa.

### 5. Entrada do Usuário e Saída

O chatbot solicita o input do usuário, que é processado pelo RAG chain (criado anteriormente) para fornecer uma resposta baseada no contexto:

```python
human_input = input()
response = rag_chain.invoke({"input": human_input})
print(response['answer']) ## No código original é usado o Streamlit para mostrar a resposta
```

### 6. Implementação da Interface com Streamlit

Para tornar o chatbot mais acessível e interativo, foi implementada uma interface web utilizando o framework Streamlit. Este framework permite que a aplicação seja executada diretamente em um navegador, sem a necessidade de interface por terminal, melhorando a usabilidade do sistema.

Algumas configurações básicas da interface foram feitas da seguinte forma:

```python
streamlit.title("Chatbot Vestibular Unicamp 2025")
streamlit.subheader("Tire suas dúvidas sobre o vestibular!")
human_input = streamlit.text_input("Digite sua dúvida aqui:")
submit_button = streamlit.form_submit_button("Enviar")
```

- `title():` Exibe o título da aplicação.

- `subheader():` Adiciona um subtítulo explicativo.

- `text_input():` Permite que o usuário digite sua dúvida e retorna a resposta do usuário.

- `form_submit_button:` Adiciona o botão que permite o envio da pergunta.

Já o **processamento das perguntas e respostas** foi feito da seguinte forma: 
ao submeter uma pergunta, o texto digitado pelo usuário é processado pela cadeia RAG já configurada no projeto. Para melhorar a experiência do usuário, o Streamlit exibe um spinner enquanto o chatbot processa a resposta.

```python
if submit_button and human_input.strip():
    with streamlit.spinner("Processando sua pergunta..."):
        response = rag_chain.invoke({"input": human_input})
        streamlit.markdown(f"**Resposta:** \n {response['answer']}")
```

- `form():` Cria um formulário interativo que inclui a caixa de entrada e o botão de envio.

- `spinner():` Mostra uma animação enquanto o modelo processa a resposta.

### Imagem da interface do Chatbot com Streamlit:

![Interface do chatbot](/images/chatbot_interface.png)

## 7. Pré-processamento com NLTK
Foi utilizado o pacote **NLTK (Natural Language Toolkit)**, uma biblioteca  usada para tarefas de Processamento de Linguagem Natural. No caso deste projeto, foram realizados dois downloads específicos de dados linguísticos necessários para a análise e processamento dos textos, que são necessários na primeira vez em que o código é rodado.

```python
import nltk

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
```

- `punkt_tab:` O pacote punkt é um tokenizador de frases e palavras. Ele divide um texto em sentenças e, posteriormente, em palavras, o que é essencial para pré-processamento textual. Este pacote foi utilizado, pois durante a criação de embeddings e na segmentação dos documentos em partes menores, é necessário dividir o texto em trechos ou palavras. O punkt fornece as regras necessárias para essa tarefa.

- `averaged_perceptron_tagger_eng:` Este pacote é um etiquetador de partes do discurso (Part-of-Speech Tagging), que atribui rótulos gramaticais às palavras em um texto (como substantivo, verbo, adjetivo etc.). Ele foi utilizado, pois em aplicações avançadas de PLN, essa funcionalidade pode ser útil para refinar a recuperação ou a relevância de partes específicas do texto ao extrair contextos importantes.

- **Observação:** A não utilização desses pacotes durante a primeira execução do código acarreta em erros que sugerem o uso desses pacotes como solução. Por conta disso, o uso do `nltk` é obrigatório durante a primeira execução do código.

## Testes e Avaliação

Abaixo serão mostrados alguns testes demonstrando perguntas sobre o vestibular da Unicamp 2025 e a reposta que o Chatbot gerou.

## Teste 1

### Pergunta:
```
Qual é o dia da primeira fase do vestibular?
```

### Resposta:
```
A primeira fase do vestibular da Unicamp 2025 será realizada no dia 20 de outubro de 2024.
```

### Avaliação da resposta:

- A resposta esta correta.

## Teste 2

### Pergunta:
```
Quais cursos de computação estão disponiveis para inscrição?
```

### Resposta:
```
Na Unicamp, os cursos de computação disponíveis para inscrição são geralmente agrupados na área de Ciências Exatas/Tecnológicas. Os cursos que costumam ser oferecidos nessa área incluem:

1. Ciência da Computação
2. Engenharia de Computação
3. Sistemas de Informação

É importante verificar o site oficial da Unicamp ou o edital do vestibular de 2025 para confirmar a lista atualizada dos cursos, pois pode haver alterações ou novos cursos disponíveis.
```

### Avaliação da resposta:

- A resposta esta correta.

## Teste 3

### Pergunta:
```
Qual curso com a maior relação de candidato por vaga?
```

### Resposta:
```
O curso com a maior relação de candidato por vaga na Unicamp 2025 é o de Administração (Noturno), que possui uma concorrência de 49 candidatos por vaga.
```

### Avaliação da resposta:

- A resposta esta errada.
- O correto seria Medicina com 266,5 C/V (ampla concorrência)
- O curso de Administração tem 7,3 C/V (ampla concorrência)

## Teste 4

### Pergunta:
```
É permitido celular durante a prova?
```

### Resposta:
```
Não, o uso de celular durante a prova não é permitido. Os candidatos devem evitar trazer dispositivos eletrônicos, pois qualquer infração relacionada ao uso de meios ilícitos pode resultar em desclassificação do vestibular. É importante seguir as orientações e levar apenas os materiais permitidos, como lápis, borracha e caneta esferográfica azul ou preta.
```

### Avaliação da resposta:

- A resposta esta correta.

## Teste 5

### Pergunta:
```
Como eu posso verificar o local de realização da minha prova?
```

### Resposta:
```
Você pode verificar o local de realização da sua prova acessando o site oficial da Unicamp, na seção dedicada ao vestibular. Normalmente, a informação sobre os locais de prova é divulgada alguns dias antes da data das provas. Você também pode conferir seu local no cartão de inscrição, que é enviado por e-mail ou pode ser acessado na área do candidato no site da Unicamp. É importante ficar atento às orientações e prazos estabelecidos pela instituição.
```

### Avaliação da resposta:

- A resposta esta correta.

## Teste 6

### Pergunta:
```
Qual o curso do vestibular com mais vagas?
```

### Resposta:
```
O curso do vestibular da Unicamp 2025 com mais vagas é Administração (Noturno), que oferece um total de 180 vagas.
```

### Avaliação da resposta:

- A resposta esta parcialmente correta.
- O curso de Administração é relamente o curso com mais vagas, porém são 170 e não 180.

## Teste 7

### Pergunta:
```
Quais documentos eu preciso levar no dia da prova?
```

### Resposta:
```
1. **Documento de identificação com fotografia**: Este deve ser o mesmo que você indicou no Formulário de Inscrição.
2. **Materiais para a prova**:
   - Lápis
   - Borracha
   - Caneta esferográfica azul ou preta

Certifique-se de ter todos esses itens preparados para evitar contratempos no dia da prova. 
```

### Avaliação da resposta:

- A resposta esta correta.

## Teste 8

### Pergunta:
```
Quantas questões tem a primeira fase?
```

### Resposta:
```
A primeira fase do vestibular da Unicamp 2025 é composta por 72 questões objetivas.
```

### Avaliação da resposta:

- A resposta esta correta.

## Teste 9

### Pergunta:
```
Quais materias caem na prova da primeira fase e quantas perguntas tem de cada materia?
```

### Resposta:
```
Na prova da primeira fase do vestibular da Unicamp 2025, as matérias e a quantidade de questões de cada uma são as seguintes:

- Matemática: 12 questões
- Língua Portuguesa e Literaturas de Língua Portuguesa: 12 questões
- Inglês: 7 questões
- Ciências da Natureza (Biologia, Química e Física): 21 questões
- Ciências Humanas (História, Geografia, Filosofia e Sociologia): 20 questões

No total, a prova é composta por 72 questões objetivas. 
```

### Avaliação da resposta:

- A resposta esta correta.

## Teste 10

### Pergunta:
```
O curso de Engenharia de Computação (Integral) tem quantas vagas e qual a relação C/V (canditato por vaga)?
```

### Resposta:
```
O curso de Engenharia de Computação (Integral) não aparece na tabela que você forneceu, portanto, não é possível informar o número de vagas ou a relação candidato por vaga (C/V) para este curso específico. Recomendo verificar o edital ou o site oficial da Unicamp para obter informações detalhadas sobre o curso de Engenharia de Computação. Se precisar de informações sobre outros cursos, estou à disposição!
```

### Avaliação da resposta:

- A resposta esta errada.
- O curso de Engenharia de Computação parece na tabela presente no edital

## Teste 11

### Pergunta:
```
Qual o curso mais concorrido da UNICAMP?
```

### Resposta:
```
O curso mais concorrido da UNICAMP geralmente é o de Medicina. Este curso atrai um grande número de candidatos todos os anos, resultando em uma alta relação candidato/vaga. Além de Medicina, outros cursos que costumam ter alta concorrência incluem Engenharia e Direito. É sempre bom verificar as estatísticas específicas do vestibular mais recente para ter informações atualizadas sobre a concorrência.
```

### Avaliação da resposta:

- A resposta esta parcialmente correta.
- O curso mais concorrido é relamente Medicina, mas o Chatbot afirma que Direito também é um curso concorrido, porém o curso de Direito não é oferecido na UNICAMP.

## Teste 12

### Pergunta:
```
O curso de Direito é oferecido no vestibular da UNICAMP?
```

### Resposta:
```
Sim, o curso de Direito é oferecido no vestibular da UNICAMP. A instituição disponibiliza esse curso na área de Ciências Humanas. Se precisar de mais informações sobre o processo seletivo ou detalhes do curso, fique à vontade para perguntar!
```

### Avaliação da resposta:

- A resposta esta errada.
- O curso de Direito não é oferecido na UNICAMP.

## Teste 13

### Pergunta:
```
O que o estudante precisa fazer para avançar para a segunda fase?
```

### Resposta:
```
Para avançar para a segunda fase do vestibular da Unicamp, o estudante precisa alcançar uma pontuação mínima estabelecida na primeira fase da prova. A primeira fase consiste em uma prova objetiva, que avalia conhecimentos gerais em diversas disciplinas. Os candidatos que obtiverem as melhores notas nessa etapa são convocados para a segunda fase, que é composta por questões dissertativas e inclui uma prova de Redação e questões específicas de acordo com o curso escolhido. É importante que o estudante consulte o edital do vestibular para informações detalhadas sobre as notas mínimas e critérios de seleção, pois esses podem variar a cada edição do vestibular.
```

### Avaliação da resposta:

- A resposta esta correta.

## Teste 14

### Pergunta:
```
Quais as materias especificas que estarão na prova de um estudante que avançou para a segunda fase e está inscrito no curso de engenharia de computação?
```

### Resposta:
```
Para o curso de Engenharia de Computação na Unicamp, os candidatos que avançam para a segunda fase do vestibular devem se preparar para as seguintes disciplinas específicas:

1. Língua Portuguesa e Literaturas de Língua Portuguesa - Esta prova é aplicada a todos os candidatos.
2. Matemática - Também é uma prova comum a todos os candidatos.
3. Física - Disciplinas específicas para os cursos da área de Exatas e Tecnológicas.
4. Química - Outra disciplina específica para os cursos da área de Exatas e Tecnológicas.
5. Questões interdisciplinares com Inglês - Inclui questões que envolvem a interpretação e compreensão do idioma.

As provas da segunda fase são realizadas em dois dias consecutivos, e é importante que os candidatos se preparem adequadamente para cada uma dessas disciplinas.
```

### Avaliação da resposta:

- A resposta esta correta.

## Teste 15

### Pergunta:
```
Qual o número mínimo e máximo de vagas para Ampla Concorrência no curso de Engenharia de Computação?
```

### Resposta:
```
As informações que você forneceu não incluem dados sobre o curso de Engenharia de Computação. Para obter o número mínimo e máximo de vagas para Ampla Concorrência nesse curso, você precisaria consultar o edital do vestibular da Unicamp 2025 ou entrar em contato diretamente com a instituição. Se precisar de informações sobre os cursos que foram listados, estou à disposição para ajudar!
```

### Avaliação da resposta:

- A resposta esta errada.
- O número de vagas mínimas e máximas para o curso de Engenharia de Computação constam no edital como 39 e 49, respectivamente.

## Avaliação Geral:
O Chatbot se mostra muito eficiente para responder perguntas objetivas, principalmente sobre regras e informações da prova, porém ele se mostra ineficiente quando precisa analisar as tabelas presentes no edital, uma vez que ele errou todas as perguntas envolvendo tabelas sobre número de vagas e sobre a relação C/V.