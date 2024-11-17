from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate 
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_openai import ChatOpenAI
import nltk

nltk.download('averaged_perceptron_tagger_eng')

# Load the markdown file
markdown_path = "norma_convest2025.md"
loader = UnstructuredMarkdownLoader(markdown_path, mode="elements")  # Use elements mode if needed
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_documents = text_splitter.split_documents(documents)
split_documents = filter_complex_metadata(split_documents)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Create a vector store from the split documents using the embedding function
vector_store = Chroma.from_documents(split_documents, embedding=embeddings)

# Initialize your language model
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_base="https://api.openai.com/v1")

# Create a prompt template for the combine_docs_chain
system_prompt = "What is the main topic of the document? Context: {context}"
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Create a RetrievalQA chain with a combine_docs_chain
combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template)
rag_chain = create_retrieval_chain(retriever=vector_store.as_retriever(), combine_docs_chain=combine_docs_chain)

query = input()
try:
    response = rag_chain.invoke({"input": query})
    print(response['answer'])
except KeyError as e:
    print(f"KeyError: {e} - Verifique se a estrutura do input está correta.")
except Exception as e:
    print(f"An error occurred: {e}")