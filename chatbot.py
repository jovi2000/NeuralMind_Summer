import streamlit
import nltk
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate 
from langchain_community.vectorstores.utils import filter_complex_metadata

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Load the markdown file
markdown_path = "norma_convest2025.md"
loader = UnstructuredMarkdownLoader(markdown_path, mode="elements") 
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_documents = text_splitter.split_documents(documents)

# Filter metadata
split_documents = filter_complex_metadata(split_documents)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Create a vector store from the split documents using the embedding function
vector_store = Chroma.from_documents(split_documents, embedding=embeddings)

# Initialize your language model
llm = ChatOpenAI(model="gpt-4o-mini")

# Create a prompt template for the combine_docs_chain
system_prompt = "Responda dúvidas sobre o vestibular da Unicamp 2025. Context: {context}"
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Create a RetrievalQA chain with a combine_docs_chain
combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template)
rag_chain = create_retrieval_chain(retriever=vector_store.as_retriever(), combine_docs_chain=combine_docs_chain)

# Chatbot interface using Streamlit
streamlit.title("Chatbot Vestibular Unicamp 2025")
streamlit.subheader("Tire suas dúvidas sobre o vestibular!")

with streamlit.form("chat_form"):
    human_input = streamlit.text_input("Digite sua dúvida aqui:")
    submit_button = streamlit.form_submit_button("Enviar")
    
    if submit_button and human_input.strip():
        with streamlit.spinner("Processando sua pergunta..."):
            response = rag_chain.invoke({"input": human_input})
            streamlit.markdown(f"**Resposta:** \n {response['answer']}")