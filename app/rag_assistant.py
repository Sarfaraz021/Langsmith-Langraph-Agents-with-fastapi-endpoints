# app/rag_assistant.py

import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from fastapi import HTTPException
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from app.prompt import prompt_template


class RAGAssistant:
    def __init__(self):
        self.load_env_variables()
        self.setup_prompt_template()
        self.retriever = None  # Define retriever as an instance variable
        # Specify the absolute path of the file
        absolute_path = r'D:\RAG-based-Personal-AI-Assistant\app\data\dummy.txt'
        # Get the current working directory
        current_directory = os.getcwd()
        # Calculate the relative path
        relative_path = os.path.relpath(absolute_path, current_directory)
        # default_documents_directory = r"D:\RAG-based-Personal-AI-Assistant\app\data\dummy.txt"
        self.initialize_retriever(relative_path)
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    def load_env_variables(self):
        """Loads environment variables from .env file."""
        load_dotenv('var.env')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

    def setup_prompt_template(self):
        """Sets up the prompt template for chat completions."""
        self.template = prompt_template
        self.prompt_template = PromptTemplate(
            input_variables=["history", "context", "question"],
            template=self.template,
        )

    def initialize_retriever(self, directory_path):
        """Initializes the retriever with documents from the specified directory path."""
        loader = TextLoader(directory_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()

        Pinecone(api_key=self.pinecone_api_key, environment='us-east-1-aws')
        vectbd = PineconeVectorStore.from_documents(
            docs, embeddings, index_name=self.pinecone_index_name)
        self.retriever = vectbd.as_retriever()

    async def generate_response(self, query):
        if not self.retriever:
            raise HTTPException(
                status_code=500, detail="Retriever not initialized")

        # Here we create an instance of the RetrievalQA chain
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type='stuff',
            retriever=self.retriever,  # Use the instance variable here
            chain_type_kwargs={"verbose": False, "prompt": self.prompt_template,
                               "memory": ConversationBufferMemory(memory_key="history", input_key="question")}
        )

        # We use the chain to invoke the model and generate a response
        assistant_response = chain.invoke(query)
        return assistant_response.get('result', 'No response generated')

    async def finetune(self, file_path):
        """Determines the document type and uses the appropriate loader to fine-tune the model."""
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        elif file_path.endswith('.csv'):
            loader = CSVLoader(file_path=file_path)
        elif file_path.endswith('.xlsx'):
            loader = UnstructuredExcelLoader(file_path, mode="elements")
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError("Unsupported file type.")

        documents = loader.load_and_split() if hasattr(
            loader, 'load_and_split') else loader.load()

        self.process_documents(documents)

    def process_documents(self, documents):
        """Process and index the documents."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        Pinecone(api_key=self.pinecone_api_key, environment='us-east-1-aws')
        vectbd = PineconeVectorStore.from_documents(
            docs, embeddings, index_name=self.pinecone_index_name)
        self.retriever = vectbd.as_retriever()
