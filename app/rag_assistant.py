# app/adaptive_rag_assistant.py

import os
from dotenv import load_dotenv
from typing import List, Literal
from typing_extensions import TypedDict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, UnstructuredExcelLoader, Docx2txtLoader
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from fastapi import HTTPException
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore"] = Field(
        ...,
        description="Given a user question choose to route it to vectorstore.",
    )


class GraphState(TypedDict):
    """Represents the state of our graph."""
    question: str
    generation: str
    documents: List[str]


class RAGAssistant:
    def __init__(self):
        self.load_env_variables()
        self.setup_components()
        self.setup_graph()

    def load_env_variables(self):
        load_dotenv('var.env')
        os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
        os.environ["PINECONE_API_KEY"] = os.getenv('PINECONE_API_KEY')
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

    def setup_components(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        self.structured_llm_router = self.llm.with_structured_output(
            RouteQuery)

        Pinecone(environment='us-east-1-aws')
        self.vectorstore = PineconeVectorStore.from_existing_index(
            self.pinecone_index_name, self.embeddings
        )
        self.retriever = self.vectorstore.as_retriever()

        system = """You are an expert at routing a user question to a vectorstore.
        The vectorstore contains documents about varuns data.
        Use the vectorstore for questions on these topics. Otherwise, use your own knowledge"""
        self.route_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "{question}"),
        ])
        self.question_router = self.route_prompt | self.structured_llm_router

        self.rag_prompt = hub.pull("indya-cleo")
        self.rag_chain = self.rag_prompt | self.llm | StrOutputParser()

    def setup_graph(self):
        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("generate", self.generate)
        workflow.add_conditional_edges(
            START,
            self.route_question,
            {
                "vectorstore": "retrieve",
            },
        )
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        self.app = workflow.compile()

    def retrieve(self, state: GraphState) -> GraphState:
        question = state["question"]
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question}

    def generate(self, state: GraphState) -> GraphState:
        question = state["question"]
        documents = state["documents"]
        generation = self.rag_chain.invoke(
            {"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    def route_question(self, state: GraphState) -> str:
        question = state["question"]
        source = self.question_router.invoke({"question": question})
        return "vectorstore" if source.datasource == "vectorstore" else "generate"

    async def process_query(self, query: str):
        inputs = {"question": query}
        result = self.app.invoke(inputs)
        return result["generation"]

    async def update_index(self, file_path: str):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        self.vectorstore.add_documents(docs)

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

        documents = loader.load()
        self.process_documents(documents)

    def process_documents(self, documents):
        """Process and index the documents."""
        if not self.vectorstore:
            raise HTTPException(
                status_code=500, detail="Vectorstore not initialized")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        self.vectorstore.add_documents(docs)

# Usage example:
# assistant = AdaptiveRAGAssistant()
# response = await assistant.process_query("Tell me about bluescarf and who is its ceo")
# print(response)
# await assistant.finetune("path/to/new/document.pdf")
