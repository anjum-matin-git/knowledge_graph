from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.config import Config
from typing import Any

def get_chat_agent(vector_db: Any, k: int = 5) -> "Runnable":
    """
    Create a chat agent using OpenAI GPT-4 and the vector DB as retriever.

    Args:
        vector_db: A vector database object with an as_retriever method.
        k: Number of top documents to retrieve for context (default: 5).

    Returns:
        A LangChain Runnable representing the RAG agent.
    """
    if not Config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    template = """You are a helpful AI assistant that answers questions based on the provided context from a knowledge graph.

Context information:
{context}

Question: {question}

Please answer the question based on the context provided. If the context doesn't contain enough information to answer the question, say so. Be concise and accurate in your response.

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    
    retriever = vector_db.as_retriever(search_kwargs={"k": k})
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def chat_with_agent(agent: Any, user_input: str) -> str:
    """
    Get response from the agent for user input.

    Args:
        agent: The RAG agent created by get_chat_agent.
        user_input: The user's question as a string.

    Returns:
        The agent's response as a string.
    """
    response = agent.invoke(user_input)
    return response