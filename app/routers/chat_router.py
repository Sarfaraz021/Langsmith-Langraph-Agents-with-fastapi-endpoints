# app/routers/chat_router.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.rag_assistant import RAGAssistant
from app.models.models import ChatQuery


router = APIRouter()


@router.post("/chat/")
async def chat(chat_query: ChatQuery):  # Use the request model here
    assistant = RAGAssistant()
    try:
        # Use the query attribute from the model
        response = await assistant.generate_response(chat_query.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
