# app/routers/finetune_routers.py
import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.rag_assistant import RAGAssistant

router = APIRouter()


@router.post("/finetune/")
async def finetune(file: UploadFile = File(...)):
    assistant = RAGAssistant()

    # Ensure the directory for temporary storage exists
    temp_dir = "app/temp"
    os.makedirs(temp_dir, exist_ok=True)

    # Generate a path for the temporary file
    if file.filename is None:
        raise HTTPException(
            status_code=400, detail="File must have a filename.")
    tmp_file_path = os.path.join(temp_dir, file.filename)

    # Save the uploaded file temporarily
    with open(tmp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Close the file to ensure it's saved and released
    file.file.close()

    # Determine the file type based on the extension
    # Index [1] to get the extension part
    file_extension = os.path.splitext(file.filename)[1]
    file_type = file_extension.lstrip('.')

    # Check if the file type is supported
    if file_type not in ['pdf', 'txt', 'csv', 'xlsx', 'docx']:
        os.remove(tmp_file_path)  # Clean up the temporary file
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    try:
        # Call finetune with the path to the temporary file
        await assistant.finetune(tmp_file_path)
        # Clean up the temporary file after processing
        os.remove(tmp_file_path)
        return {"message": "Fine-tuning successful."}
    except Exception as e:
        # Ensure the temporary file is removed even on failure
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
        raise HTTPException(status_code=500, detail=str(e))
