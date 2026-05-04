from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.document.extractor import extract_text_from_pdf
from app.services.document.chunker import chunk_text
from app.services.document.embedder import embed_and_store

router = APIRouter(prefix="/document", tags=["Document Processing"])

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file format. Only PDF files are accepted.")
    
    try:
        file_bytes = await file.read()
        raw_text = extract_text_from_pdf(file_bytes)
        
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="Text extraction failed. Ensure the PDF contains selectable text and is not a scanned image.")

        chunks = chunk_text(raw_text)
        response = embed_and_store(chunks)
        
        print(response)
        
        return {
            "status": "success",
            "message": "Document processed and stored successfully.",
            "filename": file.filename,
            "total_chunks": len(chunks)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")