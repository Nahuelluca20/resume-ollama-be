import pymupdf
from fastapi import HTTPException


class PDFService:
    @staticmethod
    async def extract_text_from_pdf(pdf_content: bytes) -> str:
        """Extract text content from PDF bytes using PyMuPDF."""
        try:
            doc = pymupdf.open(stream=pdf_content, filetype="pdf")
            text_content = ""
            
            for page in doc:
                text_content += page.get_text() + "\n"
            
            doc.close()
            return text_content.strip()
        except Exception as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Error extracting text from PDF: {str(e)}"
            )
    
    @staticmethod
    def validate_pdf_file(file_content_type: str, file_size: int) -> None:
        """Validate PDF file type and size."""
        if file_content_type != "application/pdf":
            raise HTTPException(
                status_code=400, 
                detail="Only PDF files are accepted"
            )
        
        # Limit to 10MB
        max_size = 10 * 1024 * 1024
        if file_size and file_size > max_size:
            raise HTTPException(
                status_code=400, 
                detail="File size too large. Maximum 10MB allowed"
            )