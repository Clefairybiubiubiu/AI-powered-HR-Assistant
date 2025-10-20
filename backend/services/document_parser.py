"""
Document parsing service for resumes
"""
import PyPDF2
import docx
from typing import Dict, Any, Optional
import re
import io


class DocumentParser:
    """Parse various document formats and extract text"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.doc', '.txt']
    
    def parse_document(self, content: bytes, filename: str) -> Dict[str, Any]:
        """
        Parse document content and extract structured information
        """
        file_extension = self._get_file_extension(filename)
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Extract text based on file type
        if file_extension == '.pdf':
            text = self._parse_pdf(content)
        elif file_extension == '.docx':
            text = self._parse_docx(content)
        elif file_extension == '.txt':
            text = self._parse_txt(content)
        else:
            raise ValueError(f"Parser not implemented for {file_extension}")
        
        # Extract structured information
        parsed_data = self._extract_structured_info(text)
        parsed_data['text'] = text
        parsed_data['filename'] = filename
        
        return parsed_data
    
    def _get_file_extension(self, filename: str) -> str:
        """Get file extension from filename"""
        if '.' in filename:
            return '.' + filename.lower().split('.')[-1]
        return ''
    def _parse_pdf(self, content: bytes) -> str:
        """Parse PDF content"""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise ValueError(f"Error parsing PDF: {str(e)}")
    
    def _parse_docx(self, content: bytes) -> str:
        """Parse DOCX content"""
        try:
            doc = docx.Document(io.BytesIO(content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise ValueError(f"Error parsing DOCX: {str(e)}")
    
    def _parse_txt(self, content: bytes) -> str:
        """Parse TXT content"""
        try:
            return content.decode('utf-8')
        except UnicodeDecodeError:
            return content.decode('latin-1')
    
    def _extract_structured_info(self, text: str) -> Dict[str, Any]:
        """Extract structured information from resume text"""
        info = {}
        
        # Extract name (first line or after "Name:")
        name_match = re.search(r'(?:Name|Full Name):\s*([^\n]+)', text, re.IGNORECASE)
        if not name_match:
            # Try to get first line as name
            lines = text.split('\n')
            if lines:
                info['name'] = lines[0].strip()
        else:
            info['name'] = name_match.group(1).strip()
        
        # Extract email
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if email_match:
            info['email'] = email_match.group()
        
        # Extract phone
        phone_match = re.search(r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})', text)
        if phone_match:
            info['phone'] = phone_match.group()
        
        # Extract summary/objective
        summary_patterns = [
            r'(?:Summary|Objective|Profile):\s*([^\n]+(?:\n(?!\n)[^\n]+)*)',
            r'(?:About|Overview):\s*([^\n]+(?:\n(?!\n)[^\n]+)*)'
        ]
        
        for pattern in summary_patterns:
            summary_match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if summary_match:
                info['summary'] = summary_match.group(1).strip()
                break
        
        return info
