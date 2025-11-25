"""
Document processing utilities with proper error handling and logging.
Handles text extraction from various file formats.
"""
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import PyPDF2
from docx import Document

from ..config import config
from ..logging_config import get_logger
from .exceptions import DocumentProcessingError

logger = get_logger(__name__)


class DocumentProcessor:
    """Handles document parsing for different file formats with proper error handling."""
    
    @staticmethod
    @lru_cache(maxsize=config.text_cache_size)
    def normalize_text(text: str) -> str:
        """
        Normalize text for consistent parsing.
        
        Args:
            text: Raw text to normalize
        
        Returns:
            Normalized text string
        """
        if not text:
            return ""
        
        # 1. Normalize line breaks
        text = re.sub(r'\r\n', '\n', text)  # Windows line endings
        text = re.sub(r'\r', '\n', text)     # Mac line endings
        
        # 2. Insert line breaks before section headers
        section_headers = [
            'PROFESSIONAL SUMMARY', 'PROFILE', 'SUMMARY', 'OBJECTIVE',
            'EDUCATION', 'EXPERIENCE', 'WORK EXPERIENCE', 'EMPLOYMENT',
            'SKILLS', 'TECHNICAL SKILLS', 'COMPETENCIES', 'EXPERTISE',
            'CERTIFICATIONS', 'PROJECTS', 'ACHIEVEMENTS', 'AWARDS',
            'PUBLICATIONS', 'LANGUAGES', 'INTERESTS', 'REFERENCES',
            'CONTACT', 'PERSONAL INFORMATION', 'CAREER OBJECTIVE',
            'PROFESSIONAL EXPERIENCE', 'ACADEMIC BACKGROUND'
        ]
        
        for header in section_headers:
            pattern = r'(?<!\n)(' + re.escape(header) + r')(?!\n)'
            text = re.sub(pattern, r'\n\1', text, flags=re.IGNORECASE)
        
        # Handle mixed case headers
        mixed_case_headers = [
            'Professional Summary', 'Profile', 'Summary', 'Objective',
            'Education', 'Experience', 'Work Experience', 'Employment',
            'Skills', 'Technical Skills', 'Competencies', 'Expertise',
            'Certifications', 'Projects', 'Achievements', 'Awards',
            'Publications', 'Languages', 'Interests', 'References',
            'Contact', 'Personal Information', 'Career Objective',
            'Professional Experience', 'Academic Background'
        ]
        
        for header in mixed_case_headers:
            pattern = r'(?<!\n)(' + re.escape(header) + r')(?!\n)'
            text = re.sub(pattern, r'\n\1', text)
        
        # 3. Remove bullet symbols
        bullet_symbols = ['•', '–', '○', '▪', '▫', '‣', '⁃', '◦', '‥', '…']
        for bullet in bullet_symbols:
            text = text.replace(bullet, '')
        
        # 4. Normalize spaces within lines
        lines = text.split('\n')
        normalized_lines = []
        for line in lines:
            normalized_line = re.sub(r'[ \t]+', ' ', line.strip())
            normalized_lines.append(normalized_line)
        
        text = '\n'.join(normalized_lines)
        
        # 5. Clean up multiple consecutive empty lines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text.strip()
    
    @staticmethod
    def detect_encoding(file_path: Path) -> str:
        """
        Detect file encoding.
        
        Args:
            file_path: Path to file
        
        Returns:
            Detected encoding string
        """
        try:
            import chardet
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result.get('encoding', 'utf-8')
                confidence = result.get('confidence', 0)
                
                if confidence < 0.7:
                    logger.warning(
                        f"Low confidence encoding detection for {file_path}: "
                        f"{encoding} (confidence: {confidence:.2f}), using utf-8"
                    )
                    return 'utf-8'
                
                logger.debug(f"Detected encoding for {file_path}: {encoding} (confidence: {confidence:.2f})")
                return encoding if encoding else 'utf-8'
        except ImportError:
            logger.warning("chardet not available, using utf-8")
            return 'utf-8'
        except Exception as e:
            logger.warning(f"Encoding detection failed for {file_path}: {e}, using utf-8")
            return 'utf-8'
    
    @staticmethod
    def extract_text_from_txt(file_path: Path) -> str:
        """
        Extract text from TXT file with encoding detection and normalization.
        
        Args:
            file_path: Path to TXT file
        
        Returns:
            Extracted and normalized text
        
        Raises:
            DocumentProcessingError: If text extraction fails
        """
        try:
            # Detect encoding
            encoding = DocumentProcessor.detect_encoding(file_path)
            
            # Read file with detected encoding
            with open(file_path, 'r', encoding=encoding) as file:
                text = file.read()
            
            # Normalize text
            normalized_text = DocumentProcessor.normalize_text(text)
            
            logger.info(
                f"TXT file {file_path.name} - Encoding: {encoding}, "
                f"Length: {len(normalized_text)} chars"
            )
            return normalized_text
            
        except UnicodeDecodeError as e:
            # Fallback to latin-1
            logger.warning(f"Unicode decode error for {file_path}, trying latin-1: {e}")
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    text = file.read()
                normalized_text = DocumentProcessor.normalize_text(text)
                logger.info(
                    f"TXT file {file_path.name} - Fallback encoding: latin-1, "
                    f"Length: {len(normalized_text)} chars"
                )
                return normalized_text
            except Exception as fallback_error:
                logger.error(f"Failed to read TXT file {file_path} with fallback encoding: {fallback_error}")
                raise DocumentProcessingError(f"Cannot read TXT file: {file_path}") from fallback_error
        except Exception as e:
            logger.error(f"Failed to read TXT file {file_path}: {e}", exc_info=True)
            raise DocumentProcessingError(f"Cannot read TXT file: {file_path}") from e
    
    @staticmethod
    def extract_text_from_pdf(file_path: Path) -> str:
        """
        Extract text from PDF file with normalization.
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            Extracted and normalized text
        
        Raises:
            DocumentProcessingError: If PDF extraction fails
        """
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                if not pdf_reader.pages:
                    raise DocumentProcessingError(f"PDF has no pages: {file_path}")
                
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as page_error:
                        logger.warning(
                            f"Error extracting text from page {page_num + 1} of {file_path}: {page_error}"
                        )
                
                if not text.strip():
                    raise DocumentProcessingError(f"No text extracted from PDF: {file_path}")
                
                # Normalize text
                normalized_text = DocumentProcessor.normalize_text(text)
                
                logger.info(f"PDF file {file_path.name} - Length: {len(normalized_text)} chars")
                return normalized_text
                
        except PyPDF2.errors.PdfReadError as e:
            logger.error(f"PDF read error for {file_path}: {e}")
            raise DocumentProcessingError(f"Cannot read PDF: {file_path}") from e
        except Exception as e:
            logger.error(f"Failed to read PDF file {file_path}: {e}", exc_info=True)
            raise DocumentProcessingError(f"Cannot process PDF: {file_path}") from e
    
    @staticmethod
    def extract_text_from_docx(file_path: Path) -> str:
        """
        Extract text from DOCX file with normalization.
        
        Args:
            file_path: Path to DOCX file
        
        Returns:
            Extracted and normalized text
        
        Raises:
            DocumentProcessingError: If DOCX extraction fails
        """
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                if paragraph.text:
                    text += paragraph.text + "\n"
            
            if not text.strip():
                raise DocumentProcessingError(f"No text extracted from DOCX: {file_path}")
            
            # Normalize text
            normalized_text = DocumentProcessor.normalize_text(text)
            
            logger.info(f"DOCX file {file_path.name} - Length: {len(normalized_text)} chars")
            return normalized_text
            
        except Exception as e:
            logger.error(f"Failed to read DOCX file {file_path}: {e}", exc_info=True)
            raise DocumentProcessingError(f"Cannot process DOCX: {file_path}") from e
    
    @classmethod
    def extract_text(cls, file_path: Path) -> str:
        """
        Extract text from any supported file format.
        
        Args:
            file_path: Path to file (str or Path object)
        
        Returns:
            Extracted and normalized text
        
        Raises:
            DocumentProcessingError: If file cannot be processed
        """
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.txt':
            return cls.extract_text_from_txt(file_path)
        elif file_ext == '.pdf':
            return cls.extract_text_from_pdf(file_path)
        elif file_ext == '.docx':
            return cls.extract_text_from_docx(file_path)
        else:
            error_msg = f"Unsupported file format: {file_ext}. Supported: {', '.join(config.supported_formats)}"
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg)

