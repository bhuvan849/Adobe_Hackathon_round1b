# document_parser.py
import fitz  # PyMuPDF
from collections import defaultdict
import re

def parse_document_by_structure(pdf_path):
    """
    Parses a PDF by its structural hierarchy with improved text extraction.
    """
    doc = fitz.open(pdf_path)
    sections = []
    
    for page_num, page in enumerate(doc, 1):
        # Get text blocks with formatting information
        blocks = page.get_text("dict")["blocks"]
        if not blocks:
            continue

        # Collect font information for body text detection
        font_info = []
        for block in blocks:
            if block.get('type') == 0:  # text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("text", "").strip():
                            font_info.append({
                                'size': span.get('size', 12),
                                'flags': span.get('flags', 0),
                                'text': span.get('text', '').strip()
                            })
        
        if not font_info:
            continue
            
        # Determine body text size (most common font size)
        font_sizes = [info['size'] for info in font_info]
        body_size = max(set(font_sizes), key=font_sizes.count) if font_sizes else 12

        # Extract potential headings and content
        current_sections = []
        
        for block in blocks:
            if block.get('type') == 0:  # text block
                block_text = ""
                is_heading = False
                
                for line in block.get("lines", []):
                    line_text = ""
                    line_sizes = []
                    
                    for span in line.get("spans", []):
                        span_text = span.get("text", "").strip()
                        if span_text:
                            line_text += span_text + " "
                            line_sizes.append(span.get('size', 12))
                    
                    if line_text.strip():
                        # Check if this line could be a heading
                        avg_size = sum(line_sizes) / len(line_sizes) if line_sizes else body_size
                        is_bold = any(span.get('flags', 0) & 2**4 for span in line.get("spans", []))
                        
                        # Heading heuristics
                        if (avg_size > body_size * 1.1 or is_bold) and len(line_text.split()) < 15:
                            if current_sections and current_sections[-1].get('content'):
                                # Save previous section
                                sections.append(current_sections[-1])
                            
                            # Start new section
                            current_sections.append({
                                "document": doc.name,
                                "page": page_num,
                                "title": line_text.strip(),
                                "content": ""
                            })
                            is_heading = True
                        else:
                            block_text += line_text
                
                # Add content to current section
                if not is_heading and block_text.strip():
                    if current_sections:
                        current_sections[-1]['content'] += " " + block_text.strip()
                    else:
                        # No heading found, create a generic section
                        current_sections.append({
                            "document": doc.name,
                            "page": page_num,
                            "title": f"Page {page_num} Content",
                            "content": block_text.strip()
                        })
        
        # Add remaining sections
        for section in current_sections:
            if section.get('content', '').strip():
                sections.append(section)
    
    doc.close()
    
    # Clean up sections
    cleaned_sections = []
    for section in sections:
        content = re.sub(r'\s+', ' ', section['content']).strip()
        if content and len(content) > 20:  # Filter out very short sections
            section['content'] = content
            cleaned_sections.append(section)
    
    return cleaned_sections