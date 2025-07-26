# enhanced_document_parser.py
import fitz  # PyMuPDF
import re
from collections import defaultdict, Counter
import numpy as np
from typing import List, Dict, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDocumentParser:
    def __init__(self):
        self.font_analysis = {}
        self.page_layouts = {}
        
    def parse_document_by_structure(self, pdf_path: str) -> List[Dict]:
        """
        Advanced PDF parsing with multiple extraction strategies
        """
        doc = fitz.open(pdf_path)
        doc_name = doc.name or pdf_path.split('/')[-1]
        
        # First pass: Analyze document structure
        self._analyze_document_structure(doc)
        
        # Second pass: Extract sections using multiple strategies
        toc_sections = self._extract_from_toc(doc, doc_name)
        structural_sections = self._extract_by_visual_structure(doc, doc_name)
        text_pattern_sections = self._extract_by_text_patterns(doc, doc_name)
        
        # Combine and deduplicate sections
        all_sections = self._merge_extraction_results(
            toc_sections, structural_sections, text_pattern_sections
        )
        
        doc.close()
        
        # Post-process and clean sections
        cleaned_sections = self._post_process_sections(all_sections)
        
        logger.info(f"Extracted {len(cleaned_sections)} sections from {doc_name}")
        return cleaned_sections
    
    def _analyze_document_structure(self, doc):
        """Analyze document to understand its structure patterns"""
        font_sizes = []
        font_families = []
        text_positions = []
        
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            page_fonts = []
            
            for block in blocks:
                if block.get('type') == 0:  # text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            if span.get("text", "").strip():
                                font_info = {
                                    'size': span.get('size', 12),
                                    'font': span.get('font', ''),
                                    'flags': span.get('flags', 0),
                                    'bbox': span.get('bbox', [0, 0, 0, 0]),
                                    'text': span.get('text', '').strip()
                                }
                                page_fonts.append(font_info)
                                font_sizes.append(font_info['size'])
                                font_families.append(font_info['font'])
                                text_positions.append(font_info['bbox'][1])  # y-position
            
            self.page_layouts[page_num] = page_fonts
        
        # Determine common patterns
        if font_sizes:
            self.font_analysis = {
                'body_size': self._get_body_font_size(font_sizes),
                'heading_threshold': self._calculate_heading_threshold(font_sizes),
                'common_fonts': Counter(font_families).most_common(3)
            }
    
    def _get_body_font_size(self, font_sizes: List[float]) -> float:
        """Determine the most common body text font size"""
        size_counter = Counter([round(size, 1) for size in font_sizes])
        return size_counter.most_common(1)[0][0]
    
    def _calculate_heading_threshold(self, font_sizes: List[float]) -> float:
        """Calculate threshold for heading detection"""
        body_size = self._get_body_font_size(font_sizes)
        return body_size + 1.5  # Headings typically 1.5+ pts larger
    
    def _extract_from_toc(self, doc, doc_name: str) -> List[Dict]:
        """Extract sections from PDF table of contents"""
        sections = []
        try:
            toc = doc.get_toc()
            if not toc:
                return sections
            
            for level, title, page_num in toc:
                # Find content for this section
                content = self._extract_section_content_from_toc(doc, title, page_num, toc)
                
                if content.strip():
                    sections.append({
                        'document': doc_name,
                        'page': page_num,
                        'title': title.strip(),
                        'content': content,
                        'extraction_method': 'toc',
                        'confidence': 0.9,
                        'level': level
                    })
        except Exception as e:
            logger.warning(f"TOC extraction failed: {e}")
        
        return sections
    
    def _extract_section_content_from_toc(self, doc, section_title: str, 
                                        start_page: int, toc: List) -> str:
        """Extract content between TOC sections"""
        # Find next section to determine end page
        end_page = None
        current_index = None
        
        for i, (level, title, page) in enumerate(toc):
            if title.strip() == section_title.strip():
                current_index = i
                break
        
        if current_index is not None and current_index + 1 < len(toc):
            end_page = toc[current_index + 1][2]
        else:
            end_page = len(doc)
        
        # Extract content from pages
        content_parts = []
        for page_num in range(start_page - 1, min(end_page, len(doc))):
            page = doc[page_num]
            page_text = page.get_text()
            content_parts.append(page_text)
        
        return " ".join(content_parts)
    
    def _extract_by_visual_structure(self, doc, doc_name: str) -> List[Dict]:
        """Extract sections based on visual formatting patterns"""
        sections = []
        current_section = None
        
        for page_num, page in enumerate(doc, 1):
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if block.get('type') != 0:  # Skip non-text blocks
                    continue
                
                # Analyze each line in the block
                for line in block.get("lines", []):
                    line_text = ""
                    line_props = []
                    
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            line_text += text + " "
                            line_props.append({
                                'size': span.get('size', 12),
                                'flags': span.get('flags', 0),
                                'bbox': span.get('bbox', [0, 0, 0, 0])
                            })
                    
                    line_text = line_text.strip()
                    if not line_text or len(line_text) < 3:
                        continue
                    
                    # Check if this line is a heading
                    is_heading = self._is_potential_heading(line_text, line_props)
                    
                    if is_heading:
                        # Save previous section
                        if current_section and current_section['content'].strip():
                            sections.append(current_section)
                        
                        # Start new section
                        current_section = {
                            'document': doc_name,
                            'page': page_num,
                            'title': line_text,
                            'content': "",
                            'extraction_method': 'visual',
                            'confidence': self._calculate_heading_confidence(line_text, line_props)
                        }
                    else:
                        # Add to current section content
                        if current_section:
                            current_section['content'] += " " + line_text
                        else:
                            # Create section for orphaned content
                            current_section = {
                                'document': doc_name,
                                'page': page_num,
                                'title': f"Page {page_num} Content",
                                'content': line_text,
                                'extraction_method': 'visual',
                                'confidence': 0.3
                            }
        
        # Add final section
        if current_section and current_section['content'].strip():
            sections.append(current_section)
        
        return sections
    
    def _is_potential_heading(self, text: str, props: List[Dict]) -> bool:
        """Determine if text line is likely a heading"""
        if not props:
            return False
        
        # Text characteristics
        word_count = len(text.split())
        has_ending_punct = text.endswith(('.', '!', '?'))
        is_title_case = text.istitle() or text.isupper()
        
        # Font characteristics
        avg_size = sum(p['size'] for p in props) / len(props)
        is_bold = any(p['flags'] & 16 for p in props)  # Bold flag
        
        # Heading heuristics
        size_check = avg_size > self.font_analysis.get('heading_threshold', 12)
        length_check = 2 <= word_count <= 15
        format_check = is_title_case or is_bold
        punct_check = not has_ending_punct or text.endswith(':')
        
        # Pattern matching for common heading patterns
        pattern_check = bool(re.match(r'^(\d+\.?\d*\s+|[A-Z]+\.\s+|Chapter\s+\d+|Section\s+\d+)', text, re.IGNORECASE))
        
        return (size_check or format_check or pattern_check) and length_check and punct_check
    
    def _calculate_heading_confidence(self, text: str, props: List[Dict]) -> float:
        """Calculate confidence score for heading detection"""
        confidence = 0.5
        
        if not props:
            return confidence
        
        # Size factor
        avg_size = sum(p['size'] for p in props) / len(props)
        if avg_size > self.font_analysis.get('heading_threshold', 12):
            confidence += 0.2
        
        # Bold factor
        if any(p['flags'] & 16 for p in props):
            confidence += 0.15
        
        # Pattern factor
        if re.match(r'^(\d+\.?\d*\s+|[A-Z]+\.\s+|Chapter\s+\d+|Section\s+\d+)', text, re.IGNORECASE):
            confidence += 0.2
        
        # Format factor
        if text.istitle() or text.isupper():
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _extract_by_text_patterns(self, doc, doc_name: str) -> List[Dict]:
        """Extract sections using text pattern recognition"""
        sections = []
        
        # Common heading patterns
        heading_patterns = [
            r'^(\d+\.?\d*\.?\d*)\s+(.+)$',  # Numbered headings
            r'^([A-Z]{1,3}\.)\s+(.+)$',     # Letter headings
            r'^(Chapter|Section|Part)\s+(\d+):?\s*(.*)$',  # Named sections
            r'^([A-Z\s]{3,30})$',           # ALL CAPS headings
            r'^(.{3,50}):$',                # Colon-terminated headings
        ]
        
        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            current_content = []
            
            for line in lines:
                is_heading = False
                
                for pattern in heading_patterns:
                    if re.match(pattern, line, re.IGNORECASE):
                        # Save previous section
                        if current_content:
                            content_text = " ".join(current_content)
                            if len(content_text.strip()) > 50:  # Minimum content length
                                sections.append({
                                    'document': doc_name,
                                    'page': page_num,
                                    'title': f"Content Section - Page {page_num}",
                                    'content': content_text,
                                    'extraction_method': 'pattern',
                                    'confidence': 0.6
                                })
                        
                        # Start new section
                        current_content = []
                        sections.append({
                            'document': doc_name,
                            'page': page_num,
                            'title': line,
                            'content': "",
                            'extraction_method': 'pattern',
                            'confidence': 0.7
                        })
                        is_heading = True
                        break
                
                if not is_heading:
                    current_content.append(line)
            
            # Add remaining content
            if current_content:
                content_text = " ".join(current_content)
                if len(content_text.strip()) > 50:
                    sections.append({
                        'document': doc_name,
                        'page': page_num,
                        'title': f"Content Section - Page {page_num}",
                        'content': content_text,
                        'extraction_method': 'pattern',
                        'confidence': 0.5
                    })
        
        return sections
    
    def _merge_extraction_results(self, *section_lists) -> List[Dict]:
        """Merge and deduplicate sections from different extraction methods"""
        all_sections = []
        for section_list in section_lists:
            all_sections.extend(section_list)
        
        # Sort by confidence and remove duplicates
        all_sections.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Deduplicate based on content similarity
        unique_sections = []
        for section in all_sections:
            is_duplicate = False
            for existing in unique_sections:
                if self._are_sections_similar(section, existing):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_sections.append(section)
        
        return unique_sections
    
    def _are_sections_similar(self, section1: Dict, section2: Dict) -> bool:
        """Check if two sections are similar (potential duplicates)"""
        # Same page and similar titles
        if (section1['page'] == section2['page'] and 
            self._text_similarity(section1['title'], section2['title']) > 0.8):
            return True
        
        # Similar content
        content_sim = self._text_similarity(
            section1['content'][:200], 
            section2['content'][:200]
        )
        return content_sim > 0.9
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _post_process_sections(self, sections: List[Dict]) -> List[Dict]:
        """Clean and validate extracted sections"""
        cleaned_sections = []
        
        for section in sections:
            # Clean content
            content = re.sub(r'\s+', ' ', section['content']).strip()
            title = re.sub(r'\s+', ' ', section['title']).strip()
            
            # Validation checks
            if len(content) < 30:  # Minimum content length
                continue
            
            if len(title) > 200:  # Maximum title length
                title = title[:200] + "..."
            
            # Remove sections with too much repeated content
            words = content.split()
            if len(set(words)) / len(words) < 0.3:  # Less than 30% unique words
                continue
            
            cleaned_section = {
                'document': section['document'],
                'page': section['page'],
                'title': title,
                'content': content
            }
            
            cleaned_sections.append(cleaned_section)
        
        return cleaned_sections