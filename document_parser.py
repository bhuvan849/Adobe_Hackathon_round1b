import fitz  # PyMuPDF
import re
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

class EnhancedDocumentParser:
    def __init__(self):
        # Enhanced heading patterns with better coverage
        self.heading_patterns = [
            # Traditional numbered headings
            r'^\d+\.?\s+[A-Z][^.!?]*$',
            r'^[IVX]+\.?\s+[A-Z][^.!?]*$',
            r'^[A-Z]\.?\s+[A-Z][^.!?]*$',
            
            # Chapter/Section patterns
            r'^(Chapter|Section|Part)\s+\d+[:\-\s].*$',
            r'^(CHAPTER|SECTION|PART)\s+\d+[:\-\s].*$',
            
            # ALL CAPS headings
            r'^[A-Z][A-Z\s\-]{8,}[A-Z]$',
            
            # Title Case headings
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*[:\-]?$',
            
            # Special patterns for academic/technical documents
            r'^(Abstract|Introduction|Methodology|Results|Discussion|Conclusion|References)$',
            r'^(ABSTRACT|INTRODUCTION|METHODOLOGY|RESULTS|DISCUSSION|CONCLUSION|REFERENCES)$',
            
            # Recipe/ingredients patterns
            r'^.*Ingredients?:?\s*$',
            r'^.*Instructions?:?\s*$',
            r'^.*Preparation:?\s*$',
            
            # Question patterns
            r'^[A-Z][^.!?]*\?$',
            
            # Colon-terminated headings
            r'^[A-Z][^:]*:$',
        ]
        
        # Enhanced noise patterns
        self.noise_patterns = [
            r'^\s*\d+\s*$',  # Page numbers
            r'^(Figure|Table|Chart|Graph)\s+\d+',
            r'^(Fig\.|Tab\.)\s*\d+',
            r'^(Source:|Note:|Copyright:|©)',
            r'^\s*[•\-\*\+]\s*$',  # Bullet points only
            r'^\s*(continued|cont\.|\.\.\.)\s*$',
            r'^\s*Page\s+\d+\s*$',
            r'^\s*\d+\s*of\s*\d+\s*$',
            r'^https?://',  # URLs
            r'^www\.',
            r'^\s*[()[\]{}]\s*$',  # Brackets only
            r'^\s*[-=_]{3,}\s*$',  # Dividers
        ]
        
        # Common stop words that shouldn't be headings
        self.stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had'
        }

    def parse_document_by_structure(self, pdf_path: str) -> List[Dict]:
        """Main parsing method with multiple fallback strategies."""
        try:
            doc = fitz.open(pdf_path)
            name = self._extract_filename(pdf_path)
            logger.info(f"Parsing document: {name} ({len(doc)} pages)")

            # Strategy 1: Extract from bookmarks/TOC
            sections = self._extract_from_bookmarks(doc, name)
            
            # Strategy 2: Advanced visual analysis if bookmarks insufficient
            if not sections or len(sections) < 3:
                logger.info("Bookmark extraction insufficient, using advanced visual analysis")
                sections = self._extract_by_advanced_visual_analysis(doc, name)
                
            # Strategy 3: Content-based extraction as fallback
            if len(sections) < 5:
                logger.info("Visual analysis insufficient, using content-based extraction")
                content_sections = self._extract_by_content_analysis(doc, name)
                sections.extend(content_sections)

            doc.close()
            
            # Post-process and validate
            sections = self._post_process_and_validate(sections)
            logger.info(f"Final extraction: {len(sections)} valid sections from {name}")
            
            return sections

        except Exception as e:
            logger.error(f"Error parsing {pdf_path}: {e}")
            return []

    def _extract_from_bookmarks(self, doc, name: str) -> List[Dict]:
        """Extract sections from PDF bookmarks/TOC."""
        sections = []
        try:
            toc = doc.get_toc(simple=False)  # Get full TOC with hierarchy
            
            for entry in toc:
                level, title, page_num = entry
                if page_num < len(doc):
                    # Get content from this page and potentially next few pages
                    content = self._extract_section_content(doc, page_num, title, toc)
                    
                    sections.append({
                        'document': name,
                        'title': title.strip(),
                        'page': page_num + 1,  # Convert to 1-based
                        'content': content.strip(),
                        'extraction_method': 'bookmark',
                        'confidence': 0.95
                    })
        except Exception as e:
            logger.warning(f"Bookmark extraction failed: {e}")
            
        return sections

    def _extract_section_content(self, doc, start_page: int, title: str, toc: List) -> str:
        """Extract content for a section based on TOC information."""
        # Find the next section to determine where this section ends
        end_page = len(doc)
        for entry in toc:
            _, _, page_num = entry
            if page_num > start_page:
                end_page = page_num
                break
        
        # Limit section to reasonable size (max 3 pages)
        end_page = min(end_page, start_page + 3)
        
        content_parts = []
        for page_idx in range(start_page, end_page):
            if page_idx < len(doc):
                page_text = doc[page_idx].get_text()
                # Remove the title from content if it appears at the beginning
                if page_idx == start_page:
                    lines = page_text.split('\n')
                    filtered_lines = []
                    title_found = False
                    for line in lines:
                        if not title_found and title.lower() in line.lower():
                            title_found = True
                            continue
                        if title_found:
                            filtered_lines.append(line)
                    page_text = '\n'.join(filtered_lines)
                
                content_parts.append(page_text)
        
        return '\n'.join(content_parts)

    def _extract_by_advanced_visual_analysis(self, doc, name: str) -> List[Dict]:
        """Extract sections using advanced visual analysis of PDF structure."""
        all_sections = []
        
        for page_idx, page in enumerate(doc):
            try:
                sections = self._analyze_page_structure(page, name, page_idx + 1)
                all_sections.extend(sections)
            except Exception as e:
                logger.warning(f"Error analyzing page {page_idx + 1}: {e}")
                continue
        
        # Merge sections that span multiple pages
        merged_sections = self._merge_multi_page_sections(all_sections)
        return merged_sections

    def _analyze_page_structure(self, page, doc_name: str, page_num: int) -> List[Dict]:
        """Analyze the structure of a single page."""
        try:
            # Get text blocks with formatting information
            blocks = page.get_text("dict").get("blocks", [])
            if not blocks:
                return []
            
            # Analyze font characteristics across the page
            font_analysis = self._analyze_page_fonts(blocks)
            
            # Detect structural elements
            elements = self._detect_structural_elements(blocks, font_analysis)
            
            # Extract sections from detected structure
            sections = self._extract_sections_from_elements(elements, blocks, doc_name, page_num)
            
            return sections
            
        except Exception as e:
            logger.warning(f"Error in page structure analysis: {e}")
            return []

    def _analyze_page_fonts(self, blocks) -> Dict:
        """Analyze font characteristics to identify heading patterns."""
        font_sizes = []
        font_flags = []
        
        for block in blocks:
            if block.get('type') != 0:  # Only text blocks
                continue
                
            for line in block.get('lines', []):
                for span in line.get('spans', []):
                    size = span.get('size', 0)
                    flags = span.get('flags', 0)
                    
                    if size > 0:  # Valid font size
                        font_sizes.append(size)
                        font_flags.append(flags)
        
        if not font_sizes:
            return {
                'body_size': 12,
                'heading_threshold': 14,
                'large_heading_threshold': 16,
                'bold_flag_common': False
            }
        
        font_array = np.array(font_sizes)
        
        # Calculate thresholds
        body_size = np.percentile(font_array, 40)  # Most common text size
        heading_threshold = np.percentile(font_array, 75)  # Larger text
        large_heading_threshold = np.percentile(font_array, 90)  # Very large text
        
        # Check if bold is commonly used
        bold_count = sum(1 for flags in font_flags if flags & 16)  # Bold flag
        bold_ratio = bold_count / len(font_flags) if font_flags else 0
        
        return {
            'body_size': body_size,
            'heading_threshold': heading_threshold,
            'large_heading_threshold': large_heading_threshold,
            'bold_flag_common': bold_ratio > 0.3
        }

    def _detect_structural_elements(self, blocks, font_analysis) -> List[Dict]:
        """Detect headings and other structural elements."""
        elements = []
        
        for block_idx, block in enumerate(blocks):
            if block.get('type') != 0:  # Only text blocks
                continue
            
            # Extract text and formatting info
            text_info = self._extract_block_text_info(block)
            if not text_info['text'] or self._is_noise(text_info['text']):
                continue
            
            # Determine if this is a heading
            is_heading, confidence, level = self._classify_heading(
                text_info, font_analysis, block_idx, blocks
            )
            
            element = {
                'type': 'heading' if is_heading else 'paragraph',
                'text': text_info['text'],
                'level': level if is_heading else None,
                'confidence': confidence,
                'block_index': block_idx,
                'font_info': text_info
            }
            
            elements.append(element)
        
        return elements

    def _extract_block_text_info(self, block) -> Dict:
        """Extract text and formatting information from a block."""
        text_parts = []
        font_sizes = []
        bold_spans = 0
        total_spans = 0
        
        for line in block.get('lines', []):
            for span in line.get('spans', []):
                span_text = span.get('text', '').strip()
                if span_text:
                    text_parts.append(span_text)
                    font_sizes.append(span.get('size', 12))
                    
                    # Check for bold
                    if span.get('flags', 0) & 16:  # Bold flag
                        bold_spans += 1
                    total_spans += 1
        
        combined_text = ' '.join(text_parts).strip()
        avg_font_size = np.mean(font_sizes) if font_sizes else 12
        bold_ratio = bold_spans / total_spans if total_spans > 0 else 0
        
        return {
            'text': combined_text,
            'avg_font_size': avg_font_size,
            'bold_ratio': bold_ratio,
            'word_count': len(combined_text.split())
        }

    def _classify_heading(self, text_info: Dict, font_analysis: Dict, 
                         block_idx: int, blocks: List) -> Tuple[bool, float, Optional[int]]:
        """Classify whether a text block is a heading."""
        text = text_info['text']
        font_size = text_info['avg_font_size']
        bold_ratio = text_info['bold_ratio']
        word_count = text_info['word_count']
        
        confidence = 0.0
        
        # Rule 1: Pattern matching
        pattern_match = any(re.match(pattern, text, re.IGNORECASE) 
                           for pattern in self.heading_patterns)
        if pattern_match:
            confidence += 0.4
        
        # Rule 2: Font size analysis
        if font_size > font_analysis['large_heading_threshold']:
            confidence += 0.3
        elif font_size > font_analysis['heading_threshold']:
            confidence += 0.2
        
        # Rule 3: Bold formatting
        if bold_ratio > 0.7:
            confidence += 0.2
        elif bold_ratio > 0.3:
            confidence += 0.1
        
        # Rule 4: Length constraints (headings are usually shorter)
        if word_count <= 10:
            confidence += 0.1
        elif word_count <= 20:
            confidence += 0.05
        else:
            confidence -= 0.1  # Long text less likely to be heading
        
        # Rule 5: Position on page (headings often at top or after whitespace)
        if block_idx < 3:  # Near top of page
            confidence += 0.1
        
        # Rule 6: Capitalization patterns
        if text.isupper() and len(text) > 5:
            confidence += 0.15
        elif text.istitle():
            confidence += 0.1
        
        # Rule 7: Ends with colon (common heading pattern)
        if text.endswith(':'):
            confidence += 0.15
        
        # Rule 8: Avoid common false positives
        first_word = text.split()[0].lower() if text.split() else ""
        if first_word in self.stop_words:
            confidence -= 0.1
        
        # Determine heading level based on font size and formatting
        level = None
        if confidence > 0.5:  # Threshold for being a heading
            if font_size > font_analysis['large_heading_threshold'] or bold_ratio > 0.8:
                level = 1
            elif font_size > font_analysis['heading_threshold'] or bold_ratio > 0.5:
                level = 2
            else:
                level = 3
        
        return confidence > 0.5, confidence, level

    def _extract_sections_from_elements(self, elements: List[Dict], blocks: List, 
                                      doc_name: str, page_num: int) -> List[Dict]:
        """Extract sections from detected structural elements."""
        sections = []
        current_section = None
        
        for element in elements:
            if element['type'] == 'heading' and element['confidence'] > 0.5:
                # Save previous section if exists
                if current_section:
                    content = self._gather_section_content(
                        blocks, current_section['start_block'], element['block_index']
                    )
                    current_section['content'] = content
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    'document': doc_name,
                    'title': element['text'],
                    'page': page_num,
                    'start_block': element['block_index'],
                    'extraction_method': 'visual_analysis',
                    'confidence': element['confidence']
                }
        
        # Handle last section
        if current_section:
            content = self._gather_section_content(
                blocks, current_section['start_block'], len(blocks)
            )
            current_section['content'] = content
            sections.append(current_section)
        
        return sections

    def _gather_section_content(self, blocks: List, start_idx: int, end_idx: int) -> str:
        """Gather content between two block indices."""
        content_parts = []
        
        for i in range(start_idx + 1, end_idx):  # Skip the heading block itself
            if i >= len(blocks):
                break
                
            block = blocks[i]
            if block.get('type') != 0:  # Only text blocks
                continue
            
            block_text = ""
            for line in block.get('lines', []):
                for span in line.get('spans', []):
                    span_text = span.get('text', '').strip()
                    if span_text:
                        block_text += span_text + " "
            
            if block_text.strip():
                content_parts.append(block_text.strip())
        
        return ' '.join(content_parts)

    def _extract_by_content_analysis(self, doc, name: str) -> List[Dict]:
        """Content-based extraction as final fallback."""
        sections = []
        
        for page_idx, page in enumerate(doc):
            try:
                text = page.get_text().strip()
                if not text:
                    continue
                
                # Split into paragraphs
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                
                for para_idx, paragraph in enumerate(paragraphs):
                    if len(paragraph) > 100:  # Substantial content
                        # Use first sentence or first 50 chars as title
                        sentences = paragraph.split('. ')
                        title = sentences[0][:50] + "..." if len(sentences[0]) > 50 else sentences[0]
                        
                        sections.append({
                            'document': name,
                            'title': title,
                            'page': page_idx + 1,
                            'content': paragraph,
                            'extraction_method': 'content_analysis',
                            'confidence': 0.3
                        })
                        
            except Exception as e:
                logger.warning(f"Error in content analysis for page {page_idx + 1}: {e}")
                continue
        
        return sections

    def _merge_multi_page_sections(self, sections: List[Dict]) -> List[Dict]:
        """Merge sections that might span multiple pages."""
        if not sections:
            return sections
        
        merged = []
        current_group = [sections[0]]
        
        for i in range(1, len(sections)):
            current = sections[i]
            previous = current_group[-1]
            
            # Check if sections should be merged
            should_merge = (
                current['document'] == previous['document'] and
                current['page'] == previous['page'] + 1 and
                len(previous['content']) < 200 and  # Previous section is short
                current['extraction_method'] == previous['extraction_method']
            )
            
            if should_merge:
                current_group.append(current)
            else:
                # Finalize current group
                if len(current_group) == 1:
                    merged.append(current_group[0])
                else:
                    # Merge the group
                    merged_section = self._merge_section_group(current_group)
                    merged.append(merged_section)
                
                current_group = [current]
        
        # Handle last group
        if len(current_group) == 1:
            merged.append(current_group[0])
        else:
            merged_section = self._merge_section_group(current_group)
            merged.append(merged_section)
        
        return merged

    def _merge_section_group(self, group: List[Dict]) -> Dict:
        """Merge a group of related sections."""
        if not group:
            return {}
        
        first = group[0]
        last = group[-1]
        
        # Combine titles
        titles = [section['title'] for section in group]
        combined_title = ' - '.join(titles[:2])  # Use first two titles
        
        # Combine content
        combined_content = ' '.join(section['content'] for section in group)
        
        return {
            'document': first['document'],
            'title': combined_title,
            'page': first['page'],
            'content': combined_content,
            'extraction_method': first['extraction_method'],
            'confidence': np.mean([s['confidence'] for s in group])
        }

    def _is_noise(self, text: str) -> bool:
        """Check if text is likely noise/unwanted content."""
        if not text or len(text.strip()) < 3:
            return True
        
        for pattern in self.noise_patterns:
            if re.match(pattern, text.strip(), re.IGNORECASE):
                return True
        
        # Additional noise detection
        words = text.split()
        if len(words) == 1 and (words[0].isdigit() or len(words[0]) < 3):
            return True
        
        return False

    def _post_process_and_validate(self, sections: List[Dict]) -> List[Dict]:
        """Post-process and validate extracted sections."""
        valid_sections = []
        seen_combinations = set()
        
        for section in sections:
            # Basic validation
            if not section.get('title') or not section.get('content'):
                continue
            
            title = section['title'].strip()
            content = section['content'].strip()
            
            # Skip too short content
            if len(content) < 30:
                continue
            
            # Skip duplicates
            combo = (section['document'], title, section['page'])
            if combo in seen_combinations:
                continue
            seen_combinations.add(combo)
            
            # Clean up title
            title = re.sub(r'\s+', ' ', title)
            title = title[:200]  # Limit title length
            
            # Clean up content
            content = re.sub(r'\s+', ' ', content)
            content = content[:5000]  # Limit content length
            
            section['title'] = title
            section['content'] = content
            
            valid_sections.append(section)
        
        # Sort by document, then by page, then by confidence
        valid_sections.sort(key=lambda x: (
            x['document'], 
            x['page'], 
            -x.get('confidence', 0)
        ))
        
        logger.info(f"Post-processing: {len(valid_sections)} valid sections")
        return valid_sections

    def _extract_filename(self, path: str) -> str:
        """Extract filename from path."""
        return path.split('/')[-1].replace('.pdf', '')