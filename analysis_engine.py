# advanced_analysis_engine.py
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
from typing import List, Dict, Any, Tuple
from collections import Counter
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedAnalysisEngine:
    def __init__(self, model_name='all-mpnet-base-v2'):
        """Initialize with multiple analysis models"""
        try:
            self.sentence_model = SentenceTransformer(model_name)
            logger.info(f"Loaded sentence transformer: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load {model_name}, using fallback")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize TF-IDF for keyword-based scoring
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.95
        )
        
        # Try to load spaCy model for NER and linguistic analysis
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.use_spacy = True
            logger.info("Loaded spaCy model for advanced text analysis")
        except OSError:
            logger.warning("spaCy model not available, using basic text processing")
            self.nlp = None
            self.use_spacy = False
    
    def get_ranked_sections(self, persona: str, job_to_be_done: str, 
                          all_sections: List[Dict]) -> List[Dict]:
        """
        Advanced relevance ranking using multiple scoring methods
        """
        if not all_sections:
            return []
        
        # Create comprehensive query
        query = self._create_enhanced_query(persona, job_to_be_done)
        
        # Extract persona keywords and job keywords
        persona_keywords = self._extract_keywords(persona)
        job_keywords = self._extract_keywords(job_to_be_done)
        
        logger.info(f"Ranking {len(all_sections)} sections for persona: {persona}")
        logger.info(f"Job to be done: {job_to_be_done}")
        
        # Score sections using multiple methods
        scored_sections = []
        
        for i, section in enumerate(all_sections):
            scores = self._calculate_multi_dimensional_score(
                query, section, persona_keywords, job_keywords
            )
            
            section_copy = section.copy()
            section_copy.update(scores)
            section_copy['section_id'] = i
            scored_sections.append(section_copy)
        
        # Sort by composite score
        scored_sections.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Assign importance ranks
        for i, section in enumerate(scored_sections):
            section['importance_rank'] = i + 1
        
        # Log top sections for debugging
        logger.info("Top 5 sections by relevance:")
        for i, section in enumerate(scored_sections[:5]):
            logger.info(f"{i+1}. {section['title'][:50]}... (Score: {section['composite_score']:.3f})")
        
        return scored_sections
    
    def _create_enhanced_query(self, persona: str, job_to_be_done: str) -> str:
        """Create a comprehensive query for relevance matching"""
        # Extract key components
        persona_role = self._extract_role(persona)
        persona_domain = self._extract_domain(persona)
        task_verbs = self._extract_task_verbs(job_to_be_done)
        task_objects = self._extract_task_objects(job_to_be_done)
        
        # Build weighted query
        query_parts = [
            f"Role: {persona_role}",
            f"Domain expertise: {persona_domain}",
            f"Task: {job_to_be_done}",
            f"Action words: {' '.join(task_verbs)}",
            f"Focus areas: {' '.join(task_objects)}",
            f"Professional context: {persona}",
            job_to_be_done,  # Original task for reference
        ]
        
        return " ".join(query_parts)
    
    def _extract_role(self, persona: str) -> str:
        """Extract role/profession from persona"""
        role_patterns = [
            r'(researcher|analyst|student|professor|engineer|developer|manager|consultant)',
            r'(phd|undergraduate|graduate|senior|junior|lead|principal)',
            r'(doctor|scientist|specialist|expert|practitioner)'
        ]
        
        roles = []
        for pattern in role_patterns:
            matches = re.findall(pattern, persona.lower())
            roles.extend(matches)
        
        return " ".join(set(roles)) if roles else "professional"
    
    def _extract_domain(self, persona: str) -> str:
        """Extract domain/field from persona"""
        domain_patterns = [
            r'(biology|chemistry|physics|mathematics|computer science|finance|medicine)',
            r'(business|marketing|sales|legal|engineering|research)',
            r'(data science|machine learning|artificial intelligence|biotechnology)'
        ]
        
        domains = []
        for pattern in domain_patterns:
            matches = re.findall(pattern, persona.lower())
            domains.extend(matches)
        
        return " ".join(set(domains)) if domains else "general"
    
    def _extract_task_verbs(self, job_to_be_done: str) -> List[str]:
        """Extract action verbs from job description"""
        action_verbs = [
            'analyze', 'review', 'summarize', 'identify', 'extract', 'compare',
            'evaluate', 'assess', 'investigate', 'research', 'study', 'examine',
            'prepare', 'create', 'develop', 'build', 'design', 'implement',
            'find', 'discover', 'determine', 'calculate', 'measure', 'quantify'
        ]
        
        found_verbs = []
        job_lower = job_to_be_done.lower()
        for verb in action_verbs:
            if verb in job_lower:
                found_verbs.append(verb)
        
        return found_verbs
    
    def _extract_task_objects(self, job_to_be_done: str) -> List[str]:
        """Extract key objects/topics from job description"""
        if self.use_spacy and self.nlp:
            doc = self.nlp(job_to_be_done)
            entities = [ent.text.lower() for ent in doc.ents 
                       if ent.label_ in ['ORG', 'PRODUCT', 'EVENT', 'WORK_OF_ART']]
            nouns = [token.lemma_.lower() for token in doc 
                    if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 2]
            return list(set(entities + nouns))
        else:
            # Fallback: simple noun extraction
            words = re.findall(r'\b[a-zA-Z]+\b', job_to_be_done.lower())
            return [word for word in words if len(word) > 3]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key terms from text"""
        if not text:
            return []
        
        # Remove common stop words and extract meaningful terms
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return list(set(keywords))
    
    def _calculate_multi_dimensional_score(self, query: str, section: Dict, 
                                         persona_keywords: List[str], 
                                         job_keywords: List[str]) -> Dict:
        """Calculate comprehensive relevance score using multiple methods"""
        section_text = f"{section.get('title', '')} {section.get('content', '')}"
        
        # 1. Semantic similarity using sentence transformers
        semantic_score = self._calculate_semantic_score(query, section_text)
        
        # 2. Keyword-based relevance
        keyword_score = self._calculate_keyword_score(
            section_text, persona_keywords + job_keywords
        )
        
        # 3. Title relevance (titles are often more indicative)
        title_score = self._calculate_title_relevance(
            section.get('title', ''), persona_keywords + job_keywords
        )
        
        # 4. Content quality and informativeness
        quality_score = self._calculate_content_quality(section_text)
        
        # 5. Domain-specific scoring
        domain_score = self._calculate_domain_relevance(section_text, query)
        
        # 6. Section structure bonus (well-structured content is often more valuable)
        structure_score = self._calculate_structure_score(section)
        
        # Weighted composite score
        weights = {
            'semantic': 0.35,
            'keyword': 0.25,
            'title': 0.15,
            'quality': 0.10,
            'domain': 0.10,
            'structure': 0.05
        }
        
        composite_score = (
            weights['semantic'] * semantic_score +
            weights['keyword'] * keyword_score +
            weights['title'] * title_score +
            weights['quality'] * quality_score +
            weights['domain'] * domain_score +
            weights['structure'] * structure_score
        )
        
        return {
            'semantic_score': semantic_score,
            'keyword_score': keyword_score,
            'title_score': title_score,
            'quality_score': quality_score,
            'domain_score': domain_score,
            'structure_score': structure_score,
            'composite_score': composite_score
        }
    
    def _calculate_semantic_score(self, query: str, section_text: str) -> float:
        """Calculate semantic similarity using sentence transformers"""
        try:
            query_embedding = self.sentence_model.encode(query, convert_to_tensor=True,show_progress_bar=False)
            section_embedding = self.sentence_model.encode(section_text, convert_to_tensor=True,show_progress_bar=False)
            similarity = util.cos_sim(query_embedding, section_embedding)[0][0]
            return float(similarity)
        except Exception as e:
            logger.warning(f"Semantic scoring failed: {e}")
            return 0.0
    
    def _calculate_keyword_score(self, section_text: str, keywords: List[str]) -> float:
        """Calculate keyword-based relevance score"""
        if not keywords:
            return 0.0
        
        section_lower = section_text.lower()
        total_matches = 0
        total_importance = 0
        
        for keyword in keywords:
            # Count occurrences
            count = section_lower.count(keyword.lower())
            if count > 0:
                # Weight by inverse frequency (rare words are more important)
                importance = 1.0 / (section_lower.count(keyword.lower()) + 1)
                total_matches += count * importance
                total_importance += importance
        
        return min(total_matches / max(len(keywords), 1), 1.0)
    
    def _calculate_title_relevance(self, title: str, keywords: List[str]) -> float:
        """Calculate how relevant the section title is"""
        if not title or not keywords:
            return 0.0
        
        title_lower = title.lower()
        matches = sum(1 for keyword in keywords if keyword.lower() in title_lower)
        return matches / len(keywords)
    
    def _calculate_content_quality(self, section_text: str) -> float:
        """Assess content quality and informativeness"""
        if not section_text:
            return 0.0
        
        words = section_text.split()
        if len(words) < 10:
            return 0.2
        
        # Factors for quality assessment
        word_count = len(words)
        unique_words = len(set(word.lower() for word in words))
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Quality indicators
        has_numbers = bool(re.search(r'\d+', section_text))
        has_technical_terms = bool(re.search(r'[A-Z][a-z]+[A-Z]', section_text))
        sentence_count = len(re.split(r'[.!?]+', section_text))
        
        # Normalize scores
        length_score = min(word_count / 200, 1.0)  # Prefer substantial content
        diversity_score = unique_words / word_count if word_count > 0 else 0
        complexity_score = min(avg_word_length / 6, 1.0)  # Prefer complex vocabulary
        
        bonus = 0.1 * (has_numbers + has_technical_terms)
        
        return min(length_score * 0.4 + diversity_score * 0.3 + 
                  complexity_score * 0.3 + bonus, 1.0)
    
    def _calculate_domain_relevance(self, section_text: str, query: str) -> float:
        """Calculate domain-specific relevance"""
        # Domain-specific terms that indicate relevance
        domain_indicators = {
            'research': ['study', 'analysis', 'findings', 'methodology', 'results', 'conclusion'],
            'business': ['revenue', 'profit', 'market', 'strategy', 'performance', 'investment'],
            'technical': ['algorithm', 'method', 'implementation', 'system', 'model', 'framework'],
            'academic': ['theory', 'concept', 'principle', 'hypothesis', 'evidence', 'literature'],
            'medical': ['patient', 'treatment', 'diagnosis', 'clinical', 'therapeutic', 'medical'],
            'financial': ['financial', 'economic', 'fiscal', 'monetary', 'budget', 'cost']
        }
        
        section_lower = section_text.lower()
        query_lower = query.lower()
        
        max_relevance = 0.0
        for domain, indicators in domain_indicators.items():
            if domain in query_lower:
                domain_score = sum(1 for indicator in indicators if indicator in section_lower)
                relevance = domain_score / len(indicators)
                max_relevance = max(max_relevance, relevance)
        
        return max_relevance
    
    def _calculate_structure_score(self, section: Dict) -> float:
        """Calculate bonus for well-structured sections"""
        title = section.get('title', '')
        content = section.get('content', '')
        
        score = 0.0
        
        # Title quality
        if title and 5 <= len(title.split()) <= 15:
            score += 0.3
        
        # Content structure indicators
        if content:
            # Has bullet points or numbered lists
            if re.search(r'[•\-\*]\s+|^\d+[\.\)]\s+', content, re.MULTILINE):
                score += 0.2
            
            # Has proper paragraphs
            paragraphs = content.split('\n\n')
            if len(paragraphs) > 1:
                score += 0.2
            
            # Has concluding statements
            if re.search(r'(conclusion|summary|therefore|thus|finally)', content.lower()):
                score += 0.3
        
        return min(score, 1.0)
    
    def get_refined_summary(self, query: str, section_content: str, 
                          num_sentences: int = 3) -> str:
        """
        Advanced extractive summary with context-aware sentence selection
        """
        if not section_content or not section_content.strip():
            return ""
        
        sentences = self._advanced_sentence_split(section_content)
        
        if len(sentences) <= num_sentences:
            return section_content.strip()
        
        # Score sentences using multiple criteria
        sentence_scores = []
        query_embedding = self.sentence_model.encode(query, convert_to_tensor=True, show_progress_bar=False)

        for i, sentence in enumerate(sentences):
            score = self._calculate_sentence_score(
                sentence, query, query_embedding, i, len(sentences)
            )
            sentence_scores.append((i, sentence, score))
        
        # Sort by score and select top sentences
        sentence_scores.sort(key=lambda x: x[2], reverse=True)
        selected_indices = [x[0] for x in sentence_scores[:num_sentences]]
        selected_indices.sort()  # Maintain original order
        
        # Create coherent summary
        summary_sentences = [sentences[i] for i in selected_indices]
        summary = self._create_coherent_summary(summary_sentences)
        
        return summary
    
    def _advanced_sentence_split(self, text: str) -> List[str]:
        """Advanced sentence splitting with better handling of edge cases"""
        # First, protect abbreviations and numbers
        text = re.sub(r'\b([A-Z][a-z]{1,2}\.)', r'\1<PERIOD>', text)
        text = re.sub(r'(\d+\.\d+)', r'\1<DECIMAL>', text)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore protected periods
        sentences = [s.replace('<PERIOD>', '.').replace('<DECIMAL>', '.') for s in sentences]
        
        # Clean and filter
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 15 and len(sentence.split()) > 3:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _calculate_sentence_score(self, sentence: str, query: str, 
                                query_embedding, position: int, total_sentences: int) -> float:
        """Calculate comprehensive sentence relevance score"""
        # 1. Semantic similarity
        try:
            sentence_embedding = self.sentence_model.encode(sentence, convert_to_tensor=True, show_progress_bar=False)
            semantic_score = float(util.cos_sim(query_embedding, sentence_embedding)[0][0])
        except:
            semantic_score = 0.0
        
        # 2. Position bias (first and last sentences often important)
        if position == 0:
            position_score = 0.3
        elif position == total_sentences - 1:
            position_score = 0.2
        else:
            position_score = 0.1
        
        # 3. Content indicators
        content_score = 0.0
        sentence_lower = sentence.lower()
        
        # Important phrases
        important_phrases = [
            'results show', 'findings indicate', 'analysis reveals', 'study demonstrates',
            'research shows', 'data suggests', 'evidence indicates', 'conclusion',
            'significantly', 'important', 'key', 'main', 'primary', 'critical'
        ]
        
        for phrase in important_phrases:
            if phrase in sentence_lower:
                content_score += 0.1
        
        # 4. Sentence quality
        words = sentence.split()
        quality_score = 0.0
        
        if 10 <= len(words) <= 30:  # Optimal length
            quality_score += 0.2
        
        if re.search(r'\d+', sentence):  # Contains numbers/data
            quality_score += 0.1
        
        # Combine scores
        total_score = (
            0.5 * semantic_score +
            0.2 * position_score +
            0.2 * min(content_score, 0.5) +
            0.1 * quality_score
        )
        
        return total_score
    
    def _create_coherent_summary(self, sentences: List[str]) -> str:
        """Create a coherent summary from selected sentences"""
        if not sentences:
            return ""
        
        # Basic coherence improvements
        summary = " ".join(sentences)
        
        # Fix spacing and punctuation
        summary = re.sub(r'\s+', ' ', summary).strip()
        
        # Ensure proper ending
        if summary and not summary.endswith(('.', '!', '?')):
            summary += "."
        
        return summary
    
    def analyze_subsections(self, persona: str, job_to_be_done: str, 
                          sections: List[Dict], top_n_sections: int = 10, 
                          sentences_per_section: int = 3) -> List[Dict]:
        """
        Advanced subsection analysis with context-aware refinement
        """
        query = f"Role: {persona}. Task: {job_to_be_done}. Context: {persona} {job_to_be_done}"
        
        subsections = []
        for i, section in enumerate(sections[:top_n_sections]):
            # Get refined text using advanced summarization
            refined_text = self.get_refined_summary(
                query, 
                section.get('content', ''), 
                sentences_per_section
            )
            
            if refined_text:
                subsection = {
                    'document': section.get('document', ''),
                    'page': section.get('page', 1),
                    'section_title': section.get('title', ''),
                    'refined_text': refined_text,
                    'importance_rank': section.get('importance_rank', i + 1),
                    'relevance_score': section.get('composite_score', 0.0)
                }
                subsections.append(subsection)
        
        # Additional post-processing for better results
        subsections = self._post_process_subsections(subsections, persona, job_to_be_done)
        
        return subsections
    
    def _post_process_subsections(self, subsections: List[Dict], 
                                persona: str, job_to_be_done: str) -> List[Dict]:
        """Post-process subsections for optimal results"""
        if not subsections:
            return subsections
        
        # Remove duplicates based on content similarity
        unique_subsections = []
        for subsection in subsections:
            is_duplicate = False
            for existing in unique_subsections:
                similarity = self._calculate_text_similarity(
                    subsection['refined_text'], 
                    existing['refined_text']
                )
                if similarity > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_subsections.append(subsection)
        
        # Ensure minimum content quality
        quality_filtered = []
        for subsection in unique_subsections:
            text = subsection['refined_text']
            if len(text.split()) >= 10 and len(text) >= 50:
                quality_filtered.append(subsection)
        
        return quality_filtered
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity for duplicate detection"""
        if not text1 or not text2:
            return 0.0
        
        # Use TF-IDF for similarity if available, otherwise use simple word overlap
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            # Fallback to word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            return len(intersection) / len(union) if union else 0.0