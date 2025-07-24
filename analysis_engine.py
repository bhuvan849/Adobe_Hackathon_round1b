# analysis_engine.py
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
from typing import List, Dict, Any

class AnalysisEngine:
    def __init__(self, model_name='all-mpnet-base-v2'):
        """Initialize with a lightweight sentence transformer model"""
        self.model = SentenceTransformer(model_name)

    def get_ranked_sections(self, persona: str, job_to_be_done: str, all_sections: List[Dict]) -> List[Dict]:
        """
        Performs relevance ranking on document sections based on persona and job.
        """
        if not all_sections:
            return []

        # Create a comprehensive query combining persona and job
        query = f"Role: {persona}. Task: {job_to_be_done}. Requirements: {job_to_be_done}"
        
        # Get query embedding
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Prepare section texts for embedding
        section_texts = []
        for section in all_sections:
            # Combine title and content for better matching
            section_text = f"{section.get('title', '')} {section.get('content', '')}"
            section_texts.append(section_text)

        # Get embeddings for all sections
        corpus_embeddings = self.model.encode(section_texts, convert_to_tensor=True)

        # Compute cosine similarity
        similarities = util.cos_sim(query_embedding, corpus_embeddings)[0]

        # Attach relevance scores to sections
        ranked_sections = []
        for i, section in enumerate(all_sections):
            section_copy = section.copy()
            section_copy['relevance_score'] = float(similarities[i])
            section_copy['importance_rank'] = 0  # Will be set after sorting
            ranked_sections.append(section_copy)
        
        # Sort by relevance score (descending)
        ranked_sections.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Assign importance ranks
        for i, section in enumerate(ranked_sections):
            section['importance_rank'] = i + 1
        
        return ranked_sections

    def get_refined_summary(self, query: str, section_content: str, num_sentences: int = 3) -> str:
        """
        Creates an extractive summary by finding the most relevant sentences.
        """
        if not section_content or not section_content.strip():
            return ""
            
        # Enhanced sentence splitting
        sentences = self._split_into_sentences(section_content)
        
        if len(sentences) <= num_sentences:
            return section_content.strip()

        # Get embeddings
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True)
        
        # Calculate similarities
        similarities = util.cos_sim(query_embedding, sentence_embeddings)[0]

        # Get top sentences (maintain original order)
        top_indices = np.argsort(similarities)[-num_sentences:]
        top_indices_sorted = sorted(top_indices)

        # Create summary
        summary_sentences = [sentences[i] for i in top_indices_sorted]
        summary = ". ".join(summary_sentences)
        
        # Ensure proper ending
        if summary and not summary.endswith('.'):
            summary += "."
            
        return summary

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Enhanced sentence splitting that handles various edge cases.
        """
        # Basic sentence splitting with regex
        sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s+'
        sentences = re.split(sentence_pattern, text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter very short sentences
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences

    def analyze_subsections(self, persona: str, job_to_be_done: str, sections: List[Dict], 
                          top_n_sections: int = 10, sentences_per_section: int = 3) -> List[Dict]:
        """
        Analyze subsections within the top-ranked sections.
        """
        query = f"Role: {persona}. Task: {job_to_be_done}"
        
        subsections = []
        for i, section in enumerate(sections[:top_n_sections]):
            refined_text = self.get_refined_summary(
                query, 
                section.get('content', ''), 
                sentences_per_section
            )
            
            if refined_text:
                subsections.append({
                    'document': section.get('document', ''),
                    'page': section.get('page', 1),
                    'section_title': section.get('title', ''),
                    'refined_text': refined_text,
                    'importance_rank': section.get('importance_rank', i + 1)
                })
        
        return subsections