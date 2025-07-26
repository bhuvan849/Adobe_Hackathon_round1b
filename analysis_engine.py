import re
import logging
from typing import List, Dict, Tuple, Optional, Set
from sentence_transformers import SentenceTransformer, util
import numpy as np
from collections import defaultdict
import math

logger = logging.getLogger(__name__)

class AdvancedAnalysisEngine:
    def __init__(self, model_name: str = "all-mpnet-base-v2", constraint_patterns: Dict = None):
        """
        Initialize the advanced analysis engine with persona-driven intelligence.
        
        Args:
            model_name: Sentence transformer model name
            constraint_patterns: Domain-specific filtering and boosting rules
        """
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded sentence transformer model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load {model_name}, falling back to smaller model")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.constraint_patterns = constraint_patterns or {}
        
        # Pre-compile constraint patterns for efficiency
        self._compile_constraint_patterns()
        
        # Domain-specific keywords for different personas
        self.persona_keywords = {
            'researcher': ['method', 'analysis', 'result', 'conclusion', 'hypothesis', 'study', 'research', 'data'],
            'student': ['concept', 'definition', 'example', 'theory', 'principle', 'learn', 'understand'],
            'analyst': ['trend', 'performance', 'metric', 'growth', 'revenue', 'market', 'strategy'],
            'developer': ['implementation', 'code', 'algorithm', 'architecture', 'design', 'system'],
            'manager': ['strategy', 'plan', 'objective', 'resource', 'team', 'project', 'goal'],
            'contractor': ['specification', 'requirement', 'material', 'process', 'quality', 'standard']
        }
        
        # Task-specific keywords
        self.task_keywords = {
            'review': ['summary', 'overview', 'comparison', 'evaluation', 'assessment'],
            'analysis': ['pattern', 'trend', 'insight', 'correlation', 'impact', 'factor'],
            'planning': ['strategy', 'approach', 'timeline', 'resource', 'step', 'phase'],
            'preparation': ['material', 'ingredient', 'tool', 'equipment', 'setup', 'requirement']
        }

    def _compile_constraint_patterns(self):
        """Compile regex patterns for efficient matching."""
        self.hard_filters = {}
        self.soft_boosts = {}
        
        for category, patterns in self.constraint_patterns.items():
            for name, pattern in patterns.items():
                try:
                    compiled_pattern = re.compile(pattern, re.IGNORECASE)
                    
                    if name.startswith("non_"):
                        # Hard filter (exclude content matching this pattern)
                        self.hard_filters[f"{category}_{name}"] = compiled_pattern
                    else:
                        # Soft boost (prioritize content matching this pattern)
                        boost_value = 1.3 if 'gluten' in name else 1.2
                        self.soft_boosts[f"{category}_{name}"] = (compiled_pattern, boost_value)
                        
                except re.error as e:
                    logger.warning(f"Invalid regex pattern for {category}.{name}: {e}")

    def get_ranked_sections(self, persona: str, job_to_be_done: str, 
                          sections: List[Dict], top_k: int = 5) -> Tuple[List[Dict], List[Dict]]:
        """
        Rank sections based on persona and job-to-be-done with advanced scoring.
        
        Args:
            persona: User persona/role
            job_to_be_done: Specific task to accomplish
            sections: List of document sections
            top_k: Number of top sections to return
            
        Returns:
            Tuple of (ranked_sections, subsection_analysis)
        """
        if not sections:
            logger.warning("No sections provided for ranking")
            return [], []

        logger.info(f"Ranking {len(sections)} sections for persona: {persona}, task: {job_to_be_done}")

        # Create enhanced query
        query = self._create_enhanced_query(persona, job_to_be_done)
        
        # Apply hard filters first
        filtered_sections = self._apply_hard_filters(sections, job_to_be_done)
        if not filtered_sections:
            logger.warning("All sections were excluded by hard filters")
            return [], []

        logger.info(f"After filtering: {len(filtered_sections)} sections remain")

        # Compute semantic similarities
        semantic_scores = self._compute_semantic_scores(query, filtered_sections)
        
        # Apply persona-specific scoring
        persona_scores = self._compute_persona_scores(persona, filtered_sections)
        
        # Apply task-specific scoring
        task_scores = self._compute_task_scores(job_to_be_done, filtered_sections)
        
        # Apply soft boosts from constraints
        boost_scores = self._apply_soft_boosts(filtered_sections, job_to_be_done)
        
        # Apply quality metrics
        quality_scores = self._compute_quality_scores(filtered_sections)
        
        # Combine all scores with weights
        final_scores = self._combine_scores({
            'semantic': semantic_scores,
            'persona': persona_scores,
            'task': task_scores,
            'boost': boost_scores,
            'quality': quality_scores
        })

        # Create ranked sections
        ranked_sections = []
        for i, section in enumerate(filtered_sections):
            section_copy = section.copy()
            section_copy['final_score'] = final_scores[i]
            section_copy['semantic_score'] = semantic_scores[i]
            section_copy['persona_score'] = persona_scores[i]
            section_copy['task_score'] = task_scores[i]
            ranked_sections.append(section_copy)

        # Sort by final score
        ranked_sections = sorted(ranked_sections, key=lambda x: x['final_score'], reverse=True)
        
        # Apply diversity filter to avoid too similar sections
        diverse_sections = self._apply_diversity_filter(ranked_sections, query)
        
        # Select top_k and assign ranks
        top_sections = diverse_sections[:top_k]
        for idx, section in enumerate(top_sections, 1):
            section['importance_rank'] = idx

        # Generate subsection analysis
        subsections = self._generate_subsection_analysis(query, top_sections, persona, job_to_be_done)

        logger.info(f"Final ranking: {len(top_sections)} sections selected")
        return top_sections, subsections

    def _create_enhanced_query(self, persona: str, job_to_be_done: str) -> str:
        """Create an enhanced query string for better semantic matching."""
        # Extract persona keywords
        persona_lower = persona.lower()
        persona_context = []
        
        for role, keywords in self.persona_keywords.items():
            if role in persona_lower:
                persona_context.extend(keywords[:3])  # Top 3 keywords
        
        # Extract task keywords
        task_lower = job_to_be_done.lower()
        task_context = []
        
        for task_type, keywords in self.task_keywords.items():
            if task_type in task_lower:
                task_context.extend(keywords[:3])
        
        # Combine into enhanced query
        query_parts = [f"As a {persona}", job_to_be_done]
        
        if persona_context:
            query_parts.append(f"focusing on {' '.join(persona_context)}")
        
        if task_context:
            query_parts.append(f"related to {' '.join(task_context)}")
        
        return " ".join(query_parts).strip()

    def _apply_hard_filters(self, sections: List[Dict], job: str) -> List[Dict]:
        """Apply hard filters to exclude unwanted content."""
        if not self.hard_filters:
            return sections

        job_lower = job.lower()
        
        # Determine which filter categories are active based on job
        active_filters = {}
        for filter_name, pattern in self.hard_filters.items():
            category = filter_name.split('_')[0]
            if category in job_lower:
                active_filters[filter_name] = pattern

        if not active_filters:
            return sections

        filtered_sections = []
        for section in sections:
            text_to_check = f"{section.get('title', '')} {section.get('content', '')}"
            
            # Check if section should be excluded
            exclude = False
            for filter_name, pattern in active_filters.items():
                if pattern.search(text_to_check):
                    logger.debug(f"Section '{section.get('title', '')[:50]}...' excluded by filter: {filter_name}")
                    exclude = True
                    break
            
            if not exclude:
                filtered_sections.append(section)

        logger.info(f"Hard filters: {len(sections)} -> {len(filtered_sections)} sections")
        return filtered_sections

    def _compute_semantic_scores(self, query: str, sections: List[Dict]) -> List[float]:
        """Compute semantic similarity scores using sentence transformers."""
        try:
            # Prepare texts for encoding
            section_texts = []
            for section in sections:
                title = section.get('title', '')
                content = section.get('content', '')
                # Combine title and content, with title getting more weight
                combined_text = f"{title} {title} {content}"[:1000]  # Limit length
                section_texts.append(combined_text)

            # Encode query and sections
            query_embedding = self.model.encode(query, convert_to_tensor=True, show_progress_bar=False)
            section_embeddings = self.model.encode(section_texts, convert_to_tensor=True, show_progress_bar=False)

            # Compute cosine similarities
            similarities = util.cos_sim(query_embedding, section_embeddings)[0]
            scores = similarities.cpu().numpy().tolist()

            return scores

        except Exception as e:
            logger.error(f"Error computing semantic scores: {e}")
            return [0.5] * len(sections)  # Fallback scores

    def _compute_persona_scores(self, persona: str, sections: List[Dict]) -> List[float]:
        """Compute persona-specific relevance scores."""
        persona_lower = persona.lower()
        
        # Find relevant keywords for this persona
        relevant_keywords = []
        for role, keywords in self.persona_keywords.items():
            if role in persona_lower:
                relevant_keywords.extend(keywords)
        
        if not relevant_keywords:
            return [0.0] * len(sections)  # No persona-specific boost

        scores = []
        for section in sections:
            text = f"{section.get('title', '')} {section.get('content', '')}".lower()
            
            # Count keyword matches
            matches = sum(1 for keyword in relevant_keywords if keyword in text)
            
            # Normalize score
            max_possible_matches = min(len(relevant_keywords), 5)  # Cap at 5
            score = matches / max_possible_matches if max_possible_matches > 0 else 0.0
            
            scores.append(score)

        return scores

    def _compute_task_scores(self, job_to_be_done: str, sections: List[Dict]) -> List[float]:
        """Compute task-specific relevance scores."""
        job_lower = job_to_be_done.lower()
        
        # Find relevant task keywords
        relevant_keywords = []
        for task_type, keywords in self.task_keywords.items():
            if task_type in job_lower:
                relevant_keywords.extend(keywords)

        # Add specific terms from job description
        job_words = [word for word in job_lower.split() 
                    if len(word) > 3 and word not in {'the', 'and', 'for', 'with', 'that', 'this'}]
        relevant_keywords.extend(job_words)

        if not relevant_keywords:
            return [0.0] * len(sections)

        scores = []
        for section in sections:
            text = f"{section.get('title', '')} {section.get('content', '')}".lower()
            
            # Count keyword matches with different weights
            title_text = section.get('title', '').lower()
            content_text = section.get('content', '').lower()
            
            title_matches = sum(2 for keyword in relevant_keywords if keyword in title_text)  # Title matches worth more
            content_matches = sum(1 for keyword in relevant_keywords if keyword in content_text)
            
            total_matches = title_matches + content_matches
            
            # Normalize score
            max_possible = min(len(relevant_keywords) * 2, 10)  # Cap at 10
            score = total_matches / max_possible if max_possible > 0 else 0.0
            
            scores.append(min(score, 1.0))  # Cap at 1.0

        return scores

    def _apply_soft_boosts(self, sections: List[Dict], job: str) -> List[float]:
        """Apply soft boosts based on constraint patterns."""
        if not self.soft_boosts:
            return [1.0] * len(sections)

        job_lower = job.lower()
        
        # Determine which boost categories are active
        active_boosts = {}
        for boost_name, (pattern, multiplier) in self.soft_boosts.items():
            category = boost_name.split('_')[0]
            if category in job_lower:
                active_boosts[boost_name] = (pattern, multiplier)

        if not active_boosts:
            return [1.0] * len(sections)

        scores = []
        for section in sections:
            text_to_check = f"{section.get('title', '')} {section.get('content', '')}"
            
            boost_multiplier = 1.0
            for boost_name, (pattern, multiplier) in active_boosts.items():
                if pattern.search(text_to_check):
                    boost_multiplier *= multiplier
                    logger.debug(f"Section '{section.get('title', '')[:30]}...' boosted by {boost_name}")

            scores.append(boost_multiplier)

        return scores

    def _compute_quality_scores(self, sections: List[Dict]) -> List[float]:
        """Compute quality scores based on content characteristics."""
        scores = []
        
        for section in sections:
            title = section.get('title', '')
            content = section.get('content', '')
            
            score = 0.0
            
            # Length score (prefer substantial content)
            content_length = len(content)
            if content_length > 500:
                score += 0.3
            elif content_length > 200:
                score += 0.2
            elif content_length > 100:
                score += 0.1
            
            # Title quality score
            if len(title) > 10 and not title.lower().startswith('page'):
                score += 0.2
            
            # Structure score (look for structured content)
            if ':' in content or '•' in content or '\n' in content:
                score += 0.1
            
            # Confidence score from extraction method
            confidence = section.get('confidence', 0.5)
            score += confidence * 0.4
            
            scores.append(min(score, 1.0))  # Cap at 1.0
        
        return scores

    def _combine_scores(self, score_dict: Dict[str, List[float]]) -> List[float]:
        """Combine different score types with appropriate weights."""
        weights = {
            'semantic': 0.4,    # Primary semantic matching
            'persona': 0.2,     # Persona-specific relevance
            'task': 0.2,        # Task-specific relevance
            'boost': 0.1,       # Constraint-based boosts
            'quality': 0.1      # Content quality
        }
        
        num_sections = len(next(iter(score_dict.values())))
        combined_scores = []
        
        for i in range(num_sections):
            total_score = 0.0
            for score_type, scores in score_dict.items():
                weight = weights.get(score_type, 0.1)
                score_value = scores[i] if i < len(scores) else 0.0
                
                if score_type == 'boost':
                    # Boost is a multiplier, not additive
                    total_score *= score_value
                else:
                    total_score += weight * score_value
            
            combined_scores.append(total_score)
        
        return combined_scores

    def _apply_diversity_filter(self, ranked_sections: List[Dict], query: str, 
                              similarity_threshold: float = 0.85) -> List[Dict]:
        """Apply diversity filter to reduce redundant sections."""
        if len(ranked_sections) <= 3:
            return ranked_sections  # Too few sections to filter
        
        try:
            # Get embeddings for all sections
            section_texts = [
                f"{s.get('title', '')} {s.get('content', '')}"[:500] 
                for s in ranked_sections
            ]
            embeddings = self.model.encode(section_texts, convert_to_tensor=True, show_progress_bar=False)
            
            # Select diverse sections
            selected_sections = [ranked_sections[0]]  # Always include top section
            selected_embeddings = [embeddings[0]]
            
            for i, section in enumerate(ranked_sections[1:], 1):
                current_embedding = embeddings[i]
                
                # Check similarity with already selected sections
                similarities = util.cos_sim(current_embedding, selected_embeddings)[0]
                max_similarity = float(similarities.max()) if len(similarities) > 0 else 0.0
                
                # Include if sufficiently different
                if max_similarity < similarity_threshold:
                    selected_sections.append(section)
                    selected_embeddings.append(current_embedding)
                    
                    # Stop if we have enough diverse sections
                    if len(selected_sections) >= 8:
                        break
            
            logger.info(f"Diversity filter: {len(ranked_sections)} -> {len(selected_sections)} sections")
            return selected_sections
            
        except Exception as e:
            logger.warning(f"Diversity filtering failed: {e}")
            return ranked_sections  # Return original if filtering fails

    def _generate_subsection_analysis(self, query: str, sections: List[Dict], 
                                    persona: str, job_to_be_done: str) -> List[Dict]:
        """Generate detailed subsection analysis with intelligent summarization."""
        subsections = []
        
        for section in sections:
            content = section.get('content', '')
            if not content:
                continue
            
            # Extract key sentences based on multiple criteria
            summary = self._extract_key_sentences(
                content, query, persona, job_to_be_done, num_sentences=3
            )
            
            subsection = {
                'document': section.get('document', ''),
                'page_number': section.get('page', 1),
                'refined_text': summary
            }
            
            subsections.append(subsection)
        
        return subsections

    def _extract_key_sentences(self, content: str, query: str, persona: str, 
                             job_to_be_done: str, num_sentences: int = 3) -> str:
        """Extract key sentences using multiple ranking criteria."""
        try:
            # Split into sentences
            sentences = self._split_into_sentences(content)
            
            if len(sentences) <= num_sentences:
                return ' '.join(sentences)
            
            # Score sentences using multiple criteria
            sentence_scores = []
            
            # Get embeddings for query and sentences
            query_emb = self.model.encode(query, convert_to_tensor=True, show_progress_bar=False)
            sentence_embs = self.model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
            
            # Compute semantic similarities
            semantic_sims = util.cos_sim(query_emb, sentence_embs)[0].cpu().numpy()
            
            for i, sentence in enumerate(sentences):
                score = 0.0
                
                # Semantic similarity (40% weight)
                score += 0.4 * semantic_sims[i]
                
                # Position score (first and last sentences often important) (20% weight)
                position_score = 0.0
                if i == 0:  # First sentence
                    position_score = 0.8
                elif i == len(sentences) - 1:  # Last sentence
                    position_score = 0.6
                elif i < len(sentences) * 0.3:  # Early sentences
                    position_score = 0.4
                
                score += 0.2 * position_score
                
                # Length score (prefer substantial sentences) (15% weight)
                words = sentence.split()
                if 10 <= len(words) <= 30:
                    length_score = 1.0
                elif 5 <= len(words) < 10:
                    length_score = 0.7
                elif len(words) > 30:
                    length_score = 0.8
                else:
                    length_score = 0.3
                
                score += 0.15 * length_score
                
                # Keyword relevance (25% weight)
                keyword_score = self._compute_sentence_keyword_score(
                    sentence, persona, job_to_be_done
                )
                score += 0.25 * keyword_score
                
                sentence_scores.append((i, score, sentence))
            
            # Sort by score and select top sentences
            top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:num_sentences]
            
            # Sort selected sentences by original order
            selected_sentences = sorted(top_sentences, key=lambda x: x[0])
            
            return ' '.join([sent[2] for sent in selected_sentences])
            
        except Exception as e:
            logger.warning(f"Error in key sentence extraction: {e}")
            # Fallback: return first few sentences
            sentences = self._split_into_sentences(content)
            return ' '.join(sentences[:num_sentences])

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with better handling of edge cases."""
        # Basic sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter and clean sentences
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and len(sentence.split()) >= 3:  # Minimum requirements
                clean_sentences.append(sentence)
        
        return clean_sentences

    def _compute_sentence_keyword_score(self, sentence: str, persona: str, job_to_be_done: str) -> float:
        """Compute keyword relevance score for a sentence."""
        sentence_lower = sentence.lower()
        score = 0.0
        
        # Persona keywords
        persona_lower = persona.lower()
        for role, keywords in self.persona_keywords.items():
            if role in persona_lower:
                matches = sum(1 for keyword in keywords if keyword in sentence_lower)
                score += matches / len(keywords) * 0.5
        
        # Job keywords
        job_words = [word for word in job_to_be_done.lower().split() 
                    if len(word) > 3]
        if job_words:
            matches = sum(1 for word in job_words if word in sentence_lower)
            score += matches / len(job_words) * 0.5
        
        return min(score, 1.0)  # Cap at 1.0