#!/usr/bin/env python3
# optimized_main.py
import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from document_parser import EnhancedDocumentParser
from analysis_engine import AdvancedAnalysisEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"

class OptimizedDocumentAnalyzer:
    def __init__(self):
        self.parser = EnhancedDocumentParser()
        self.analyzer = AdvancedAnalysisEngine()
        self.stats = {
            'total_documents': 0,
            'processing_time': 0,
            'average_relevance_score': 0
        }
    
    def run_analysis(self, config_path: str) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            config = self._load_config(config_path)
            persona = config.get("persona", {}).get("role", "Professional")
            job_to_be_done = config.get("job_to_be_done", {}).get("task", "Analyze documents")
            doc_objects = config.get("documents", [])
            
            logger.info(f"Starting analysis for persona: {persona}")
            logger.info(f"Job to be done: {job_to_be_done}")
            
            all_sections = self._process_documents(doc_objects)
            
            if not all_sections:
                logger.warning("No sections extracted from documents")
                return self._create_empty_output(config)
            
            ranked_sections = self.analyzer.get_ranked_sections(
                persona, job_to_be_done, all_sections
            )
            
            output_data = self._create_output(
                config, ranked_sections, persona, job_to_be_done
            )
            
            self.stats['processing_time'] = time.time() - start_time
            self.stats['average_relevance_score'] = self._calculate_avg_relevance(ranked_sections)
            
            # --- FIX: Update total_sections count at the end ---
            output_data['metadata']['processing_stats']['total_sections_extracted'] = len(ranked_sections)
            
            logger.info(f"Analysis completed successfully in {self.stats['processing_time']:.2f}s")
            return output_data
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        required_fields = ['persona', 'job_to_be_done', 'documents']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in config: {field}")
        
        return config
    
    def _process_documents(self, doc_objects: List[Dict]) -> List[Dict]:
        all_sections = []
        pdf_files_dir = Path(INPUT_DIR) / "PDFs"
        
        doc_filenames = [doc.get("filename") for doc in doc_objects if doc.get("filename")]
        
        if not doc_filenames:
            raise ValueError("No document filenames found in configuration")
        
        self.stats['total_documents'] = len(doc_filenames)
        
        for filename in doc_filenames:
            pdf_path = pdf_files_dir / filename
            if not pdf_path.exists():
                logger.warning(f"PDF file not found: {filename}")
                continue
            
            logger.info(f"Processing document: {filename}")
            try:
                sections = self.parser.parse_document_by_structure(str(pdf_path))
                for section in sections:
                    section['document'] = filename
                all_sections.extend(sections)
                logger.info(f"Extracted {len(sections)} sections from {filename}")
            except Exception as e:
                logger.error(f"Failed to process {filename}: {str(e)}")
                continue
        
        return all_sections
    
    def _create_output(self, config: Dict, ranked_sections: List[Dict], 
                      persona: str, job_to_be_done: str) -> Dict[str, Any]:
        
        top_sections = ranked_sections[:5]
        
        extracted_sections = []
        for i, section in enumerate(top_sections):
            extracted_sections.append({
                "document": section["document"],
                "section_title": section["title"],
                "importance_rank": i + 1,
                "page_number": section["page"],
                "relevance_score": round(section.get('composite_score', 0), 4)
            })
        
        sub_section_analysis = self.analyzer.analyze_subsections(
            persona, job_to_be_done, top_sections, 
            top_n_sections=5, sentences_per_section=3
        )
        
        output_data = {
            "metadata": {
                **config,
                "processing_stats": {
                    "total_documents_processed": self.stats['total_documents'],
                    # --- FIX: Get the final, correct count here ---
                    "total_sections_extracted": len(ranked_sections), 
                    "processing_time_seconds": round(self.stats['processing_time'], 2),
                    "average_relevance_score": round(self.stats['average_relevance_score'], 4)
                },
                "timestamp": datetime.utcnow().isoformat() + "Z"
            },
            "extracted_sections": extracted_sections,
            "sub_section_analysis": sub_section_analysis
        }
        
        return output_data
    
    def _create_empty_output(self, config: Dict) -> Dict[str, Any]:
        return {
            "metadata": {
                **config,
                "processing_stats": {
                    "total_documents_processed": self.stats['total_documents'],
                    "total_sections_extracted": 0,
                    "processing_time_seconds": round(self.stats.get('processing_time', 0), 2),
                    "warning": "No sections were extracted from the provided documents"
                },
                "timestamp": datetime.utcnow().isoformat() + "Z"
            },
            "extracted_sections": [],
            "sub_section_analysis": []
        }
    
    def _calculate_avg_relevance(self, sections: List[Dict]) -> float:
        if not sections:
            return 0.0
        
        scores = [s.get('composite_score', 0) for s in sections]
        return sum(scores) / len(scores)

def main():
    """Main execution function"""
    print("="*60)
    print("Starting analysis for Challenge 1b")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Initialize analyzer
        analyzer = OptimizedDocumentAnalyzer()
        
        # Set up paths
        input_config_path = os.path.join(INPUT_DIR, "challenge1b_input.json")
        
        # Run analysis
        logger.info(f"Starting analysis with input directory: {INPUT_DIR}")
        output_data = analyzer.run_analysis(input_config_path)
        
        # Save output
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "challenge1b_output.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Print summary
        total_time = time.time() - start_time
        stats = output_data['metadata']['processing_stats']
        
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"📄 Documents processed: {stats['total_documents_processed']}")
        
        print(f"⚡ Processing time: {total_time:.2f} seconds")
        
        print(f"💾 Output saved to: {output_path}")
        print(f"{'='*60}")
        
        
        
        return output_data
        
    except Exception as e:
        # Create error output
        error_output = {
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "processing_time": time.time() - start_time
        }
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        error_path = os.path.join(OUTPUT_DIR, "error.json")
        
        with open(error_path, 'w') as f:
            json.dump(error_output, f, indent=2)
        
        logger.error(f"Analysis failed: {e}")
        print(f"\n❌ ANALYSIS FAILED: {e}")
        print(f"Error details saved to: {error_path}")
        
        raise

if __name__ == "__main__":
    main()