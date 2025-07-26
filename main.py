import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import traceback
import sys

# Import our enhanced modules
from document_parser import EnhancedDocumentParser
from analysis_engine import AdvancedAnalysisEngine

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/processing.log', mode='w')
    ]
)

# Suppress noisy library logs
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Configuration
INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"
MAX_PROCESSING_TIME = 55  # seconds
MAX_SECTIONS_PER_DOC = 100  # Increased for better coverage

class DocumentAnalysisSystem:
    """Enhanced document analysis system with robust error handling and optimization."""
    
    def __init__(self):
        self.parser = None
        self.engine = None
        self.processing_stats = {
            'start_time': time.time(),
            'documents_attempted': 0,
            'documents_successful': 0,
            'total_sections_extracted': 0,
            'errors': [],
            'warnings': []
        }

    def initialize_components(self, constraint_patterns: Dict = None) -> bool:
        """Initialize analysis components with comprehensive error handling."""
        try:
            logger.info("=== Initializing Document Analysis System ===")
            
            # Initialize document parser
            logger.info("Initializing enhanced document parser...")
            self.parser = EnhancedDocumentParser()
            
            # Initialize analysis engine with constraints
            logger.info("Initializing advanced analysis engine...")
            self.engine = AdvancedAnalysisEngine(constraint_patterns=constraint_patterns)
            
            # Test components
            logger.info("Testing system components...")
            self._test_components()
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize components: {str(e)}"
            logger.error(error_msg)
            self.processing_stats['errors'].append(error_msg)
            return False

    def _test_components(self):
        """Test system components to ensure they're working."""
        try:
            # Test sentence transformer
            test_embedding = self.engine.model.encode("test sentence", show_progress_bar=False)
            logger.info(f"Sentence transformer test successful, embedding shape: {test_embedding.shape}")
        except Exception as e:
            logger.warning(f"Sentence transformer test failed: {e}")

    def load_configuration(self) -> Optional[Dict]:
        """Load and validate input configuration with enhanced error handling."""
        try:
            config_path = Path(INPUT_DIR) / "challenge1b_input.json"
            
            if not config_path.exists():
                # Try alternative locations
                alternative_paths = [
                    Path(INPUT_DIR) / "input.json",
                    Path("/app") / "challenge1b_input.json",
                    Path(".") / "challenge1b_input.json"
                ]
                
                for alt_path in alternative_paths:
                    if alt_path.exists():
                        config_path = alt_path
                        break
                else:
                    raise FileNotFoundError(f"Configuration file not found in expected locations")
            
            logger.info(f"Loading configuration from: {config_path}")
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Validate configuration structure
            validated_config = self._validate_configuration(config)
            
            logger.info(f"Configuration loaded successfully: {len(validated_config['doc_filenames'])} documents")
            return validated_config
            
        except Exception as e:
            error_msg = f"Failed to load configuration: {str(e)}"
            logger.error(error_msg)
            self.processing_stats['errors'].append(error_msg)
            return None

    def _validate_configuration(self, config: Dict) -> Dict:
        """Validate and normalize configuration structure."""
        required_fields = ['persona', 'job_to_be_done', 'documents']
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        # Extract persona
        persona_data = config.get('persona', {})
        if isinstance(persona_data, dict):
            persona = persona_data.get('role', str(persona_data))
        else:
            persona = str(persona_data)
        
        # Extract job to be done
        job_data = config.get('job_to_be_done', {})
        if isinstance(job_data, dict):
            job_to_be_done = job_data.get('task', str(job_data))
        else:
            job_to_be_done = str(job_data)
        
        # Extract and validate documents
        documents = config.get('documents', [])
        if not documents:
            raise ValueError("No documents specified in configuration")
        
        doc_filenames = []
        for doc in documents:
            if isinstance(doc, dict):
                filename = doc.get('filename', '')
            else:
                filename = str(doc)
            
            if filename and filename.endswith('.pdf'):
                doc_filenames.append(filename)
            elif filename:
                logger.warning(f"Skipping non-PDF document: {filename}")
        
        if not doc_filenames:
            raise ValueError("No valid PDF documents found in configuration")
        
        return {
            'original_config': config,
            'persona': persona.strip(),
            'job_to_be_done': job_to_be_done.strip(),
            'doc_filenames': doc_filenames
        }

    def create_constraint_patterns(self, persona: str, job_to_be_done: str) -> Dict:
        """Create domain-specific constraint patterns based on persona and job."""
        constraint_patterns = {}
        
        job_lower = job_to_be_done.lower()
        persona_lower = persona.lower()
        
        # Food/dietary constraints
        if any(keyword in job_lower for keyword in ['food', 'menu', 'recipe', 'vegetarian', 'vegan', 'meal']):
            constraint_patterns['dietary'] = {
                # Hard filters (exclusions)
                'non_vegetarian': r'\b(chicken|turkey|beef|pork|lamb|fish|salmon|tuna|shrimp|seafood|bacon|prosciutto|salami|ham|meat)\b',
                'non_vegan': r'\b(cheese|milk|butter|cream|egg|dairy|yogurt|mozzarella|parmesan)\b',
                
                # Soft boosts (preferences)
                'gluten_free': r'\b(gluten[\-\s]*free|gluten[\-\s]*friendly|celiac|wheat[\-\s]*free)\b',
                'vegetarian_friendly': r'\b(vegetable|veggie|plant[\-\s]*based|vegan|vegetarian|salad|bean|lentil|tofu)\b',
                'healthy': r'\b(organic|fresh|natural|healthy|nutritious|vitamin|mineral)\b'
            }
        
        # Academic/research constraints
        if any(keyword in persona_lower for keyword in ['researcher', 'student', 'academic', 'scholar']):
            constraint_patterns['academic'] = {
                # Boost academic content
                'methodology': r'\b(method|methodology|approach|technique|analysis|study|research|experiment)\b',
                'results': r'\b(result|finding|conclusion|outcome|data|evidence|statistic)\b',
                'theory': r'\b(theory|hypothesis|model|framework|concept|principle)\b'
            }
        
        # Business/financial constraints
        if any(keyword in persona_lower for keyword in ['analyst', 'manager', 'consultant', 'executive']):
            constraint_patterns['business'] = {
                'performance': r'\b(revenue|profit|growth|performance|metric|KPI|ROI|strategy)\b',
                'market': r'\b(market|competition|customer|client|segment|trend|forecast)\b',
                'operational': r'\b(process|efficiency|optimization|resource|cost|budget)\b'
            }
        
        # Technical constraints
        if any(keyword in persona_lower for keyword in ['developer', 'engineer', 'technical', 'architect']):
            constraint_patterns['technical'] = {
                'implementation': r'\b(code|algorithm|architecture|design|system|implementation|framework)\b',
                'performance': r'\b(performance|optimization|scalability|efficiency|benchmark)\b',
                'security': r'\b(security|authentication|encryption|vulnerability|privacy)\b'
            }
        
        logger.info(f"Created constraint patterns: {list(constraint_patterns.keys())}")
        return constraint_patterns

    def process_documents(self, doc_filenames: List[str]) -> List[Dict]:
        """Process documents with enhanced error handling and time management."""
        all_sections = []
        pdf_search_paths = [
            Path(INPUT_DIR) / "PDFs",
            Path(INPUT_DIR),
            Path("/app/input/PDFs"),
            Path("/app/PDFs"),
            Path(".")
        ]
        
        for filename in doc_filenames:
            # Check time limit
            elapsed_time = time.time() - self.processing_stats['start_time']
            if elapsed_time > MAX_PROCESSING_TIME:
                warning_msg = f"Approaching time limit ({elapsed_time:.1f}s), stopping document processing"
                logger.warning(warning_msg)
                self.processing_stats['warnings'].append(warning_msg)
                break
            
            self.processing_stats['documents_attempted'] += 1
            
            # Find the PDF file
            pdf_path = None
            for search_path in pdf_search_paths:
                candidate_path = search_path / filename
                if candidate_path.exists():
                    pdf_path = candidate_path
                    break
            
            if not pdf_path:
                error_msg = f"Document not found: {filename}"
                logger.warning(error_msg)
                self.processing_stats['errors'].append(error_msg)
                continue
            
            try:
                logger.info(f"Processing document: {filename} ({pdf_path})")
                start_time = time.time()
                
                sections = self.parser.parse_document_by_structure(str(pdf_path))
                
                processing_time = time.time() - start_time
                logger.info(f"Processed {filename} in {processing_time:.2f}s, extracted {len(sections)} sections")
                
                if sections:
                    # Limit sections per document to prevent memory issues
                    if len(sections) > MAX_SECTIONS_PER_DOC:
                        sections = sections[:MAX_SECTIONS_PER_DOC]
                        logger.info(f"Limited {filename} to {MAX_SECTIONS_PER_DOC} sections")
                    
                    all_sections.extend(sections)
                    self.processing_stats['documents_successful'] += 1
                    self.processing_stats['total_sections_extracted'] += len(sections)
                else:
                    warning_msg = f"No sections extracted from {filename}"
                    logger.warning(warning_msg)
                    self.processing_stats['warnings'].append(warning_msg)
                
            except Exception as e:
                error_msg = f"Error processing {filename}: {str(e)}"
                logger.error(error_msg)
                self.processing_stats['errors'].append(error_msg)
                continue
        
        logger.info(f"Document processing complete: {len(all_sections)} total sections from {self.processing_stats['documents_successful']} documents")
        return all_sections

    def analyze_sections(self, persona: str, job_to_be_done: str, all_sections: List[Dict]) -> Dict:
        """Analyze and rank sections with comprehensive error handling."""
        try:
            if not all_sections:
                logger.warning("No sections available for analysis")
                return {'ranked_sections': [], 'subsection_analysis': []}
            
            logger.info(f"Starting analysis of {len(all_sections)} sections")
            start_time = time.time()
            
            # Get ranked sections and subsection analysis
            ranked_sections, subsection_analysis = self.engine.get_ranked_sections(
                persona, job_to_be_done, all_sections, top_k=5
            )
            
            analysis_time = time.time() - start_time
            logger.info(f"Analysis completed in {analysis_time:.2f}s, selected {len(ranked_sections)} sections")
            
            return {
                'ranked_sections': ranked_sections,
                'subsection_analysis': subsection_analysis
            }
            
        except Exception as e:
            error_msg = f"Error in section analysis: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.processing_stats['errors'].append(error_msg)
            
            # Return fallback results
            fallback_sections = all_sections[:5] if all_sections else []
            for i, section in enumerate(fallback_sections, 1):
                section['importance_rank'] = i
                section['final_score'] = 1.0 - (i * 0.1)
            
            return {
                'ranked_sections': fallback_sections,
                'subsection_analysis': []
            }

    def create_output(self, config: Dict, analysis_results: Dict) -> Dict:
        """Create comprehensive output structure."""
        try:
            # Create extracted sections summary
            extracted_sections = []
            for section in analysis_results['ranked_sections']:
                extracted_sections.append({
                    'document': section.get('document', ''),
                    'section_title': section.get('title', ''),
                    'importance_rank': section.get('importance_rank', 0),
                    'page_number': section.get('page', 1)
                })
            
            # Calculate processing statistics
            total_processing_time = time.time() - self.processing_stats['start_time']
            
            # Create comprehensive metadata
            metadata = config['original_config'].copy()
            metadata.update({
                'processing_timestamp': datetime.utcnow().isoformat() + 'Z',
                'processing_stats': {
                    'total_documents_processed': self.processing_stats['documents_successful'],
                    'total_sections_extracted': self.processing_stats['total_sections_extracted'],
                    'processing_time_seconds': round(total_processing_time, 2),
                    'documents_attempted': self.processing_stats['documents_attempted']
                }
            })
            
            # Add error information if present
            if self.processing_stats['errors']:
                metadata['processing_stats']['errors_encountered'] = len(self.processing_stats['errors'])
                metadata['processing_stats']['error_summary'] = self.processing_stats['errors'][:3]  # Top 3 errors
            
            if self.processing_stats['warnings']:
                metadata['processing_stats']['warnings_encountered'] = len(self.processing_stats['warnings'])
                metadata['processing_stats']['warning_summary'] = self.processing_stats['warnings'][:3]
            
            # Create final output structure
            output_data = {
                'metadata': metadata,
                'extracted_sections': extracted_sections,
                'sub_section_analysis': analysis_results['subsection_analysis']
            }
            
            return output_data
            
        except Exception as e:
            error_msg = f"Error creating output: {str(e)}"
            logger.error(error_msg)
            self.processing_stats['errors'].append(error_msg)
            
            # Return minimal fallback output
            return {
                'metadata': config.get('original_config', {}),
                'extracted_sections': [],
                'sub_section_analysis': []
            }

    def save_output(self, output_data: Dict) -> bool:
        """Save output with comprehensive error handling."""
        try:
            output_path = Path(OUTPUT_DIR) / "challenge1b_output.json"
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with proper formatting
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            file_size = output_path.stat().st_size
            logger.info(f"Output saved successfully to {output_path} ({file_size} bytes)")
            
            # Validate the saved file
            self._validate_output_file(output_path)
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to save output: {str(e)}"
            logger.error(error_msg)
            self.processing_stats['errors'].append(error_msg)
            return False

    def _validate_output_file(self, output_path: Path):
        """Validate the saved output file."""
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            
            # Check required fields
            required_fields = ['metadata', 'extracted_sections', 'sub_section_analysis']
            for field in required_fields:
                if field not in loaded_data:
                    logger.warning(f"Output validation: Missing field {field}")
            
            logger.info("Output file validation successful")
            
        except Exception as e:
            logger.warning(f"Output file validation failed: {e}")

    def generate_processing_summary(self) -> Dict:
        """Generate a comprehensive processing summary."""
        total_time = time.time() - self.processing_stats['start_time']
        
        summary = {
            'total_processing_time': round(total_time, 2),
            'documents_attempted': self.processing_stats['documents_attempted'],
            'documents_successful': self.processing_stats['documents_successful'],
            'total_sections_extracted': self.processing_stats['total_sections_extracted'],
            'success_rate': (
                self.processing_stats['documents_successful'] / 
                max(self.processing_stats['documents_attempted'], 1)
            ),
            'avg_sections_per_document': (
                self.processing_stats['total_sections_extracted'] / 
                max(self.processing_stats['documents_successful'], 1)
            ),
            'errors': len(self.processing_stats['errors']),
            'warnings': len(self.processing_stats['warnings'])
        }
        
        return summary


def create_constraint_patterns_from_config(persona: str, job_to_be_done: str) -> Dict:
    """Create constraint patterns based on the specific configuration."""
    constraint_patterns = {}
    
    job_lower = job_to_be_done.lower()
    persona_lower = persona.lower()
    
    # Vegetarian/dietary constraints
    if 'vegetarian' in job_lower or 'vegan' in job_lower:
        constraint_patterns['dietary'] = {}
        
        # Hard filters for non-vegetarian items
        if 'vegetarian' in job_lower:
            constraint_patterns['dietary']['non_vegetarian'] = (
                r'\b(chicken|turkey|beef|pork|lamb|fish|salmon|tuna|shrimp|'
                r'seafood|bacon|prosciutto|salami|ham|meat|duck|goose|'
                r'anchovy|sardine|crab|lobster)\b'
            )
        
        # Additional vegan restrictions
        if 'vegan' in job_lower:
            constraint_patterns['dietary']['non_vegan'] = (
                r'\b(cheese|milk|butter|cream|egg|dairy|yogurt|mozzarella|'
                r'parmesan|cheddar|ricotta|honey|gelatin)\b'
            )
        
        # Soft boosts for preferred items
        constraint_patterns['dietary']['gluten_free'] = (
            r'\b(gluten[\-\s]*free|gluten[\-\s]*friendly|celiac[\-\s]*safe|'
            r'wheat[\-\s]*free|gf\b)\b'
        )
        
        constraint_patterns['dietary']['vegetarian_boost'] = (
            r'\b(vegetable|veggie|plant[\-\s]*based|vegan|vegetarian|'
            r'salad|bean|lentil|tofu|quinoa|chickpea|hummus)\b'
        )
    
    # Buffet/catering specific boosts
    if 'buffet' in job_lower or 'catering' in job_lower or 'corporate' in job_lower:
        if 'catering' not in constraint_patterns:
            constraint_patterns['catering'] = {}
        
        constraint_patterns['catering']['buffet_friendly'] = (
            r'\b(buffet|serve|serving|portion|tray|platter|'
            r'large[\-\s]*batch|crowd|group|party)\b'
        )
        
        constraint_patterns['catering']['easy_prep'] = (
            r'\b(easy|simple|quick|make[\-\s]*ahead|prepare|'
            r'batch|no[\-\s]*cook|minimal)\b'
        )
    
    # Academic/research patterns
    if any(term in persona_lower for term in ['researcher', 'student', 'academic']):
        constraint_patterns['academic'] = {
            'methodology': r'\b(method|methodology|approach|analysis|study|research)\b',
            'results': r'\b(result|finding|conclusion|data|evidence|experiment)\b',
            'theory': r'\b(theory|hypothesis|model|framework|concept)\b'
        }
    
    # Business/analyst patterns
    if any(term in persona_lower for term in ['analyst', 'contractor', 'consultant']):
        constraint_patterns['business'] = {
            'planning': r'\b(plan|strategy|requirement|specification|objective)\b',
            'execution': r'\b(implement|execute|deliver|process|procedure)\b',
            'quality': r'\b(quality|standard|best[\-\s]*practice|guideline)\b'
        }
    
    return constraint_patterns


def main():
    """Enhanced main function with comprehensive error handling and optimization."""
    system = DocumentAnalysisSystem()
    
    try:
        logger.info("=== Document Analysis System Starting ===")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")
        
        # Load configuration
        config = system.load_configuration()
        if not config:
            raise RuntimeError("Failed to load configuration")
        
        persona = config['persona']
        job_to_be_done = config['job_to_be_done']
        doc_filenames = config['doc_filenames']
        
        logger.info(f"Configuration: Persona='{persona}', Job='{job_to_be_done}', Documents={len(doc_filenames)}")
        
        # Create constraint patterns based on the specific test case
        constraint_patterns = create_constraint_patterns_from_config(persona, job_to_be_done)
        
        # Initialize system components with constraints
        if not system.initialize_components(constraint_patterns=constraint_patterns):
            raise RuntimeError("Failed to initialize system components")
        
        # Process documents
        logger.info("=== Starting Document Processing ===")
        all_sections = system.process_documents(doc_filenames)
        
        if not all_sections:
            logger.warning("No sections extracted from any documents")
        
        # Analyze sections
        logger.info("=== Starting Section Analysis ===")
        analysis_results = system.analyze_sections(persona, job_to_be_done, all_sections)
        
        # Create output
        logger.info("=== Creating Output ===")
        output_data = system.create_output(config, analysis_results)
        
        # Save output
        if not system.save_output(output_data):
            raise RuntimeError("Failed to save output")
        
        # Generate and log processing summary
        summary = system.generate_processing_summary()
        logger.info("=== Processing Summary ===")
        for key, value in summary.items():
            logger.info(f"{key}: {value}")
        
        logger.info("=== Document Analysis System Completed Successfully ===")
        
    except Exception as e:
        error_msg = f"System error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        # Try to save error information
        try:
            error_output = {
                'metadata': {
                    'error': error_msg,
                    'processing_timestamp': datetime.utcnow().isoformat() + 'Z',
                    'processing_stats': system.processing_stats
                },
                'extracted_sections': [],
                'sub_section_analysis': []
            }
            system.save_output(error_output)
        except:
            pass  # If we can't save error output, just continue
        
        raise


def validate_environment():
    """Comprehensive environment validation."""
    logger.info("=== Environment Validation ===")
    
    try:
        # Check Python version
        logger.info(f"Python version: {sys.version}")
        
        # Check required directories
        for directory in [INPUT_DIR, "/tmp"]:
            if not os.path.exists(directory):
                logger.warning(f"Directory does not exist: {directory}")
            else:
                logger.info(f"Directory exists: {directory}")
        
        # Check required packages
        required_packages = ['fitz', 'sentence_transformers', 'numpy', 're']
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"Package available: {package}")
            except ImportError as e:
                logger.error(f"Package missing: {package} - {e}")
        
        # Test sentence transformer (but don't keep it loaded)
        try:
            from sentence_transformers import SentenceTransformer
            test_model = SentenceTransformer('all-MiniLM-L6-v2')
            test_embedding = test_model.encode("test", show_progress_bar=False)
            logger.info(f"Sentence transformer test successful, embedding shape: {test_embedding.shape}")
            del test_model  # Free memory
        except Exception as e:
            logger.warning(f"Sentence transformer test failed: {e}")
        
        # Check available memory (approximate)
        try:
            import psutil
            memory = psutil.virtual_memory()
            logger.info(f"Available memory: {memory.available / (1024**3):.1f} GB")
        except ImportError:
            logger.info("psutil not available, cannot check memory")
        
        logger.info("Environment validation complete")
        
    except Exception as e:
        logger.warning(f"Environment validation error: {e}")


if __name__ == "__main__":
    try:
        validate_environment()
        main()
        logger.info("Execution completed successfully")
    except Exception as e:
        logger.error(f"Fatal error during execution: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)