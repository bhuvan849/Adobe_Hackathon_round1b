import os
import json
import time
from datetime import datetime
from document_parser import parse_document_by_structure
from analysis_engine import AnalysisEngine

INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"

def main():
    start_time = time.time()
    print(f"Starting precision analysis for directory: {INPUT_DIR}")

    try:
        input_config_path = os.path.join(INPUT_DIR, "challenge1b_input.json")
        if not os.path.exists(input_config_path):
            raise FileNotFoundError("challenge1b_input.json not found in input directory")

        with open(input_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        persona = config.get("persona", {}).get("role", "User")
        job_to_be_done = config.get("job_to_be_done", {}).get("task", "Find information")
        doc_objects = config.get("documents", [])
        doc_filenames = [doc.get("filename") for doc in doc_objects if doc.get("filename")]

        if not doc_filenames:
            raise ValueError("`documents` list in JSON is empty or missing filenames.")

        pdf_files_dir = os.path.join(INPUT_DIR, "PDFs")
        pdf_paths = [os.path.join(pdf_files_dir, fname) for fname in doc_filenames]

        print(f"Persona: {persona}")
        print(f"Job to be done: {job_to_be_done}")
        
        all_sections = []
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                print(f"Warning: PDF file not found and will be skipped: {os.path.basename(pdf_path)}")
                continue
            doc_name = os.path.basename(pdf_path)
            print(f"Parsing document: {doc_name}")
            sections = parse_document_by_structure(pdf_path)
            for section in sections:
                section['document'] = doc_name
            all_sections.extend(sections)

        if not all_sections:
            print("Warning: No sections were extracted from any of the provided documents.")
            output_data = {"metadata": config, "extracted_sections": [], "sub_section_analysis": []}
        else:
            print("Initializing analysis engine...")
            engine = AnalysisEngine()
            
            print(f"Ranking {len(all_sections)} sections based on relevance...")
            ranked_sections = engine.get_ranked_sections(persona, job_to_be_done, all_sections)

            output_data = {
                "metadata": config,
                "extracted_sections": [],
                "sub_section_analysis": []
            }
            query_for_summary = f"{persona} {job_to_be_done}"

            for i, section in enumerate(ranked_sections[:5]):
                output_data["extracted_sections"].append({
                    "document": section["document"],
                    "section_title": section["title"],
                    "importance_rank": i + 1,
                    "page_number": section["page"]
                })
                
                refined_text = engine.get_refined_summary(query_for_summary, section["content"])
                
                output_data["sub_section_analysis"].append({
                    "document": section["document"],
                    "refined_text": refined_text,
                    "page_number": section["page"]
                })

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "challenge1b_output.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)

        end_time = time.time()
        print(f"\nAnalysis complete. Output written to {output_path}")
        print(f"Total execution time: {end_time - start_time:.2f} seconds")

    except Exception as e:
        print(f"An error occurred: {e}")
        error_output = {"error": str(e), "timestamp": datetime.utcnow().isoformat() + "Z"}
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, "error.json"), 'w') as f:
            json.dump(error_output, f, indent=4)
        raise

if __name__ == "__main__":
    main()