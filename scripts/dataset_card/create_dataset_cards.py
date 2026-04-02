from src.dataset_card.dataset_profiler import DatasetProfiler

if __name__ == "__main__":
    # Define paths
    BASE_DIR = "data/processed/properties"
    OUTPUT_TXT = "dataset_schema_overview.txt"
    
    # Initialize and run
    profiler = DatasetProfiler(BASE_DIR)
    profiler.generate_report(OUTPUT_TXT)