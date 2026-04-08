from src.dataset_card.dataset_card_populator import DatasetCardPopulator

# Run script from project root (enzymado-benchmark/)
if __name__ == "__main__":
    # Define paths
    BASE_DIR = "data/processed/properties"
    OUTPUT_TXT = "dataset_schema_overview.txt"
    
    # Initialize and run
    profiler = DatasetCardPopulator(BASE_DIR)
    profiler.run(OUTPUT_TXT)