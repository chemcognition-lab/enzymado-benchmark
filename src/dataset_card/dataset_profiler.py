import pandas as pd
from pathlib import Path

class DatasetProfiler:
    def __init__(self, base_directory: str):
        self.base_dir = Path(base_directory)
        
    def _clean_dtype_name(self, pandas_dtype: str) -> str:
        """Translates pandas dtypes into more standard terminology."""
        dtype_str = str(pandas_dtype)
        if dtype_str == "object":
            return "str (or mixed)"
        return dtype_str

    def get_base_csv(self, directory: Path) -> Path | None:
        """Finds the base CSV in a folder, ignoring the split files."""
        for file in directory.glob("*.csv"):
            if "splits" not in file.name:
                return file
        return None

    def extract_schema(self, csv_path: Path, nrows: int = 10000) -> dict:
        """
        Reads a sample of the CSV to infer columns and data types.
        Returns a dictionary of {column_name: data_type}.
        """
        # Read only a chunk of the file for speed and memory efficiency
        df = pd.read_csv(csv_path, nrows=nrows)
        
        schema = {}
        for column_name, dtype in df.dtypes.items():
            schema[column_name] = self._clean_dtype_name(dtype)
            
        return schema

    def generate_report(self, output_file: str):
        """Loops through directories and appends findings to a text file."""
        print(f"Starting profiling. Output will be saved to {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            
            # Loop through subdirectories
            for prop_dir in sorted(self.base_dir.iterdir()):

                # Skip non-directories
                if not prop_dir.is_dir():
                    continue
                    
                # Get non-split csv
                csv_path = self.get_base_csv(prop_dir)
                if not csv_path:
                    continue
                    
                # Extract dataset card schema from csv
                print(f"Profiling {prop_dir.name}...")
                schema = self.extract_schema(csv_path)
                
                # Write schema to output file
                f.write(f"=== Property: {prop_dir.name} ===\n")
                f.write(f"Dataset File: {csv_path.name}\n")
                f.write("Columns:\n")
                
                for col, dtype in schema.items():
                    f.write(f"  - {col}: {dtype}\n")
                    
                f.write("\n" + "-"*40 + "\n\n")
                
        print("Done!")