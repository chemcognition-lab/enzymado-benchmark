import json
import argparse
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

def render_datacard(json_path: str, template_path: str, output_path: str):
    """
    Reads a JSON file and a Jinja2 markdown template, then renders the template 
    using the JSON data and outputs it to the specified file.
    """
    json_file = Path(json_path)
    template_file = Path(template_path)
    out_file = Path(output_path)

    if not json_file.exists():
        raise FileNotFoundError(f"JSON file not found: {json_file}")
    if not template_file.exists():
        raise FileNotFoundError(f"Template file not found: {template_file}")

    print(f"Loading data from: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Set up Jinja2 environment focusing on the directory of the template
    env = Environment(loader=FileSystemLoader(template_file.parent))
    template = env.get_template(template_file.name)

    print(f"Rendering template: {template_file}")
    rendered_content = template.render(**data)

    # Ensure output directory exists before saving
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(rendered_content)
    
    print(f"Successfully generated data card at: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a markdown data card from a Jinja template and JSON data.")
    parser.add_argument("--json", required=True, help="Path to the input JSON file with dataset statistics")
    parser.add_argument("--template", required=True, help="Path to the input markdown Jinja template")
    parser.add_argument("--output", default="final_datacard.md", help="Path to save the rendered output markdown file")
    
    args = parser.parse_args()
    render_datacard(args.json, args.template, args.output)
