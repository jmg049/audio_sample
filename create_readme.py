from pathlib import Path
import os
import shutil
from pprint import pprint

if __name__ == "__main__":
    bench_results = Path("./bench_results")
    if not bench_results.exists():
        shutil.copy("./README_BASE.md", "README.md")
    else:
        
        with open("./README_BASE.md", "r") as readme:
            base_content = readme.read()
            
        new_content = base_content + "\n\n"
            
        new_content +=  "## Benchmarks \n\n"
        table_files = [os.path.join(bench_results, f) for f  in os.listdir(bench_results)]
        print(f"Found {len(table_files)} markdown files")

        pprint(table_files)
        
        for table in table_files:
            with open(table, "r") as md:
                md_text = md.read()
                title = Path(table).stem.replace("_to_", " to ").strip()
                title = f"### {title} \n\n"
                new_content += title
                new_content += md_text
                new_content += "\n\n"
        new_content = new_content.strip()
        with open("README.md", "w+") as md:
            md.write(new_content)