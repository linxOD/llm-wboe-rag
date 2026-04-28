from glob import glob

files = glob("llm_corpus/*.toon")
for file in files:
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()
        print(f"Content of {file}:\n{content}\n{'-'*40}\n")
    new_content = content.replace("\"", "")
    with open(file, "w", encoding="utf-8") as f:
        f.write(new_content)
