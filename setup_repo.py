import os

# Struttura della repository
structure = {
    "data/raw": [],
    "src": [   
        "__init__.py",
        "data_utils.py",
        "model_utils.py",
        "trainer_utils.py"
        "inference_utils.py"
    ],
    "scripts": [
        "train.py",
        "predict.py"
    ],
    "outputs/models": [],
    "notebooks": []
}

print("Creazione struttura repository...")
for folder, files in structure.items():
    os.makedirs(folder, exist_ok=True)
    for filename in files:
        path = os.path.join(folder, filename)
        with open(path, "w") as f:
            f.write(f"# {filename}\n")
    print(f"Creata cartella: {folder}")

# gitignore
gitignore = """
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
data/
outputs/models/
outputs/predictions/
wandb/
.env
poetry.lock
"""
with open(".gitignore", "w") as f:
    f.write(gitignore.strip())

print("\nFatto!")