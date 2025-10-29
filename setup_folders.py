import os

for folder in ["data", "models", "logs", "history"]:
    os.makedirs(folder, exist_ok=True)
    open(os.path.join(folder, ".gitkeep"), "a").close()

print("✅ All folders created with .gitkeep files!")
