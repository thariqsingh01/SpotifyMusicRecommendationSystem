import ast
import os
import networkx as nx
import matplotlib.pyplot as plt

def find_imports(filepath):
    """Extract import statements from a Python file."""
    imports = []
    with open(filepath, "r") as file:
        tree = ast.parse(file.read(), filename=filepath)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)
    return imports

def analyze_directory(directory):
    """Analyze all Python files in the directory for import dependencies."""
    dependencies = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                imports = find_imports(filepath)
                dependencies[filepath] = imports
    return dependencies

# Adjust the directory path as needed
directory_path = "D:/Varsity/Honours/Semester 2/Comp700/SpotifyMusicRecommendationSystem/SpotifyMusicRecommendationSystem"
dependencies = analyze_directory(directory_path)

# Create a graph
G = nx.DiGraph()

# Add nodes and edges from your dependencies data
for file, imports in dependencies.items():
    for imp in imports:
        # Handle local files if needed, e.g., "app.models" to "app/models.py"
        imp_file = imp.replace(".", "/") + ".py"
        if os.path.isfile(os.path.join(directory_path, imp_file)):
            G.add_edge(file, imp_file)

# Draw the graph
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, seed=42)  # Adjust layout
nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10, font_weight="bold", arrows=True)
plt.title("Module Dependencies")
plt.show()
