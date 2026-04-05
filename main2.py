import numpy as np
import os
import faiss
from src.embedding import get_embedding
from src.search import build_index, search
from PIL import Image
import matplotlib.pyplot as plt


folder = "data"  # your main folder

images = []
for root, dirs, files in os.walk(folder):
    for file in files:
        if file.endswith((".jpg", ".png", ".jpeg")):
            images.append(os.path.join(root, file))

# Create embeddings
embeddings = []
for img in images:
    embeddings.append(get_embedding(img))

embeddings = np.array(embeddings).astype('float32')

# Build FAISS index
index = build_index(embeddings)

# Query with first image
query_path = input("Enter image path: ")
query = get_embedding(query_path).astype('float32')

distances, indices = search(index, query, k=4)

print("\n=== RESULTS ===")

print("\n=== TOP 3 MOST SIMILAR ===")

rank = 1
for i, idx in enumerate(indices[0]):
    if images[idx] == query_path:
        continue  # skip the same image

    print(f"Rank {rank}: {images[idx]}")
    print(f"Similarity: {100-distances[0][i]:.4f}%\n")

    rank += 1
    if rank >= 4:
        break

# Collect images
result_images = []

result_images = []
result_scores = []

for i, idx in enumerate(indices[0]):
    if images[idx] == query_path:
        continue
    result_images.append(images[idx])
    result_scores.append(distances[0][i])
    if len(result_images) == 3:
        break
# Create grid (2 rows)
plt.figure(figsize=(10, 6))

# 🟢 Your Bag (top, full width)
img = Image.open(query_path)
plt.subplot(2, 3, 2)
plt.imshow(img)
plt.title("Your Bag")
plt.axis("off")

# 🟢 Top 3 results (bottom row)
for i, path in enumerate(result_images):
    img = Image.open(path)
    plt.subplot(2, 3, i + 4)
    plt.imshow(img)
    plt.title(f"Bag {i+1}\nSimilarity: {100-result_scores[i]:.4f}")
    plt.axis("off")

plt.tight_layout()
plt.show()
