import matplotlib.pyplot as plt

# Data from the first image (Residual Accuracy)
sizes1 = [5, 10, 20, 50, 75, 100]
residual_accuracy = [0.998, 0.862, 0.610, 0.412, 0.410, 0.235]

# Data from the second image (Cosine Distance -> Cosine Similarity)
sizes2 = [10, 50, 100, 500, 1000]
cosine_distance = [0.01096875, 0.02450000, 0.02667100, 0.01020200, 0.02302125]
cosine_similarity = [1 - d for d in cosine_distance]

# Plotting
plt.figure(figsize=(12, 5))

# Plot 1: Residual Accuracy
plt.subplot(1, 2, 1)
plt.plot(sizes1, residual_accuracy, marker='o', color='blue', label='Phase-2 accuracy')
plt.title('Matrix Size vs Cosine similarity')
plt.xlabel('Matrix Size')
plt.ylabel('Cosine similarity')
plt.grid(True)
plt.legend()

# Plot 2: Cosine Similarity
plt.subplot(1, 2, 2)
plt.plot(sizes2, cosine_similarity, marker='s', color='green', label='Phase-3 accuracy')
plt.title('Matrix Size vs Cosine Similarity')
plt.xlabel('Matrix Size')
plt.ylabel('Cosine Similarity')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("plot.png", dpi=300)
plt.show()
