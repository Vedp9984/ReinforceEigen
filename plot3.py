import matplotlib.pyplot as plt

# Common matrix sizes
common_sizes = [10, 50, 100]

# Filtered data for common sizes
residual_accuracy = [0.862, 0.412, 0.235]  # from first image
cosine_distance = [0.01096875, 0.02450000, 0.02667100]  # from second image
cosine_similarity = [1 - d for d in cosine_distance]

# Plotting
plt.figure(figsize=(8, 6))

plt.plot(common_sizes, residual_accuracy, marker='o', linestyle='-', color='blue', label='Phase-2 Accuracy')
plt.plot(common_sizes, cosine_similarity, marker='s', linestyle='--', color='green', label='Phase-3 Accuracy')

plt.title('Matrix Size vs Accuracy Metrics')
plt.xlabel('Matrix Size')
plt.ylabel('Cosine similarity')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("plot3.png", dpi=300)
plt.show()
