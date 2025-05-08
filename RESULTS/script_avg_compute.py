import csv
from collections import defaultdict

input_file = 'data.csv'
output_file = 'average_results.csv'

# Dictionary to store the sum and count for each matrix size
sums = defaultdict(lambda: {'residual_sum': 0.0, 'cosine_sum': 0.0, 'count': 0})

# Read the data and compute sums
with open(input_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    
    # Skip header or comment lines
    for line in reader:
        # Skip empty lines or lines with comments/headers
        if not line or not line[0].isdigit():
            continue
        
        matrix_size = int(line[0])
        final_residual = float(line[2])
        cosine_distance = float(line[3])
        
        sums[matrix_size]['residual_sum'] += final_residual
        sums[matrix_size]['cosine_sum'] += cosine_distance
        sums[matrix_size]['count'] += 1

# Compute averages
results = []
for matrix_size, data in sorted(sums.items()):
    avg_residual = data['residual_sum'] / data['count']
    avg_cosine = data['cosine_sum'] / data['count']
    results.append([matrix_size, avg_residual, avg_cosine])

# Write the results to a CSV file
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Matrix_Size', 'Avg_Final_Residual', 'Avg_Cosine_Distance'])
    for row in results:
        writer.writerow(row)

print(f"Averages computed and saved to {output_file}")

# Also print the results to console
print("\nResults:")
print("Matrix_Size | Avg_Final_Residual | Avg_Cosine_Distance")
print("-" * 50)
for row in results:
    print(f"{row[0]:10d} | {row[1]:.8f} | {row[2]:.8f}")
