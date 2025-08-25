# main.py
# Enhanced K-Means Clustering Application with Data Cleaning

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# --- 1. Load the Dataset ---
print("Loading dataset 'data.csv'...")
try:
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv('data.csv')
    print("Dataset loaded successfully.")
    print(f"Dataset shape: {df.shape}")  # Shows (number_of_rows, number_of_columns)
except FileNotFoundError:
    print("Error: File 'data.csv' not found. Please ensure it's in the project directory.")
    exit()

# Display the first few rows and column names to understand the data
print("\nFirst 5 rows of the dataset:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())

# --- 2. Data Preprocessing with Missing Value Handling ---
print("\nStarting data preprocessing with missing value handling...")

# Select relevant numerical features for clustering.
# Common weather features: temperature (TEMP, MAX, MIN), dew point (DEWP), pressure (SLP), wind speed (WDSP), precipitation (PRCP)
# IMPORTANT: Adjust these feature names based on the actual columns in your downloaded CSV!
selected_features = ['TEMP', 'DEWP', 'SLP', 'WDSP', 'PRCP']

# Extract the selected features into a new DataFrame
X = df[selected_features].copy()

# Identify and replace missing value codes with NaN
missing_value_codes = [9999.9, 99.99]
print(f"\nReplacing missing value codes {missing_value_codes} with NaN...")

for feature in selected_features:
    # Count values that match missing value codes before replacement
    for code in missing_value_codes:
        count_before = (X[feature] == code).sum()
        if count_before > 0:
            print(f"Found {count_before} instances of {code} in {feature}")

    # Replace missing value codes with NaN
    X[feature] = X[feature].replace(missing_value_codes, np.nan)

# Report missing values after replacement
print("\nMissing values after replacing codes with NaN:")
print(X.isna().sum())

# Handle remaining missing values: Replace NaNs with the mean of the column
print("\nImputing remaining missing values with column means...")
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
# Convert back to DataFrame for clarity
X_clean = pd.DataFrame(X_imputed, columns=X.columns)

# Standardize the features (Crucial for K-Means as it uses distance metrics)
# This scales the data to have a mean of 0 and a standard deviation of 1.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)
print("Data cleaned, imputed, and scaled.")


# --- 3. Custom K-Means Implementation ---
def custom_kmeans(data, k, max_iterations=100, tolerance=1e-4, distance_metric='euclidean'):
    """
    Custom implementation of the K-Means algorithm

    Parameters:
    data: numpy array of scaled data points
    k: number of clusters
    max_iterations: maximum number of iterations
    tolerance: convergence threshold for centroid movement
    distance_metric: 'euclidean' or 'manhattan'

    Returns:
    centroids: final cluster centers
    labels: cluster assignments for each data point
    wcss: within-cluster sum of squares
    history: tracking of centroid movements and WCSS over iterations
    """
    # Initialize centroids using k-means++ initialization
    n_samples, n_features = data.shape
    centroids = np.zeros((k, n_features))

    # First centroid: choose a random data point
    np.random.seed(42)  # For reproducibility
    first_idx = np.random.randint(n_samples)
    centroids[0] = data[first_idx]

    # Select remaining centroids using k-means++ algorithm
    for i in range(1, k):
        # Calculate distances to the nearest centroid for each point
        distances = np.min(cdist(data, centroids[:i], metric=distance_metric), axis=1)
        # probabilities proportional to distance squared
        probabilities = distances ** 2
        probabilities /= probabilities.sum()
        # Choose next centroid
        cumulative_prob = np.cumsum(probabilities)
        r = np.random.rand()
        for j, p in enumerate(cumulative_prob):
            if r < p:
                centroids[i] = data[j]
                break

    # Initialize variables for tracking convergence
    history = {
        'centroids': [centroids.copy()],
        'wcss': [],
        'labels': []
    }

    # Iterate until convergence or max iterations reached
    for iteration in range(max_iterations):
        # Step 1: Assignment - Assign each data point to the nearest centroid
        if distance_metric == 'euclidean':
            distances = cdist(data, centroids, metric='euclidean')
        elif distance_metric == 'manhattan':
            distances = cdist(data, centroids, metric='cityblock')

        labels = np.argmin(distances, axis=1)

        # Step 2: Update - Recalculate centroids as mean of assigned points
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                new_centroids[i] = centroids[i]  # Keep centroid unchanged if no points assigned

        # Calculate WCSS (Within-Cluster Sum of Squares)
        wcss = 0
        for i in range(k):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                if distance_metric == 'euclidean':
                    wcss += np.sum((cluster_points - new_centroids[i]) ** 2)
                elif distance_metric == 'manhattan':
                    wcss += np.sum(np.abs(cluster_points - new_centroids[i]))

        # Store history
        history['centroids'].append(new_centroids.copy())
        history['wcss'].append(wcss)
        history['labels'].append(labels.copy())

        # Check for convergence (if centroids don't change much)
        centroid_shift = np.sqrt(np.sum((new_centroids - centroids) ** 2, axis=1)).max()
        if centroid_shift < tolerance:
            print(f"Converged after {iteration + 1} iterations.")
            break

        centroids = new_centroids

    else:
        print(f"Reached maximum iterations ({max_iterations}) without full convergence.")

    return centroids, labels, wcss, history


# --- 4. Determining the Optimal Number of Clusters (k) ---
print("\nCalculating elbow method and silhouette scores to find optimal k...")
inertia = []
silhouette_scores = []
# Try k values from 2 to 10 (silhouette score requires at least 2 clusters)
k_range = range(2, 11)

for k in k_range:
    # Use scikit-learn for faster evaluation of multiple k values
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(silhouette_avg)
    print(f"For k = {k}, Silhouette Score = {silhouette_avg:.4f}")

# Plot the Elbow Method graph
plt.figure(figsize=(15, 5))

# Subplot 1: Elbow Method
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, 'bx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia (WCSS)')
plt.title('Elbow Method For Optimal k')

# Subplot 2: Silhouette Scores
plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'rx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores For Different k')
plt.tight_layout()
plt.show()

# Find the optimal k based on silhouette score (higher is better)
optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
print(f"\nOptimal k based on silhouette score: {optimal_k_silhouette} (score = {max(silhouette_scores):.4f})")

# --- 5. Applying Custom K-Means with the chosen k ---
chosen_k = optimal_k_silhouette
distance_metric = 'euclidean'  # Can change to 'manhattan' for different distance metric

print(f"\nApplying Custom K-Means with optimal k = {chosen_k} and {distance_metric} distance...")
centroids, custom_labels, custom_wcss, history = custom_kmeans(
    X_scaled, chosen_k, distance_metric=distance_metric
)

# Also run scikit-learn implementation for comparison
sklearn_kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
sklearn_kmeans.fit(X_scaled)
sklearn_labels = sklearn_kmeans.labels_
sklearn_wcss = sklearn_kmeans.inertia_

# Add the cluster labels back to the original CLEANED DataFrame
X_clean['Custom_Cluster'] = custom_labels
X_clean['SKLearn_Cluster'] = sklearn_labels
df['Cluster'] = custom_labels  # Using custom clusters for further analysis

# --- 6. Evaluating the Clusters ---
print("\n--- Cluster Evaluation ---")

# Compare custom implementation with scikit-learn
print("Comparison between Custom and Scikit-learn implementations:")
print(f"Custom K-Means WCSS: {custom_wcss:.4f}")
print(f"Scikit-Learn K-Means WCSS: {sklearn_wcss:.4f}")

# Calculate silhouette scores for both implementations
custom_silhouette = silhouette_score(X_scaled, custom_labels)
sklearn_silhouette = silhouette_score(X_scaled, sklearn_labels)

print(f"Custom K-Means Silhouette Score: {custom_silhouette:.4f}")
print(f"Scikit-Learn K-Means Silhouette Score: {sklearn_silhouette:.4f}")

# Check if clusters are similar (they might have different labels but same grouping)
# We'll use adjusted rand score to compare cluster assignments
from sklearn.metrics import adjusted_rand_score

ari_score = adjusted_rand_score(custom_labels, sklearn_labels)
print(f"Adjusted Rand Index between implementations: {ari_score:.4f}")


# Interpret the silhouette score
def interpret_silhouette(score):
    if score > 0.7:
        return "Strong cluster structure"
    elif score > 0.5:
        return "Reasonable cluster structure"
    elif score > 0.25:
        return "Weak cluster structure"
    else:
        return "No substantial cluster structure"


print(f"Custom implementation interpretation: {interpret_silhouette(custom_silhouette)}")
print(f"Scikit-learn implementation interpretation: {interpret_silhouette(sklearn_silhouette)}")

# --- 7. Visualizing the Algorithm Convergence ---
print("\nVisualizing algorithm convergence...")
plt.figure(figsize=(15, 5))

# Subplot 1: WCSS over iterations
plt.subplot(1, 2, 1)
plt.plot(range(1, len(history['wcss']) + 1), history['wcss'], 'b-')
plt.xlabel('Iteration')
plt.ylabel('WCSS')
plt.title('WCSS over Iterations')

# Subplot 2: Centroid movement over iterations
centroid_movement = []
for i in range(1, len(history['centroids'])):
    movement = np.sqrt(np.sum((history['centroids'][i] - history['centroids'][i - 1]) ** 2, axis=1)).mean()
    centroid_movement.append(movement)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(centroid_movement) + 1), centroid_movement, 'r-')
plt.xlabel('Iteration')
plt.ylabel('Average Centroid Movement')
plt.title('Centroid Convergence')

plt.tight_layout()
plt.show()

# --- 8. Analyzing the Results ---
print("\n--- Cluster Analysis ---")
# See the size of each cluster
cluster_counts = X_clean['Custom_Cluster'].value_counts().sort_index()
print("Number of data points in each cluster (Custom implementation):")
print(cluster_counts)

# Analyze the mean values of features for each cluster (in original scale, not standardized)
# First, let's get the unscaled data with clusters
X_unscaled = X_clean.drop(['Custom_Cluster', 'SKLearn_Cluster'], axis=1, errors='ignore')
X_unscaled['Cluster'] = custom_labels

cluster_means = X_unscaled.groupby('Cluster').mean()
print("\nMean values for features per cluster (original scale):")
print(cluster_means)

# --- 9. Visualizing the Clusters ---
print("\nGenerating visualization of clusters...")
plt.figure(figsize=(15, 10))

# Subplot 1: Scatter plot of two features with custom clusters
plt.subplot(2, 2, 1)
scatter = plt.scatter(X_unscaled['TEMP'], X_unscaled['PRCP'], c=custom_labels, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cluster Label')
plt.xlabel('Temperature (TEMP)')
plt.ylabel('Precipitation (PRCP)')
plt.title(f'Custom K-Means Clustering (k={chosen_k})')

# Subplot 2: Scatter plot of two features with sklearn clusters
plt.subplot(2, 2, 2)
scatter = plt.scatter(X_unscaled['TEMP'], X_unscaled['PRCP'], c=sklearn_labels, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cluster Label')
plt.xlabel('Temperature (TEMP)')
plt.ylabel('Precipitation (PRCP)')
plt.title(f'Scikit-Learn K-Means Clustering (k={chosen_k})')

# Subplot 3: Bar chart of cluster sizes
plt.subplot(2, 2, 3)
plt.bar(cluster_counts.index.astype(str), cluster_counts.values)
plt.xlabel('Cluster')
plt.ylabel('Number of Data Points')
plt.title('Cluster Sizes (Custom Implementation)')
for i, v in enumerate(cluster_counts.values):
    plt.text(i, v + 0.01 * max(cluster_counts.values), str(v), ha='center')

# Subplot 4: Feature means by cluster
plt.subplot(2, 2, 4)
for feature in cluster_means.columns:
    plt.plot(cluster_means.index, cluster_means[feature], 'o-', label=feature)
plt.xlabel('Cluster')
plt.ylabel('Original Value')
plt.title('Feature Means by Cluster (Original Scale)')
plt.legend()

plt.tight_layout()
plt.show()

# --- 10. Detailed Silhouette Analysis ---
print("\n--- Detailed Silhouette Analysis ---")
from sklearn.metrics import silhouette_samples

# Compute the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(X_scaled, custom_labels)

# Create a subplot with silhouette plot
plt.figure(figsize=(11, 9))

# The silhouette coefficient can range from -1 to 1
plt.xlim([-0.2, 1])
# The (n_clusters+1)*10 is for inserting blank space between silhouette plots
plt.ylim([0, len(X_scaled) + (chosen_k + 1) * 10])

y_lower = 10
for i in range(chosen_k):
    # Aggregate the silhouette scores for samples belonging to cluster i
    ith_cluster_silhouette_values = sample_silhouette_values[custom_labels == i]
    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = plt.cm.viridis(float(i) / chosen_k)
    plt.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)

    # Label the silhouette plots with their cluster numbers at the middle
    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples

plt.title("Silhouette Plot for the Custom K-Means Clusters")
plt.xlabel("Silhouette Coefficient Values")
plt.ylabel("Cluster Label")

# The vertical line for average silhouette score of all the values
plt.axvline(x=custom_silhouette, color="red", linestyle="--")
plt.text(custom_silhouette + 0.02, y_lower * 0.5, f'Average: {custom_silhouette:.4f}')

# Add some whitespace for readability
plt.yticks([])
plt.tight_layout()
plt.show()

# --- 11. Interpret the Clusters ---
print("\n--- Cluster Interpretation ---")
print("Based on the feature means for each cluster, we can interpret the clusters as follows:")

# Create a description for each cluster based on its characteristics
for cluster_id in range(chosen_k):
    cluster_data = cluster_means.loc[cluster_id]
    print(f"\nCluster {cluster_id} (n={cluster_counts[cluster_id]}):")

    # Temperature interpretation
    temp = cluster_data['TEMP']
    if temp < 10:
        temp_desc = "very cold"
    elif temp < 20:
        temp_desc = "cold"
    elif temp < 30:
        temp_desc = "moderate"
    elif temp < 40:
        temp_desc = "warm"
    else:
        temp_desc = "hot"

    # Precipitation interpretation
    prcp = cluster_data['PRCP']
    if prcp < 0.1:
        prcp_desc = "dry"
    elif prcp < 1.0:
        prcp_desc = "light precipitation"
    elif prcp < 5.0:
        prcp_desc = "moderate precipitation"
    else:
        prcp_desc = "heavy precipitation"

    # Dew point interpretation (related to humidity)
    dewp = cluster_data['DEWP']
    if dewp < 10:
        dewp_desc = "low humidity"
    elif dewp < 20:
        dewp_desc = "moderate humidity"
    else:
        dewp_desc = "high humidity"

    # Wind speed interpretation
    wdsp = cluster_data['WDSP']
    if wdsp < 5:
        wdsp_desc = "calm"
    elif wdsp < 10:
        wdsp_desc = "breezy"
    elif wdsp < 20:
        wdsp_desc = "windy"
    else:
        wdsp_desc = "very windy"

    print(f"  - {temp_desc} (avg temp: {temp:.1f}°F)")
    print(f"  - {prcp_desc} (avg precip: {prcp:.2f} inches)")
    print(f"  - {dewp_desc} (avg dew point: {dewp:.1f}°F)")
    print(f"  - {wdsp_desc} (avg wind speed: {wdsp:.1f} knots)")

print("\nK-Means clustering application finished successfully with cleaned data!")