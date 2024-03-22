from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.cluster import KMeans

def cluster_kmeans_sentence(data, num_clusters):
    """
    Clusters sentences using the K-Means algorithm.

    Parameters:
    data (dict): A dictionary where keys represent unique identifiers for sentences,
                   and values are the corresponding sentence data.
    num_clusters (int): The number of clusters to form.

    Returns:
    clustered_dict (dict): A dictionary where keys are the same as the input 'data' keys,
                            and values are dictionaries containing the original sentence data ('data')
                            and the assigned cluster label ('cluster').
    """
    if data.__len__() < num_clusters:
        print(
            f"Error! Cannot cluster into {num_clusters} Clusters in a document with {data.__len__()} Sentences. Choose num_cluster to be less than {data.__len__()}")
        return

    # Convert data to numpy array
    data_array = list(data)

    # Create and fit K-Means model
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)
    labels = kmeans.fit_predict(data_array)

    # Create a dictionary with original data and assigned cluster labels
    clustered_dict = {key: {'data': data[key], 'cluster': label} for key, label in zip(range(0, len(data)), labels)}

    return clustered_dict

def cluster_kmeans_word(data, num_clusters):
    """
    Clusters words using the K-Means algorithm.

    Parameters:
    data (list): A list containing word data to be clustered.
    num_clusters (int): The number of clusters to form.

    Returns:
    clustered_dict (dict): A dictionary where keys are indices of the input 'data' list,
                            and values are dictionaries containing the original word data ('data')
                            and the assigned cluster label ('cluster').
    """
    if data.__len__() < num_clusters:
        print(
            f"Error! Cannot cluster into {num_clusters} Clusters in a document with {len(data)} Sentences. Choose num_cluster to be less than {len(data)}")
        return

    # Convert data to numpy array
    data_array = np.array(list(data))

    # Create and fit K-Means model
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)
    labels = kmeans.fit_predict(data_array)
    centers = kmeans.cluster_centers_

    # Create a dictionary with original data and assigned cluster labels
    clustered_dict = {i: {'data': (row[0], row[1]), 'cluster': labels[i]} for i, row in enumerate(data)}

    return clustered_dict

def calc_silhouette_scores(data, max_clusters=3):
    """
    Calculate silhouette scores for different numbers of clusters using K-Means algorithm.

    Parameters:
    data (array-like): Input data for clustering.
    max_clusters (int, optional): The maximum number of clusters to consider. Default is 3.

    Returns:
    scores (list): A list of silhouette scores for each number of clusters from 2 to 'max_clusters'.
    """
    scores = []

    for i in range(2, max_clusters + 1):
        # Create and fit K-Means model
        kmeans = KMeans(n_clusters=i, n_init=10)
        kmeans.fit_predict(data)

        # Calculate silhouette score and append to the scores list
        scores.append(silhouette_score(data, kmeans.labels_))

    return scores

def tsne_plot(word_cloud, cluster_count=None):
    """
    Generate a TSNE plot for different types of embeddings in a word cloud.

    Parameters:
    word_cloud (dict): A dictionary containing embeddings for each type.
       Format: {embedding_type: {key: {element: {'data': (x, y)}}}}
    cluster_count (int): The number of clusters to form for K-Means clustering.

    Returns:
    None: Displays a TSNE plot using Matplotlib.
    """
    plot_data = {}
    used_k = {}
    inertia = {}
    color_mappings = {}
    for embedding_typ in word_cloud:
        x_means = {}
        y_means = {}
        for key in word_cloud[embedding_typ].keys():
            if word_cloud[embedding_typ] is None:
                print("the given document withholds no data")
                return

            # Prepare data for TSNE
            data_array = []
            for element in word_cloud[embedding_typ][key]:
                data_array.append(word_cloud[embedding_typ][key][element]["data"])

            tsne_model = TSNE(perplexity=data_array.__len__() - 1, n_components=2, init='pca', n_iter=3500,
                              random_state=10)
            tsne_values = tsne_model.fit_transform(np.array(data_array))

            # Get mean coordinates for every document
            x_means[key] = np.mean(tsne_values[:, 0])
            y_means[key] = np.mean(tsne_values[:, 1])

        # use silhouette score or determined K
        data = [(x_means[key], y_means[key]) for key in x_means]
        data = np.array(data)
        if cluster_count == None:
            silhouette_scores = calc_silhouette_scores(data, len(data) - 1)
            kmeans = KMeans(n_clusters=silhouette_scores.index(max(silhouette_scores)) + 2, n_init='auto')
            used_k[embedding_typ] = silhouette_scores.index(max(silhouette_scores)) + 2
        else:
            kmeans = KMeans(n_clusters=cluster_count, n_init='auto')

        kmeans.fit(data)
        labels = kmeans.labels_
        point_labels = {point: label for point, label in zip(x_means.keys(), labels)}
        centers = kmeans.cluster_centers_
        color_mappings[embedding_typ] = create_random_color_list(len(centers))
        print(f"Inertia für {embedding_typ}: {kmeans.inertia_}")
        plot_data[embedding_typ] = [x_means, y_means, point_labels, centers]

    # Generate plot
    fig, axes = plt.subplots(2, 2, figsize=(20, 20), sharey=True)
    axes.flatten()
    for i, embedding_typ in enumerate(plot_data):
        row = i // 2
        col = i % 2
        axes[row, col].set_title(f"TSNE Plot ({embedding_typ})")

        for key in plot_data[embedding_typ][0].keys():
            axes[row, col].scatter(plot_data[embedding_typ][0][key], plot_data[embedding_typ][1][key],
                                   color=color_mappings[embedding_typ][plot_data[embedding_typ][2][key]])

            axes[row, col].annotate(plot_data[embedding_typ][2][key],
                                    xy=(plot_data[embedding_typ][0][key], plot_data[embedding_typ][1][key]))

        for center in plot_data[embedding_typ][3]:
            axes[row, col].scatter(center[0], center[1], marker='x', c="red", s=100, alpha=0.5)

    plt.show()
    if cluster_count == None:
        return used_k

def tsne_plot_document(word_cloud, cluster_count):
    """
    Generate a TSNE plot for a given document using word embeddings.

    Parameters:
    word_cloud (dict): A dictionary containing word embeddings for the document.
        Format: {element_id: {'data': (x, y)}}
    cluster_count (int): The number of clusters to form for K-Means clustering.

    Returns:
    None: Displays a TSNE plot using Matplotlib and prints the inertia value.
    """
    color_pallet = create_random_color_list(cluster_count)
    for key in word_cloud.keys():
        if word_cloud[key] is None:
            print("the given document withholds no data")
            return
        data_array = []

        # Prepare data for TSNE
        for element in word_cloud[key]:
            data_array.append(word_cloud[key][element]["data"])

        tsne_model = TSNE(perplexity=data_array.__len__() - 1, n_components=2, init='pca', n_iter=3500, random_state=10)
        tsne_values = tsne_model.fit_transform(np.array(data_array))

        # Extract x and y coordinates
        x = [value[0] for value in tsne_values]
        y = [value[1] for value in tsne_values]

        # Prepare data for K-Means clustering
        data = list(zip(x, y))
        data = np.array(data)
        kmeans = KMeans(n_clusters=cluster_count, n_init='auto')
        kmeans.fit(data)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        inertia = kmeans.inertia_

        merged_data = [{'point': point, 'cluster': cluster} for point, cluster in zip(data, labels)]

    # Generate plot
    plt.figure(figsize=(16, 10))
    plt.title("TSNE Plot")
    for i in range(0, len(merged_data)):
        plt.scatter(merged_data[i]['point'][0], merged_data[i]['point'][1],
                    color=color_pallet[merged_data[i]['cluster']])

        plt.annotate(merged_data[i]['cluster'], xy=(merged_data[i]['point'][0], merged_data[i]['point'][1]))

    for center in centers:
        plt.scatter(center[0], center[1], marker='x', c="red", s=100, alpha=0.5)
    plt.show()
    print(f"Inertia: {inertia}")

def generate_random_color():
    """
    Generate a random RGB color code in hexadecimal format.

    Returns:
    str: A string representing a random color code in the format '#RRGGBB'.
    """
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return '#%02x%02x%02x' % (r, g, b)

def create_random_color_list(n):
    """
    Generate a list of random RGB color codes in hexadecimal format.

    Parameters:
    n (int): The number of random colors to generate.

    Returns:
    list: A list of strings, each representing a random color code in the format '#RRGGBB'.
    """
    return [generate_random_color() for _ in range(n)]

def find_nearest_neighbors(cluster_center, embedding_points, n_neighbors=2, metric='euclidean'):
    """
    Find the n nearest neighbors of a given point from a list of points.
    Parameters:
        cluster_center: Cluster center the neighbors should be calculated for
        embedding_points: List of points that could possibly be nearest neighbors for the cluster center
        n_neighbors: Amount of neighbors that should be calculated
        metric: The distance metric to use (default is 'euclidean')
    Returns:
        neighbors: The n nearest neighbors of the center from embedding_points
    """
    #Change shape for NearestNeighbors classificator
    cluster_center = np.array(cluster_center).reshape(1, -1) 

    #Init model and calculate the indices for the nearest neighbors
    if len(embedding_points) < n_neighbors:
        n_neighbors = len(embedding_points) #If there are not enough samples, take len of samples - 1
    knn_model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    knn_model.fit(embedding_points)
    _, indices = knn_model.kneighbors(cluster_center)

    #Extract the nearest neighbors
    nearest_neighbors = []
    for ind in indices[0]:
        nearest_neighbors.append(embedding_points[ind])

    return nearest_neighbors

def plot_n_neighbors(word_cloud, cluster_count=None, n_neighbors=1):
    """
    Plot the nearest n_neighbors for clusters resulting from word_cloud and cluster_count.
    Parameters:
        word_cloud: Calculated clusters
        cluster_count: Count of the clusters
        n_neighbors: Amount of neighbors that should be calculated
    """
    plot_data = {}
    used_k = {}
    inertia = {}
    color_mappings = {}
    for embedding_typ in word_cloud:
        x_means = {}
        y_means = {}
        for key in word_cloud[embedding_typ].keys():
            if word_cloud[embedding_typ] is None:
                print("the given document withholds no data")
                return

            # Prepare data for TSNE
            data_array = []
            for element in word_cloud[embedding_typ][key]:
                data_array.append(word_cloud[embedding_typ][key][element]["data"])

            tsne_model = TSNE(perplexity=data_array.__len__() - 1, n_components=2, init='pca', n_iter=3500,
                              random_state=10)
            tsne_values = tsne_model.fit_transform(np.array(data_array))

            # Get mean coordinates for every document
            x_means[key] = np.mean(tsne_values[:, 0])
            y_means[key] = np.mean(tsne_values[:, 1])

        # use silhouette score or determined K
        data = [(x_means[key], y_means[key]) for key in x_means]
        data = np.array(data)
        if cluster_count == None:
            silhouette_scores = calc_silhouette_scores(data, len(data) - 1)
            kmeans = KMeans(n_clusters=silhouette_scores.index(max(silhouette_scores)) + 2, n_init='auto')
            used_k[embedding_typ] = silhouette_scores.index(max(silhouette_scores)) + 2
        else:
            kmeans = KMeans(n_clusters=cluster_count, n_init='auto')

        kmeans.fit(data)
        labels = kmeans.labels_
        point_labels = {point: label for point, label in zip(x_means.keys(), labels)}
        centers = kmeans.cluster_centers_
        color_mappings[embedding_typ] = create_random_color_list(len(centers))
        plot_data[embedding_typ] = [x_means, y_means, point_labels, centers]

    # Generate plot
    fig, axes = plt.subplots(2, 2, figsize=(20, 20), sharey=True)
    axes.flatten()
    for i, embedding_typ in enumerate(plot_data):
        row = i // 2
        col = i % 2
        axes[row, col].set_title(f"TSNE Plot ({embedding_typ})")
        
        x_means = plot_data[embedding_typ][0]
        y_means = plot_data[embedding_typ][1]
        point_labels = plot_data[embedding_typ][2]
        centers = plot_data[embedding_typ][3]

        #Map points to their corresponding clusters
        point_mappings = {}
        for index, center in enumerate(centers):
            point_mappings[index] = []
            for key, center_label in point_labels.items():
                if center_label == index:
                    point = (x_means[key], y_means[key])
                    if point not in point_mappings[index]:
                        point_mappings[index].append([x_means[key], y_means[key]])

        #Plot each cluster center and its corresponding nearest neighbors
        for index, center in enumerate(centers):
            center_point = (center[0], center[1])
            cluster_points = point_mappings[index]
            axes[row, col].scatter(center[0], center[1], marker='x', s=100, alpha=0.5,c=color_mappings[embedding_typ][index])
            axes[row, col].annotate(index, xy=(center[0], center[1]))
            n_nearest_neighbors = find_nearest_neighbors(center_point, cluster_points, n_neighbors=n_neighbors)
            for point in n_nearest_neighbors:
                axes[row, col].scatter(point[0], point[1], c=color_mappings[embedding_typ][index])
                axes[row, col].annotate(index, xy=(point[0], point[1]))
        