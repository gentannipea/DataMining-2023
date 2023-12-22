"""
k-SubMix clustering

Implementation of the k-SubMix algorithm as described in the Paper
'k-SubMix: Common Subspace Clustering on Mixed-type data'
"""

import sys
import numpy as np
from scipy.stats import ortho_group
from sklearn.utils import check_random_state
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_mutual_info_score as ami, \
    adjusted_rand_score as rand, accuracy_score
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from itertools import permutations
import timeit
import os
from kmodes.kprototypes import KPrototypes

ACCEPTED_NUMERICAL_ERROR = 1e-6
NOISE_SPACE_THRESHOLD = -1e-7

GAMMA: int = None
EARLY_STOPPING_NMI: float = None


def subMix(X, X_num, X_cat, ground_truth, n_clusters, V, m, l, Pc, Pn, centers_num, centers_cat, max_iter,random_state):
    """
    Execute the subMix algorithm. The algorithm will search for the optimal cluster subspaces and assignments
    depending on the input number of clusters and subspaces. The number of subspaces will automatically be traced by the
    length of the input n_clusters array.
    :param X: input data
    :param X_num: numerical input data
    :param X_cat: categorical input data
    :param n_clusters: list containing number of clusters for each subspace
    :param V: orthogonal rotation matrix
    :param m: list containing number of numerical dimensionalities for each subspace
    :param l: list containing number of categorical dimensionalities for each subspace
    :param Pc: list containing categorical projections for each subspace
    :param Pn: list containing numerical projections for each subspace
    :param centers_num: list containing the numerical cluster centers for each subspace
    :param centers_num: list containing the categorical cluster centers for each subspace
    :param max_iter: maximum number of iterations for the algorithm
    :param random_state: use a fixed random state to get a repeatable solution
    :return: labels, centers_num, centers_cat, V, m, l, Pn, Pc, n_clusters, scatter_matrices, scatter_cat
    """
    V, m, l, Pn, Pc, centers_num, centers_cat, random_state, subspaces, labels, scatter_matrices, scatter_cat = _initialize_kSubMix_parameters(
        X, X_num, X_cat, n_clusters, V, m, l, Pn, Pc, centers_num, centers_cat, max_iter, random_state)
    # Check if labels stay the same (break condition)
    old_labels = None
    eigenvalues_cat = []
    eigenvalues_num = []
    # Repeat actions until convergence or max_iter
    for iteration in range(max_iter):
        # Execute basic kmeans steps
        for i in range(subspaces):
            # Assign each point to closest cluster center
            labels[i] = _assign_labels(X_num, X_cat, V, centers_num[i], centers_cat[i], Pn[i], Pc[i], n_clusters[i],labels[i])
            # Update centers and scatter matrices depending on cluster assignments
            centers_num[i], centers_cat[i], scatter_matrices[i], scatter_cat[i] = _update_centers_and_scatter_matrices(X_num, X_cat, n_clusters[i], labels[i])
            # Remove empty clusters
            # TODO ADD CAT SCATTER
            # n_clusters[i], centers_num[i], centers_cat[i], scatter_matrices[i], labels[i] = _remove_empty_cluster(n_clusters[i], centers_num[i],centers_cat[i],scatter_matrices[i],labels[i])
        # Check if labels have not changed
        if _are_labels_equal(labels, old_labels):
            print(f'Iterations at exit: {iteration + 1}')
            break
        else:
            old_labels = labels.copy()
        # Update rotation for each pair of subspaces
        for i in range(subspaces - 1):
            for j in range(i + 1, subspaces):
                # Do rotation calculations
                P_1_new_num, P_2_new_num, P_1_new_cat, P_2_new_cat, V_new, eigenvalues_num, eigenvalues_cat = _update_rotation(X_num, X_cat, V, i, j, n_clusters, labels, Pn, Pc, scatter_matrices, scatter_cat, eigenvalues_num,eigenvalues_cat)
                # Update V, m,,l, Pc,Pn
                m[i] = len(P_1_new_num)
                m[j] = len(P_2_new_num)
                l[i] = len(P_1_new_cat)
                l[j] = len(P_2_new_cat)

                Pn[i] = P_1_new_num
                Pn[j] = P_2_new_num
                Pc[i] = P_1_new_cat
                Pc[j] = P_2_new_cat
                V = V_new
        # Handle empty subspaces (no dimensionalities left) -> Should be removed
        subspaces, n_clusters, m, l, Pn, Pc, centers_num, centers_cat, labels, scatter_matrices = _remove_empty_subspace(subspaces,n_clusters,m, l, Pn, Pc, centers_num, centers_cat,labels,scatter_matrices)
    #print("[k-SubMix] Converged in iteration " + str(iteration + 1))
    # Return relevant values

    return labels, centers_num, centers_cat, V, m, l, Pn, Pc, n_clusters, scatter_matrices, scatter_cat


def _initialize_kSubMix_parameters(X, X_num, X_cat, n_clusters, V, m, l, Pn, Pc, centers_num, centers_cat, max_iter,random_state):
    """
    Initialize the input parameters form Nk-SubMix. This means that all input values which are None must be defined.
    Also all input parameters which are not None must be checked, if a correct execution is possible.
    :param X: input data
    :param X_num: numerical input data
    :param X_cat: categorical input data
    :param n_clusters: list containing number of clusters for each subspace
    :param V: orthogonal rotation matrix
    :param m: list containing number of numerical dimensionalities for each subspace
    :param l: list containing number of categorical dimensionalities for each subspace
    :param Pc: list containing categorical projections for each subspace
    :param Pn: list containing numerical projections for each subspace
    :param centers_num: list containing the numerical cluster centers for each subspace
    :param centers_num: list containing the categorical cluster centers for each subspace
    :param max_iter: maximum number of iterations for the algorithm
    :param random_state: use a fixed random state to get a repeatable solution
    :return: checked V, m, P, centers, random_state, number of subspaces, labels, scatter_matrices
    """
    data_dimensionality_num = X_num.shape[1]
    data_dimensionality_cat = X_cat.shape[1]
    random_state = check_random_state(random_state)
    # Check if n_clusters is a list
    if not type(n_clusters) is list:
        raise ValueError(
            "Number of clusters must be specified for each subspace and therefore be a list.\nYour input:\n" + str(
                n_clusters))
    # Check if n_clusters contains negative values
    if len([x for x in n_clusters if x < 1]) > 0:
        raise ValueError(
            "Number of clusters must not contain negative values or 0.\nYour input:\n" + str(
                n_clusters))
    # Check if n_clusters contains more than one noise space
    noise_spaces = len([x for x in n_clusters if x == 1])
    if noise_spaces > 1:
        raise ValueError(
            "Only one subspace can be the noise space (number of clusters = 1).\nYour input:\n" + str(n_clusters))
    # Check if noise space is not the last member in n_clusters
    if noise_spaces != 0 and n_clusters[-1] != 1:
        raise ValueError(
            "Noise space (number of clusters = 1) must be the last entry in n_clsuters.\nYour input:\n" + str(n_clusters))
    # Get number of subspaces
    subspaces = len(n_clusters)
    # Check if V is orthogonal
    if V is None:
        if data_dimensionality_num > 1:
            V = ortho_group.rvs(dim=data_dimensionality_num,random_state=random_state)
        else:
            V = np.ones((1, 1))
    if not _is_matrix_orthogonal(V):
        raise ValueError("Your input matrix V is not orthogonal.\nV:\n" + str(V))
    if V.shape[0] != data_dimensionality_num or V.shape[1] != data_dimensionality_num:
        raise ValueError(
            "The shape of the input matrix V must equal the data dimensionality.\nShape of V:\n" + str(V.shape))
    # Calculate dimensionalities m
    if m is None and Pn is None:
        m = [int(data_dimensionality_num / subspaces)] * subspaces
        if data_dimensionality_num % subspaces != 0:
            choices = random_state.choice(range(subspaces), data_dimensionality_num - np.sum(m))
            for choice in choices:
                m[choice] += 1
    # If m is None but Pn is defined use Pn's dimensionality
    elif m is None:
        m = [len(x) for x in Pn]
    if not type(m) is list or not len(m) is subspaces:
        raise ValueError("A dimensionality list m must be specified for each subspace.\nYour input:\n" + str(m))
    # Calculate dimensionalities l
    if l is None and Pc is None:
        l = [int(data_dimensionality_cat / subspaces)] * subspaces
        if data_dimensionality_cat % subspaces != 0:
            choices = random_state.choice(range(subspaces), data_dimensionality_cat - np.sum(l))
            for choice in choices:
                l[choice] += 1
    # If l is None but P is defined use P's dimensionality
    elif l is None:
        l = [len(x) for x in Pc]
    # Calculate projections Pn
    if Pn is None:
        possible_projections = list(range(data_dimensionality_num))
        Pn = []
        for dimensionality in m:
            choices = random_state.choice(possible_projections, dimensionality, replace=False)
            Pn.append(choices)
            possible_projections = list(set(possible_projections) - set(choices))
    if not type(Pn) is list or not len(Pn) is subspaces:
        raise ValueError("Projection lists must be specified for each subspace.\nYour input:\n" + str(Pn))
    else:
        # Check if the length of entries in Pn matches values of m
        used_dimensionalities = []
        for i, dimensionality in enumerate(m):
            used_dimensionalities.extend(Pn[i])
            if not len(Pn[i]) == dimensionality:
                raise ValueError(
                    "Values for dimensionality m and length of projection list P do not match.\nDimensionality m:\n" + str(
                        dimensionality) + "\nDimensionality P:\n" + str(Pn[i]))
        # Check if every dimension in considered in Pn
        if sorted(used_dimensionalities) != list(range(data_dimensionality_num)):
            raise ValueError("Projections P must include all dimensionalities.\nYour used dimensionalities:\n" + str(
                used_dimensionalities))
    # Calculate projections Pc
    if Pc is None:
        possible_projections = list(range(data_dimensionality_cat))
        Pc = []
        for dimensionality in l:
            choices = random_state.choice(possible_projections, dimensionality, replace=False)
            Pc.append(choices)
            possible_projections = list(set(possible_projections) - set(choices))
    if not type(Pc) is list or not len(Pc) is subspaces:
        raise ValueError("Projection lists must be specified for each subspace.\nYour input:\n" + str(Pc))
    else:
        # Check if the length of entries in Pc matches values of l
        used_dimensionalities = []
        for i, dimensionality in enumerate(l):
            used_dimensionalities.extend(Pc[i])
            if not len(Pc[i]) == dimensionality:
                raise ValueError(
                    "Values for dimensionality m and length of projection list P do not match.\nDimensionality l:\n" + str(
                        dimensionality) + "\nDimensionality P:\n" + str(Pc[i]))
        # Check if every dimension in considered in Pc
        if sorted(used_dimensionalities) != list(range(data_dimensionality_cat)):
            raise ValueError("Projections P must include all dimensionalities.\nYour used dimensionalities:\n" + str(
                used_dimensionalities))
    X = np.concatenate([X_num, X_cat], axis=1)
    cat_dims=list(range(X_num.shape[1],X.shape[1]))
    # Define initial cluster centers with k-Mode for each subspace (mixed-type cluster center initialization method)
    if centers_num is None or centers_cat is None:
        centers_num = []
        centers_cat = []
        for k in n_clusters:
            km_huang = KPrototypes(n_clusters=k, init="Huang", n_init=1)
            prediction = km_huang.fit_predict(X,categorical=cat_dims)
            center_huang = km_huang.cluster_centroids_
            subspace_center_num = []
            subspace_center_cat = []
            center_huang = center_huang.tolist()
            for i in center_huang:
                nums_huang = i[:data_dimensionality_num]
                cat_huang = i[-data_dimensionality_cat:]
                subspace_center_num.append(nums_huang)
                subspace_center_cat.append(cat_huang)
            centers_num.append(subspace_center_num)
            centers_cat.append(subspace_center_cat)
    if not type(centers_num) is list or not len(centers_num) is subspaces:
        raise ValueError("Cluster centers must be specified for each subspace.\nYour input:\n" + str(centers_num))
    else:
        # Check if number of centers for subspaces matches value in n_clusters
        for i, subspace_centers in enumerate(centers_num):
            if not n_clusters[i] == len(subspace_centers):
                raise ValueError(
                    "Values for number of clusters n_clusters and number of centers do not match.\nNumber of clusters:\n" + str(
                        n_clusters[i]) + "\nNumber of centers:\n" + str(len(subspace_centers)))
    # Check max iter
    if max_iter is None or type(max_iter) is not int or max_iter <= 0:
        raise ValueError(
            "Max_iter must be an integer larger than 0. Your Max_iter:\n" + str(max_iter))
    # Initial labels and scatter matrices
    labels = [None] * subspaces
    scatter_matrices = [None] * subspaces
    scatter_cat = [None] * subspaces
    return V, m, l, Pn, Pc, centers_num, centers_cat, random_state, subspaces, labels, scatter_matrices, scatter_cat


def remap_labels(pred_labels, true_labels):
    """Rename prediction labels (clustered output) to best match true labels.
    :param pred_labels: predicted labels
    :param true_labels: ground truth labels
    :return: remapped labels and ground truth labels"""
    #true_labels=[int(s)for s in true_labels]
    #true_labels = [item for sublist in true_labels for item in sublist]
    true_labels = true_labels.flatten()
    pred_labels, true_labels = np.array(pred_labels), np.array(true_labels)
    cluster_names = np.unique(pred_labels)
    accuracy = 0
    perms = np.array(list(permutations(np.unique(true_labels))))
    remapped_labels = true_labels
    for perm in perms:
        flipped_labels = np.zeros(len(true_labels))
        for label_index, label in enumerate(cluster_names):
            flipped_labels[pred_labels == label] = perm[label_index]
            testAcc = np.sum(flipped_labels == true_labels) / len(true_labels)
            if testAcc > accuracy:
                accuracy = testAcc
                remapped_labels = flipped_labels
    return remapped_labels, true_labels

def _assign_labels(X_num, X_cat, V, centers_num, centers_cat, Pn_subspace, Pc_subspace, n_clusters,old_labels):
    """
    Assign each point in each subspace to its nearest cluster center.
    :param X_num: numerical input data
    :param X_cat: categorical input data
    :param V: orthogonal rotation matrix
    :param centers_num: list containing the numerical cluster centers for each subspace
    :param centers_num: list containing the categorical cluster centers for each subspace
    :param Pn_subspace: numerical projections of the subspace
    :param Pc_subspace: categorical projections of the subspace
    :param n_clusters: list containing number of clusters for each subspace
    :param old_labels: labels of last cluster iteration to calculate categoricla costs
    :return: list with cluster assignments
    """
    cropped_X = np.matmul(X_num, V[:, Pn_subspace])
    cropped_centers_num = np.matmul(centers_num, V[:, Pn_subspace])
    cropped_X_cat = X_cat[:, Pc_subspace]
    np_centers = np.asarray(centers_cat)
    cropped_centers_cat = np_centers[:, Pc_subspace.tolist()]
    n_points = cropped_X.shape[0]
    labels = np.empty(n_points, dtype=np.uint16)
    total_num = 0
    total_cat = 0
    for point_i in range(n_points):
        num_costs = _num_dissim(cropped_centers_num, cropped_X[point_i])
        total_num = total_num + num_costs
        cat_costs = _cat_dissim(cropped_X_cat, cropped_X_cat[point_i], cropped_centers_cat,old_labels,n_clusters)
        total_cat = total_cat + cat_costs
        tot_costs = num_costs + GAMMA * cat_costs
        clust = np.argmin(tot_costs)
        labels[point_i] = clust
    labels = labels.astype(np.int32)
    return labels

def _cat_dissim(X_cat, point, center,old_labels,n_clusters):
    """Categorical dissimilarity measure based on how often a feature value appears within a certain cluster"""
    if np.isnan(center).any() or np.isnan(point).any():
        raise ValueError("Missing values detected in categorical columns.")
    X_cat_T = np.transpose(X_cat)
    cost_point_centers = []
    for center_id, _ in enumerate(center):
        points_in_cluster = np.where(old_labels == center_id)[0]
        x_cluster = np.array(X_cat)[points_in_cluster]
        cost_center = 0
        for col in range(len(X_cat_T)):
            #Labels are initally None but costs depend on labels -> Initial label generation for the first iteration (Hamming distance to initalize categorical costs)
            if old_labels is None:
                if center[center_id][col] == point[col]:
                    cost_col_point = 0
                else:
                    cost_col_point = 1
            else: #Actual categorical cost funciton to calculate distance from a point to a cluster
                center_column_X = x_cluster[:, col]
                point_column = point[col]
                count_point=np.count_nonzero(center_column_X == point_column)
                cost_col_point =((len(points_in_cluster)-count_point)/len(points_in_cluster))
            cost_center = cost_center + cost_col_point
        cost_point_centers.append(cost_center)
    return np.asarray(cost_point_centers)


def _num_dissim(center, point):
    """Euclidean distance dissimilarity function"""
    # if np.isnan(center).any():
    #   center=0
    # if np.isnan(point).any():
    #   point=0
    #   raise ValueError("Missing values detected in numerical columns.")
    return np.sum((center - point) ** 2, axis=1)


def _update_centers_and_scatter_matrices(X_num, X_cat, n_clusters_subspace, labels_subspace):
    """
    Update the cluster centers within this subspace depending on the labels of the data points. Also updates the the
    scatter matrix of each cluster by summing up the outer product of the distance between each point and center.
    :param X_num: numerical input data
    :param X_cat: categorical input data
    :param n_clusters_subspace: number of clusters of the subspace
    :param labels_subspace: cluster assignments of the subspace
    :return: centers_num, centers_cat, scatter_matrices, scatter_categorical - Updated cluster center and scatter matrices (one scatter matrix for each cluster)
    """
    # Create empty matrices
    centers_num = np.zeros((n_clusters_subspace, X_num.shape[1]))
    centers_cat = np.zeros((n_clusters_subspace, X_cat.shape[1]))
    scatter_matrices = np.zeros((n_clusters_subspace, X_num.shape[1], X_num.shape[1]))
    scatter_categorical = np.zeros(X_cat.shape[1])
    #Numerical
    for center_id, _ in enumerate(centers_num):
        # Get points in this cluster
        points_in_cluster = np.where(labels_subspace == center_id)[0]
        if len(points_in_cluster) == 0:
            centers_num[center_id] = np.nan
            continue
        # Update numeric center
        centers_num[center_id] = np.average(X_num[points_in_cluster], axis=0)
        centered_points = X_num[points_in_cluster] - centers_num[center_id]
        for entry in centered_points:
            rank1 = np.outer(entry, entry)
            scatter_matrices[center_id] += rank1
    #Categorical
    #n_points = X_cat.shape[0]
    X_cat_T = np.transpose(X_cat)
    for center_id, _ in enumerate(centers_cat):
        points_in_cluster = np.where(labels_subspace == center_id)[0]
        x_cluster = np.array(X_cat)[points_in_cluster]
        for point in x_cluster:
            cost_point_centers = []
            for col in range(len(X_cat_T)):
                center_column_X = x_cluster[:, col]
                point_column = point[col]
                count_point=np.count_nonzero(center_column_X == point_column)
                cost_col_point =((len(points_in_cluster)-count_point)/len(points_in_cluster))
                cost_point_centers.append(cost_col_point)
            scatter_categorical += cost_point_centers
    return centers_num, centers_cat, scatter_matrices, scatter_categorical

#TODO
def _remove_empty_cluster(n_clusters_subspace, num_centers_subspace, cat_centers_subspace, scatter_matrices_subspace,labels_subspace):
    """
    Check if after label assignemnt and center update a cluster got lost. Empty clusters will be
    removed for the following rotation und iterations. Therefore all necessary lists will be updated.
    :param n_clusters_subspace: number of clusters of the subspace
    :param num_centers_subspace: numerical cluster centers of the subspace
    :param cat_centers_subspace: cluster centers of the subspace
    :param scatter_matrices_subspace: scatter matrices of the subspace
    :param labels_subspace: cluster assignments of the subspace
    :return: n_clusters_subspace, centers_subspace, scatter_matrices_subspace, labels_subspace (updated)
    """
    # Check if any cluster is lost
    if np.any(np.isnan(num_centers_subspace)):
        # Get ids of lost clusters
        empty_clusters = np.where(np.any(np.isnan(num_centers_subspace), axis=1))[0]
        print(
            "[k-SubMix] ATTENTION: Clusters were lost! Number of lost clusters: " + str(
                len(empty_clusters)) + " out of " + str(
                len(num_centers_subspace)))
        # Update necessary lists
        n_clusters_subspace -= len(empty_clusters)
        for cluster_id in reversed(empty_clusters):
            num_centers_subspace = np.delete(num_centers_subspace, cluster_id, axis=0)
            num_centers_subspace = np.delete(num_centers_subspace, cluster_id, axis=0)
            scatter_matrices_subspace = np.delete(scatter_matrices_subspace, cluster_id, axis=0)
            labels_subspace[labels_subspace > cluster_id] -= 1
    return n_clusters_subspace, num_centers_subspace, cat_centers_subspace, scatter_matrices_subspace, labels_subspace


def _update_rotation(X_num, X_cat, V, first_index, second_index, n_clusters, labels, Pn, Pc, scatter_num, scatter_cat,
                     eigenvalues_num, eigenvalues_cat):
    """
    Update the rotation of the subspaces. Updates V and m and P for the input subspaces.
    :param X_num: numerical input data
    :param X_cat: categorical input data
    :param V: orthogonal rotation matrix4
    :param first_index: index of the first subspace
    :param second_index: index of the second subspace (can be noise space)
    :param n_clusters: list containing number of clusters for each subspace
    :param labels: list containing cluster assignments for each subspace
    :param Pn: list containing numerical projections for each subspace
    :param Pc:list containing categorical projections for each subspace
    :param scatter_num: list containing numerical scatter matrices for each subspace
    :param scatter_cat: list containing categorical scatter matrices for each subspace
    :return: P_1_new, P_2_new, V_new - new P for the first subspace, new P for the second subspace and new V
    """

    # print(f"-----\nscatter_cat: {scatter_cat}\n-----")

    # Check if second subspace is the noise space
    is_noise_space = (n_clusters[second_index] == 1)
    # Get combined projections and combined_cropped_V
    P_1_num = Pn[first_index]
    P_1_cat = Pc[first_index]
    P_2_num = Pn[second_index]
    P_2_cat = Pc[second_index]
    P_combined_num = np.append(P_1_num, P_2_num)
    P_combined_cat = np.append(P_1_cat, P_2_cat)
    # Check if both Pn's are empty
    # TODOo for non redundant clustering
    if len(P_combined_num) == 0:
        return P_1_num, P_2_num, V
    cropped_V_combined = V[:, P_combined_num]

    # Prepare numerical input for eigenvalue decomposition.
    sum_scatter_matrices_1_num = np.sum(scatter_num[first_index], 0)
    sum_scatter_matrices_2_num = np.sum(scatter_num[second_index], 0)
    diff_scatter_matrices_num = sum_scatter_matrices_1_num - sum_scatter_matrices_2_num
    projected_diff_scatter_matrices_num = np.matmul(
        np.matmul(cropped_V_combined.transpose(), diff_scatter_matrices_num),
        cropped_V_combined)
    if not _is_matrix_symmetric(projected_diff_scatter_matrices_num):
        raise Exception(
            "Input for eigenvalue decomposition is not symmetric.\nInput:\n" + str(projected_diff_scatter_matrices_num))
    # Get eigenvalues and eigenvectors (already sorted by eigh)
    e, V_C = np.linalg.eigh(projected_diff_scatter_matrices_num)

    # Use transitions and eigenvectors to build V full
    V_F = _create_full_rotation_matrix(X_num.shape[1], P_combined_num, V_C)
    # Calculate new V
    V_new = np.matmul(V, V_F)
    if not _is_matrix_orthogonal(V_new):
        raise Exception("New V is not othogonal.\nNew V:\n" + str(V_new))
    # Use number of negative eigenvalues to get new projections
    n_negative_e = len(e[e < 0])
    if is_noise_space:
        # n_negative_e = len(e[e < NOISE_SPACE_THRESHOLD])
        n_negative_e = len(e[e <= 0])
    P_1_new_num, P_2_new_num = _update_projections(P_combined_num, n_negative_e)

    #Categorical Feature selection
    negative_costs = np.where(scatter_cat[first_index]+n_clusters[0] < scatter_cat[second_index])
    positive_costs = np.where(scatter_cat[first_index]+n_clusters[0] >= scatter_cat[second_index])
    #Update categorical dimensions for clustered subspace
    P_1_new_cat, P_2_new_cat = _update_categorical_projections(P_combined_cat, negative_costs,positive_costs)
    # Return new dimensionalities, projections and V
    return P_1_new_num, P_2_new_num, P_1_new_cat, P_2_new_cat, V_new, eigenvalues_num, eigenvalues_cat


def _get_cost_function_result(cropped_V, scatter_matrices_subspace):
    """
    Calculate the result of the k-SubMix loss function for a certain subspace.
    Depends on the rotation and the scattermatrices. Calculates:
    P^T*V^T*S*V*P
    :param cropped_V: cropped orthogonal rotation matrix
    :param scatter_matrices_subspace: scatter matrices of the subspace
    :return: result of the k-SubMix cost function
    """
    scatter_matrix = np.sum(scatter_matrices_subspace, 0)
    return np.trace(np.matmul(np.matmul(cropped_V.transpose(), scatter_matrix),
                              cropped_V))


def _create_full_rotation_matrix(dimensionality, P_combined, V_C):
    """
    Create full rotation matrix out of the found eigenvectors. Set diagonal to 1 and overwrite columns and rows with
    indices in P_combined (consider the order) with the values from V_C. All other values should be 0.
    :param dimensionality: dimensionality of the full rotation matrix
    :param P_combined: combined projections of the subspaces
    :param V_C: the calculated eigenvectors
    :return: the new full rotation matrix
    """
    V_F = np.identity(dimensionality)
    V_F[np.ix_(P_combined, P_combined)] = V_C
    return V_F
def _update_categorical_projections(P_combined, negative_costs,positive_costs):
    """
    Create the categorical new projections for the subspaces. First subspace gets all features with
      negative costs for clustered subspace. Second subspace gets all other projections in reversed order.
        :param P_combined: combined projections of the subspaces
        :param negative_costs: indices of features with smaller costs in clustered subspace
        :param positive_costs: indices of features with bigger costs in clustered subspace
        :return: P_1_new, P_2_new - categorical projections for the subspaces
        """
    P_1_new = P_combined[negative_costs]
    P_2_new = P_combined[positive_costs]
    return P_1_new, P_2_new


def _update_projections(P_combined, number_negative_e):
    """
    Create the new numerical projections for the subspaces. First subspace gets all as many projections as there are negative
    eigenvalues. Second subspace gets all other projections in reversed order.
    :param P_combined: combined projections of the subspaces
    :param number_negative_e: number of negative eigenvalues
    :return: P_1_new, P_2_new - projections for the subspaces
    """
    P_1_new = np.array([P_combined[x] for x in range(number_negative_e)], dtype=int)
    P_2_new = np.array([P_combined[x] for x in reversed(range(number_negative_e, len(P_combined)))], dtype=int)
    return P_1_new, P_2_new


def _remove_empty_subspace(subspaces, n_clusters, m, l, Pn, Pc, centers_num, centers_cat, labels, scatter_matrices):
    """
    Check if after rotation and rearranging the dimensionalities a empty subspaces occurs. Empty subspaces will be
    removed for the next iteration. Therefore all necessary lists will be updated.
    :param subspaces: number of subspaces
    :param n_clusters:
    :param m: list containing number of nunmerical dimensionalities for each subspace
    :param m: list containing number of categorical dimensionalities for each subspace
    :param Pn: list containing numerical projections for each subspace
    :param Pc: list containing categorical projections for each subspace
    :param centers_num: list containing the numerical cluster centers for each subspace
    :param centers_cat: list containing the categorical cluster centers for each subspace
    :param labels: list containing cluster assignments for each subspace
    :param scatter_matrices: list containing scatter matrices for each subspace
    :return: subspaces, n_clusters, m,l, Pn,Pc, centers_num, centers_cat, labels, scatter_matrices
    """
    for j in range(len(m)):
        if m[j] == 0 and l[j] == 0:
            np_m = np.array(m)
            empty_spaces = np.where(np_m == 0)[0]
            print(
                "[k-SubMix] ATTENTION: Subspaces were lost! Number of lost subspaces: " + str(
                    len(empty_spaces)) + " out of " + str(
                    len(m)))
            subspaces -= len(empty_spaces)
            n_clusters = [x for i, x in enumerate(n_clusters) if i not in empty_spaces]
            m = [x for i, x in enumerate(m) if i not in empty_spaces]
            l = [x for i, x in enumerate(l) if i not in empty_spaces]
            Pn = [x for i, x in enumerate(Pn) if i not in empty_spaces]
            Pc = [x for i, x in enumerate(Pc) if i not in empty_spaces]
            centers_num = [x for i, x in enumerate(centers_num) if i not in empty_spaces]
            centers_cat = [x for i, x in enumerate(centers_cat) if i not in empty_spaces]
            labels = [x for i, x in enumerate(labels) if i not in empty_spaces]
            scatter_matrices = [x for i, x in enumerate(scatter_matrices) if i not in empty_spaces]
    return subspaces, n_clusters, m, l, Pn, Pc, centers_num, centers_cat, labels, scatter_matrices


def _is_matrix_orthogonal(matrix):
    """
    Check whether a matrix is orthogonal by comparing the multiplication of the matrix and its transpose and
    the identity matrix.
    :param matrix: input matrix
    :return: True if matrix is orthogonal
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False
    matrix_product = np.matmul(matrix, matrix.transpose())
    return np.allclose(matrix_product, np.identity(matrix.shape[0]), atol=ACCEPTED_NUMERICAL_ERROR)


def _is_matrix_symmetric(matrix):
    """
    Check whether a matrix is symmetric by comparing the matrix with its transpose.
    :param matrix: input matrix
    :return: True if matrix is symmetric
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False
    return np.allclose(matrix, matrix.T, atol=ACCEPTED_NUMERICAL_ERROR)


def _are_labels_equal(labels_new, labels_old):
    """
    Check if the old labels and new labels are equal. Therefore check the nmi for each subspace. If all are 1, labels
    have not changed.
    :param labels_new: new labels list
    :param labels_old: old labels list
    :return: True if labels for all subspaces are the same
    """
    if labels_new is None or labels_old is None:
        return False
    return all([nmi(labels_new[i], labels_old[i]) >= EARLY_STOPPING_NMI for i in range(len(labels_new))])


def _f1_score(prediction, ground_truth, check_all_subspaces=False):
    ground_truth = ground_truth.T
    n_elements = len(prediction[0]) if check_all_subspaces else len(prediction)
    n_tp = 0
    n_fp = 0
    n_fn = 0
    # n_tn = 0
    for i in range(n_elements - 1):
        for j in range(i + 1, n_elements):
            if not check_all_subspaces:
                if prediction[i] == prediction[j]:
                    if ground_truth[i] == ground_truth[j]:
                        n_tp += 1
                    else:
                        n_fp += 1
                else:
                    if ground_truth[i] == ground_truth[j]:
                        n_fn += 1
            else:
                if _f1_same_cluster(prediction, i, j):
                    if _f1_same_cluster(ground_truth, i, j):
                        n_tp += 1
                    else:
                        n_fp += 1
                else:
                    if _f1_same_cluster(ground_truth, i, j):
                        n_fn += 1
    precision = 0 if (n_tp + n_fp) == 0 else n_tp / (n_tp + n_fp)
    recall = 0 if (n_tp + n_fn) == 0 else n_tp / (n_tp + n_fn)
    return 0 if (precision == 0 and recall == 0) else 2 * precision * recall / (precision + recall)


def _f1_same_cluster(labels, i, j):
    for s in range(len(labels)):
        if labels[s][i] == labels[s][j]:
            return True
    return False


"""
==================== k-SubMixObject ====================
"""


class kSubMix():
    def __init__(self, n_clusters, V=None, m=None, l=None, Pn=None, Pc=None, centers_num=None, centers_cat=None,
                 max_iter=300,
                 random_state=None, gamma = None, earlyStoppingNMI = 1):
        """
        Create new k-SubMix instance. Gives the opportunity to use the fit() method to cluster a dataset.
        :param n_clusters: list containing number of clusters for each subspace
        :param V: orthogonal rotation matrix (optional)
        :param m: list containing number of numerical  dimensionalities for each subspace (optional)
        :param l: list containing number of categorical dimensionalities for each subspace (optional)
        :param Pn: list containing numerical projections for each subspace (optional)
        :param Pc: list containing categorical projections for each subspace (optional)
        :param centers_num: list containing the numerical cluster centers for each subspace (optional)
        :param centers_cat: list containing the categorical cluster centers for each subspace (optional)
        :param max_iter: maximum number of iterations for the k-SubMix algorithm (default: 300)
        :param random_state: use a fixed random state to get a repeatable solution (optional)
        """

        global GAMMA, EARLY_STOPPING_NMI
        GAMMA = gamma
        EARLY_STOPPING_NMI = earlyStoppingNMI

        # Fixed attributes
        self.input_n_clusters = n_clusters.copy()
        self.max_iter = max_iter
        self.random_state = random_state
        # Variables
        self.n_clusters = n_clusters
        self.labels = None
        # self.centers = centers
        self.centers_num = centers_num
        self.centers_cat = centers_cat
        self.V = V
        self.m = m
        self.l = l
        self.Pn = Pn
        self.Pc = Pc
        self.scatter_matrices = None


    def fit(self, X, X_num, X_cat, ground_truth):
        """
        Cluster the input dataset with the k-SubMix algorithm. Saves the labels, centers, V, m,l, Pn,Pc and scatter matrices
        in the kSubMix object.
        :param X: input data
        :param X_num: numercial input data
        :param X_cat: categorical input data
        :param ground_truth: Ground truth labels
        :return: the k-SubMix object
        """
        labels, centers_num, centers_cat, V, m, l, Pn, Pc, n_clusters, scatter_matrices, scatter_cat = subMix(X,
                                                                                                                X_num,
                                                                                                                X_cat,
                                                                                                                ground_truth,
                                                                                                                self.n_clusters,
                                                                                                                self.V,
                                                                                                                self.m,
                                                                                                                self.l,
                                                                                                                self.Pn,
                                                                                                                self.Pc,
                                                                                                                self.centers_num,
                                                                                                                self.centers_cat,
                                                                                                                self.max_iter,
                                                                                                                self.random_state)
        # Update class variables
        self.labels = labels
        self.centers_num = centers_num
        self.centers_cat = centers_cat
        self.V = V
        self.m = m
        self.l = l
        self.Pn = Pn
        self.Pc = Pc
        self.n_clusters = n_clusters
        self.scatter_matrices = scatter_matrices
        self.scatter_cat = scatter_cat
        return self

    def transform_full_space(self, X):
        """
        Transfrom the input dataset with the orthogonal rotation matrix V from the k-SubMix object.
        :param X: input data
        :return: the rotated dataset
        """
        return np.matmul(X, self.V)

    def transform_clustered_space(self, X, subspace_index):
        """
        Transform the input dataset with the orthogonal rotation matrix V projected onto a special subspace.
        :param X: input data
        :param subspace_index: index of the subspace
        :return: the rotated dataset
        """
        cluster_space_V = self.V[:, self.P[subspace_index]]
        return np.matmul(X, cluster_space_V)

    def have_subspaces_been_lost(self):
        """
        Check whether subspaces have been lost during k-SubMix execution.
        :return: True if at least one subspace has been lost
        """
        return len(self.n_clusters) != len(self.input_n_clusters)

    def have_clusters_been_lost(self):
        """
        Check whether clusteres within a subspace have been lost during k-SubMix execution.
        Will also return true if subspaces have been lost (check have_subspaces_been_lost())
        :return: True if at least one cluster has been lost
        """
        return not np.array_equal(self.input_n_clusters, self.n_clusters)

    def get_cluster_count_of_changed_subspaces(self):
        """
        Get the Number of clusters of the changed subspaces. If no subspace/cluster is lost, empty list will be
        returned.
        :return: list with the changed cluster count
        """
        changed_subspace = self.input_n_clusters.copy()
        for x in self.n_clusters:
            if x in changed_subspace:
                changed_subspace.remove(x)
        return changed_subspace

    def plot_subspace(self, X, subspace_index, labels=None, title=None):
        """
        Plot the specified subspace as scatter matrix plot.
        :param X: input data
        :param subspace_index: index of the subspace
        :param labels: the labels to use for the plot (default: labels found by k-SubMix)
        :return: a scatter matrix plot of the input data
        """
        if self.labels is None:
            raise Exception("The k-SubMix algorithm has not run yet. Use the fit() function first.")
        if labels is None:
            labels = self.labels[subspace_index]
        if X.shape[0] != len(labels):
            raise Exception("Number of data objects must match the number of labels.")
        if self.m[subspace_index] > 10:
            print("[k-SubMix] Info: Subspace has a dimensionality > 10 and will not be plotted.")
            return
        sns.set(style="ticks")
        plot_data = pd.DataFrame(self.transform_clustered_space(X, subspace_index))
        columns = plot_data.columns.values
        plot_data["labels"] = labels
        # If group contains only one element KDE Plot will not work -> Use histrogram instead
        diag = "auto"
        if 1 in np.unique(labels, return_counts=True)[1]:
            diag = "hist"
        g = sns.pairplot(plot_data, hue="labels", vars=columns, diag_kind=diag)
        # Add title
        my_title = "Subspace {0}".format(subspace_index) if title is None else "{0} / Subspace {1}".format(title,
                                                                                                           subspace_index)
        g.fig.suptitle(my_title)
        plt.show()

    def compare_ground_truth(self, ground_truth, scoring_function="nmi"):
        """
        Compare ground truth of a dataset with the results from k-SubMix. Checks for each subspace the chosen scoring
        function and calculates the average over all subspaces. A perfect result will always be 1, the worst 0.
        :param ground_truth: dataset with the true labels for each subspace
        :param scoring_function: the scoring function can be "nmi", "ami", "rand", "f1" or "pc-f1" (default: "nmi")
        :return: scring result. Between 0 <= score <= 1
        """
        labels = self.labels.copy()
        # Check number of points
        if len(labels[0]) != ground_truth.shape[0]:
            raise Exception(
                "Number of objects of the dataset and ground truth are not equal.\nNumber of dataset objects: " + str(
                    len(labels[0])) + "\nNumber of ground truth objects: " + str(ground_truth.shape[0]))
        if scoring_function not in ["nmi", "ami", "rand", "f1", "pc-f1", "acc"]:
            raise Exception("Your input scoring function is not supported.")
        # Don't check noise spaces
        for i in range(len(self.n_clusters) - 1, -1, -1):
            if self.n_clusters[i] == 1:
                del labels[i]
        for c in range(ground_truth.shape[1] - 1, -1, -1):
            unique_labels = np.unique(ground_truth[:, c])
            len_unique_labels = len(unique_labels) if np.all(unique_labels >= 0) else len(unique_labels) - 1
            if len_unique_labels == 1:
                ground_truth = np.delete(ground_truth, c, axis=1)
        if len(labels) == 0 or ground_truth.shape[1] == 0:
            return 0
        # Calculate scores
        if scoring_function == "pc-f1":
            return _f1_score(labels, ground_truth, True)
        confusion_matrix = np.zeros((len(labels), ground_truth.shape[1]))
        for i in range(len(labels)):
            for j in range(ground_truth.shape[1]):
                #labels[i], ground_truth[:, j]=remap_labels(labels[i],(ground_truth[:, j]))
                if scoring_function == "nmi":
                    confusion_matrix[i, j] = nmi(labels[i], ground_truth[:, j].flatten().astype(int))
                elif scoring_function == "ami":
                    confusion_matrix[i, j] = ami(labels[i], ground_truth[:, j].flatten().astype(int))
                elif scoring_function == "rand":
                    confusion_matrix[i, j] = rand(labels[i], ground_truth[:, j].flatten().astype(int))
                elif scoring_function == "acc":
                    confusion_matrix[i, j] = accuracy_score(labels[i], ground_truth[:, j].flatten().astype(int), normalize=True)
                elif scoring_function == "f1":
                    confusion_matrix[i, j] = _f1_score(labels[i], ground_truth[:, j].flatten().astype(int), False)
        # Save best possible score
        best_score = 0
        max_sub = max(len(labels), ground_truth.shape[1])
        min_sub = min(len(labels), ground_truth.shape[1])
        for permut in itertools.permutations(range(max_sub)):
            score_sum = 0
            for m in range(min_sub):
                if len(labels) >= ground_truth.shape[1]:
                    i = permut[m]
                    j = m
                else:
                    i = m
                    j = permut[m]
                score_sum += confusion_matrix[i, j]
            if score_sum > best_score:
                best_score = score_sum
        # Return best found score. Get average score over all subspaces
        return best_score / ground_truth.shape[1]

    def calculate_cost_function(self):
        """
        Calculate the result of the k-SubMix numerical loss function. Depends on the rotation and the scattermatrices.
        Calculates for each subspace:
        P^T*V^T*S*V*P
        :return: result of the numerical k-SubMix cost function
        """
        if self.labels is None:
            raise Exception("The k-SubMix algorithm has not run yet. Use the fit() function first.")
        costs = np.sum(
            [_get_cost_function_result(self.V[:, self.P[i]], s) for i, s in enumerate(self.scatter_matrices)])
        return costs

def plotEigenvalues(eigenvalues_num, eigenvalues_cat):
    x = list(range(0, len(eigenvalues_num)))
    cat = sum(sum(c) for c in eigenvalues_cat)
    num = sum(sum(n) for n in eigenvalues_num)
    labels_cat = ["Cat1", "Cat2", "Cat3", "Cat4", "Cat5", "Cat6", "Cat7", "Cat8", ]
    labels_num = ["Num1", "Num2", "Num3", "Num4", "Num5", "Num6", "Num7", "Num8", ]
    for i in range(len(eigenvalues_num)):
        x = list(range(0, len(eigenvalues_num[i])))
        plt.plot(x, eigenvalues_num[i], marker='o', label=labels_cat[i])
        plt.plot(x, eigenvalues_cat[i], marker='x', label=labels_num[i])
    plt.show()


if __name__ == "__main__":
    user_args = sys.argv[1:]
    k_list = [int(s) for s in user_args[0].split(',')]
    numerical_dims_cols = [int(s) for s in user_args[1].split(',')]
    categorical_dims_cols=[int(s) for s in user_args[2].split(',')]
    ground_truth_cols=[int(s) for s in user_args[3].split(',')]
    gamma = float(str((user_args[4])))
    file = str(user_args[5])
    evaluation_file = str(user_args[6])

    dataset = np.genfromtxt(file, delimiter=",")
    cur_path = os.path.dirname(__file__)
    data_path = os.path.join(cur_path, '..', '..', 'Datasets')
    gt_data= np.genfromtxt(os.path.join(data_path,file), delimiter=",")
    data = np.delete(dataset, (0), axis=0)
    dataset = dataset[~np.isnan(dataset).any(axis=1)]


    numerical_data = dataset[:, numerical_dims_cols]
    categorical_data = dataset[:, categorical_dims_cols]
    ground_truth = gt_data[:, [ground_truth_cols]]
    start = timeit.default_timer()
    result = kSubMix(k_list)
    print(f'dataset: {dataset}\nnumerical_data: {numerical_data}\ncategorical_data: {categorical_data}\nground_truth: {ground_truth}')
    exit(0)
    result.fit(dataset, numerical_data, categorical_data, ground_truth)
    stop = timeit.default_timer()
    #print("RUNTIME")
    #print(stop - start)
    print("Results")
    print("Pn")
    print(result.Pn)
    print("Pc")
    print(result.Pc)
    ami = result.compare_ground_truth(ground_truth, "ami")
    #print("==> AMI: " + str(ami))
    nmi = result.compare_ground_truth(ground_truth)
    print("==> NMI: " + str(nmi))
    #acc = result.compare_ground_truth(ground_truth, "acc") #Check remap labels function
    #print("==> Accuracy: " + str(acc))
    acc=0
    rand = result.compare_ground_truth(ground_truth, "rand")
    #print("==> RAND: " + str(rand))
    # score = result.compare_ground_truth(ground_truth, "f1")
    # print("==> f1: " + str(score))
    print()
    evaluation_df = pd.DataFrame.from_records({
                                          'Dataset': os.path.basename(file[:len(file) - 4]),
                                          'Algo name': "k-SubMix",
                                          'NMI': nmi,
                                          'RAND': rand,
                                          'AMI': ami,
                                          'Accuracy': acc,
                                          "Numerical Features": result.m[0],
                                          "Categorical Features": result.l[0],
                                          "Total Features": result.m[0] + result.l[0],
                                          "Runtime": stop - start,
                                          "Gamma": gamma}
        ,columns=['Dataset', 'Algo name','NMI', 'RAND', 'AMI', 'Accuracy', "Numerical Features",
                     "Categorical Features", "Total Features", "Runtime","Gamma"],index=[0])
    if os.path.isfile(evaluation_file):
        evaluation_df.to_csv(evaluation_file, header=False, mode='a', index=False)
    else:
        evaluation_df = pd.DataFrame(columns=['Dataset','Algo name', 'NMI', 'RAND', 'AMI', 'Accuracy', "Numerical Features",
                     "Categorical Features", "Total Features", "Runtime","Gamma"])
        evaluation_df.to_csv(evaluation_file, header=True, mode='a', index=False)
