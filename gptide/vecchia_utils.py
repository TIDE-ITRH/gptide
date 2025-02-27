#### Utils for GP Tide vecchia implementation
#
# Mainly nearest neighbour lookup code
# and forward and backward dense and sparse solvers

from numba import njit, config, set_num_threads
import numpy as np
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    from sklearn.neighbors import NearestNeighbors
    FAISS_AVAILABLE = False
from psutil import cpu_count
from sklearn.neighbors import NearestNeighbors

# Set the number of threads for Numba
core_num = cpu_count(logical=False)  # Get the number of physical CPU cores
max_threads = config.NUMBA_NUM_THREADS  # Get the maximum number of threads Numba can use
core_num = min(core_num, max_threads)  # Use the minimum of the two values
config.THREADING_LAYER = 'workqueue'  # Set the threading layer for Numba
set_num_threads(core_num)  # Set the number of threads for Numba



#### Neighbours #############################################################


def get_pred_nn(query, x, m=50, method='exact', size=40, efSearch=100, n_jobs=-1):
    """
    Find the nearest neighbors for the query points in the dataset x.
    
    query: Array of query points.
    x: Array of dataset points.
    m: Number of nearest neighbors to find.
    method: Method for nearest neighbor search ('exact' or 'approx').
    size: Size parameter for approximate search.
    efSearch: efSearch parameter for approximate search.
    n_jobs: Number of parallel jobs for nearest neighbor search.
    
    Returns:
    Array of nearest neighbor indices for each query point.
    """
    n, d = x.shape  # Get the number of points and dimensions
    m = min(m, n)  # Ensure m is not greater than the number of points
    if m == n:
        k = query.shape[0]
        NN = np.arange(m) + np.arange(k)[:, np.newaxis]
        NN %= m
    else:
        if (method != 'sklearn') & FAISS_AVAILABLE:
            if method == 'exact':
                neigh = faiss.IndexFlatL2(d)  # Exact nearest neighbor search
            elif method == 'approx':
                neigh = faiss.IndexHNSWFlat(d, size)  # Approximate nearest neighbor search
                neigh.hnsw.efSearch = efSearch
            neigh.add(x)
            _, NN = neigh.search(query, k=int(m))
        else:
            neigh = NearestNeighbors(algorithm='kd_tree', n_jobs=n_jobs)  # Initialize the nearest neighbors search
            neigh.fit(x)  # Fit the model to the data
            NN = neigh.kneighbors(query, n_neighbors=m, return_distance=False)  # Find the nearest neighbors
    return NN

@njit(cache=True)
def nn_brute(x, m):
    """
    Brute-force nearest neighbor search.
    
    x: Array of dataset points.
    m: Number of nearest neighbors to find.
    
    Returns:
    Array of nearest neighbor indices for each point.
    """
    n = x.shape[0]  # Get the number of points
    m = min(m, n-1)  # Ensure m is not greater than the number of points minus one
    NNarray = np.full((n, m+1), -1)  # Initialize the nearest neighbor array with -1
    for i in range(n):
        dist = np.sum((x[:(i+1),:] - x[i,:])**2, axis=1)  # Compute the squared distances
        order = np.argsort(dist)  # Get the indices that would sort the distances
        NNarray[i,:min(m+1, i+1)] = order[:min(m+1, i+1)]  # Store the nearest neighbors
    return NNarray

@njit(cache=True)
def extract_NN_m(NN_mask, less_than_k_mask, m):
    """
    Extract the nearest neighbors from the mask.
    
    NN_mask: Mask of nearest neighbors.
    less_than_k_mask: Mask of valid neighbors.
    m: Number of nearest neighbors to extract.
    
    Returns:
    Array of nearest neighbor indices.
    """
    n = NN_mask.shape[0]  # Get the number of points
    NN_m = np.empty((n, m))  # Initialize the nearest neighbor array
    for i in range(n):
        NN_m[i,] = NN_mask[i][less_than_k_mask[i]][:m]  # Extract the nearest neighbors
    return NN_m

def find_nn(x, m, method='sklearn', size=40, efSearch=100, n_jobs=-1):
    """
    Find the nearest neighbors for the dataset points.
    
    x: Array of dataset points.
    m: Number of nearest neighbors to find.
    method: Method for nearest neighbor search ('sklearn' or FAISS methods 'exact' or 'approx').
    size: Size parameter for FAISS approximate search.
    efSearch: efSearch parameter for FAISS approximate search.
    n_jobs: Number of parallel jobs for nearest neighbor search.
    
    Returns:
    Array of nearest neighbor indices for each point.
    """
    n, d = x.shape  # Get the number of points and dimensions
    m, mult = min(m, n-1), 2  # Ensure m is not greater than the number of points minus one

    NNarray = np.full((n, m + 1), -1, dtype=int)  # Initialize the nearest neighbor array with -1

    maxval = min(mult * m + 1, n)  # Determine the maximum value for the nearest neighbor search
    NNarray[:maxval] = nn_brute(x[:maxval], m)  # Perform brute-force nearest neighbor search for the initial points

    query_inds, msearch = np.arange(maxval, n), m  # Initialize the query indices and search parameter

    if (method != 'sklearn') & FAISS_AVAILABLE:
        while len(query_inds) > 0:
            max_query_inds = np.max(query_inds) + 1  # Get the maximum query index
            msearch = min(max_query_inds, 2*msearch)  # Update the search parameter
            data_inds = np.arange(min(max_query_inds, n))  # Get the data indices
            if method == 'exact':
                neigh = faiss.IndexFlatL2(d)  # Exact nearest neighbor search
            elif method == 'approx':
                neigh = faiss.IndexHNSWFlat(d, size)  # Approximate nearest neighbor search
                neigh.hnsw.efSearch = efSearch
            neigh.add(x[data_inds,:])  # Add the data points to the search index
            _, NN = neigh.search(x[query_inds,:], k=int(msearch))  # Search for the nearest neighbors
            less_than_k = NN <= query_inds[:, np.newaxis]  # Check if the neighbors are within the query indices
            less_than_k_valid = NN >= 0  # Check if the neighbors are valid
            less_than_k = np.logical_and(less_than_k, less_than_k_valid)  # Combine the masks
            sum_less_than_k = np.sum(less_than_k, axis=1)  # Sum the valid neighbors
            ind_less_than_k = sum_less_than_k >= m+1  # Check if the number of valid neighbors is greater than or equal to m+1
            NN_mask, less_than_k_mask, query_inds_mask = NN[ind_less_than_k,:], less_than_k[ind_less_than_k,:], query_inds[ind_less_than_k]  # Get the masks for the valid neighbors
            NN_m = extract_NN_m(NN_mask, less_than_k_mask, m+1)  # Extract the nearest neighbors
            NNarray[query_inds_mask,:] = NN_m  # Store the nearest neighbors
            query_inds = query_inds[~ind_less_than_k]  # Update the query indices
    else:
        neigh = NearestNeighbors(algorithm='kd_tree', n_jobs=n_jobs)  # Initialize the nearest neighbors search
        while len(query_inds) > 0:
            max_query_inds = np.max(query_inds) + 1  # Get the maximum query index
            msearch = min(max_query_inds, 2*msearch)  # Update the search parameter
            data_inds = np.arange(min(max_query_inds, n))  # Get the data indices
            neigh.fit(x[data_inds,:])  # Fit the model to the data
            NN = neigh.kneighbors(x[query_inds,:], n_neighbors=msearch, return_distance=False)  # Find the nearest neighbors
            less_than_k = NN <= query_inds[:, np.newaxis]  # Check if the neighbors are within the query indices
            sum_less_than_k = np.sum(less_than_k, axis=1)  # Sum the valid neighbors
            ind_less_than_k = sum_less_than_k >= m+1  # Check if the number of valid neighbors is greater than or equal to m+1
            NN_mask, less_than_k_mask, query_inds_mask = NN[ind_less_than_k,:], less_than_k[ind_less_than_k,:], query_inds[ind_less_than_k]  # Get the masks for the valid neighbors
            NN_m = extract_NN_m(NN_mask, less_than_k_mask, m+1)  # Extract the nearest neighbors
            NNarray[query_inds_mask,:] = NN_m  # Store the nearest neighbors
            query_inds = query_inds[~ind_less_than_k]  # Update the query indices
    NNarray = np.fliplr(np.sort(NNarray))  # Sort the nearest neighbors in descending order
    return NNarray




### Solvers #############################################################


@njit(cache=True)
def backward_solve(U, b):
    """Dense backward solve for a lower triangular matrix U and a vector b."""
    n = U.shape[0]
    x = np.zeros((n,1))
    for i in range(n-1, -1, -1):
        sumj = 0.0
        for j in range(i+1, n):
            sumj += U[i, j] * x[j,0]
        x[i] = (b[i] - sumj) / U[i, i]
    return x


@njit(cache=True)
def forward_solve(L, b):
    n = L.shape[0]
    x = np.zeros((n,1))
    for i in range(n):
        sumj = 0.0
        for j in range(i):
            sumj += L[i, j] * x[j,0]
        x[i] = (b[i] - sumj) / L[i, i]
    return x


@njit(cache=True)
def forward_solve_sp(L, NNarray, b):
    """
    Forward solve for a lower triangular matrix L and a vector b.
    """
    n, m = L.shape
    x = np.zeros(n)
    for i in range(n):
        sumj = 0.0
        for j in range(1, min(i+1, m)):
            sumj += L[i, j] * x[NNarray[i, j]]
        x[i] = (b[i] - sumj) / L[i,0]
    return x


def build_mahalanobis(diag_elements, offdiag_element=None):
    """
    Build a Mahalanobis matrix with specified diagonal and optional off-diagonal elements.
    
    diag_elements: List or array of diagonal elements.
    offdiag_element: Optional off-diagonal element (default is None, meaning no off-diagonal elements).
    
    Returns:
    Mahalanobis matrix of shape (dim, dim).
    """
    dim = len(diag_elements)
    # Create a diagonal matrix with the specified diagonal elements
    A = np.diag(diag_elements)
    # If offdiag_element is specified, set the off-diagonal elements
    if offdiag_element is not None:
        A += offdiag_element * (np.ones((dim, dim)) - np.eye(dim))
    return A