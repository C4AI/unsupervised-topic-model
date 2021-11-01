# based on: https://scikit-learn.org/stable/auto_examples/bicluster/plot_spectral_coclustering.html

import numpy as np
from numpy.linalg import inv, norm
from numpy.random import default_rng
from matplotlib import pyplot as plt
from scipy.linalg import fractional_matrix_power as fmp

from sklearn.datasets import make_biclusters
from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import consensus_score, silhouette_score, accuracy_score, adjusted_rand_score, v_measure_score, adjusted_mutual_info_score

from dataclasses import dataclass, field
from pprint import pprint
import sys, logging

RNG_SEED=996535595 #297395776
NO_CLUSTERS = 5 # 4
MAT_SIZE = 500 # 500
ATTEMPTS_MAX = 10 # it'll work eventually!
ITER_MAX = 5 # 5 seems ok..? (at least for synthetic data)

#TODO: test '# or maybe just keep going but dont recalculate S, although i dont think that makes a difference'
#TODO: make it work with numba


# NOTE: 297395776 leads to singular newQ and newP, both with no ones in the 0th column
# so everything breaks on iter 1, step 1
# question: are all cases doable? also maybe check convergence individually? (not sure if it matters actually)

# NOTE: 233735426 has 2 null columns ;;;-;;;
# HMMM: but on attempt 2, it works (perfectly) after only 5 iterations

# seed=1045625548, c=4, N=6000 works on first attempt; low c seems good

# for c=7, seed=97132067, N=1000, it works on attempt 4

"""
SpectralCoclustering.biclusters_ is the tuple: (row_truths, col_truths), where row_truths is a boolean matrix
with NO_CLUSTERS lines indicating whether the jth row belongs to the ith cluster (and
col_truths is analogous).
Caution: data was shuffled using row_idx, so rows won't have the real labels.

SpectralCoclustering.row_labels_ is an N array (if data is N x M) 
with the integer (0-indexed) indices for each row. (And column_labels_ works analogously.)
"""
# DO: remember to flip P and Q so they have the correct shape

def start_default_rng (seed=None):
    """ Start the default RNG with a (user-provided / randomly-generated) seed 
    so we don't have to store the RNG's starting state (which is less human-friendly)."""

    if seed:
        new_seed = seed
    else:
        temp_rng = default_rng()
        new_seed = temp_rng.integers(low=0, high=2**30) # get a seed so we don't have to store
    new_rng = default_rng(new_seed)
    print(f"RNG seed is: {new_seed}\n")
    return new_rng

def cool_header_thing ():
    print_rng = default_rng()
    n1 = print_rng.integers(low=14, high=21)
    n2 = print_rng.integers(low=14, high=21)
    s = "".join(print_rng.permutation(list(n1* "#" + n2* "@")))
    return s

def mat_debug (M: np.ndarray, name=None):
    name = name if name else "matrix"
    s = cool_header_thing()
    header =  f"\n\n{s}  {name}  {s}\n"
    print(header)
    print(f"dtype, min, max, norm: {M.dtype} | {np.abs(M).min()} | {np.abs(M).max()} | {np.linalg.norm(M)}")
    if M.shape[0] * M.shape[1] < 2000:
        pprint(M)

def count_zero_columns (M : np.ndarray):
    M_col_sum = np.sum(M, axis=0)
    zero_cols = np.argwhere( (M_col_sum == 0))
    return zero_cols.shape[0]

@dataclass(eq=False) # generated eq method might not be ideal?
class WBKM_coclustering:
    # init arguments
    data: np.ndarray
    n_clusters: int
    iter_max: int = ITER_MAX
    n_attempts: int = ATTEMPTS_MAX
    random_state: int = None
    verbose: bool = False
    logger : logging.Logger = None

    # properties calculated post init
    biclusters_: np.ndarray = field(init=False)
    row_labels_: np.ndarray = field(init=False)
    column_labels_: np.ndarray = field(init=False)
    P : np.ndarray = field(init=False)
    Q : np.ndarray = field(init=False)
    S : np.ndarray = field(init=False)
    D1 : np.ndarray = field(init=False)
    D2 : np.ndarray = field(init=False)
    best_max_iter_reached : int = field(init=False)
    best_no_zero_cols : int = field(init=False)

    def print_or_log(self, s):
        # NOTE: logging might break if there's an exception while logging
        if self.logger:
            self.logger.info(s)
        else:
            print(s)

    def rand_ind_matrix (shape, rng):
        """Create a random zero matrix with NO_CLUSTERS columns and 
        a randomly placed 1 for each line."""
        ret = np.zeros(shape)
        c = shape[1]
        chosen_idx = rng.integers(low=0, high=c, size=(shape[0],))
        chosen_idx = np.resize(chosen_idx, (shape[1], shape[0])).T
        _, j_idx = np.mgrid[slice(ret.shape[0]), slice(ret.shape[1])] # prefer anything over for loop

        ret = np.where((chosen_idx == j_idx) , 1, 0)
        return ret

    def getNewQ(self, Q, R, X, D1, D2):
        ### iter over lines..?
        ###_, j_idx = np.mgrid[slice(Q.shape[0]), slice(Q.shape[1])] # prefer anything over for loop
        self.print_or_log("    Getting new Q ...")
        first_term_mat = inv(D1) @ X @ inv(D2) # first term is actually the ith column
        
        # one column of DXD at a time
        newQ = np.zeros(Q.shape) # might be R.shape? idk
        for i in range(Q.shape[0]):
            a = first_term_mat.shape[0] # column size
            b = R.shape[1] # number of columns

            # this is the shape for the appropriate resize but we get the column repeated as matrix rows, 
            # so we need to transpose it
            first_term_ith_column_many_times = np.resize(first_term_mat[:,i], (b,a)).T
            #print("ith many times shape:", first_term_ith_column_many_times.shape) # (500,c)
            #NOTE: numba doe snot support np.resize :c
            

            # we use arange() to get the values that k assumes in the formula
            # (pick between the c columns of R the one that gives the smallest thing)
            cond = (np.argmin(np.sum((first_term_ith_column_many_times - R)**2, axis=0)) == np.arange(R.shape[1])) # bool array
            newQ[i, :] = np.where(cond, 1, 0)
        return newQ

    def getNewP(self, P, L, X, D1, D2):
        self.print_or_log("    Getting new P ...")

        first_term_mat = inv(D1) @ X @ inv(D2) # first term is actually the jth row

        # one row of DXD at a time
        newP = np.zeros(P.shape) # might be L.shape? idk
        for j in range(P.shape[0]):
            a = first_term_mat.shape[1] # row size
            b = L.shape[0] # number of rows
            first_term_ith_row_many_times = np.resize(first_term_mat[j,:], (b,a)) # FIXME: use repeat instead?
            

            # NOTE:assuming ith row od DXD instead of jth row
            # otherwise, we might have more than 1 chosen j in a single line (so its not an indicator matrix)

            # we use arange() to get the values that k assumes in the formula
            # (pick between the c rows of L the one that gives the smallest thing)
            cond = (np.argmin(np.sum((first_term_ith_row_many_times - L)**2, axis=1)) == np.arange(L.shape[0])) # bool array
            newP[j, :] = np.where(cond, 1, 0)

        return newP

    def get_stuff (P, Q):
        bic = (np.array(P, dtype=bool).T, np.array(Q, dtype=bool).T)
        row = np.argmax(P, axis=1)
        col = np.argmax(Q, axis=1)
        return bic, row, col
    
    def objective (self, X, D1, D2, S, Q, P):
        try:
            obj = norm(
                fmp(D1,-0.5)@X@fmp(D2,-0.5) - fmp(D1,0.5)@P@S@Q.T@fmp(D2,0.5)
                )
        except Exception as e:
            self.print_or_log(str(e))
            obj = np.nan
        return obj
    
    def attempt_coclustering (self, X, D1, D2, c, attempt_no=0):
        s = cool_header_thing()
        self.print_or_log(f"\n{s}\nAttempt #{attempt_no}:\n{s}\n")
        d, n = X.shape

        # generate random indicator matrices
        P = WBKM_coclustering.rand_ind_matrix((d,c), self.RNG)
        Q = WBKM_coclustering.rand_ind_matrix((n,c), self.RNG)

        f = lambda M: np.diag(np.diag(M)) # function to make matrix out of diagonal
        newQ, newP = None, None
        stop_everything, no_zero_cols = False, 0
        iteration = 0
        S = inv(P.T @ D1 @ P @ Q.T @ D2 @ Q) @ f(P.T @ X @ Q)
        obj = self.objective(X, D1, D2, S, Q, P)
        self.print_or_log(f"\nobjective: {obj}")
        while iteration < self.iter_max: # FIXME: add convergence condition # frobenius on newP,P and newQ,Q?
            if iteration != 0:
                Q = newQ
                P = newP
            self.print_or_log(f"iteration #{iteration+1}")

            # step 1
            self.print_or_log("  step 1")
            #S = inv(P.T @ D1 @ P @ Q.T @ D2 @ Q) @ f(P.T @ X @ Q) # FIXME: singular matrix in iter 1

            to_be_inv = P.T @ D1 @ P @ Q.T @ D2 @ Q
            if np.linalg.det(to_be_inv) == 0:
                self.print_or_log("WARN: det is 0; step 1 failed")
                #mat_debug(to_be_inv, "to_be_inv")

                no_zero_cols = count_zero_columns(to_be_inv)
                self.print_or_log(f"number of zero columns: {no_zero_cols}")

                stop_everything = True # NOTE: maybe this conveniently indicates convergence? c:
                break # or maybe just keep going but dont recalculate S, although i dont think that makes a difference
                
            S = inv(to_be_inv) @ f(P.T @ X @ Q) # FIXME: possibly singular matrix in the inverse calculation

            # step 2
            self.print_or_log("  step 2")
            R = P @ S
            newQ = self.getNewQ(Q, R, X, D1, D2)

            # step 3
            self.print_or_log("  step 3")
            # NOTE: at this point, we must use newQ, but note that we have already calculated S
            # for this iteration, and so we mustn't recalculate it
            L = S @ newQ.T # FIXME: newQ or Q?
            newP = self.getNewP(P, L, X, D1, D2)
            
            obj = self.objective(X, D1, D2, S, newQ, newP)
            self.print_or_log(f"objective: {obj}\n")
            iteration += 1 # increase iteration counter
        self.print_or_log(f"stuff: {stop_everything}, {iteration}, {no_zero_cols}")
        
        return (newP, newQ, S, stop_everything, iteration, no_zero_cols)

    # run after auto-generated init
    def __post_init__(self):
        # pre-initialization

        # initialization

        """
        # float128 is just longdouble; it's platform-dependent; but most of all it's unsupported by linalg
        # and commenting out the type check (in np.linalg._realType) doesn't seem to work :(  
        # (_umath_linalg is object code)
        # possible alternative: https://mpmath.org/
        # https://stackoverflow.com/questions/9062562/what-is-the-internal-precision-of-numpy-float128/17023995#17023995
        """
        ##################### #OMEGA NOTE #########################################
        # a lot of outdated info ;-;
        ##################### #OMEGA NOTE #########################################
        self.RNG = default_rng(self.random_state)
        preferred_dtype = np.float64
        X = np.array(self.data, dtype=preferred_dtype) if not isinstance(self.data, np.ndarray) else self.data.astype(preferred_dtype)      

        c = self.n_clusters

        # NOTE: D1 and D2 have VERY LARGE entries with default values for make_biclusters
        # (minval=10 and maxval=100 for the clusters leads to entries around 5e3 in D1, 
        # so np.linalg.det(D1)==inf  c: )
        ## inv can still work though
        sum_over_columns = np.sum(X, axis=0) # shape=(n,)
        sum_over_rows = np.sum(X, axis=1) # shape=(d,)
        D1 = np.diag(sum_over_rows) # diagonal weight matrix, shape=(d,d)
        D2 = np.diag(sum_over_columns) # diagonal weight matrix, shape=(n,n)
        self.D1, self.D2 = D1, D2
        
        bestP, bestQ = None, None
        forced_exit, best_max_iter_reached, best_no_zero_cols = True, 0, c
        attempt = 0
        while attempt < self.n_attempts and forced_exit:
            P, Q, S, forced_exit, max_iter_reached, no_zero_cols = self.attempt_coclustering(X, D1, D2, c, attempt_no=attempt+1)
            if max_iter_reached > best_max_iter_reached:
                self.print_or_log("is best because 1") 
                # NOTE: forced_exit=False will always go further and get the better max_iter
                best_max_iter_reached, best_no_zero_cols = max_iter_reached, no_zero_cols
                bestP, bestQ, bestS = P, Q, S
                if self.verbose:
                    self.print_or_log("\n__is__ best!")
            elif max_iter_reached == best_max_iter_reached and no_zero_cols < best_no_zero_cols:
                self.print_or_log("best because 2")
                best_no_zero_cols = no_zero_cols
                bestP, bestQ, bestS = P, Q, S
                if self.verbose:
                    self.print_or_log("\n__is__ best!")
            attempt += 1

        if forced_exit:
            self.print_or_log("｡ﾟ(ﾟ´Д｀ﾟ)ﾟ｡　singular matrix every time nuu :c\n")
        self.P, self.Q, self.S = bestP, bestQ, bestS
        self.best_no_zero_cols, self.best_max_iter_reached = best_no_zero_cols, best_max_iter_reached
        self.biclusters_, self.row_labels_, self.column_labels_ = WBKM_coclustering.get_stuff(P, Q)
        
if __name__ == '__main__':
    RNG = start_default_rng(seed=RNG_SEED+1)
    
    data, rows, columns = make_biclusters(
        shape=(MAT_SIZE, MAT_SIZE), n_clusters=NO_CLUSTERS, shuffle=False, random_state=RNG_SEED,
        noise=0.01,
        minval=0.3,
        maxval=0.7
    )
    

    plt.matshow(data, cmap=plt.cm.Blues)
    plt.title("Original dataset")
    plt.savefig("bla1.png")

    # shuffle clusters
    row_idx = RNG.permutation(data.shape[0])
    col_idx = RNG.permutation(data.shape[1])
    data = data[row_idx][:, col_idx]

    plt.matshow(data, cmap=plt.cm.Blues)
    plt.title("Shuffled dataset")
    plt.savefig("bla2.png")

    # do co-clustering
    model = WBKM_coclustering(data, n_clusters=NO_CLUSTERS, random_state=RNG_SEED, n_attempts=ATTEMPTS_MAX,
        verbose=True)
    #pprint(model.biclusters_)
    #pprint(model.row_labels_)
    #pprint(model.column_labels_)

    #########################
    # evaluate results 
    #########################
        
    # bicluster-specific:
    bic_true = rows[:, row_idx], columns[:, col_idx]
    bic_pred = model.biclusters_
    con_score = consensus_score(bic_true, bic_pred)
    print("\nconsensus score: {:.3f}".format(con_score))

    sil_score_row = silhouette_score(data, model.row_labels_) # move below? do external/internal?
    sil_score_col = silhouette_score(data.T, model.column_labels_)
    print(f"\nsilhouette score:\n\trows: {sil_score_row:.3f}\n\tcols: {sil_score_col:.3f}")

    # retrieve integer labels
    def bic_boolean_to_labels (bic):
        rows, cols = bic
        labelize = lambda a: np.argmax(a, axis=0)
        row_labels, col_labels = labelize(rows), labelize(cols)
        return row_labels, col_labels
    pred_rows, pred_columns = model.row_labels_, model.column_labels_
    true_rows, true_columns = bic_boolean_to_labels(bic_true)

    # rows (samples):
    #acc = accuracy_score(bic_true[0].T, bic_pred[0].T) # acc not work :c
    #acc = accuracy_score(true_rows, pred_rows)
    ari = adjusted_rand_score(true_rows, pred_rows)
    ami = adjusted_mutual_info_score(true_rows, pred_rows)
    vmeasure = v_measure_score(true_rows, pred_rows)
    print(f"rows:\n  ARI= {ari:.3f}\n  AMI= {ami:.3f}\n  VMs= {vmeasure:.3f}\n")

    # columns (features/attributes):
    #acc = accuracy_score(bic_true[1], bic_pred[1])
    ari = adjusted_rand_score(true_columns, pred_columns)
    vmeasure = v_measure_score(true_columns, pred_columns)
    ami = adjusted_mutual_info_score(true_columns, pred_columns)
    print(f"columns:\n  ARI= {ari:.3f}\n  AMI= {ami:.3f}\n  VMs= {vmeasure:.3f}\n")

    # rearranged data
    fit_data = data[np.argsort(model.row_labels_)]
    fit_data = fit_data[:, np.argsort(model.column_labels_)]
    plt.matshow(fit_data, cmap=plt.cm.Blues)
    plt.title("After biclustering; rearranged to show biclusters")
    plt.savefig("bla3.png")

    plt.show()