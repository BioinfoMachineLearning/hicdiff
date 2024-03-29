import gzip
import numpy as np
import copy
from sklearn import metrics
import scipy.sparse as sps

# Code taken from https://github.com/kundajelab/genomedisco
def to_transition(mtogether):
    sums = mtogether.sum(axis=1)
    # make the ones that are 0, so that we don't divide by 0
    sums[sums == 0.0] = 1.0
    D = sps.spdiags(1.0 / sums.flatten(), [0], mtogether.shape[0], mtogether.shape[1], format='csr')
    return D.dot(mtogether)


def random_walk(m_input, t):
    # return m_input.__pow__(t)
    # return np.linalg.matrix_power(m_input,t)
    return m_input.__pow__(t)


def write_diff_vector_bedfile(diff_vector, nodes, nodes_idx, out_filename):
    out = gzip.open(out_filename, 'w')
    for i in range(diff_vector.shape[0]):
        node_name = nodes_idx[i]
        node_dict = nodes[node_name]
        out.write(str(node_dict['chr']) + '\t' + str(node_dict['start']) + '\t' + str(
            node_dict['end']) + '\t' + node_name + '\t' + str(diff_vector[i][0]) + '\n')
    out.close()


def compute_reproducibility(m1_csr, m2_csr, transition, tmax=3, tmin=3):
    # make symmetric
    m1up = m1_csr # m1_csr is output,
    m1down = m1up.transpose()
    #m1down.setdiag(0)
    m1 = m1up

    m2up = m2_csr # m2_csr is target
    m2down = m2up.transpose()
    #m2down.setdiag(0)
    m2 = m2up

    # convert to an actual transition matrix
    if transition:
        m1 = to_transition(m1)
        m2 = to_transition(m2)

    # count nonzero nodes (note that we take the average number of nonzero nodes in the 2 datasets)
    rowsums_1 = m1.sum(axis=1)
    nonzero_1 = [i for i in range(rowsums_1.shape[0]) if rowsums_1[i] > 0.0]
    rowsums_2 = m2.sum(axis=1)
    nonzero_2 = [i for i in range(rowsums_2.shape[0]) if rowsums_2[i] > 0.0]
    nonzero_total = len(list(set(nonzero_1).union(set(nonzero_2))))
    nonzero_total = 0.5 * (1.0 * len(list(set(nonzero_1))) + 1.0 * len(list(set(nonzero_2))))
    if nonzero_total == 0:
        nonzero_total = 1.0

    scores = []
    if True:
        diff_vector = np.zeros((m1.shape[0], 1))
        for t in range(1, tmax + 1):  # range(args.tmin,args.tmax+1):
            extra_text = ' (not included in score calculation)'
            if t == 1:
                rw1 = copy.deepcopy(m1)
                rw2 = copy.deepcopy(m2)

            else:
                rw1 = rw1.dot(m1)
                rw2 = rw2.dot(m2)

            if t >= tmin:
                # diff_vector += (abs(rw1 - rw2)).sum(axis=1)
                diff = abs(rw1 - rw2).sum()  # +euclidean(rw1.toarray().flatten(),rw2.toarray().flatten()))
                scores.append(1.0 * float(diff) / float(nonzero_total))
                extra_text = ' | score=' + str('{:.3f}'.format(1.0 - float(diff) / float(nonzero_total)))
    #             print('GenomeDISCO | ' + strftime("%c") + ' | done t=' + str(t) + extra_text)

    # compute final score
    ts = range(tmin, tmax + 1)
    denom = len(ts) - 1
    if tmin == tmax:
        auc = scores[0]

        if 2 < auc:
            #auct = int(auc)
            auc = 2

        elif 0 <= auc <= 2:
            auc = auc

    else:
        auc = metrics.auc(range(len(ts)), scores) / denom

    reproducibility = (1 - auc)

    #gds_hic = open("GenomeDISCO.txt", 'a')
    #gds_hic.write("scores_0:\t"+str(scores[0])+"\t"+"denom:\t"+str(denom)+"\t"+"auc:\t"+str(auc)+"\t"+"gds:\t"+str(reproducibility)+"\n")
    return reproducibility

