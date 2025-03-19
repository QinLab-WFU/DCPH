import scipy.io as scio
import os


def save_mat(query_img, query_labels, retrieval_img, retrieval_labels, method="CSQ", dataset="cifar10", bit=16):

    save_dir = './result/code'
    os.makedirs(save_dir, exist_ok=True)

    query_img = query_img
    retrieval_img = retrieval_img
    query_labels = query_labels
    retrieval_labels = retrieval_labels

    result_dict = {
        'q_img': query_img,
        'r_img': retrieval_img,
        'q_l': query_labels,
        'r_l': retrieval_labels
    }

    scio.savemat(os.path.join(save_dir, method + dataset + str(bit) + ".mat"), result_dict)