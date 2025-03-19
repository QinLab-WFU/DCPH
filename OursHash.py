# VTS (CSQ with ViT Backbone - ICME 2022)
# paper [Vision Transformer Hashing for Image Retrieval, ICME 2022](https://arxiv.org/pdf/2109.12564.pdf)
# CSQ basecode considered from https://github.com/swuxyj/DeepHash-pytorch

from utils.tools import *
import argparse
import os
import random
import torch
import torch.optim as optim
import time
import numpy as np
from scipy.linalg import hadamard
import scipy.io as scio

from utils.SCPloss import Criterion
from TransformerModel.Nolotransformer import Network


torch.multiprocessing.set_sharing_strategy('file_system')


def get_config():
    config = {
        # "dataset": "cifar10",
        # "dataset": "mirflickr",
        # "dataset": "cifar10-2",
        "dataset": "coco",
        # "dataset": "nuswide_21",
        # "dataset": "imagenet",

        "net": Network, "net_print": "Nolo",

        "bit_list": [64],
        "optimizer": {"type": optim.Adam, "optim_params": {"lr": 1e-5, "weight_decay": 1e-5}},
        "device": torch.device("cuda:0"), "save_path": "Checkpoints_Results",
        "epoch": 200, "test_map": 100, "batch_size": 128, "resize_size": 256, "crop_size": 224,
        "info": "OURS", "alpha": 0.1,
    }
    config = config_dataset(config)
    return config


def save_mat(query_img, query_labels, retrieval_img, retrieval_labels, bit):
    save_dir = './result/PR-curve/ours'
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
    scio.savemat(os.path.join(save_dir, "Ours" + str(bit) + "-vit-" + "coco" + "-" + ".mat"), result_dict)


def train_val(config, bit):
    start_epoch = 1
    Best_mAP = 0
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train

    num_classes = config["n_class"]
    hash_bit = bit

    net = Network(embed_dim=128, hash_bit=hash_bit)
    net = net.to(device)

    if not os.path.exists(config["save_path"]):
        os.makedirs(config["save_path"])

    results_path = os.path.join(config["save_path"],
                                config["dataset"] + "_" + config["info"] + "_" + config["net_print"] + "_Bit" + str(
                                    bit) + ".txt")
    f = open(results_path, 'a')

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    SCPloss = Criterion(embed_dim=hash_bit, n_classes=num_classes, device=device)
    optimizer_CPFloss = torch.optim.Adam(params=SCPloss.parameters(), lr=1e-5)

    for epoch in range(start_epoch, config["epoch"] + 1):
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print("%s-%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], config["net_print"], epoch, config["epoch"], current_time, bit, config["dataset"]), end="")
        net.train()
        train_loss = 0
        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)
            label = label.float()

            optimizer.zero_grad()
            optimizer_CPFloss.zero_grad()

            u = net(image)

            loss1 = SCPloss(u, label)
            loss = loss1

            train_loss += loss.item()

            loss.backward()

            optimizer.step()
            optimizer_CPFloss.step()
        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))
        f.write('Train | Epoch: %d | Loss: %.3f\n' % (epoch, train_loss))

        if (epoch) % config["test_map"] == 0:
            # print("calculating test binary code......")
            tst_binary, tst_label = compute_result(test_loader, net, device=device)

            # print("calculating dataset binary code.......")\
            trn_binary, trn_label = compute_result(dataset_loader, net, device=device)

            # print("calculating map.......")
            mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                             config["topK"])

            if mAP > Best_mAP:
                Best_mAP = mAP
                # save_mat(tst_binary.numpy(), tst_label.numpy(), trn_binary.numpy(), trn_label.numpy(), bit=hash_bit)
            print("%s epoch:%d, bit:%d, dataset:%s, MAP:%.3f, Best MAP: %.3f" % (
                config["info"], epoch, bit, config["dataset"], mAP, Best_mAP))
            f.write('Test | Epoch %d | MAP: %.3f | Best MAP: %.3f\n'
                    % (epoch, mAP, Best_mAP))

    f.close()


if __name__ == "__main__":
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        train_val(config, bit)