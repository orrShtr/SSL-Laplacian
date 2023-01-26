import random
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms


from utils_datasets import generate_2moons, generate_3moons
from utils import createAffinity, createAffinitySSL, createAffinityWNLL, ev_calculation_L, SpectralClusteringFromEV, SSL_GL_Clustering

import warnings
warnings.simplefilter("ignore", UserWarning)

#MNIST
plotFlag = True
model_path = r".\results\MNIST_all_0212"
if not os.path.exists(model_path):
    os.mkdir(model_path)
    os.mkdir(model_path + '/images')

print("MNIST")

classes = ('0', '1', '2', '3', '4',
           '5', '6', '7', '8', '9')
classNum = len(classes)

nodes_num = 70000
data_dir = '../dataset'
train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True)
data_transform = transforms.Compose([
    transforms.ToTensor(),
])
train_dataset.transform = data_transform
test_dataset.transform = data_transform

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=60000, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10000, shuffle=False)

train_iter = iter(train_loader)
test_iter = iter(test_loader)

images_train, y_train = train_iter.next()
X_train = images_train.view(60000, -1)

images_test, y_test = test_iter.next()
X_test = images_test.view(10000, -1)

y = torch.cat((y_train, y_test), 0)
X = torch.cat((X_train, X_test), 0)

print("y shape", y.shape)
print("X shape", X.shape)

ms = 50 #10
ms_normal = 20
sigmaFlag = 0
mu1 = 2.0
# mu2 = 2.0
# number_of_labeled_nodes_array = [0, 50, 100, 200, 300, 400, 500]
number_of_labeled_nodes_array = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
number_of_labeled_nodes_array_len = len(number_of_labeled_nodes_array)


drawNum = 10

nmi_ssl_mean_array = []
nmi_ssl_std_array = []
acc_ssl_mean_array = []
acc_ssl_std_array = []

nmi_wnll_mean_array = []
nmi_wnll_std_array = []
acc_wnll_mean_array = []
acc_wnll_std_array = []

nmi_GL_us_mean_array = []
nmi_GL_us_std_array = []
acc_GL_us_mean_array = []
acc_GL_us_std_array = []

nmi_GL_ssl_mean_array = []
nmi_GL_ssl_std_array = []
acc_GL_ssl_mean_array = []
acc_GL_ssl_std_array = []

nmi_GL_wnll_mean_array = []
nmi_GL_wnll_std_array = []
acc_GL_wnll_mean_array = []
acc_GL_wnll_std_array = []

for i in range(number_of_labeled_nodes_array_len):
    number_of_labeled_nodes = number_of_labeled_nodes_array[i]
    print("number_of_labeled_nodes : ", number_of_labeled_nodes)

    if number_of_labeled_nodes == 0:
        mu2 = 0
    else:
        mu2 = (nodes_num / number_of_labeled_nodes) - 1

    nmi_ssl_array_curr_param = []
    acc_ssl_array_curr_param = []

    nmi_wnll_array_curr_param = []
    acc_wnll_array_curr_param = []

    nmi_GL_us_array_curr_param = []
    acc_GL_us_array_curr_param = []

    nmi_GL_ssl_array_curr_param = []
    acc_GL_ssl_array_curr_param = []

    nmi_GL_wnll_array_curr_param = []
    acc_GL_wnll_array_curr_param = []

    for j in range(drawNum):
        train_iter = iter(train_loader)
        images, y = train_iter.next()
        X = images.view(nodes_num, -1)

        nodes_indx_list = range(0, nodes_num)
        labeled_index = random.sample(nodes_indx_list, number_of_labeled_nodes)
        unlabeled_index = [indx for indx in nodes_indx_list if indx not in labeled_index]

        # L_SSL
        W_ssl = createAffinitySSL(X, y, ms, ms_normal, sigmaFlag, labeled_index, classNum, mu1, mu2)
        ev = ev_calculation_L(W_ssl, classNum)
        ev_unlabeled = ev[unlabeled_index]
        RCut_labels, model_nmi, model_acc = SpectralClusteringFromEV(ev_unlabeled, y[unlabeled_index], classNum)
        nmi_ssl_array_curr_param.append(model_nmi)
        acc_ssl_array_curr_param.append(model_acc)

        # L_WNLL
        W_WNLL = createAffinityWNLL(X, ms, ms_normal, sigmaFlag, labeled_index)
        ev = ev_calculation_L(W_WNLL, classNum)
        ev_unlabeled = ev[unlabeled_index]
        RCut_labels, model_nmi, model_acc = SpectralClusteringFromEV(ev_unlabeled, y[unlabeled_index], classNum)
        nmi_wnll_array_curr_param.append(model_nmi)
        acc_wnll_array_curr_param.append(model_acc)

        # GL US
        W_US = createAffinity(X, ms, ms_normal, sigmaFlag)
        s0 = torch.sum(W_US, axis=0)
        D = torch.diag(s0)
        L = D - W_US
        clusteringRes, model_nmi, model_acc = SSL_GL_Clustering(L, labeled_index, unlabeled_index, y, classes)
        nmi_GL_us_array_curr_param.append(model_nmi)
        acc_GL_us_array_curr_param.append(model_acc)

        # GL SSL
        s0 = torch.sum(W_ssl, axis=0)
        D = torch.diag(s0)
        L = D - W_ssl
        clusteringRes, model_nmi, model_acc = SSL_GL_Clustering(L, labeled_index, unlabeled_index, y, classes)
        nmi_GL_ssl_array_curr_param.append(model_nmi)
        acc_GL_ssl_array_curr_param.append(model_acc)

        # GL WNLL
        s0 = torch.sum(W_WNLL, axis=0)
        D = torch.diag(s0)
        L = D - W_WNLL
        clusteringRes, model_nmi, model_acc = SSL_GL_Clustering(L, labeled_index, unlabeled_index, y, classes)
        nmi_GL_wnll_array_curr_param.append(model_nmi)
        acc_GL_wnll_array_curr_param.append(model_acc)

    # L SSL
    nmi_ssl_array_curr_param = torch.Tensor(nmi_ssl_array_curr_param)
    acc_ssl_array_curr_param = torch.Tensor(acc_ssl_array_curr_param)
    nmi_ssl_array_curr_param_mean = torch.mean(nmi_ssl_array_curr_param)
    nmi_ssl_array_curr_param_std = torch.std(nmi_ssl_array_curr_param)
    nmi_ssl_mean_array.append(nmi_ssl_array_curr_param_mean)
    nmi_ssl_std_array.append(nmi_ssl_array_curr_param_std)
    acc_ssl_array_curr_param_mean = torch.mean(acc_ssl_array_curr_param)
    acc_ssl_array_curr_param_std = torch.std(acc_ssl_array_curr_param)
    acc_ssl_mean_array.append(acc_ssl_array_curr_param_mean)
    acc_ssl_std_array.append(acc_ssl_array_curr_param_std)
    print("L SSL : ")
    print("NMI mean : ", nmi_ssl_array_curr_param_mean)
    print("NMI std : ", nmi_ssl_array_curr_param_std)
    print("ACC mean : ", acc_ssl_array_curr_param_mean)
    print("ACC std : ", acc_ssl_array_curr_param_std)

    # L wnll
    nmi_wnll_array_curr_param = torch.Tensor(nmi_wnll_array_curr_param)
    acc_wnll_array_curr_param = torch.Tensor(acc_wnll_array_curr_param)
    nmi_wnll_array_curr_param_mean = torch.mean(nmi_wnll_array_curr_param)
    nmi_wnll_array_curr_param_std = torch.std(nmi_wnll_array_curr_param)
    nmi_wnll_mean_array.append(nmi_wnll_array_curr_param_mean)
    nmi_wnll_std_array.append(nmi_wnll_array_curr_param_std)
    acc_wnll_array_curr_param_mean = torch.mean(acc_wnll_array_curr_param)
    acc_wnll_array_curr_param_std = torch.std(acc_wnll_array_curr_param)
    acc_wnll_mean_array.append(acc_wnll_array_curr_param_mean)
    acc_wnll_std_array.append(acc_wnll_array_curr_param_std)
    print("L wnll : ")
    print("NMI mean : ", nmi_wnll_array_curr_param_mean)
    print("NMI std : ", nmi_wnll_array_curr_param_std)
    print("ACC mean : ", acc_wnll_array_curr_param_mean)
    print("ACC std : ", acc_wnll_array_curr_param_std)

    # GL us
    nmi_GL_us_array_curr_param = torch.Tensor(nmi_GL_us_array_curr_param)
    acc_GL_us_array_curr_param = torch.Tensor(acc_GL_us_array_curr_param)
    nmi_GL_us_array_curr_param_mean = torch.mean(nmi_GL_us_array_curr_param)
    nmi_GL_us_array_curr_param_std = torch.std(nmi_GL_us_array_curr_param)
    nmi_GL_us_mean_array.append(nmi_GL_us_array_curr_param_mean)
    nmi_GL_us_std_array.append(nmi_GL_us_array_curr_param_std)
    acc_GL_us_array_curr_param_mean = torch.mean(acc_GL_us_array_curr_param)
    acc_GL_us_array_curr_param_std = torch.std(acc_GL_us_array_curr_param)
    acc_GL_us_mean_array.append(acc_GL_us_array_curr_param_mean)
    acc_GL_us_std_array.append(acc_GL_us_array_curr_param_std)
    print("GL us : ")
    print("NMI mean : ", nmi_GL_us_array_curr_param_mean)
    print("NMI std : ", nmi_GL_us_array_curr_param_std)
    print("ACC mean : ", acc_GL_us_array_curr_param_mean)
    print("ACC std : ", acc_GL_us_array_curr_param_std)

    # GL ssl
    nmi_GL_ssl_array_curr_param = torch.Tensor(nmi_GL_ssl_array_curr_param)
    acc_GL_ssl_array_curr_param = torch.Tensor(acc_GL_ssl_array_curr_param)
    nmi_GL_ssl_array_curr_param_mean = torch.mean(nmi_GL_ssl_array_curr_param)
    nmi_GL_ssl_array_curr_param_std = torch.std(nmi_GL_ssl_array_curr_param)
    nmi_GL_ssl_mean_array.append(nmi_GL_ssl_array_curr_param_mean)
    nmi_GL_ssl_std_array.append(nmi_GL_ssl_array_curr_param_std)
    acc_GL_ssl_array_curr_param_mean = torch.mean(acc_GL_ssl_array_curr_param)
    acc_GL_ssl_array_curr_param_std = torch.std(acc_GL_ssl_array_curr_param)
    acc_GL_ssl_mean_array.append(acc_GL_ssl_array_curr_param_mean)
    acc_GL_ssl_std_array.append(acc_GL_ssl_array_curr_param_std)
    print("GL ssl : ")
    print("NMI mean : ", nmi_GL_ssl_array_curr_param_mean)
    print("NMI std : ", nmi_GL_ssl_array_curr_param_std)
    print("ACC mean : ", acc_GL_ssl_array_curr_param_mean)
    print("ACC std : ", acc_GL_ssl_array_curr_param_std)

    # GL wnll
    nmi_GL_wnll_array_curr_param = torch.Tensor(nmi_GL_wnll_array_curr_param)
    acc_GL_wnll_array_curr_param = torch.Tensor(acc_GL_wnll_array_curr_param)
    nmi_GL_wnll_array_curr_param_mean = torch.mean(nmi_GL_wnll_array_curr_param)
    nmi_GL_wnll_array_curr_param_std = torch.std(nmi_GL_wnll_array_curr_param)
    nmi_GL_wnll_mean_array.append(nmi_GL_wnll_array_curr_param_mean)
    nmi_GL_wnll_std_array.append(nmi_GL_wnll_array_curr_param_std)
    acc_GL_wnll_array_curr_param_mean = torch.mean(acc_GL_wnll_array_curr_param)
    acc_GL_wnll_array_curr_param_std = torch.std(acc_GL_wnll_array_curr_param)
    acc_GL_wnll_mean_array.append(acc_GL_wnll_array_curr_param_mean)
    acc_GL_wnll_std_array.append(acc_GL_wnll_array_curr_param_std)
    print("GL wnll : ")
    print("NMI mean : ", nmi_GL_wnll_array_curr_param_mean)
    print("NMI std : ", nmi_GL_wnll_array_curr_param_std)
    print("ACC mean : ", acc_GL_wnll_array_curr_param_mean)
    print("ACC std : ", acc_GL_wnll_array_curr_param_std)



nmi_ssl_mean_array = torch.Tensor(nmi_ssl_mean_array)
nmi_ssl_std_array = torch.Tensor(nmi_ssl_std_array)
acc_ssl_mean_array = torch.Tensor(acc_ssl_mean_array)
acc_ssl_std_array = torch.Tensor(acc_ssl_std_array)

nmi_wnll_mean_array = torch.Tensor(nmi_wnll_mean_array)
nmi_wnll_std_array = torch.Tensor(nmi_wnll_std_array)
acc_wnll_mean_array = torch.Tensor(acc_wnll_mean_array)
acc_wnll_std_array = torch.Tensor(acc_wnll_std_array)

nmi_GL_us_mean_array = torch.Tensor(nmi_GL_us_mean_array)
nmi_GL_us_std_array = torch.Tensor(nmi_GL_us_std_array)
acc_GL_us_mean_array = torch.Tensor(acc_GL_us_mean_array)
acc_GL_us_std_array = torch.Tensor(acc_GL_us_std_array)

nmi_GL_ssl_mean_array = torch.Tensor(nmi_GL_ssl_mean_array)
nmi_GL_ssl_std_array = torch.Tensor(nmi_GL_ssl_std_array)
acc_GL_ssl_mean_array = torch.Tensor(acc_GL_ssl_mean_array)
acc_GL_ssl_std_array = torch.Tensor(acc_GL_ssl_std_array)

nmi_GL_wnll_mean_array = torch.Tensor(nmi_GL_wnll_mean_array)
nmi_GL_wnll_std_array = torch.Tensor(nmi_GL_wnll_std_array)
acc_GL_wnll_mean_array = torch.Tensor(acc_GL_wnll_mean_array)
acc_GL_wnll_std_array = torch.Tensor(acc_GL_wnll_std_array)



print("L_WNLL")
print("NMI Mean : ", nmi_wnll_mean_array)
print("NMI STD : ", nmi_wnll_std_array)
print("ACC Mean : ", acc_wnll_mean_array)
print("ACC STD : ", acc_wnll_std_array)

print("L_ssl")
print("NMI Mean : ", nmi_ssl_mean_array)
print("NMI STD : ", nmi_ssl_std_array)
print("ACC Mean : ", acc_ssl_mean_array)
print("ACC STD : ", acc_ssl_std_array)

print("GL_US")
print("NMI Mean : ", nmi_GL_us_mean_array)
print("NMI STD : ", nmi_GL_us_std_array)
print("ACC Mean : ", acc_GL_us_mean_array)
print("ACC STD : ", acc_GL_us_std_array)

print("GL_WNLL")
print("NMI Mean : ", nmi_GL_wnll_mean_array)
print("NMI STD : ", nmi_GL_wnll_std_array)
print("ACC Mean : ", acc_GL_wnll_mean_array)
print("ACC STD : ", acc_GL_wnll_std_array)

print("GL_ssl")
print("NMI Mean : ", nmi_GL_ssl_mean_array)
print("NMI STD : ", nmi_GL_ssl_std_array)
print("ACC Mean : ", acc_GL_ssl_mean_array)
print("ACC STD : ", acc_GL_ssl_std_array)


fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(number_of_labeled_nodes_array, nmi_wnll_mean_array, label="$L_{WNLL}$", color='green')
ax.fill_between(number_of_labeled_nodes_array, nmi_wnll_mean_array - nmi_wnll_std_array, torch.minimum(nmi_wnll_mean_array + nmi_wnll_std_array,
                                                                                         torch.tensor(1.0)), color='green', alpha=0.3)
ax.plot(number_of_labeled_nodes_array, nmi_ssl_mean_array, label="$L_{SSL}$", color='blue')
ax.fill_between(number_of_labeled_nodes_array, nmi_ssl_mean_array - nmi_ssl_std_array, torch.minimum(nmi_ssl_mean_array + nmi_ssl_std_array,
                                                                                         torch.tensor(1.0)), color='blue', alpha=0.3)
ax.set_xlabel("$|S|$", fontsize=22)
ax.set_ylabel("NMI", fontsize=22)
ax.grid(True)
plt.legend(loc='best', fontsize=20)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
savefig_path = model_path + "/images/nmi_vs_S_size_L.png"
plt.savefig(savefig_path)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(number_of_labeled_nodes_array, nmi_GL_us_mean_array, label="$L_{US}$", color='red')
ax.fill_between(number_of_labeled_nodes_array, nmi_GL_us_mean_array - nmi_GL_us_std_array, torch.minimum(nmi_GL_us_mean_array + nmi_GL_us_std_array,
                                                                                       torch.tensor(1.0)), color='red', alpha=0.3)
ax.plot(number_of_labeled_nodes_array, nmi_GL_wnll_mean_array, label="$L_{WNLL}$", color='green')
ax.fill_between(number_of_labeled_nodes_array, nmi_GL_wnll_mean_array - nmi_GL_wnll_std_array, torch.minimum(nmi_GL_wnll_mean_array + nmi_GL_wnll_std_array,
                                                                                         torch.tensor(1.0)), color='green', alpha=0.3)
ax.plot(number_of_labeled_nodes_array, nmi_GL_ssl_mean_array, label="$L_{SSL}$", color='blue')
ax.fill_between(number_of_labeled_nodes_array, nmi_GL_ssl_mean_array - nmi_GL_ssl_std_array, torch.minimum(nmi_GL_ssl_mean_array + nmi_GL_ssl_std_array,
                                                                                         torch.tensor(1.0)), color='blue', alpha=0.3)
ax.set_xlabel("$|S|$", fontsize=22)
ax.set_ylabel("NMI", fontsize=22)
ax.grid(True)
plt.legend(loc='best', fontsize=20)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
savefig_path = model_path + "/images/nmi_vs_S_size_GL.png"
plt.savefig(savefig_path)
plt.show()


fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(number_of_labeled_nodes_array, acc_wnll_mean_array, label="$L_{WNLL}$", color='green')
ax.fill_between(number_of_labeled_nodes_array, acc_wnll_mean_array - acc_wnll_std_array, torch.minimum(acc_wnll_mean_array + acc_wnll_std_array,
                                                                                         torch.tensor(1.0)), color='green', alpha=0.3)
ax.plot(number_of_labeled_nodes_array, acc_ssl_mean_array, label="$L_{SSL}$", color='blue')
ax.fill_between(number_of_labeled_nodes_array, acc_ssl_mean_array - acc_ssl_std_array, torch.minimum(acc_ssl_mean_array + acc_ssl_std_array,
                                                                                         torch.tensor(1.0)), color='blue', alpha=0.3)
ax.set_xlabel("$|S|$", fontsize=22)
ax.set_ylabel("ACC", fontsize=22)
ax.grid(True)
plt.legend(loc='best', fontsize=20)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
savefig_path = model_path + "/images/acc_vs_S_size_L.png"
plt.savefig(savefig_path)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(number_of_labeled_nodes_array, acc_GL_us_mean_array, label="$L_{US}$", color='red')
ax.fill_between(number_of_labeled_nodes_array, acc_GL_us_mean_array - acc_GL_us_std_array, torch.minimum(acc_GL_us_mean_array + acc_GL_us_std_array,
                                                                                       torch.tensor(1.0)), color='red', alpha=0.3)
ax.plot(number_of_labeled_nodes_array, acc_GL_wnll_mean_array, label="$L_{WNLL}$", color='green')
ax.fill_between(number_of_labeled_nodes_array, acc_GL_wnll_mean_array - acc_GL_wnll_std_array, torch.minimum(acc_GL_wnll_mean_array + acc_GL_wnll_std_array,
                                                                                         torch.tensor(1.0)), color='green', alpha=0.3)
ax.plot(number_of_labeled_nodes_array, acc_GL_ssl_mean_array, label="$L_{SSL}$", color='blue')
ax.fill_between(number_of_labeled_nodes_array, acc_GL_ssl_mean_array - acc_GL_ssl_std_array, torch.minimum(acc_GL_ssl_mean_array + acc_GL_ssl_std_array,
                                                                                         torch.tensor(1.0)), color='blue', alpha=0.3)
ax.set_xlabel("$|S|$", fontsize=22)
ax.set_ylabel("ACC", fontsize=22)
ax.grid(True)
plt.legend(loc='best', fontsize=20)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
savefig_path = model_path + "/images/acc_vs_S_size_GL.png"
plt.savefig(savefig_path)
plt.show()


###
nmi_us_mean_array = torch.ones(len(number_of_labeled_nodes_array)-1)*nmi_ssl_mean_array[0]
nmi_us_std_array = torch.ones(len(number_of_labeled_nodes_array)-1)*nmi_ssl_std_array[0]
acc_us_mean_array = torch.ones(len(number_of_labeled_nodes_array)-1)*acc_ssl_mean_array[0]
acc_us_std_array = torch.ones(len(number_of_labeled_nodes_array)-1)*acc_ssl_std_array[0]

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(number_of_labeled_nodes_array[1:], nmi_GL_us_mean_array[1:], label="$Dirichlet (L_{US})$", color='red')
ax.fill_between(number_of_labeled_nodes_array[1:], nmi_GL_us_mean_array[1:] - nmi_GL_us_std_array[1:], torch.minimum(nmi_GL_us_mean_array[1:] + nmi_GL_us_std_array[1:],
                                                                                       torch.tensor(1.0)), color='red', alpha=0.3)
ax.plot(number_of_labeled_nodes_array[1:], nmi_GL_ssl_mean_array[1:], label="$Dirichlet (L_{SSL})$", color='blue')
ax.fill_between(number_of_labeled_nodes_array[1:], nmi_GL_ssl_mean_array[1:] - nmi_GL_ssl_std_array[1:], torch.minimum(nmi_GL_ssl_mean_array[1:] + nmi_GL_ssl_std_array[1:],
                                                                                         torch.tensor(1.0)), color='blue', alpha=0.3)
ax.plot(number_of_labeled_nodes_array[1:], nmi_us_mean_array, label="$Spectral (L_{US})$", color='orange')
ax.fill_between(number_of_labeled_nodes_array[1:], nmi_us_mean_array - nmi_us_std_array, torch.minimum(nmi_us_mean_array + nmi_us_std_array,
                                                                                         torch.tensor(1.0)), color='orange', alpha=0.3)
ax.plot(number_of_labeled_nodes_array[1:], nmi_ssl_mean_array[1:], label="$Spectral (L_{SSL})$", color='green')
ax.fill_between(number_of_labeled_nodes_array[1:], nmi_ssl_mean_array[1:] - nmi_ssl_std_array[1:], torch.minimum(nmi_ssl_mean_array[1:] + nmi_ssl_std_array[1:],
                                                                                         torch.tensor(1.0)), color='green', alpha=0.3)

ax.set_xlabel("$|S|$", fontsize=22)
ax.set_ylabel("NMI", fontsize=22)
ax.grid(True)
plt.legend(loc='best', fontsize=16)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
savefig_path = model_path + "/images/g_new_1.png"
plt.savefig(savefig_path)
plt.show()


fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(number_of_labeled_nodes_array[1:], acc_GL_us_mean_array[1:], label="$Dirichlet (L_{US})$", color='red')
ax.fill_between(number_of_labeled_nodes_array[1:], acc_GL_us_mean_array[1:] - acc_GL_us_std_array[1:], torch.minimum(acc_GL_us_mean_array[1:] + acc_GL_us_std_array[1:],
                                                                                       torch.tensor(1.0)), color='red', alpha=0.3)
ax.plot(number_of_labeled_nodes_array[1:], acc_GL_ssl_mean_array[1:], label="$Dirichlet (L_{SSL})$", color='blue')
ax.fill_between(number_of_labeled_nodes_array[1:], acc_GL_ssl_mean_array[1:] - acc_GL_ssl_std_array[1:], torch.minimum(acc_GL_ssl_mean_array[1:] + acc_GL_ssl_std_array[1:],
                                                                                         torch.tensor(1.0)), color='blue', alpha=0.3)
ax.plot(number_of_labeled_nodes_array[1:], acc_us_mean_array, label="$Spectral (L_{US})$", color='orange')
ax.fill_between(number_of_labeled_nodes_array[1:], acc_us_mean_array - acc_us_std_array, torch.minimum(acc_us_mean_array + acc_us_std_array,
                                                                                         torch.tensor(1.0)), color='orange', alpha=0.3)
ax.plot(number_of_labeled_nodes_array[1:], acc_ssl_mean_array[1:], label="$Spectral (L_{SSL})$", color='green')
ax.fill_between(number_of_labeled_nodes_array[1:], acc_ssl_mean_array[1:] - acc_ssl_std_array[1:], torch.minimum(acc_ssl_mean_array[1:] + acc_ssl_std_array[1:],
                                                                                         torch.tensor(1.0)), color='green', alpha=0.3)

ax.set_xlabel("$|S|$", fontsize=22)
ax.set_ylabel("ACC", fontsize=22)
ax.grid(True)
plt.legend(loc='best', fontsize=16)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
savefig_path = model_path + "/images/g_new_2.png"
plt.savefig(savefig_path)
plt.show()
