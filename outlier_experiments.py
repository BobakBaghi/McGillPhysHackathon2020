import argparse
import os
from datetime import datetime
from multiprocessing import Manager
import numpy as np
from sklearn.ensemble import IsolationForest
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout
from utils import save_roc_pr_curve_data, get_class_name_from_index, get_channels_axis
from models.encoders_decoders import conv_encoder, conv_decoder, RSRAE, RSRAE_plus, RSR1Loss, RSR2Loss, L21Loss
from outlier_datasets import load_cifar10_with_outliers, load_cifar100_with_outliers, \
    load_fashion_mnist_with_outliers, load_mnist_with_outliers, load_svhn_with_outliers
from models import dagmm, dsebm
from transformations import RA, RA_IA, RA_IA_PR
from models.encoders_decoders import CAE_pytorch
from models.drae_loss import DRAELossAutograd

from models.wrn_pytorch import WideResNet
from models.resnet_pytorch import ResNet
from models.densenet_pytorch import DenseNet
import torchvision.transforms as transforms
from keras2pytorch_dataset import trainset_pytorch, testset_pytorch
import torch.utils.data as data
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from misc import AverageMeter
from eval_accuracy import simple_accuracy
import pandas as pd
# from models.pytorch_gan.GAN import discriminator, generator
from models.pytorch_gan.SNGAN import Discriminator, Generator
import torchvision.utils as tu
import itertools

parser = argparse.ArgumentParser(description='Run UOD experiments.')
parser.add_argument('--results_dir', type=str, default='./results', help='Directory to save results.')
parser.add_argument('--transform_backend', type=str, default='wrn', help='Backbone network for SSD.')
parser.add_argument('--operation_type', type=str, default='RA+IA+PR',
                    choices=['RA', 'RA+IA', 'RA+IA+PR'], help='Type of operations.')
parser.add_argument('--score_mode', type=str, default='neg_entropy',
                    choices=['pl_mean', 'max_mean', 'neg_entropy'],
                    help='Score mode for E3Outlier: pl_mean/max_mean/neg_entropy.')
args = parser.parse_args()
RESULTS_DIR = args.results_dir
BACKEND = args.transform_backend
OP_TYPE = args.operation_type
SCORE_MODE = args.score_mode

transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

def train_cae(trainloader, model, criterion, optimizer, epochs):
    """Valid for both CAE+MSELoss and CAE+DRAELoss"""
    model.train()
    losses = AverageMeter()
    for epoch in range(epochs):
        for batch_idx, (inputs, _) in enumerate(trainloader):
            inputs = torch.autograd.Variable(inputs.cuda())

            outputs = model(inputs)

            loss = criterion(inputs, outputs)

            losses.update(loss.item(), inputs.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx+1) % 10 == 0:
                print('Epoch: [{} | {}], batch: {}, loss: {}'.format(epoch + 1, epochs, batch_idx + 1, losses.avg))

def test_cae_pytorch(testloader, model):
    """Yield reconstruction loss as well as representations"""
    model.eval()
    losses = []
    reps = []
    for batch_idx, (inputs, _) in enumerate(testloader):
        inputs = torch.autograd.Variable(inputs.cuda())
        rep = model.encode(inputs)
        outputs = model.decode(rep)
        loss = outputs.sub(inputs).pow(2).view(outputs.size(0), -1)
        loss = loss.sum(dim=1, keepdim=False)
        losses.append(loss.data.cpu())
        reps.append(rep.data.cpu())
    losses = torch.cat(losses, dim=0)
    reps = torch.cat(reps, dim=0)
    return losses.numpy(), reps.numpy()

def train_pytorch(trainloader, model, criterion, optimizer, epochs):
    # train the model
    model.train()
    top1 = AverageMeter()
    losses = AverageMeter()
    for epoch in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = torch.autograd.Variable(inputs.cuda()), torch.autograd.Variable(targets.cuda())

            outputs, _ = model(inputs)

            loss = criterion(outputs, targets)

            prec1 = simple_accuracy(outputs.data.cpu(), targets.data.cpu())

            top1.update(prec1, inputs.size(0))
            losses.update(loss.data.cpu(), inputs.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print('Epoch: [{} | {}], batch: {}, loss: {}, Accuracy: {}'.format(epoch + 1, epochs, batch_idx + 1, losses.avg, top1.avg))

def test_pytorch(testloader, model):
        model.eval()
        res = torch.Tensor()
        for batch_idx, (inputs) in enumerate(testloader):
            inputs = torch.autograd.Variable(inputs.cuda())
            outputs, _ = model(inputs)
            res = torch.cat((res, outputs.data.cpu()), dim=0)
        return res


def get_features_pytorch(testloader, model):
    model.eval()
    features = []
    for inputs in testloader:
        inputs = torch.autograd.Variable(inputs.cuda())
        _, rep = model(inputs)
        features.append(rep.data.cpu())
    features = torch.cat(features, dim=0)
    return features


def train_rsrae_plus(trainloader, model, criterion, optimizer, epochs):
    """Valid for both RSRAE+"""
    model.train()
    losses_ae = AverageMeter()
    losses_rsr = AverageMeter()
    losses = AverageMeter()
    L_ae = criterion[0]
    L_rsr1 = criterion[1]
    L_rsr2 = criterion[2]
    lambda1 = 0.1
    lambda2 = 0.1
    for epoch in range(epochs):
        for batch_idx, (inputs, _) in enumerate(trainloader):
            inputs = inputs.cuda()
            rec, h, hr, AA = model(inputs)

            loss_ae = L_ae(inputs, rec)
            loss_rsr1 = L_rsr1(hr, h)
            loss_rsr2 = L_rsr2(AA)
            loss_rsr = lambda1 * loss_rsr1 + lambda2 * loss_rsr2
            loss = loss_ae + loss_rsr
            try:
                losses_ae.update(loss_ae.item(), inputs.size(0))
                losses_rsr.update(loss_rsr.item(), inputs.size(0))
                losses.update(loss.item(), inputs.size(0))
            except:
                pass

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx+1) % 10 == 0:
                print('Epoch: [{} | {}], batch: {}, loss_ae: {}, loss_rsr: {}, loss: {}'.format(epoch + 1, epochs, batch_idx + 1, losses_ae.avg, losses_rsr.avg, losses.avg))


def train_rsrae(trainloader, model, criterion, optimizer, epochs):
    """Valid for RSRAE"""
    model.train()
    losses_ae = AverageMeter()
    losses_rsr1 = AverageMeter()
    losses_rsr2 = AverageMeter()
    L_ae = criterion[0]
    L_rsr1 = criterion[1]
    L_rsr2 = criterion[2]
    for epoch in range(epochs):
        for batch_idx, (inputs, _) in enumerate(trainloader):
            inputs = inputs.cuda()
            rec, hh, hr, AA = model(inputs)

            # update by AE loss
            loss_ae = L_ae(inputs, rec)
            try:
                losses_ae.update(loss_ae.item(), inputs.size(0))
            except:
                pass
            optimizer.zero_grad()
            loss_ae.backward()
            optimizer.step()

            # update by RSR loss 1
            loss_rsr1 = L_rsr1(hr, hh)
            try:
                losses_rsr1.update(loss_rsr1.item(), inputs.size(0))
            except:
                pass
            optimizer.zero_grad()
            loss_rsr1.backward()
            optimizer.step()

            # update by RSR loss 2
            loss_rsr2 = L_rsr2(AA)
            try:
                losses_rsr2.update(loss_rsr2.item(), inputs.size(0))
            except:
                pass
            optimizer.zero_grad()
            loss_rsr2.backward()
            optimizer.step()

            if (batch_idx+1) % 10 == 0:
                print('Epoch: [{} | {}], batch: {}, loss_ae: {}, loss_rsr1: {}, loss_rsr2: {}'.format(epoch + 1, epochs, batch_idx + 1, losses_ae.avg, losses_rsr1.avg, losses_rsr2.avg))


def prior_generator(batch_size, zdim, rand_type='rand'):
    """Generate prior distribution for G."""
    if rand_type == 'rand':
        return torch.rand(batch_size, zdim).mul(2).sub(1)  # [-1,1]
    elif rand_type == 'randn':
        return torch.randn(batch_size, zdim)
    else:
        raise KeyError('Unknown rand type: {}'.format(rand_type))


def softmax(input_tensor):
    act = nn.Softmax(dim=1)
    return act(input_tensor).numpy()

def neg_entropy(score):
    if len(score.shape) != 1:
        score = np.squeeze(score)
    return score@np.log2(score)

def dist_calc(feats1, feats2):
    nb_data1 = feats1.shape[0]
    nb_data2 = feats2.shape[0]
    omega = np.dot(np.sum(feats1 ** 2, axis=1)[:, np.newaxis], np.ones(shape=(1, nb_data2)))
    omega += np.dot(np.sum(feats2 ** 2, axis=1)[:, np.newaxis], np.ones(shape=(1, nb_data1))).T
    omega -= 2 * np.dot(feats1, feats2.T)
    return omega


def prox_l21(S, lmbda):
    """L21 proximal operator."""
    Snorm = np.sqrt((S ** 2).sum(axis=tuple(range(1, S.ndim)), keepdims=False))
    multiplier = 1 - 1 / np.minimum(Snorm/lmbda, 1)
    out = S * multiplier.reshape((S.shape[0],)+(1,)*(S.ndim-1))
    return out


def train_robust_cae(x_train, model, criterion, optimizer, lmbda, inner_epochs, outer_epochs, reinit=True):
    batch_size = 128
    S = np.zeros_like(x_train)  # reside on numpy as x_train

    def get_reconstruction(loader):
        model.eval()
        rc = []
        for batch, _ in loader:
            with torch.no_grad():
                rc.append(model(batch.cuda()).cpu().numpy())
        out = np.concatenate(rc, axis=0)
        # NOTE: transform_train swaps the channel axis, swap back to yield the same shape
        out = out.transpose((0, 2, 3, 1))
        return out

    for oe in range(outer_epochs):
        # update AE
        if reinit:
            # Since our CAE_pytorch does not implement reset_parameters, regenerate a new model if reinit.
            del model
            model = CAE_pytorch(in_channels=x_train.shape[get_channels_axis()]).cuda()
        model.train()
        trainset = trainset_pytorch(x_train-S, train_labels=np.ones((x_train.shape[0], )), transform=transform_train)
        trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        for ie in range(inner_epochs):
            for batch_idx, (inputs, _) in enumerate(trainloader):
                inputs = inputs.cuda()
                outputs = model(inputs)
                loss = criterion(inputs, outputs)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (batch_idx + 1) % 10 == 0:
                    print('Epoch: [{} | {} ({} | {})], batch: {}, loss: {}'.format(
                        ie+1, inner_epochs, oe+1, outer_epochs, batch_idx+1, loss.item())
                    )
        # update S via l21 proximal operator
        testloader = data.DataLoader(trainset, batch_size=1024, shuffle=False)
        recon = get_reconstruction(testloader)
        S = prox_l21(x_train - recon, lmbda)

    # get final reconstruction
    finalset = trainset_pytorch(x_train - S, train_labels=np.ones((x_train.shape[0],)), transform=transform_train)
    finalloader = data.DataLoader(finalset, batch_size=1024, shuffle=False)
    reconstruction = get_reconstruction(finalloader)
    losses = ((x_train-S-reconstruction) ** 2).sum(axis=(1, 2, 3), keepdims=False)
    return losses


# ######################### functions to perform different deep outlier detection methods ############################


def _RDAE_experiment(x_train, y_train, dataset_name, single_class_ind, gpu_q, p):
    gpu_to_use = gpu_q.get()
    cudnn.benchmark = True

    n_channels = x_train.shape[get_channels_axis()]
    model = CAE_pytorch(in_channels=n_channels)
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), eps=1e-7, weight_decay=0.0005)
    criterion = nn.MSELoss()
    epochs = 20
    inner_epochs = 1
    lmbda = 0.00065

    # train RDAE
    losses = train_robust_cae(x_train, model, criterion, optimizer, lmbda, inner_epochs, epochs//inner_epochs, False)
    losses = losses - losses.min()
    losses = losses / (1e-8 + losses.max())
    scores = 1 - losses

    res_file_name = '{}_rdae-{}_{}_{}.npz'.format(dataset_name, p,
                                                  get_class_name_from_index(single_class_ind, dataset_name),
                                                  datetime.now().strftime('%Y-%m-%d-%H%M'))
    res_file_path = os.path.join(RESULTS_DIR, dataset_name, res_file_name)
    os.makedirs(os.path.join(RESULTS_DIR, dataset_name), exist_ok=True)
    save_roc_pr_curve_data(scores, y_train, res_file_path)

    gpu_q.put(gpu_to_use)

def _DRAE_experiment(x_train, y_train, dataset_name, single_class_ind, gpu_q, p):
    gpu_to_use = gpu_q.get()

    n_channels = x_train.shape[get_channels_axis()]
    model = CAE_pytorch(in_channels=n_channels)
    batch_size = 128

    model = model.cuda()
    trainset = trainset_pytorch(train_data=x_train, train_labels=y_train, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    cudnn.benchmark = True
    criterion = DRAELossAutograd(lamb=0.1)
    optimizer = optim.Adam(model.parameters(), eps=1e-7, weight_decay=0.0005)
    epochs = 250

    # #########################Training########################
    train_cae(trainloader, model, criterion, optimizer, epochs)

    # #######################Testin############################
    testloader = data.DataLoader(trainset, batch_size=1024, shuffle=False)
    losses, reps = test_cae_pytorch(testloader, model)
    losses = losses - losses.min()
    losses = losses / (1e-8+losses.max())
    scores = 1 - losses

    res_file_name = '{}_drae-{}_{}_{}.npz'.format(dataset_name, p,
                                                  get_class_name_from_index(single_class_ind, dataset_name),
                                                  datetime.now().strftime('%Y-%m-%d-%H%M'))
    res_file_path = os.path.join(RESULTS_DIR, dataset_name, res_file_name)
    os.makedirs(os.path.join(RESULTS_DIR, dataset_name), exist_ok=True)
    save_roc_pr_curve_data(scores, y_train, res_file_path)

    gpu_q.put(gpu_to_use)

def _E3Outlier_experiment(x_train, y_train, dataset_name, single_class_ind, gpu_q, p):
    """Surrogate Supervision Discriminative Network training."""
    gpu_to_use = gpu_q.get()

    n_channels = x_train.shape[get_channels_axis()]

    if OP_TYPE == 'RA':
        transformer = RA(8, 8)
    elif OP_TYPE == 'RA+IA':
        transformer = RA_IA(8, 8, 12)
    elif OP_TYPE == 'RA+IA+PR':
        transformer = RA_IA_PR(8, 8, 12, 23, 2)
    else:
        raise NotImplementedError
    print(transformer.n_transforms)

    if BACKEND == 'wrn':
        n, k = (10, 4)
        model = WideResNet(num_classes=transformer.n_transforms, depth=n, widen_factor=k, in_channel=n_channels)
    elif BACKEND == 'resnet20':
        n = 20
        model = ResNet(num_classes=transformer.n_transforms, depth=n, in_channels=n_channels)
    elif BACKEND == 'resnet50':
        n = 50
        model = ResNet(num_classes=transformer.n_transforms, depth=n, in_channels=n_channels)
    elif BACKEND == 'densenet22':
        n = 22
        model = DenseNet(num_classes=transformer.n_transforms, depth=n, in_channels=n_channels)
    elif BACKEND == 'densenet40':
        n = 40
        model = DenseNet(num_classes=transformer.n_transforms, depth=n, in_channels=n_channels)
    else:
        raise NotImplementedError('Unimplemented backend: {}'.format(BACKEND))
    print('Using backend: {} ({})'.format(type(model).__name__, BACKEND))

    x_train_task = x_train
    transformations_inds = np.tile(np.arange(transformer.n_transforms), len(x_train_task))
    x_train_task_transformed = transformer.transform_batch(np.repeat(x_train_task, transformer.n_transforms, axis=0), transformations_inds)

    # parameters for training
    trainset = trainset_pytorch(train_data=x_train_task_transformed, train_labels=transformations_inds, transform=transform_train)
    batch_size = 128
    trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()
    model = torch.nn.DataParallel(model).cuda()
    if dataset_name in ['mnist', 'fashion-mnist']:
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    else:
        optimizer = optim.Adam(model.parameters(), eps=1e-7, weight_decay=0.0005)
    epochs = int(np.ceil(250 / transformer.n_transforms))
    train_pytorch(trainloader, model, criterion, optimizer, epochs)

    # SSD-IF
    test_set = testset_pytorch(test_data=x_train_task, transform=transform_test)
    x_train_task_rep = get_features_pytorch(
        testloader=data.DataLoader(test_set, batch_size=batch_size, shuffle=False), model=model
    ).numpy()
    clf = IsolationForest(contamination=p, n_jobs=4).fit(x_train_task_rep)
    if_scores = clf.decision_function(x_train_task_rep)
    res_file_name = '{}_ssd-iforest-{}_{}_{}.npz'.format(dataset_name, p,
                                                       get_class_name_from_index(single_class_ind, dataset_name),
                                                       datetime.now().strftime('%Y-%m-%d-%H%M'))
    res_file_path = os.path.join(RESULTS_DIR, dataset_name, res_file_name)
    os.makedirs(os.path.join(RESULTS_DIR, dataset_name), exist_ok=True)
    save_roc_pr_curve_data(if_scores, y_train, res_file_path)

    # E3Outlier
    if SCORE_MODE == 'pl_mean':
        preds = np.zeros((len(x_train_task), transformer.n_transforms))
        original_preds = np.zeros((transformer.n_transforms, len(x_train_task), transformer.n_transforms))
        for t in range(transformer.n_transforms):
            idx = np.squeeze(np.array([range(x_train_task.shape[0])]) * transformer.n_transforms + t)
            test_set = testset_pytorch(test_data=x_train_task_transformed[idx, :],
                                       transform=transform_test)
            original_preds[t, :, :] = softmax(test_pytorch(testloader=data.DataLoader(test_set, batch_size=batch_size, shuffle=False), model=model))
            preds[:, t] = original_preds[t, :, :][:, t]
        scores = preds.mean(axis=-1)
    elif SCORE_MODE == 'max_mean':
        preds = np.zeros((len(x_train_task), transformer.n_transforms))
        original_preds = np.zeros((transformer.n_transforms, len(x_train_task), transformer.n_transforms))
        for t in range(transformer.n_transforms):
            idx = np.squeeze(np.array([range(x_train_task.shape[0])]) * transformer.n_transforms + t)
            test_set = testset_pytorch(test_data=x_train_task_transformed[idx, :],
                                       transform=transform_test)
            original_preds[t, :, :] = softmax(test_pytorch(testloader=data.DataLoader(test_set, batch_size=batch_size, shuffle=False), model=model))
            preds[:, t] = np.max(original_preds[t, :, :], axis=1)
        scores = preds.mean(axis=-1)
    elif SCORE_MODE == 'neg_entropy':
        preds = np.zeros((len(x_train_task), transformer.n_transforms))
        original_preds = np.zeros((transformer.n_transforms, len(x_train_task), transformer.n_transforms))
        for t in range(transformer.n_transforms):
            idx = np.squeeze(np.array([range(x_train_task.shape[0])]) * transformer.n_transforms + t)
            test_set = testset_pytorch(test_data=x_train_task_transformed[idx, :],
                                       transform=transform_test)
            original_preds[t, :, :] = softmax(test_pytorch(testloader=data.DataLoader(test_set, batch_size=batch_size, shuffle=False), model=model))
            for s in range(len(x_train_task)):
                preds[s, t] = neg_entropy(original_preds[t, s, :])
        scores = preds.mean(axis=-1)
    else:
        raise NotImplementedError

    res_file_name = '{}_e3outlier-{}_{}_{}.npz'.format(dataset_name, p,
                                                       get_class_name_from_index(single_class_ind, dataset_name),
                                                       datetime.now().strftime('%Y-%m-%d-%H%M'))
    res_file_path = os.path.join(RESULTS_DIR, dataset_name, res_file_name)
    os.makedirs(os.path.join(RESULTS_DIR, dataset_name), exist_ok=True)
    save_roc_pr_curve_data(scores, y_train, res_file_path)

    gpu_q.put(gpu_to_use)


def _cae_pytorch_experiment(x_train, y_train, dataset_name, single_class_ind, gpu_q, p):
    gpu_to_use = gpu_q.get()

    n_channels = x_train.shape[get_channels_axis()]
    model = CAE_pytorch(in_channels=n_channels)
    batch_size = 128

    model = model.cuda()
    trainset = trainset_pytorch(train_data=x_train, train_labels=y_train, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    cudnn.benchmark = True
    criterion = nn.MSELoss()
    # use adam always
    optimizer = optim.Adam(model.parameters(), eps=1e-7, weight_decay=0.0005)
    epochs = 250

    # #########################Training########################
    train_cae(trainloader, model, criterion, optimizer, epochs)

    # #######################Testin############################
    testloader = data.DataLoader(trainset, batch_size=1024, shuffle=False)
    losses, reps = test_cae_pytorch(testloader, model)
    losses = losses - losses.min()
    losses = losses / (1e-8+losses.max())
    scores = 1 - losses

    res_file_name = '{}_cae-{}_{}_{}.npz'.format(dataset_name, p,
                                                 get_class_name_from_index(single_class_ind, dataset_name),
                                                 datetime.now().strftime('%Y-%m-%d-%H%M'))
    res_file_path = os.path.join(RESULTS_DIR, dataset_name, res_file_name)
    os.makedirs(os.path.join(RESULTS_DIR, dataset_name), exist_ok=True)
    save_roc_pr_curve_data(scores, y_train, res_file_path)

    # Use reps to train iforest
    clf = IsolationForest(contamination=p, n_jobs=4).fit(reps)
    scores_iforest = clf.decision_function(reps)
    iforest_file_name = '{}_cae-iforest-{}_{}_{}.npz'.format(dataset_name, p,
                                                             get_class_name_from_index(single_class_ind, dataset_name),
                                                             datetime.now().strftime('%Y-%m-%d-%H%M'))
    iforest_file_path = os.path.join(RESULTS_DIR, dataset_name, iforest_file_name)
    save_roc_pr_curve_data(scores_iforest, y_train, iforest_file_path)

    gpu_q.put(gpu_to_use)


def _dagmm_experiment(x_train, y_train, dataset_name, single_class_ind, gpu_q, p):
    gpu_to_use = gpu_q.get()
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use

    n_channels = x_train.shape[get_channels_axis()]
    input_side = x_train.shape[2]  # image side will always be at shape[2]
    enc = conv_encoder(input_side, n_channels, representation_dim=5,
                       representation_activation='linear')
    dec = conv_decoder(input_side, n_channels=n_channels, representation_dim=enc.output_shape[-1])
    n_components = 3
    estimation = Sequential([Dense(64, activation='tanh', input_dim=enc.output_shape[-1] + 2), Dropout(0.5),
                             Dense(10, activation='tanh'), Dropout(0.5),
                             Dense(n_components, activation='softmax')]
                            )

    batch_size = 1024
    epochs = 200
    lambda_diag = 0.005
    lambda_energy = 0.1
    dagmm_mdl = dagmm.create_dagmm_model(enc, dec, estimation, lambda_diag)
    optimizer = keras.optimizers.Adam(lr=1e-4)  # default config
    dagmm_mdl.compile(optimizer, ['mse', lambda y_true, y_pred: lambda_energy*y_pred])

    x_train_task = x_train
    x_test_task = x_train  # This is just for visual monitoring
    dagmm_mdl.fit(x=x_train_task, y=[x_train_task, np.zeros((len(x_train_task), 1))],  # second y is dummy
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test_task, [x_test_task, np.zeros((len(x_test_task), 1))]),
                  # verbose=0
                  )

    energy_mdl = Model(dagmm_mdl.input, dagmm_mdl.output[-1])

    scores = -energy_mdl.predict(x_train, batch_size)
    scores = scores.flatten()
    if not np.all(np.isfinite(scores)):
        min_finite = np.min(scores[np.isfinite(scores)])
        scores[~np.isfinite(scores)] = min_finite - 1
    labels = y_train.flatten()
    res_file_name = '{}_dagmm-{}_{}_{}.npz'.format(dataset_name, p,
                                                   get_class_name_from_index(single_class_ind, dataset_name),
                                                   datetime.now().strftime('%Y-%m-%d-%H%M'))
    res_file_path = os.path.join(RESULTS_DIR, dataset_name, res_file_name)
    save_roc_pr_curve_data(scores, labels, res_file_path)

    gpu_q.put(gpu_to_use)

# ######################### newly added methods ############################

def _rsrae_plus_experiment(x_train, y_train, dataset_name, single_class_ind, gpu_q, p):
    gpu_to_use = gpu_q.get()

    n_channels = x_train.shape[get_channels_axis()]
    model = RSRAE_plus(in_channels=n_channels)
    batch_size = 128

    model = model.cuda()
    trainset = trainset_pytorch(train_data=x_train, train_labels=y_train, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    cudnn.benchmark = True
    ae_criterion = L21Loss().cuda()
    rsr1_criterion = RSR1Loss().cuda()
    rsr2_criteriom = RSR2Loss().cuda()
    # use adam always
    optimizer = optim.Adam(model.parameters(), lr=0.00025, eps=1e-7, weight_decay=0.0005)
    epochs = 250

    # #########################Training########################
    train_rsrae_plus(trainloader, model, [ae_criterion, rsr1_criterion, rsr2_criteriom], optimizer, epochs)

    # #######################Testin############################
    testloader = data.DataLoader(trainset, batch_size=1024, shuffle=False)
    losses, reps = test_cae_pytorch(testloader, model)
    losses = losses - losses.min()
    losses = losses / (1e-8+losses.max())
    scores = 1 - losses

    res_file_name = '{}_rsrae_plus-outlier-{}_final_{}_{}.npz'.format(dataset_name, p,
                                                               get_class_name_from_index(single_class_ind, dataset_name),
                                                               datetime.now().strftime('%Y-%m-%d-%H%M'))
    res_file_path = os.path.join(RESULTS_DIR, dataset_name, res_file_name)
    os.makedirs(os.path.join(RESULTS_DIR, dataset_name), exist_ok=True)
    save_roc_pr_curve_data(scores, y_train, res_file_path)

    gpu_q.put(gpu_to_use)


def _rsrae_experiment(x_train, y_train, dataset_name, single_class_ind, gpu_q, p):
    gpu_to_use = gpu_q.get()

    n_channels = x_train.shape[get_channels_axis()]
    model = RSRAE(in_channels=n_channels)
    batch_size = 128

    model = model.cuda()
    trainset = trainset_pytorch(train_data=x_train, train_labels=y_train, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    cudnn.benchmark = True
    ae_criterion = L21Loss().cuda()
    rsr1_criterion = RSR1Loss().cuda()
    rsr2_criteriom = RSR2Loss().cuda()
    # use adam always
    optimizer = optim.Adam(model.parameters(), lr=0.00025, eps=1e-7, weight_decay=0.0005)
    epochs = 250

    # #########################Training########################
    train_rsrae(trainloader, model, [ae_criterion, rsr1_criterion, rsr2_criteriom], optimizer, epochs)

    # #######################Testin############################
    testloader = data.DataLoader(trainset, batch_size=1024, shuffle=False)
    losses, reps = test_cae_pytorch(testloader, model)
    losses = losses - losses.min()
    losses = losses / (1e-8+losses.max())
    scores = 1 - losses

    res_file_name = '{}_rsrae-outlier-{}_final_{}_{}.npz'.format(dataset_name, p,
                                                               get_class_name_from_index(single_class_ind, dataset_name),
                                                               datetime.now().strftime('%Y-%m-%d-%H%M'))
    res_file_path = os.path.join(RESULTS_DIR, dataset_name, res_file_name)
    os.makedirs(os.path.join(RESULTS_DIR, dataset_name), exist_ok=True)
    save_roc_pr_curve_data(scores, y_train, res_file_path)

    gpu_q.put(gpu_to_use)


def _mo_gaal_experiment(x_train, y_train, dataset_name, single_class_ind, gpu_q, p):
    gpu_to_use = gpu_q.get()

    try:
        # from tensorboardX import SummaryWriter
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(RESULTS_DIR, dataset_name, 'tensorboard-{}'.format(
            get_class_name_from_index(single_class_ind, dataset_name),
        )))
    except ImportError:
        writer = None
    n_channels = x_train.shape[get_channels_axis()]
    input_size = x_train.shape[2]
    trainset = trainset_pytorch(train_data=x_train, train_labels=y_train, transform=transform_train)

    # setup GAN
    no_G = 5  # Number of generator
    zdim = 128
    batch_size = 64
    joint_epochs = 100  # Train D and Gs jointly.
    D_epochs = joint_epochs * 3  # Train D only. Refer to the MO-GAAL paper.
    sample_z_ = prior_generator(batch_size, zdim).cuda()  # for visualization
    y_real_ = torch.ones(batch_size, 1, requires_grad=False).cuda()
    y_fake_ = torch.zeros(batch_size, 1, requires_grad=False).cuda()
    D = Discriminator(n_channels).cuda()
    Gs = [Generator(zdim, n_channels).cuda() for _ in range(no_G)]
    # D = discriminator(input_dim=n_channels, output_dim=1, input_size=input_size).cuda()
    # Gs = [generator(input_dim=zdim, output_dim=n_channels, input_size=input_size).cuda() for _ in range(no_G)]
    D_optimizer = optim.Adam(filter(lambda p: p.requires_grad, D.parameters()), lr=0.0002, betas=(0., 0.9))
    Gs_optimizer = optim.Adam(itertools.chain(*[G.parameters() for G in Gs]), lr=0.0002, betas=(0., 0.9))
    D_scheduler = optim.lr_scheduler.ExponentialLR(D_optimizer, gamma=0.99)
    Gs_scheduler = optim.lr_scheduler.ExponentialLR(Gs_optimizer, gamma=0.99)
    print(f'Using {no_G} generators, G iter: {joint_epochs}, D iter: {D_epochs}.')

    def _generate(z):
        """Generate images using Gs.
        Note that more outliers need to be generated from ``less concentrated subset'', i.e.
        low score ones.
        """
        k = len(Gs)
        total_block = ((1 + k) * k) // 2
        sz = z.size(0)
        noise_start = [int((((k + (k - i + 1)) * i) / 2) * sz / total_block) for i in range(k)]
        noise_end = [int((((k + (k - i)) * (i + 1)) / 2) * sz / total_block) for i in range(k)]
        generated = [G(z[start:end]) for G, start, end in zip(Gs, noise_start, noise_end)]
        generated = torch.cat(generated, dim=0)
        return generated

    def _eval_D():
        """Do evaluation using D."""
        testloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            values = torch.cat([D(x.cuda()).cpu() for x, _ in testloader], dim=0)
        return values.view(-1)

    def _collect_D_stats():
        """Get the quantile of p-value from output values of D."""
        D.eval()
        values = _eval_D()
        pvalue = pd.DataFrame(values.numpy())
        stats = [float(pvalue.quantile(i/no_G)) for i in range(no_G)]
        D.train()
        return stats

    criterion = nn.BCEWithLogitsLoss().cuda()
    D.train()
    global_steps = 0
    for e in range(D_epochs):
        for G in Gs:
            G.train()
        trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
        for it, (x_, _) in enumerate(trainloader):

            # update D
            x_ = x_.cuda()
            z_ = prior_generator(batch_size, zdim).cuda()
            D_optimizer.zero_grad()
            D_real = D(x_)
            D_real_loss = criterion(D_real, y_real_)
            G_ = _generate(z_)
            D_fake = D(G_)
            D_fake_loss = criterion(D_fake, y_fake_)
            # -y_i*log(D(x_i)) + -(1-y_i)*log(1-D(G(z_i))).
            D_loss = D_real_loss + D_fake_loss
            if writer is not None:
                writer.add_scalar('D_loss', D_loss.item(), global_steps)
            D_loss.backward()
            D_optimizer.step()

            # Get global statistics
            # TODO(leoyolo): is this necessary? Should we update for each epoch?
            stats = _collect_D_stats()

            # update Gs
            # resample z_ as in original MO-GAAL
            z_ = prior_generator(batch_size, zdim).cuda()
            g_loss_avg = 0.
            if e < joint_epochs:
                Gs_optimizer.zero_grad()
                for i, G in enumerate(Gs):
                    G_ = G(z_)
                    target = torch.empty_like(y_fake_).fill_(stats[i])
                    D_fake = D(G_)
                    # -t_i*log(D(G(z_i))) - (1-t_i)log(1-D(G(z_i))).
                    G_loss = criterion(D_fake, target)
                    if writer is not None:
                        writer.add_scalar(f'G_loss_{i}', G_loss.item(), global_steps)
                    g_loss_avg += G_loss.item()
                    G_loss.backward()
                Gs_optimizer.step()

                if (global_steps + 1) % 100 == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss_avg: %.8f" %
                          ((e + 1), (it + 1), len(trainset) // batch_size, D_loss.item(), g_loss_avg/no_G))

                # log image (while Gs are still updating)
                if (global_steps + 1) % 500 == 0:
                    for G in Gs:
                        G.eval()
                    generated = _generate(sample_z_)
                    os.makedirs(os.path.join(RESULTS_DIR, dataset_name, 'generated'), exist_ok=True)
                    img_path = os.path.join(RESULTS_DIR, dataset_name, 'generated',
                                            '{}-{}.png'.format(
                                                get_class_name_from_index(single_class_ind, dataset_name),
                                                global_steps + 1))
                    tu.save_image(generated, img_path, nrow=int(np.sqrt(generated.size(0))), normalize=True,
                                  range=(-1, 1))
                    if writer is not None:
                        grid = tu.make_grid(generated, nrow=int(np.sqrt(generated.size(0))),
                                            normalize=True, range=(-1, 1))
                        writer.add_image(
                            '{}-{}'.format(dataset_name, get_class_name_from_index(single_class_ind, dataset_name)),
                            grid,
                            global_steps
                        )
                    for G in Gs:
                        G.train()
            else:
                for G in Gs:
                    G.eval()
            global_steps += 1
        # update evary epoch
        D_scheduler.step()
        Gs_scheduler.step()

    D.eval()
    scores = _eval_D().sigmoid().numpy()
    res_file_name = '{}_mo-gaal-{}_final_{}_{}.npz'.format(
        dataset_name, p, get_class_name_from_index(single_class_ind, dataset_name),
        datetime.now().strftime('%Y-%m-%d-%H%M')
    )
    res_file_path = os.path.join(RESULTS_DIR, dataset_name, res_file_name)
    save_roc_pr_curve_data(scores, y_train, res_file_path)

    if writer is not None:
        writer.close()
    gpu_q.put(gpu_to_use)


def _dsebm_experiment(x_train, y_train, dataset_name, single_class_ind, gpu_q, p):
    gpu_to_use = gpu_q.get()
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use

    n_channels = x_train.shape[get_channels_axis()]
    input_side = x_train.shape[2]  # image side will always be at shape[2]
    encoder_mdl = conv_encoder(input_side, n_channels, representation_activation='relu')
    energy_mdl = dsebm.create_energy_model(encoder_mdl)
    reconstruction_mdl = dsebm.create_reconstruction_model(energy_mdl)

    # optimization parameters
    batch_size = 128
    epochs = 250
    if dataset_name in ['mnist', 'fashion-mnist']:
        optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
    else:
        optimizer = keras.optimizers.Adam()  # default config
    reconstruction_mdl.compile(optimizer, 'mse')
    x_train_task = x_train
    x_test_task = x_train  # This is just for visual monitoring
    reconstruction_mdl.fit(x=x_train_task, y=x_train_task,
                           batch_size=batch_size,
                           epochs=epochs,
                           validation_data=(x_test_task, x_test_task))

    scores = -energy_mdl.predict(x_train, batch_size)
    labels = y_train
    res_file_name = '{}_dsebm-outlier-{}_final_{}_{}.npz'.format(dataset_name, p,
                                                get_class_name_from_index(single_class_ind, dataset_name),
                                                datetime.now().strftime('%Y-%m-%d-%H%M'))
    res_file_path = os.path.join(RESULTS_DIR, dataset_name, res_file_name)
    save_roc_pr_curve_data(scores, labels, res_file_path)

    gpu_q.put(gpu_to_use)

# ############################### Interface to run all experiments ###################################################

def run_experiments(load_dataset_fn, dataset_name, q, n_classes, abnormal_fraction, run_idx):
    max_sample_num = 12000
    os.makedirs(os.path.join(RESULTS_DIR, dataset_name), exist_ok=True)

    for c in range(n_classes):
        np.random.seed(run_idx)
        x_train, y_train = load_dataset_fn(c, abnormal_fraction)

        # random sampling if the number of data is too large
        if x_train.shape[0] > max_sample_num:
            selected = np.random.choice(x_train.shape[0], max_sample_num, replace=False)
            x_train = x_train[selected, :]
            y_train = y_train[selected]

        # SSD-IF / E3Outlier
        _E3Outlier_experiment(x_train, y_train, dataset_name, c, q, abnormal_fraction)

        # DRAE
        _DRAE_experiment(x_train, y_train, dataset_name, c, q, abnormal_fraction)

        # RDAE
        _RDAE_experiment(x_train, y_train, dataset_name, c, q, abnormal_fraction)

        # CAE / CAE-IF
        _cae_pytorch_experiment(x_train, y_train, dataset_name, c, q, abnormal_fraction)

        # DAGMM
        _dagmm_experiment(x_train, y_train, dataset_name, c, q, abnormal_fraction)


# Collections of all valid algorithms.
__ALGO_NAMES__ = ['{}-{}'.format(algo, p)
                  for algo in ('cae', 'cae-iforest', 'drae', 'rdae', 'dagmm', 'ssd-iforest', 'e3outlier')
                  for p in (0.05, 0.1, 0.15, 0.2, 0.25)]


if __name__ == '__main__':

    n_run = 5
    N_GPUS = 1  # deprecated, use one gpu only
    man = Manager()
    q = man.Queue(N_GPUS)
    for g in range(N_GPUS):
        q.put(str(g))

    experiments_list = [
        (load_mnist_with_outliers, 'mnist', 10),
        (load_fashion_mnist_with_outliers, 'fashion-mnist', 10),
        (load_cifar10_with_outliers, 'cifar10', 10),
        (load_cifar100_with_outliers, 'cifar100', 20),
        (load_svhn_with_outliers, 'svhn', 10),
    ]

    p_list = [0.05, 0.1, 0.15, 0.2, 0.25]
    for i in range(n_run):
        for data_load_fn, dataset_name, n_classes in experiments_list:
            for p in p_list:
                run_experiments(data_load_fn, dataset_name, q, n_classes, p, i)