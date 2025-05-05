from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from vanilla_distilation import distill


class DeepInversionFeatureHook:
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        r_feature = F.mse_loss(module.running_var.data, var) ** 0.5 + F.mse_loss(module.running_mean.data, mean) ** 0.5
        self.r_feature = r_feature

    def close(self):
        self.hook.remove()


def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2


class DeepInversionClass:
    def __init__(
        self,
        teacher=None, 
        student=None,
        batch_size=128,
        n_iterations=200,
        n_classes=100,
        path="./gen_images/", # do we need that???
        final_data_path="./gen_images_final/", # do we need that???
        image_shape=(3, 32, 32),
        r_feature=0.01,
        first_bn_multiplier=10,
        tv_l1=0.0,
        tv_l2=0.0001,
        l2=0.00001,
        main_loss_multiplier=1.0,
        lr=0.03,
        adi_scale=1, # paper suggests 10, but the repo just ignores "adaptive" case..
    ):
        self.path = path
        self.final_data_path = final_data_path

        self.teacher = teacher
        self.student = student

        self.batch_size = batch_size
        self.n_iterations = n_iterations
        self.n_classes = n_classes
        self.image_shape = image_shape

        self.bn_reg_scale = r_feature
        self.first_bn_multiplier = first_bn_multiplier
        self.var_scale_l1 = tv_l1
        self.var_scale_l2 = tv_l2
        self.l2_scale = l2
        self.main_loss_multiplier = main_loss_multiplier
        self.adi_scale = adi_scale

        self.lr = lr

        self.loss_r_feature_layers = []

        for module in self.net_teacher.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.loss_r_feature_layers.append(DeepInversionFeatureHook(module))

        # init dirs to save pictures???

    def get_images(
        self,
    ):
        teacher = self.teacher
        student = self.student

        student.eval()

        kl_loss = nn.KLDivLoss(reduction="batchmean").cuda()

        criterion = nn.CrossEntropyLoss()

        targets = torch.Longtensor([random.randint(0,self.n_classes) for _ in range(self.batch_size)]).cuda()
        inputs = torch.randn((self.batch_size, *self.image_shape), requires_grad=True, device="cuda", dtype=torch.float32)

        optimizer = optim.SGD(student.parameters(), lr=self.lr)


        optimizer = optim.Adam([inputs], lr=self.lr)

        for iteration_loc in range(self.n_iterations):

            if random.random() > 0.5:
                inputs = torch.flip(inputs, dims=(3,))
            
            optimizer.zero_grad()

            teacher.zero_grad()

            outputs = teacher(inputs)

            loss = criterion(outputs, targets)

            loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs)

            rescale = [self.first_bn_multiplier] + [1. for _ in range(len(self.loss_r_feature_layers)-1)]
            loss_r_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(self.loss_r_feature_layers)])


            if self.adi_scale > 0:
                outputs_student = student(inputs).detach()

                T = 3.0

                P = F.softmax(outputs_student / T, dim=1)
                Q = F.softmax(outputs / T, dim=1)

                M = 0.5 * (P + Q)

                P = torch.clamp(P, 0.01, 0.99)
                Q = torch.clamp(Q, 0.01, 0.99)
                M = torch.clamp(M, 0.01, 0.99)

                loss_verifier_cig = 0.5 * kl_loss(torch.log(P), M)  + 0.5 * kl_loss(torch.log(Q), M)
                loss_verifier_cig = 1.0 - torch.clamp(loss_verifier_cig, 0.0, 1.0)

            loss_l2 = ((inputs.view(self.batch_size, -1) ** 2).sum(dim=1)**0.5).mean()

            loss_aux = self.var_scale_l2 * loss_var_l2 + \
                        self.var_scale_l1 * loss_var_l1 + \
                        self.bn_reg_scale * loss_r_feature + \
                        self.l2_scale * loss_l2
            
            if self.adi_scale > 0:
                loss_aux += self.adi_scale * loss_verifier_cig

            loss = self.main_loss_multiplier * loss + loss_aux

            #### logging calback??

            loss.backward()
            optimizer.step()

        optimizer.state = defaultdict(dict)

        return torch.clamp(inputs, 0, 1), targets
    

def distill_deep_inversion(
    teacher=None, 
    student=None,
    config={},
    total_iterations=100000,
    batch_size=256,
    deep_inversion_batch_size=2048,
    deep_inversion_iterations=200,
    n_classes=100,
    image_shape=(3, 32, 32),
    r_feature=0.01,
    first_bn_multiplier=10,
    tv_l1=0.0,
    tv_l2=0.0001,
    l2=0.00001,
    main_loss_multiplier=1.0,
    lr=0.1,
    adi_scale=1, # paper suggests 10, but the repo just ignores "adaptive" case..
):
    
    
    deep_inversion = DeepInversionClass(
        teacher=teacher,
        student=student,
        batch_size=batch_size,
        n_iterations=deep_inversion_iterations,
        n_classes=n_classes,
        image_shape=image_shape,
        r_feature=r_feature,
        first_bn_multiplier=first_bn_multiplier,
        tv_l1=tv_l1,
        tv_l2=tv_l2,
        l2=l2,
        main_loss_multiplier=main_loss_multiplier,
        lr=lr,
        adi_scale=adi_scale,
    )   

    optimizer = torch.optim.Adam(student.parameters(), lr=config.get("lr", 1e-3), betas=config.get("betas", (0.9, 0.95)))

    alpha=config.get("alpha",0.6)
    T = config.get("T", 2.5)

    num_steps = total_iterations // batch_size

    for step in range(num_steps):
        inputs, targets = deep_inversion.get_images()   

        loader = DataLoader(
            TensorDataset(inputs, targets),
            batch_size=batch_size,
            shuffle=True
        )

        distill(
            teacher=teacher,
            student=student,
            optimizer=optimizer,
            train_loader=loader,
            test_loader=None,
            iterations=deep_inversion_batch_size // batch_size,
            test_freq=-1,
            alpha=alpha,
            T=T
        )
    
    return student