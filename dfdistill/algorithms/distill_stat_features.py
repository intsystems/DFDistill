import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal
import os
import time
from torch.utils.data import DataLoader
from tqdm import tqdm



def collect_top_layer_stats(teacher_model: nn.Module, stats_trainloader: DataLoader, device: str = 'cuda'):
    """
        Collects mean and cov from top-layer (before activations).
        Teacher should give raw LOGITS!

        returns: mean_activations, L_cholesky
    """

    # 1. to eval mode
    teacher_model.eval()

    # 2. collect top-layer activations
    print("Collecting teacher's top layer activation statistics...")
    all_teacher_activations = []

    with torch.no_grad():
        # stats_trainloader = DataLoader(trainset, batch_size=stats_batch_size, shuffle=False, num_workers=2)
        for i, data in tqdm(enumerate(stats_trainloader, 0), total=len(stats_trainloader)):
            inputs, _ = data
            inputs = inputs.to(device)
            activations = teacher_model(inputs)
            all_teacher_activations.append(activations.cpu())

    all_teacher_activations = torch.cat(all_teacher_activations, dim=0)
    print(f"Collected {all_teacher_activations.shape} teacher activations.")

    # 3. calculate stats
    mean_activations = torch.mean(all_teacher_activations, dim=0)
    cov_activations = torch.cov(all_teacher_activations.T)
    jitter = 1e-6 * torch.eye(cov_activations.shape[-1])
    cov_activations += jitter
    L_cholesky = torch.linalg.cholesky(cov_activations)

    print("Teacher statistics collected: Mean and Cholesky decomposition of Covariance Matrix.")
    print(f"Mean shape: {mean_activations.shape}")
    print(f"Covariance shape: {cov_activations.shape}")

    return mean_activations, L_cholesky


def reconstruct_data_batch(teacher_model, mean_stats, cholesky_stats, batch_size, iterations, lr, device):
    teacher_model.eval()

    reconstructed_input = nn.Parameter(torch.randn(batch_size, 3, 32, 32).to(device))
    optimizer_input = optim.Adam([reconstructed_input], lr=lr)

    reconstruction_losses = []
    for i in range(iterations):
        optimizer_input.zero_grad()

        # 1. Sampling from N(mean, cov) is mean + L_cholesky @ random_normal
        random_noise = torch.randn([batch_size, *mean_stats.shape]).to(device)
        sampled_activations = mean_stats.to(device) + torch.matmul(random_noise, cholesky_stats.T.to(device))

        # 2. apply ReLU
        target_activations = F.relu(sampled_activations)

        # 3. get logits
        teacher_output = teacher_model(reconstructed_input)
        teacher_relu_output = F.relu(teacher_output)

        # 4. Calculate MSE loss
        loss = F.mse_loss(teacher_relu_output, target_activations)

        # 5. Backward
        loss.backward()
        optimizer_input.step()

        reconstruction_losses.append(loss.item())

    # we do not need gradients
    reconstructed_input = reconstructed_input.detach()
    return reconstructed_input, reconstruction_losses


def reconstruct_train_dataset(teacher_model: nn.Module,
                              mean_stats, 
                              cholesky_stats,
                              reconstruction_size: int = 1024, 
                              reconstruction_iterations: int = 100,
                              reconstruction_lr: int = 0.1,
                              reconstruction_batch_size: int = 128,
                              device: str = 'cuda'):
    """

    """


    n_full_batches = reconstruction_size // reconstruction_batch_size
    last_batch_size = reconstruction_size % reconstruction_batch_size

    rec_train = []

    for _ in tqdm(range(n_full_batches), total=n_full_batches):
        rec_batch, _ = reconstruct_data_batch(teacher_model, mean_stats, cholesky_stats, reconstruction_batch_size, reconstruction_iterations, reconstruction_lr, device)
        rec_train.append(rec_batch)

    if last_batch_size != 0:
        rec_batch, _ = reconstruct_data_batch(teacher_model, mean_stats, cholesky_stats, last_batch_size, reconstruction_iterations, reconstruction_lr, device)
        rec_train.append(rec_batch)

    return torch.cat(rec_train, axis=0)


def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


def distill_stat_features(
    teacher: nn.Module,
    student: nn.Module,

    train_loader: DataLoader,
    test_loader: DataLoader = None,

    reconstruction_size: int = 1024, 
    reconstruction_iterations: int = 100,
    reconstruction_lr: float = 0.1,
    reconstruction_batch_size: int = 128,

    student_epochs: int = 3,
    student_lr: float = 0.001,

    temperature: float = 8.0,
    distillation_loss_fn = nn.KLDivLoss(reduction='batchmean'),

    eval_every: int = 50,

    device: str = 'cuda',
):
    teacher = teacher.to(device)
    student = student.to(device)

    # 1. collect train stats
    mean_stat, chol_stat = collect_top_layer_stats(teacher, train_loader, device)

    optimizer_student = optim.Adam(student.parameters(), lr=student_lr)
    student_train_losses = []

    for epoch in tqdm(range(student_epochs), total=student_epochs):
        
        student.train()
        running_loss = 0.0

        # 2. generate train dataset

        train_dataset_rec = reconstruct_train_dataset(
            teacher,
            mean_stat,
            chol_stat,
            reconstruction_size,
            reconstruction_iterations,
            reconstruction_lr,
            reconstruction_batch_size,
            device
        )

        gen_train_loader = DataLoader(train_dataset_rec, batch_size=reconstruction_batch_size, shuffle=False)

        for i, reconstructed_batch in tqdm(enumerate(gen_train_loader), total=len(gen_train_loader)):
            optimizer_student.zero_grad()

            # Получаем логиты учителя для реконструированных данных
            with torch.no_grad(): # Не считаем градиенты для учителя
                teacher_logits = teacher(reconstructed_batch)

            # Получаем логиты ученика для реконструированных данных
            student_logits = student(reconstructed_batch)

            # Вычисляем дистилляционный loss (KL divergence) [9]
            # Применяем softmax и log_softmax с температурой
            loss_distillation = distillation_loss_fn(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1)
            )
            loss = loss_distillation

            loss.backward()
            optimizer_student.step()

            running_loss += loss.item()

            if (i + 1) % eval_every == 0 and test_loader is not None:
                print(f'Student Epoch {epoch + 1}/{student_epochs}, Batch {i}, Student Loss: {running_loss / (i+1):.4f}')
                student_accuracy = evaluate_model(student, test_loader, device)
                print(f'Distilled Student Accuracy on original test images: {student_accuracy:.2f} %')



        epoch_loss = running_loss / len(gen_train_loader)
        student_train_losses.append(epoch_loss)
        print(f'Student Epoch {epoch + 1}/{student_epochs} finished. Avg Student Loss: {epoch_loss:.4f}')

    print("Finished Student Training.")

    return student
