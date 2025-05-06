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

def collect_all_layer_spectral_stats(teacher_model, stats_trainloader, device, spectral_keep_ratio=0.1):
    teacher_model.eval()
    teacher_model.to(device)

    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            if name not in activations:
                activations[name] = []
            activations[name].append(output.detach().cpu())
        return hook

    # Register hooks
    hooks = []
    for name, module in teacher_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(get_activation(name)))

    # Run model to collect activations
    with torch.no_grad():
        for data in tqdm(stats_trainloader, desc="Collecting activations"):
            inputs, _ = data
            inputs = inputs.to(device)
            _ = teacher_model(inputs)
    
    
    for h in hooks:
        h.remove()
    
    # Spectral info container
    layer_spectral_info = {}
    for name, acts in activations.items():
        acts_tensor = torch.cat(acts, dim=0)

        # Flatten to [N, D]
        if acts_tensor.ndim > 2:
            acts_tensor = acts_tensor.flatten(start_dim=1)

        N, D = acts_tensor.shape

        # Build adjacency matrix A (simple identity or random symmetric matrix if desired)
        A = torch.eye(D)

        # Compute eigenvectors of A (graph Fourier basis)
        eigvals, eigvecs = torch.linalg.eig(A)
        eigvecs = eigvecs.real  

        F_forward = eigvecs.T
        F_inv = eigvecs

        # Compute spectrum
        spectrum = acts_tensor @ F_forward.T  
        magnitude = torch.abs(spectrum)

        k = int(spectral_keep_ratio * D)
        topk_vals, topk_indices = torch.topk(magnitude, k=k, dim=1)

        layer_spectral_info[name] = {
            'F': F_forward,
            'F_inv': F_inv,
            'top_indices': topk_indices,
            'topk_values': topk_vals,
            'original_size': D
        }

    return layer_spectral_info

def reconstruct_data_batch(teacher_model,
                            layer_spectral_info,
                            batch_size,
                            iterations,
                            lr,
                            device='cuda'):
    teacher_model.eval()

    input_shape = (batch_size, 3, 32, 32)  # Assumes CIFAR-like images
    reconstructed_input = nn.Parameter(torch.randn(*input_shape).to(device))
    optimizer_input = optim.Adam([reconstructed_input], lr=lr)

    # Record hooks to capture teacher activations
    current_activations = {}
    hooks = []

    def get_activation_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                current_activations[name] = output.view(output.size(0), -1)
        return hook

    for name, module in teacher_model.named_modules():
        if name in layer_spectral_info:
            hooks.append(module.register_forward_hook(get_activation_hook(name)))

    for _ in range(iterations):
        optimizer_input.zero_grad()

        current_activations.clear()
        teacher_model(reconstructed_input)

        loss = 0.0
        for layer_name, act in current_activations.items():
            if act.shape[0] != batch_size:
                continue  # Skip incomplete batch

            spectral_data = layer_spectral_info[layer_name]
            F_forward = spectral_data['F'].to(device)
            F_inv = spectral_data['F_inv'].to(device)
            top_indices = spectral_data['top_indices'].to(device)
            D = spectral_data['original_size']

            # Graph Fourier Transform
            s = act  
            s_hat = (F_forward @ s.T).T 

            # Truncate and reconstruct
            s_hat_trunc = torch.zeros_like(s_hat)
            s_hat_trunc[:, top_indices] = s_hat[:, top_indices]
            s_recon = (F_inv @ s_hat_trunc.T).T

            layer_loss = F.mse_loss(s_recon, s)
            loss += layer_loss

        loss.backward()
        optimizer_input.step()

    for h in hooks:
        h.remove()

    return reconstructed_input.detach(), None 

def reconstruct_train_dataset(teacher_model: nn.Module,
                               layer_spectral_info,
                               reconstruction_size: int = 1024,
                               reconstruction_iterations: int = 100,
                               reconstruction_lr: float = 0.001,
                               reconstruction_batch_size: int = 1,
                               device: str = 'cuda'):
    n_full_batches = reconstruction_size // reconstruction_batch_size
    last_batch_size = reconstruction_size % reconstruction_batch_size

    rec_train = []

    for _ in tqdm(range(n_full_batches), desc="Reconstructing batches"):
        rec_batch, _ = reconstruct_data_batch(
            teacher_model,
            layer_spectral_info,
            reconstruction_batch_size,
            reconstruction_iterations,
            reconstruction_lr,
            device
        )
        rec_train.append(rec_batch)

    if last_batch_size != 0:
        rec_batch, _ = reconstruct_data_batch(
            teacher_model,
            layer_spectral_info,
            last_batch_size,
            reconstruction_iterations,
            reconstruction_lr,
            device
        )
        rec_train.append(rec_batch)

    return torch.cat(rec_train, dim=0)

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

def distill_stat_features_spectral(
    teacher_net: nn.Module,
    student_net: nn.Module,

    train_loader: DataLoader,
    test_loader: DataLoader,

    reconstruction_size: int = 1024,
    reconstruction_iterations: int = 100,
    reconstruction_lr: float = 0.1,
    reconstruction_batch_size: int = 1,

    student_epochs: int = 3,
    student_lr: float = 0.001,

    temperature: float = 8.0,
    distillation_loss_fn=nn.KLDivLoss(reduction='batchmean'),

    eval_every: int = 50,
    spectral_keep_ratio: float = 0.1,
    device: str = 'cuda',
):
    print("Collecting spectral activation statistics from teacher...")
    layer_spectral_info = collect_all_layer_spectral_stats(
        teacher_model=teacher_net,
        stats_trainloader=train_loader,
        spectral_keep_ratio=spectral_keep_ratio,
        device=device
    )

    optimizer_student = optim.Adam(student_net.parameters(), lr=student_lr)
    student_train_losses = []

    for epoch in tqdm(range(student_epochs), total=student_epochs):
        student_net.train()
        running_loss = 0.0

        print(f"Reconstructing synthetic training set (epoch {epoch+1})...")
        train_dataset_rec = reconstruct_train_dataset(
            teacher_net,
            layer_spectral_info,
            reconstruction_size,
            reconstruction_iterations,
            reconstruction_lr,
            reconstruction_batch_size,
            device
        )

        gen_train_loader = DataLoader(train_dataset_rec, batch_size=reconstruction_batch_size, shuffle=False)

        for i, reconstructed_batch in tqdm(enumerate(gen_train_loader), total=len(gen_train_loader)):
            optimizer_student.zero_grad()

            with torch.no_grad():
                teacher_logits = teacher_net(reconstructed_batch)

            student_logits = student_net(reconstructed_batch)

            loss_distillation = distillation_loss_fn(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1)
            )
            loss = loss_distillation

            loss.backward()
            optimizer_student.step()
            running_loss += loss.item()

            if (i + 1) % eval_every == 0:
                print(f"Student Epoch {epoch + 1}/{student_epochs}, Batch {i}, Loss: {running_loss / (i+1):.4f}")
                acc = evaluate_model(student_net, test_loader, device)
                print(f"Test accuracy: {acc:.2f}%")

        epoch_loss = running_loss / len(gen_train_loader)
        student_train_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1} finished. Avg Loss: {epoch_loss:.4f}")

    print("Finished distillation using spectral stats.")
    return student_net