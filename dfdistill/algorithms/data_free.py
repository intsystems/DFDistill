import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from IPython.display import clear_output, display


def validate(student, test_loader, device="cuda", verbose=True):
    student.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = student(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    test_loss /= total
    accuracy = correct / total

    if verbose:
        print(f'[Validate] Avg loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({100. * accuracy:.2f}%)')
    return accuracy


def plot_metrics(history):
    clear_output(wait=True)
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(history['epochs'], history['student_loss'], label='Student Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy (%)', color='tab:green')
    ax2.plot(history['epochs'], history['accuracy'], label='Accuracy', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    fig.tight_layout()
    plt.title('Training Progress')
    plt.grid(True)
    plt.show()


def train_dfad(
    teacher,
    student,
    generator,
    optimizer_s,
    optimizer_g,
    scheduler_s=None,
    scheduler_g=None,
    test_loader=None,
    device="cuda",
    epochs=500,
    iters_per_epoch=100,
    batch_size=128,
    nz=256,
    log_interval=10,
    val_interval=1,
    dataset_name='cifar10',
    model_name='resnet18_8x',
    save_dir='checkpoint/student',
):
    teacher.eval()
    student.train()
    generator.train()

    best_acc = 0.0
    history = {
        'epochs': [],
        'student_loss': [],
        'accuracy': [],
    }

    for epoch in range(1, epochs + 1):
        epoch_student_loss = 0.0

        for iteration in range(1, iters_per_epoch + 1):
            # Train Student
            for _ in range(5):
                z = torch.randn(batch_size, nz, 1, 1, device=device)
                fake_images = generator(z).detach()
                with torch.no_grad():
                    teacher_logits = teacher(fake_images)
                student_logits = student(fake_images)
                loss_student = F.l1_loss(student_logits, teacher_logits)
                optimizer_s.zero_grad()
                loss_student.backward()
                optimizer_s.step()
                epoch_student_loss += loss_student.item()

            # Train Generator
            z = torch.randn(batch_size, nz, 1, 1, device=device)
            fake_images = generator(z)
            with torch.no_grad():
                teacher_logits = teacher(fake_images)
            student_logits = student(fake_images)
            loss_generator = -F.l1_loss(student_logits, teacher_logits)
            optimizer_g.zero_grad()
            loss_generator.backward()
            optimizer_g.step()

            if iteration % log_interval == 0:
                print(
                    f'[Epoch {epoch}/{epochs}] Iter {iteration}/{iters_per_epoch} | '
                    f'G_loss: {loss_generator.item():.4f} | S_loss: {loss_student.item():.4f}'
                )

        if scheduler_s is not None:
            scheduler_s.step()
        if scheduler_g is not None:
            scheduler_g.step()

        avg_student_loss = epoch_student_loss / (iters_per_epoch * 5)
        acc = 0.0
        if test_loader is not None and (epoch % val_interval == 0 or epoch == epochs):
            acc = validate(student, test_loader, device=device, verbose=False)
            if acc > best_acc:
                best_acc = acc
                os.makedirs(save_dir, exist_ok=True)
                torch.save(student.state_dict(), os.path.join(save_dir, f"{dataset_name}-{model_name}.pt"))
                torch.save(generator.state_dict(), os.path.join(save_dir, f"{dataset_name}-{model_name}-generator.pt"))
                print(f'[Checkpoint] New best accuracy: {100. * best_acc:.2f}%')

        history['epochs'].append(epoch)
        history['student_loss'].append(avg_student_loss)
        history['accuracy'].append(acc * 100.0)
        plot_metrics(history)

    print(f'\n[Finished] Best Accuracy: {100. * best_acc:.2f}%')
    return student
