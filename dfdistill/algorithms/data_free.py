import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython.display import clear_output


def validate(student, test_loader, device="cuda", verbose=True):
    """
    Evaluate the student model on the test dataset.

    Args:
        student (torch.nn.Module): The student model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (str, optional): Device to run evaluation on ('cuda' or 'cpu'). Default is 'cuda'.
        verbose (bool, optional): Whether to print evaluation results. Default is True.

    Returns:
        float: Accuracy of the student model on the test data.
    """
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
    """
    Plot loss and accuracy curves for training progress.

    Args:
        history (dict): Dictionary containing training history with keys:
            - 'epochs': list of epoch numbers.
            - 'student_loss': list of student loss values.
            - 'accuracy': list of accuracy values.
    """
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
    """
    Train student and generator models using the DFAD method (Distillation with Feature Adversarial Distillation).

    Args:
        teacher (torch.nn.Module): Pretrained teacher model.
        student (torch.nn.Module): Student model to be trained.
        generator (torch.nn.Module): Image generator model.
        optimizer_s (torch.optim.Optimizer): Optimizer for the student.
        optimizer_g (torch.optim.Optimizer): Optimizer for the generator.
        scheduler_s (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler for student optimizer. Default is None.
        scheduler_g (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler for generator optimizer. Default is None.
        test_loader (DataLoader, optional): DataLoader for test dataset, used for validation. Default is None.
        device (str, optional): Device to run training on ('cuda' or 'cpu'). Default is 'cuda'.
        epochs (int, optional): Number of training epochs. Default is 500.
        iters_per_epoch (int, optional): Number of iterations per epoch. Default is 100.
        batch_size (int, optional): Batch size. Default is 128.
        nz (int, optional): Dimension of noise vector for generator input. Default is 256.
        log_interval (int, optional): Interval (in iterations) to log training progress. Default is 10.
        val_interval (int, optional): Interval (in epochs) to run validation. Default is 1.
        dataset_name (str, optional): Dataset name for checkpoint filenames. Default is 'cifar10'.
        model_name (str, optional): Model name for checkpoint filenames. Default is 'resnet18_8x'.
        save_dir (str, optional): Directory to save checkpoints. Default is 'checkpoint/student'.

    Returns:
        torch.nn.Module: Trained student model.
    """
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
            # Train student for 5 steps
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

            # Train generator
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
