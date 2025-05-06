import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


def train_dfad(
    teacher,
    student,
    generator,
    optimizer_s,
    optimizer_g,
    test_loader=None,
    device="cuda",
    epochs=10,
    iters_per_epoch=100,
    batch_size=128,
    nz=256,
    log_interval=10,
    val_interval=1,
    dataset_name='cifar10',
    model_name='resnet18',
    **kwargs
):
    """
    Train a student model using the Data-Free Adversarial Distillation algorithm.

    Parameters
    ----------
    teacher : torch.nn.Module
        The teacher model to use for distillation.
    student : torch.nn.Module
        The student model to train.
    generator : torch.nn.Module
        The generator model to use for generating fake images.
    optimizer_s : torch.optim.Optimizer
        The optimizer for the student model.
    optimizer_g : torch.optim.Optimizer
        The optimizer for the generator model.
    test_loader : torch.utils.data.DataLoader, optional
        The test data loader to use for evaluation. If None, no evaluation will be performed.
    device : str, optional
        The device to use for training. Defaults to "cuda".
    epochs : int, optional
        The number of epochs to train. Defaults to 10.
    iters_per_epoch : int, optional
        The number of iterations per epoch. Defaults to 100.
    batch_size : int, optional
        The batch size to use for training. Defaults to 128.
    nz : int, optional
        The number of latent variables to use for the generator. Defaults to 256.
    log_interval : int, optional
        The interval to print the loss. Defaults to 10.
    val_interval : int, optional
        The interval to evaluate the model. Defaults to 1.
    dataset_name : str, optional
        The name of the dataset to use. Defaults to "cifar10".
    model_name : str, optional
        The name of the model to use. Defaults to "resnet18".
    **kwargs
        Additional keyword arguments to pass to the optimizer.

    Returns
    -------
    The trained student model.
    """
    teacher.eval()
    student.train()
    generator.train()

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        for iteration in range(1, iters_per_epoch + 1):
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
                    f'[Epoch {epoch:3d}/{epochs}] Iter {iteration:4d}/{iters_per_epoch} '
                    f'| G_loss: {loss_generator.item():.4f} '
                    f'| S_loss: {loss_student.item():.4f}'
                )

        # if test_loader is not None and (epoch % val_interval == 0 or epoch == epochs):
        #     acc = validate(student, generator, test_loader, device)
        #     if acc > best_acc:
        #         best_acc = acc
        #         os.makedirs('checkpoint/student', exist_ok=True)
        #         torch.save(student.state_dict(), f'checkpoint/student/{dataset_name}-{model_name}.pt')
        #         torch.save(generator.state_dict(), f'checkpoint/student/{dataset_name}-{model_name}-generator.pt')
        #         print(f'[Checkpoint] New best accuracy: {best_acc:.2f}%')
    if best_acc > 0.0:
        print(f'\n[Finished] Best Accuracy: {best_acc:.2f}%')
    return student
