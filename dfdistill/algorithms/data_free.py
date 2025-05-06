import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


def train_dfad(
    teacher,
    student,
    generator,
    device,
    optimizer_s,
    optimizer_g,
    test_loader=None,
    epochs=10,
    iters_per_epoch=100,
    batch_size=128,
    nz=256,
    log_interval=10,
    val_interval=1,
    dataset_name='cifar10',
    model_name='resnet18'
):
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

        if test_loader is not None and (epoch % val_interval == 0 or epoch == epochs):
            acc = validate(student, generator, test_loader, device)
            if acc > best_acc:
                best_acc = acc
                os.makedirs('checkpoint/student', exist_ok=True)
                torch.save(student.state_dict(), f'checkpoint/student/{dataset_name}-{model_name}.pt')
                torch.save(generator.state_dict(), f'checkpoint/student/{dataset_name}-{model_name}-generator.pt')
                print(f'[Checkpoint] New best accuracy: {best_acc:.2f}%')

    print(f'\n[Finished] Best Accuracy: {best_acc:.2f}%')
    return student
