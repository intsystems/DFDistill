import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys

# sys.path.insert(0, '../algorithms')
sys.path.insert(0, '/home/ernest/Desktop/BMM_project/DFDistill')
# /home/ernest/Desktop/BMM_project/DFDistill/dfdistill/algorithms/distill_stat_features.py
from dfdistill.algorithms.distill_stat_features import collect_top_layer_stats, reconstruct_data_batch, reconstruct_train_dataset, evaluate_model, distill_stat_features


class TestStatDistillation(unittest.TestCase):
    def setUp(self):
        self.device = 'cpu'       
        self.n_classes = 10

        self.teacher = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4704, self.n_classes)
        )
        
        self.student = nn.Sequential(
            nn.Conv2d(3, 3, 5),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2352, self.n_classes)
        )

        X = torch.randn(10, 3, 32, 32)
        y = torch.randint(0, 10, (10,))
        self.dummy_dataset = TensorDataset(X, y)
        self.dummy_loader = DataLoader(self.dummy_dataset, batch_size=2)


    def test_collect_top_layer_stats(self):
        mean, chol = collect_top_layer_stats(self.teacher, self.dummy_loader, device=self.device)
        self.assertEqual(mean.shape, (10,))
        self.assertEqual(chol.shape, (10, 10))


    def test_reconstruct_data_batch(self):
        mean = torch.randn(10)
        chol = torch.eye(10)
        batch_size = 2
        iterations = 3
        reconstructed, losses = reconstruct_data_batch(
            self.teacher, mean, chol, batch_size, iterations, lr=0.1, device=self.device
        )
        self.assertEqual(reconstructed.shape, (batch_size, 3, 32, 32))
        self.assertEqual(len(losses), iterations)

    def test_reconstruct_train_dataset(self):
        reconstruction_size = 10
        mean = torch.randn(10)
        chol = torch.eye(10)
        dataset = reconstruct_train_dataset(
            self.teacher, mean, chol, 
            reconstruction_size=reconstruction_size,
            reconstruction_batch_size=3,
            device=self.device
        )
        self.assertEqual(len(dataset), reconstruction_size)

    def test_evaluate_model(self):
        class ZeroModel(nn.Module):
            def forward(self, x):
                return torch.zeros(x.size(0), 10)
        
        model = ZeroModel()
        X = torch.randn(5, 3, 32, 32)
        y = torch.tensor([5]*5)
        loader = DataLoader(TensorDataset(X, y), batch_size=2)
        accuracy = evaluate_model(model, loader, device=self.device)
        self.assertEqual(accuracy, 0.0)

    def test_distill_changes_student(self):
        student_params_before = [p.clone() for p in self.student.parameters()]
        distill_stat_features(
            self.teacher, self.student, self.dummy_loader,
            reconstruction_size=4,
            reconstruction_iterations=1,
            student_epochs=1,
            student_lr=0.1,
            device=self.device
        )
        for p_before, p_after in zip(student_params_before, self.student.parameters()):
            self.assertFalse(torch.allclose(p_before, p_after))

    def test_teacher_unchanged_after_distill(self):
        teacher_params_before = [p.clone() for p in self.teacher.parameters()]
        distill_stat_features(
            self.teacher, self.student, self.dummy_loader,
            reconstruction_size=4,
            reconstruction_iterations=1,
            student_epochs=1,
            student_lr=0.1,
            device=self.device
        )
        for p_before, p_after in zip(teacher_params_before, self.teacher.parameters()):
            self.assertTrue(torch.allclose(p_before, p_after))