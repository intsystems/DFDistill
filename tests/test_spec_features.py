import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys

sys.path.insert(0, '/home/ernest/Desktop/BMM_project/DFDistill/')

from dfdistill.algorithms.distill_spec_features import collect_all_layer_spectral_stats, reconstruct_data_batch, reconstruct_train_dataset, evaluate_model, distill_stat_features_spectral
class TestSpecDistillation(unittest.TestCase):

    def setUp(self):
        self.device = 'cpu'
        self.batch_size = 4
        self.num_classes = 10
        self.input_shape = (3, 32, 32)
        
        # dummy datasets
        self.dummy_data = torch.randn(20, *self.input_shape)
        self.dummy_labels = torch.randint(0, self.num_classes, (20,))
        self.dummy_dataset = TensorDataset(self.dummy_data, self.dummy_labels)
        self.dummy_loader = DataLoader(self.dummy_dataset, batch_size=self.batch_size)

        class TeacherNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.fc = nn.Linear(4704, 10)
                
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = x.flatten(1)
                x = self.fc(x)
                return x

        # Simple student model
        class StudentNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 3, 3)
                self.fc = nn.Linear(2700, 10)
                
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = x.flatten(1)
                x = self.fc(x)
                return x

        self.teacher = TeacherNet().to(self.device)
        self.student = StudentNet().to(self.device)

    def test_spectral_stats_collection(self):
        spectral_info = collect_all_layer_spectral_stats(
            teacher_model=self.teacher,
            stats_trainloader=self.dummy_loader,
            device=self.device
        )
        
        # all layers should present
        layer_names = [name for name, _ in self.teacher.named_modules()
                      if isinstance(_, (nn.Conv2d, nn.Linear))]
        self.assertEqual(len(spectral_info), len(layer_names))
        
        # keys in the dictionary
        for layer_name, info in spectral_info.items():
            self.assertIn('F', info)
            self.assertIn('F_inv', info)
            self.assertIn('top_indices', info)
            self.assertIn('original_size', info)
            
            # matrix dimensions
            D = info['original_size']
            self.assertEqual(info['F'].shape, (D, D))
            self.assertEqual(info['F_inv'].shape, (D, D))

    def test_data_reconstruction_batch(self):
        spectral_info = collect_all_layer_spectral_stats(
            self.teacher,
            self.dummy_loader,
            self.device
        )
        
        reconstructed, _ = reconstruct_data_batch(
            teacher_model=self.teacher,
            layer_spectral_info=spectral_info,
            batch_size=self.batch_size,
            iterations=2,
            lr=0.1,
            device=self.device
        )
        
        # check output shape
        self.assertEqual(reconstructed.shape, (self.batch_size, *self.input_shape))
        
        # # Verify values are within reasonable range
        # self.assertLess(reconstructed.abs().max().item(), 10)
        # self.assertGreater(reconstructed.std().item(), 0.01)

    def test_full_dataset_reconstruction(self):
        spectral_info = collect_all_layer_spectral_stats(
            self.teacher,
            self.dummy_loader,
            self.device
        )
        
        reconstructed = reconstruct_train_dataset(
            teacher_model=self.teacher,
            layer_spectral_info=spectral_info,
            reconstruction_size=8,
            reconstruction_iterations=2,
            reconstruction_lr=0.1,
            reconstruction_batch_size=4,
            device=self.device
        )
        
        # check total size
        self.assertEqual(len(reconstructed), 8)
        # check individual sample shape
        self.assertEqual(reconstructed[0].shape, self.input_shape)


    def test_distillation_process(self):
        # Make copy of initial student weights
        initial_weights = [p.clone() for p in self.student.parameters()]
        
        _ = distill_stat_features_spectral(
            teacher_net=self.teacher,
            student_net=self.student,
            train_loader=self.dummy_loader,
            test_loader=self.dummy_loader,
            reconstruction_size=8,
            reconstruction_iterations=2,
            student_epochs=1,
            reconstruction_batch_size=4,
            device=self.device
        )
        
        # Verify student weights have changed
        for init_p, trained_p in zip(initial_weights, self.student.parameters()):
            self.assertFalse(torch.allclose(init_p, trained_p))

    def test_spectral_reconstruction_improvement(self):
        # Verify reconstruction loss decreases
        spectral_info = collect_all_layer_spectral_stats(
            self.teacher,
            self.dummy_loader,
            self.device
        )
        
        losses = []
        for _ in range(3):  # Multiple iterations to check trend
            reconstructed, _ = reconstruct_data_batch(
                teacher_model=self.teacher,
                layer_spectral_info=spectral_info,
                batch_size=2,
                iterations=1,
                lr=0.1,
                device=self.device
            )
            
            # Calculate reconstruction loss
            with torch.no_grad():
                teacher_acts = self.teacher(reconstructed).to_sparse()
                loss = sum([torch.norm(act.to_sparse()).item() for act in teacher_acts.values()])
                losses.append(loss)
        
        # Check loss is generally decreasing (allow some noise)
        self.assertLess(losses[-1], losses[0] * 2)  # Not worse than double initial loss
