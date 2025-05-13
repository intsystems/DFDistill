import unittest
import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.insert(0, '/home/ernest/Desktop/BMM_project/DFDistill/')

from dfdistill.algorithms.data_free import train_dfad

class TestDFADTraining(unittest.TestCase):
    
    def setUp(self):
        self.device = 'cpu'
        self.batch_size = 4
        self.nz = 20 # hidden dim
        self.input_shape = (3, 32, 32)

        # Simple generator model
        class Generator(nn.Module):
            def __init__(self):
                super().__init__()
                self.main = nn.Sequential(
                    nn.ConvTranspose2d(20, 5, 4),
                    nn.ReLU(),
                    nn.ConvTranspose2d(5, 10, 4),
                    nn.ReLU(),
                    nn.ConvTranspose2d(10, 3, 4),
                    nn.Tanh()
                )
            
            def forward(self, x):
                return self.main(x)

        # Simple teacher model
        class Teacher(nn.Module):
            def __init__(self):
                super().__init__()
                self.main = nn.Sequential(
                    nn.Conv2d(3, 5, 3),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(320, 10)
                )
            
            def forward(self, x):
                return self.main(x)

        # Simple student model
        class Student(nn.Module):
            def __init__(self):
                super().__init__()
                self.main = nn.Sequential(
                    nn.Conv2d(3, 2, 3),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(128, 10)
                )
            
            def forward(self, x):
                return self.main(x)

        self.teacher = Teacher().to(self.device)
        self.student = Student().to(self.device)
        self.generator = Generator().to(self.device)

    def reset_parameters(self):
        # Reset parameters before each test
        for model in [self.teacher, self.student, self.generator]:
            for layer in model.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def test_train_dfad(self):
        optimizer_s = optim.Adam(self.student.parameters(), lr=0.001)
        optimizer_g = optim.Adam(self.generator.parameters(), lr=0.001)

        # Test basic execution
        trained_student = train_dfad(
            teacher=self.teacher,
            student=self.student,
            generator=self.generator,
            optimizer_s=optimizer_s,
            optimizer_g=optimizer_g,
            device=self.device,
            epochs=1,
            iters_per_epoch=2,
            batch_size=self.batch_size,
            nz=self.nz
        )
        
        self.assertIsInstance(trained_student, nn.Module)

    def test_parameter_updates(self):
        # Record initial parameters
        initial_student = [p.clone() for p in self.student.parameters()]
        initial_generator = [p.clone() for p in self.generator.parameters()]

        optimizer_s = optim.Adam(self.student.parameters(), lr=0.001)
        optimizer_g = optim.Adam(self.generator.parameters(), lr=0.001)

        train_dfad(
            teacher=self.teacher,
            student=self.student,
            generator=self.generator,
            optimizer_s=optimizer_s,
            optimizer_g=optimizer_g,
            device=self.device,
            epochs=1,
            iters_per_epoch=1,
            batch_size=self.batch_size,
            nz=self.nz
        )

        # Check parameter updates
        for init, trained in zip(initial_student, self.student.parameters()):
            self.assertFalse(torch.allclose(init, trained))
        
        for init, trained in zip(initial_generator, self.generator.parameters()):
            self.assertFalse(torch.allclose(init, trained))