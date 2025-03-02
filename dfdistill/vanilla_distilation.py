from abc import ABCMeta, abstractmethod

import torch
import torch.nn.functional as F
import ignite
from tqdm import trange


class BaseTorch(object, metaclass=ABCMeta):
    def __init__(self, config):
        self.metrics = {"accuracy": ignite.metrics.Accuracy()}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_model(self, teacher, student, optimizer):
        self.teacher = teacher.to(self.device)
        self.student = student.to(self.device)
        self.optimizer = optimizer

    @abstractmethod
    def teach_step(self, x, y):
        pass

    def logging_metrics(self, labels, logits):
        for name in self.metrics.keys():
            self.metrics[name].update([logits, labels])

    def reset_metrics(self):
        for name in self.metrics.keys():
            self.metrics[name].reset()

    def set_metrics(self, metrics):
        for name in metrics.keys():
            assert isinstance(metrics.keys()[name], ignite.metrics.Metric)
            self.metrics[name] = metrics

    def get_metrics(self):
        result = {}
        for name in self.metrics.keys():
            result[name] = self.metrics[name].compute()
        return result

    def test(self, dataset):
        self.reset_metrics()
        with torch.set_grad_enabled(False):
            for x, y in dataset:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.student(x)
                self.logging_metrics(y, logits)
        result = self.get_metrics()
        self.reset_metrics()
        return result


class ST(BaseTorch):
    def __init__(self, config):
        super(ST, self).__init__(config)

        self.teacher = None
        self.student = None
        self.optimizer = None

        self.alpha = config["alpha"]
        self.T = config["T"]

    def teach_step(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        self.optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            t_logits = self.teacher(x)
        s_logits = self.student(x)
        self.logging_metrics(y, s_logits)
        loss = self.compute_loss(t_logits, s_logits, y)
        loss.backward()
        self.optimizer.step()
        return loss

    def compute_loss(self, t_logits, s_logits, labels):
        t_logits = t_logits.detach()
        return (1 - self.alpha) * self.cls_loss(
            labels, s_logits
        ) + self.alpha * self.st_loss(t_logits, s_logits)

    def cls_loss(self, labels, s_logits):
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(s_logits, labels)

    def st_loss(self, t_logits, s_logits):
        assert t_logits.shape == s_logits.shape
        return (
            F.kl_div(
                F.log_softmax(s_logits / self.T, dim=1),
                F.softmax(t_logits / self.T, dim=1),
                size_average=False,
            )
            * (self.T ** 2)
            / s_logits.shape[0]
        )
    

def result_to_tqdm_template(result, training=True):
    template = ""
    for k in result.keys():
        display_name = k
        if not training:
            display_name = "val_{}".format(display_name)
        template += "{}: {:.4f} -".format(display_name, result[k])
    return template[:-1]


def _init_dist(teacher, student, algo, optimizer, iterations):
    algo.set_model(teacher, student, optimizer)
    bar_format = "{desc} - {n_fmt}/{total_fmt} [{bar:30}] ELP: {elapsed}{postfix}"
    process_log = trange(iterations, desc="Training", position=0, bar_format=bar_format)
    return algo, process_log
    

def distill(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    iterations: int = 5000, 
    test_freq: int = 1000,
    alpha: float = 0.6, 
    T: float = 2.5
):
    algo, process_log = _init_dist(teacher, student, ST(alpha, T), optimizer, iterations)

    train_tmp = ""
    test_tmp = ""

    train_iter = iter(train_loader)
    for idx in range(iterations):
        try:
            x, y = train_iter.next()
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = train_iter.next()
        process_log.update(1)
        loss = algo.teach_step(x, y)
        result = algo.get_metrics()
        train_tmp = result_to_tqdm_template(result)

        if idx % test_freq == 0 and idx != 0:
            process_log.set_description_str("Testing ")
            algo.reset_metrics()
            result = algo.test(test_loader)
            test_tmp = result_to_tqdm_template(result, training=False)
            process_log.set_description_str("Training")
        postfix = train_tmp + "- " + test_tmp
        process_log.set_postfix_str(postfix)

    return student
