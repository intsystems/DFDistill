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
        """
        Sets the teacher, student and optimizer to be used for distillation.

        Parameters
        ----------
        teacher : torch.nn.Module
            The teacher model to be used for distillation.
        student : torch.nn.Module
            The student model to be used for distillation.
        optimizer : torch.optim.Optimizer
            The optimizer to be used to update the student model.
        """
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

    def test(
        self, 
        dataset
    ):
        """
        Computes the test metrics for the student model on the given dataset.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            The dataset to be used for computing the test metrics.

        Returns
        -------
        result : dict
            A dictionary containing the test metrics for the student model.
        """
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
        
        """
        Initializes the ST (Self-Training) algorithm.

        Parameters
        ----------
        config : dict
            A dictionary containing the hyperparameters of the algorithm.

        Attributes
        ----------
        teacher : torch.nn.Module
            The teacher model to be used for distillation. Defaults to None.
        student : torch.nn.Module
            The student model to be used for distillation. Defaults to None.
        optimizer : torch.optim.Optimizer
            The optimizer to be used to update the student model. Defaults to None.
        alpha : float
            The weight of the supervised loss. Defaults to 0.5.
        T : float
            The temperature of the softmax. Defaults to 1.0.
        """
        super(ST, self).__init__(config)

        self.teacher = None
        self.student = None
        self.optimizer = None

        self.alpha = config["alpha"]
        self.T = config["T"]

    def teach_step(self, x, y):
        """
        Performs a single teaching step in the distillation process.

        Parameters
        ----------
        x : torch.Tensor
            The input data batch.
        y : torch.Tensor
            The corresponding labels for the input data batch.

        Returns
        -------
        torch.Tensor
            The computed loss for the current step.

        This method performs a forward pass through both the teacher and student models,
        computes the loss, and updates the student model's parameters using backpropagation.
        """

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

    def compute_loss(
        self, 
        t_logits, 
        s_logits, 
        labels
    ):
        """
        Computes the loss for the current distillation step.

        This method computes the supervised loss using the provided labels and the
        student model's output. Then, it computes the distillation loss using the
        teacher model's output and the student model's output. Finally, it returns
        the weighted sum of the two losses according to the alpha hyperparameter.

        Parameters
        ----------
        t_logits : torch.Tensor
            The output of the teacher model.
        s_logits : torch.Tensor
            The output of the student model.
        labels : torch.Tensor
            The true labels of the input data.

        Returns
        -------
        torch.Tensor
            The computed loss.
        """
        t_logits = t_logits.detach()
        return (1 - self.alpha) * self.cls_loss(
            labels, s_logits
        ) + self.alpha * self.st_loss(t_logits, s_logits)

    def cls_loss(
        self, 
        labels, 
        s_logits
    ):
        """
        Computes the classification loss using cross-entropy.

        Parameters
        ----------
        labels : torch.Tensor
            The true labels of the input data.
        s_logits : torch.Tensor
            The output logits from the student model.

        Returns
        -------
        torch.Tensor
            The computed cross-entropy loss between the student model's
            predicted logits and the true labels.
        """

        criterion = torch.nn.CrossEntropyLoss()
        return criterion(s_logits, labels)

    def st_loss(
        self, 
        t_logits, 
        s_logits
    ):
        """
        Computes the knowledge distillation loss between the teacher and student logits.

        The distillation loss is the KL-divergence between the teacher's and student's
        output distributions, after applying the softmax function to the logits.

        Parameters
        ----------
        t_logits : torch.Tensor
            The output logits from the teacher model.
        s_logits : torch.Tensor
            The output logits from the student model.

        Returns
        -------
        torch.Tensor
            The computed distillation loss between the teacher and student models.
        """
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
    """
    Converts a result dictionary to a string template for tqdm.

    Parameters
    ----------
    result : dict
        A dictionary of results to display in the tqdm bar.
    training : bool
        Whether the results are from the training or validation set.

    Returns
    -------
    str
        A string template to pass to the `tqdm` `bar_format` argument.
    """
    template = ""
    for k in result.keys():
        display_name = k
        if not training:
            display_name = "val_{}".format(display_name)
        template += "{}: {:.4f} -".format(display_name, result[k])
    return template[:-1]


def _init_dist(
    teacher, 
    student, 
    algo, 
    optimizer, 
    iterations,
    verbose=True
):
    """
    Initializes the distillation process.

    Parameters
    ----------
    teacher : torch.nn.Module
        The teacher model to distill knowledge from.
    student : torch.nn.Module
        The student model to distill knowledge into.
    algo : object
        The algorithm to use for distillation.
    optimizer : torch.optim.Optimizer
        The optimizer to use for updating the student model's parameters.
    iterations : int
        The number of iterations to distill for.
    verbose : bool, optional
        Whether to display a tqdm progress bar of the distillation process.

    Returns
    -------
    tuple
        A tuple containing the algorithm object and the tqdm progress bar object.
    """

    algo.set_model(teacher, student, optimizer)
    bar_format = "{desc} - {n_fmt}/{total_fmt} [{bar:30}] ELP: {elapsed}{postfix}"
    if verbose:
        process_log = trange(iterations, desc="Training", position=0, bar_format=bar_format)
    else:
        process_log = None
    return algo, process_log
    

def distill(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader = None,
    iterations: int = 5000, 
    test_freq: int = 1000,
    alpha: float = 0.6, 
    T: float = 2.5,
    verbose=True
):
    """
    Performs knowledge distillation using the provided teacher model and
    optimizer on the student model.

    Parameters
    ----------
    teacher : torch.nn.Module
        The teacher model to distill knowledge from.
    student : torch.nn.Module
        The student model to distill knowledge into.
    optimizer : torch.optim.Optimizer
        The optimizer to use for updating the student model's parameters.
    train_loader : torch.utils.data.DataLoader
        The data loader for the training data.
    test_loader : torch.utils.data.DataLoader, optional
        The data loader for the testing data. If provided, the model is tested
        every `test_freq` iterations.
    iterations : int, optional
        The number of iterations to distill for. Defaults to 5000.
    test_freq : int, optional
        The frequency at which to test the model. Defaults to 1000.
    alpha : float, optional
        The weight for the distillation loss. Defaults to 0.6.
    T : float, optional
        The temperature for the softmax in distillation. Defaults to 2.5.
    verbose : bool, optional
        Whether to display a tqdm progress bar of the distillation process. Defaults to True.

    Returns
    -------
    torch.nn.Module
        The trained student model.
    """
    algo, process_log = _init_dist(
        teacher, 
        student, 
        ST({"alpha": alpha, "T": T}), 
        optimizer, 
        iterations,
        verbose=verbose
    )

    train_tmp = ""
    test_tmp = ""

    device = next(teacher.parameters()).device

    n_iter = 0
    while True:
        for i, batch in enumerate(train_loader):

            if verbose:
                process_log.update(1)

            loss = algo.teach_step(batch[0].to(device), batch[1].to(device))
            result = algo.get_metrics()
            train_tmp = result_to_tqdm_template(result)

            if test_loader is not None and test_freq > 0:
                if n_iter % test_freq == 0 and n_iter != 0:

                    if verbose:
                        process_log.set_description_str("Testing ")

                    algo.reset_metrics()
                    result = algo.test(test_loader)
                    test_tmp = result_to_tqdm_template(result, training=False)

                    if verbose:
                        process_log.set_description_str("Training")
                        
            postfix = train_tmp + "- " + test_tmp

            n_iter += 1
            if verbose:
                process_log.set_postfix_str(postfix)

            if n_iter >= iterations:
                break

        if n_iter >= iterations:
            break

    return student
