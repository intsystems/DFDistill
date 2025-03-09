import warnings

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


class Trainer:
    def __init__(
        self, 
        model=None,
        train_config=None
    ):
        self.model = model
        self.train_config = train_config if train_config is not None else {}
    
    def train(
        self,
        train_loader=None,
        val_loader=None,
        train_data=None,
        val_data=None,
        batch_size=None,
        model=None,
        max_iters=-1,
        max_epochs=-1,
        optimizer=None,
        lr=None,
        evaluate_every=None,
        log_every=None,
        reduce_lr_each_epoch=None,
        verbose=None
    ):
        if train_loader is None:
            if train_data is not None:
                if isinstance(train_data, Dataset):
                    pass
                else:
                    raise ValueError("train_data must be torch.utils.data.Dataset")
                if batch_size is not None:
                    self.train_config["batch_size"] = batch_size
                elif "batch_size" in self.train_config:
                    batch_size = self.train_config["batch_size"]
                else:
                    warnings.warn("No batch_size provided. Defaulting batch_size for train_loader to 1", UserWarning)
                    batch_size = 1
                train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            else:
                raise ValueError("train_loader or train_data must be provided")

        if val_loader is None:
            if val_data is not None:
                if isinstance(val_data, Dataset):
                    pass
                else:
                    raise ValueError("val_data must be torch.utils.data.Dataset")
                if batch_size is not None:
                    self.train_config["batch_size"] = batch_size
                elif "batch_size" in self.train_config:
                    batch_size = self.train_config["batch_size"]
                else:
                    warnings.warn("No batch_size provided. Defaulting batch_size for val_loader to 1", UserWarning)
                    batch_size = 1
                val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
            else:
                val_loader=None
        
        if max_epochs != -1:
            self.train_config["max_epochs"] = max_epochs
        elif "max_epochs" in self.train_config:
            max_epochs = self.train_config["max_epochs"]
        else:
            max_epochs = self.train_config["max_epochs"] = -1

        if max_iters != -1:
            self.train_config["max_iters"] = max_iters
        elif "max_iters" in self.train_config:
            max_iters = self.train_config["max_iters"]
        else:
            max_iters = self.train_config["max_iters"] = -1

        if max_epochs == -1 and max_iters == -1:
            max_epochs = 1

        if verbose is not None:
            self.train_config["verbose"] = verbose
        elif "verbose" in self.train_config:
            verbose = self.train_config["verbose"]
        else:
            self.train_config["verbose"] = verbose = True

        if model is not None:
            self.model = model
        if self.model is None:
            raise ValueError("No model provided")
        
        if lr is not None:
            self.train_config["lr"] = lr
        elif "lr" in self.train_config:
            lr = self.train_config["lr"]
        else:
            warnings.warn("No lr provided. Defaulting to 1e-4", UserWarning)
            lr = self.train_config["lr"] = 1e-4
                
        if evaluate_every is not None:
            self.train_config["evaluate_every"] = evaluate_every
        elif "evaluate_every" in self.train_config:
            evaluate_every = self.train_config["evaluate_every"]
        else:
            warnings.warn("No evaluate_every provided. Defaulting to -1 (no evaluation)", UserWarning)
            evaluate_every = self.train_config["evaluate_every"] = -1

        if log_every is not None:
            self.train_config["log_every"] = log_every
        elif "log_every" in self.train_config:
            log_every = self.train_config["log_every"]
        else:
            log_every = self.train_config["log_every"] = -1

        if optimizer is not None:
            if isinstance(optimizer, torch.optim.Optimizer):
                optimizer = optimizer([param for param in self.model.parameters() if param.requires_grad], lr=lr)
            elif isinstance(optimizer, str):
                opt_name = self.train_config["optimizer"] = optimizer
            else:
                raise ValueError("optimizer must be torch.optim.Optimizer or str")
        elif "optimizer" in self.train_config:
            opt_name = self.train_config["optimizer"]
            if not isinstance(opt_name, str):
                raise ValueError("train_config['optimizer'] must be 'str'")
        else:
            warnings.warn("No optimizer provided. Defaulting to AdamW", UserWarning)
            opt_name = self.train_config["optimizer"] = "AdamW"

        if opt_name == "AdamW":
            optimizer = torch.optim.AdamW([param for param in self.model.parameters() if param.requires_grad], 
                                          lr=lr, betas=(0.8, 0.9))
        elif opt_name == "Adam":
            optimizer = torch.optim.Adam([param for param in self.model.parameters() if param.requires_grad], 
                                         lr=lr, betas=(0.8, 0.9))
        elif opt_name == "SGD":
            optimizer = torch.optim.SGD([param for param in self.model.parameters() if param.requires_grad], lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}. Available options are ['AdamW', 'Adam', 'SGD']")

        if reduce_lr_each_epoch is not None:
            self.train_config["reduce_lr_each_epoch"] = reduce_lr_each_epoch
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=reduce_lr_each_epoch)
        elif reduce_lr_each_epoch in self.train_config:
            reduce_lr_each_epoch = self.train_config["reduce_lr_each_epoch"]
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=reduce_lr_each_epoch)
        else:
            scheduler = None

        losses = []
        metrics = []
        n_iter = 0
        cur_epoch = 0.0
        device = next(self.model.parameters()).device

        self.model.train()

        with tqdm(range(max_iters if max_iters > 0 else len(train_loader.dataset) * max_epochs), 
                    desc="Training iters") as pbar:

            while True:
                for i, batch in enumerate(train_loader):
                    
                    optimizer.zero_grad()

                    inputs, labels = batch[0].to(device), batch[1].to(device)
                    outputs = self.model.forward(inputs)
                    loss = F.cross_entropy(outputs, labels)

                    loss.backward()

                    optimizer.step()

                    losses.append(loss.detach().cpu().numpy())

                    n_iter += len(batch[-1])
                    pbar.update(len(batch[-1]))
                    cur_epoch = n_iter / len(train_loader.dataset)
                    
                    if evaluate_every > 0 and (i + 1) % evaluate_every == 0:
                        eval_metrics = self.evaluate(val_loader)
                        metrics.append(eval_metrics)
                        self.model.train()

                    if self.train_config["verbose"]:
                        if log_every > 0 and (i + 1) % log_every == 0:
                            if evaluate_every > 0:
                                print(f"Loss: {losses[-1]:.3e}, Last validation accuracy: {eval_metrics.get('Accuracy', -1):.4f}, Epoch: {cur_epoch:.3f}")
                            else:
                                print(f"Loss: {losses[-1]:.3e}, Epoch: {cur_epoch:.3f}")

                    if max_iters > 0 and n_iter >= max_iters:
                        break
                    if max_epochs > 0 and cur_epoch >= max_epochs:
                        break

                if scheduler is not None:
                    scheduler.step()

                if max_iters > 0 and n_iter >= max_iters:
                    break
                if max_epochs > 0 and cur_epoch >= max_epochs:
                    break

            if self.train_config["verbose"]:
                if log_every > 0:
                    if evaluate_every > 0:
                        print(f"Loss: {losses[-1]:.3e}, Last validation accuracy: {eval_metrics.get('Accuracy', -1):.4f}, Epoch: {cur_epoch:.3f}")
                    else:
                        print(f"Loss: {losses[-1]:.3e}, Epoch: {cur_epoch:.3f}")

        self._metrics = metrics
        self._losses = losses

        return self.model

    def evaluate(
        self,
        val_loader=None,
        model=None
    ):  
        if model is None:
            model = self.model

        if val_loader is not None:
            with torch.no_grad():
                model.eval()
                device = next(model.parameters()).device

                pred_labels = []
                real_labels = []
                for batch in tqdm(val_loader, leave=False, desc="eval batch"):

                    inputs, labels = batch[0].to(device), batch[1]
                    outputs = model.forward(inputs)

                    pred_labels.append(torch.argmax(outputs, dim=1).cpu())
                    real_labels.append(labels)

                pred_labels = torch.cat(pred_labels).view(-1)
                real_labels = torch.cat(real_labels).view(-1)
                
                accuracy = (pred_labels == real_labels).numpy().mean().item()

                return {"Accuracy": accuracy}
        else:
            warnings.warn("No validation data provided, return empty metrics", UserWarning)
            return {}   