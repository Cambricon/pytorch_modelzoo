import os
import sys
from torch.cuda.amp import autocast
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../../../../tools/utils/")
from metric import MetricCollector

class Train:
    """Performs the training of ``model`` given a training dataset data
    loader, the optimizer, and the loss criterion.

    Keyword arguments:
    - model (``nn.Module``): the model instance to train.
    - data_loader (``Dataloader``): Provides single or multi-process
    iterators over the dataset.
    - optim (``Optimizer``): The optimization algorithm.
    - criterion (``Optimizer``): The loss criterion.
    - metric (```Metric``): An instance specifying the metric to return.
    - device (``torch.device``): An object representing the device on which
    tensors are allocated.

    """

    def __init__(self, model, data_loader, optim, criterion, metric, device):
        self.model = model
        self.data_loader = data_loader
        self.optim = optim
        self.criterion = criterion
        self.metric = metric
        self.device = device

    def run_epoch(self, epoch, warmup_scheduler, args, iteration_loss=False):
        """Runs an epoch of training.

        Keyword arguments:
        - iteration_loss (``bool``, optional): Prints loss at every step.

        Returns:
        - The epoch loss (float).

        """
        self.model.train()
        epoch_loss = 0.0
        self.metric.reset()

        metric_collector = MetricCollector(
            enable_only_benchmark=True,
            record_elapsed_time=True,
            record_hardware_time=True if args.device == 'mlu' else False)
        metric_collector.place()
        for step, batch_data in enumerate(self.data_loader):
            if step == args.iters:
                break
            # Get the inputs and labels
            inputs = batch_data[0].to(self.device, non_blocking=True)
            labels = batch_data[1].to(self.device, non_blocking=True)

            with autocast(enabled=args.pyamp):
                # Forward propagation
                outputs = self.model(inputs)
    
                # Loss computation
                if args.deterministic:
                    import torch
                    torch.use_deterministic_algorithms(False)
                    loss = self.criterion(outputs, labels)
                    torch.use_deterministic_algorithms(True)
                else:
                    loss = self.criterion(outputs, labels)

            # Backpropagation
            self.optim.zero_grad()
            if args.pyamp:
                args.scaler.scale(loss).backward()
            else:
                loss.backward()
            if args.pyamp:
                args.scaler.step(self.optim)
                args.scaler.update()
            else:
                self.optim.step()
            if epoch < args.warmup:
                warmup_scheduler.step()

            # Keep track of loss for current epoch
            epoch_loss += loss.item()

            # MetricCollector record
            metric_collector.record()
            metric_collector.place()
            # Keep track of the evaluation metric
            self.metric.add(outputs.detach(), labels.detach())

            if iteration_loss:
                print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))

        if args.pyamp:
            precision = "amp"
        else:
            precision = "fp32"
        metric_collector.insert_metrics(
            net = "ENet",
            batch_size = args.batch_size,
            precision = precision,
            cards = int(os.environ['WORLD_SIZE']) if args.rank == 0 else 1,
            DPF_mode = "ddp " if args.distributed == True else "single")
        if ((args.distributed == False) or (args.rank == 0)):
            metric_collector.dump()
        return epoch_loss / len(self.data_loader), self.metric.value()
