import os
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../tools/utils/")
from metric import MetricCollector

class Test:
    """Tests the ``model`` on the specified test dataset using the
    data loader, and loss criterion.

    Keyword arguments:
    - model (``nn.Module``): the model instance to test.
    - data_loader (``Dataloader``): Provides single or multi-process
    iterators over the dataset.
    - criterion (``Optimizer``): The loss criterion.
    - metric (```Metric``): An instance specifying the metric to return.
    - device (``torch.device``): An object representing the device on which
    tensors are allocated.

    """

    def __init__(self, model, data_loader, criterion, metric, device, deterministic=False):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.metric = metric
        self.device = device
        self.deterministic = deterministic

    def run_epoch(self, iteration_loss=False):
        """Runs an epoch of validation.

        Keyword arguments:
        - iteration_loss (``bool``, optional): Prints loss at every step.

        Returns:
        - The epoch loss (float), and the values of the specified metrics

        """
        self.model.eval()
        epoch_loss = 0.0
        self.metric.reset()
        for step, batch_data in enumerate(self.data_loader):
            # Get the inputs and labels
            inputs = batch_data[0].to(self.device, non_blocking=True)
            labels = batch_data[1].to(self.device, non_blocking=True)

            import torch
            with torch.no_grad():
                # Forward propagation
                outputs = self.model(inputs)

                # Loss computation
                if self.deterministic:
                    import torch
                    torch.use_deterministic_algorithms(False)
                    loss = self.criterion(outputs, labels)
                    torch.use_deterministic_algorithms(True)
                    # loss = self.criterion.cpu()(outputs.cpu(), labels.cpu()).cuda()
                else:
                    loss = self.criterion(outputs, labels)

            # Keep track of loss for current epoch
            epoch_loss += loss.item()

            # Keep track of evaluation the metric
            self.metric.add(outputs.detach(), labels.detach())

            if iteration_loss:
                print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))

        metric_collector = MetricCollector(enable_only_avglog=True)
        metric_collector.insert_metrics(net = "ENet",
                                        accuracy = [self.metric.value()[1]])
        if (int(os.environ['RANK']) == 0):
            metric_collector.dump()
        return epoch_loss / len(self.data_loader), self.metric.value()
