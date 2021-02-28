from utils import DEVICE, OUTPUT_DIR

from torch import nn, optim
import numpy as np
import torch

from typing import List
import copy
import time
import os


# modified function from
# https://github.com/Roll920/ThiNet_Code/blob/728ce546f2ab92982e97c3e681ba8d88828509de/ThiNet_ICCV/compress_model.py
def select_filters(x, y, comp_rate):
    x, y = np.mat(x), np.mat(y)
    to_keep = int(x.shape[1] * comp_rate)
    to_remove = int(x.shape[1]) - to_keep
    remove_idx = []

    x_tmp = 0
    for i in range(to_remove):
        min_value = float('inf')
        for j in range(x.shape[1]):
            if j not in remove_idx:
                tmp_value = np.linalg.norm(x_tmp + x[:, j])
                if tmp_value < min_value:
                    min_value = tmp_value
                    min_idx = j
        remove_idx.append(min_idx)
        x_tmp += x[:, min_idx]
        print('Remove num={0}, channel num={1}, i={2}, loss={3:.3f}'
              .format(to_remove, x.shape[1], i, min_value))

    idx = set(range(x.shape[1])) - set(remove_idx)
    idx = np.array(list(idx))
    idx = np.sort(idx)

    selected_x = x[:, idx]
    w = (selected_x.T * selected_x).I * (selected_x.T * y)
    w = np.array(w)

    loss = np.linalg.norm(y - selected_x * w)
    print('loss before w={0:.3f}, loss with w={1:.3f}'
          .format(np.linalg.norm(y - np.sum(selected_x, 1)), loss))

    return idx, w


# TODO
def select_features():
    pass


# TODO
def compress_layer() -> None:
    pass


class FineTuner:
    def __init__(
            self,
            dataloaders: dict,
            criterion: nn.Module,
            optimizer: optim.Optimizer,
    ) -> None:
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.softmax = nn.Softmax(dim=1)

    def tune(self, model: nn.Module, num_epochs: int = 2) -> nn.Module:
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 15)

            for phase in ['val', 'train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for sample in self.dataloaders[phase]:
                    inputs = sample[0].to(DEVICE)
                    labels = sample[1].to(DEVICE)

                    # reset the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        outputs = model(inputs).to(DEVICE)
                        loss = self.criterion(outputs, labels).to(DEVICE)
                        _, preds = torch.max(self.softmax(outputs), 1)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(self.dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc)
                )

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            torch.save(
                best_model_wts,
                os.path.join(OUTPUT_DIR, 'pruned', 'resnet20.pth')
            )
            print()

        # Print final statistics
        time_elapsed = time.time() - since
        print('Fine tuning complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60)
        )
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model parameters
        model.load_state_dict(best_model_wts)

        return model


# TODO: not tested; test
def prune(
        model: nn.Module,
        dataloaders: dict,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        comp_rate: float,
        layers: List[str, ...],
        tune_epochs: int = 2,
        inplace: bool = False
) -> nn.Module:
    if not inplace:
        model_pruned = copy.deepcopy(model)
    else:
        model_pruned = model

    fine_tuner = FineTuner(dataloaders, criterion, optimizer)

    for name, param in model_pruned.named_parameters():
        if name in layers:
            for layer_name, layer_param in model_pruned.named_parameters()[name]:
                print('Pruning {}'.format(name + '/' + layer_name))
                x, y = select_features(layer_param)
                filter_idx = select_filters(x, y, layer_param, comp_rate)
                compress_layer(filter_idx)
                model_pruned = fine_tuner.tune(model, tune_epochs)

    return model_pruned
