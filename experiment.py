import torch
from torch_geometric.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
import random
from attrdict import AttrDict

from common import STOP
from models.graph_model import GraphModel


class Experiment():
    def __init__(self, args):
        self.task = args.task
        gnn_type = args.type
        self.depth = args.depth
        num_layers = self.depth if args.num_layers is None else args.num_layers
        self.dim = args.dim
        self.unroll = args.unroll
        self.train_fraction = args.train_fraction
        self.max_epochs = args.max_epochs
        self.batch_size = args.batch_size
        self.accum_grad = args.accum_grad
        self.eval_every = args.eval_every
        self.loader_workers = args.loader_workers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stopping_criterion = args.stop
        self.patience = args.patience

        seed = 11
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.X_train, self.X_test, dim0, out_dim, self.criterion = \
            self.task.get_dataset(self.depth, self.train_fraction)

        self.model = GraphModel(gnn_type=gnn_type, num_layers=num_layers, dim0=dim0, h_dim=self.dim, out_dim=out_dim,
                                last_layer_fully_adjacent=args.last_layer_fully_adjacent, unroll=args.unroll,
                                layer_norm=not args.no_layer_norm,
                                use_activation=not args.no_activation,
                                use_residual=not args.no_residual
                                ).to(self.device)

        print(f'Starting experiment')
        self.print_args(args)
        print(f'Training examples: {len(self.X_train)}, test examples: {len(self.X_test)}')

    def print_args(self, args):
        if type(args) is AttrDict:
            for key, value in args.items():
                print(f"{key}: {value}")
        else:
            for arg in vars(args):
                print(f"{arg}: {getattr(args, arg)}")
        print()

    def run(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', threshold_mode='abs', factor=0.5, patience=10)
        print('Starting training')

        best_test_acc = 0.0
        best_train_acc = 0.0
        best_epoch = 0
        epochs_no_improve = 0
        for epoch in range(1, (self.max_epochs // self.eval_every) + 1):
            self.model.train()
            loader = DataLoader(self.X_train * self.eval_every, batch_size=self.batch_size, shuffle=True,
                                pin_memory=True, num_workers=self.loader_workers)

            total_loss = 0
            total_num_examples = 0
            train_correct = 0
            optimizer.zero_grad()
            for i, batch in enumerate(loader):
                batch = batch.to(self.device)
                out = self.model(batch)
                loss = self.criterion(input=out, target=batch.y)
                total_num_examples += batch.num_graphs
                total_loss += (loss.item() * batch.num_graphs)
                _, train_pred = out.max(dim=1)
                train_correct += train_pred.eq(batch.y).sum().item()

                loss = loss / self.accum_grad
                loss.backward()
                if (i + 1) % self.accum_grad == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            avg_training_loss = total_loss / total_num_examples
            train_acc = train_correct / total_num_examples
            scheduler.step(train_acc)

            test_acc = self.eval()
            cur_lr = [g["lr"] for g in optimizer.param_groups]

            new_best_str = ''
            stopping_threshold = 0.0001
            stopping_value = 0
            if self.stopping_criterion is STOP.TEST:
                if test_acc > best_test_acc + stopping_threshold:
                    best_test_acc = test_acc
                    best_train_acc = train_acc
                    best_epoch = epoch
                    epochs_no_improve = 0
                    stopping_value = test_acc
                    new_best_str = ' (new best test)'
                else:
                    epochs_no_improve += 1
            elif self.stopping_criterion is STOP.TRAIN:
                if train_acc > best_train_acc + stopping_threshold:
                    best_train_acc = train_acc
                    best_test_acc = test_acc
                    best_epoch = epoch
                    epochs_no_improve = 0
                    stopping_value = train_acc
                    new_best_str = ' (new best train)'
                else:
                    epochs_no_improve += 1
            print(
                f'Epoch {epoch * self.eval_every}, LR: {cur_lr}: Train loss: {avg_training_loss:.7f}, Train acc: {train_acc:.4f}, Test accuracy: {test_acc:.4f}{new_best_str}')
            if stopping_value == 1.0:
                break
            if epochs_no_improve >= self.patience:
                print(
                    f'{self.patience} * {self.eval_every} epochs without {self.stopping_criterion} improvement, stopping. ')
                break
        print(f'Best train acc: {best_train_acc}, epoch: {best_epoch * self.eval_every}')

        return best_train_acc, best_test_acc, best_epoch

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            loader = DataLoader(self.X_test, batch_size=self.batch_size, shuffle=False,
                                pin_memory=True, num_workers=self.loader_workers)

            total_correct = 0
            total_examples = 0
            for batch in loader:
                batch = batch.to(self.device)
                _, pred = self.model(batch).max(dim=1)
                total_correct += pred.eq(batch.y).sum().item()
                total_examples += batch.y.size(0)
            acc = total_correct / total_examples
            return acc
