import main
from common import Task, STOP, GNN_TYPE
from attrdict import AttrDict
from experiment import Experiment
import torch

override_params = {
    2: {'batch_size': 64, 'eval_every': 1000},
    3: {'batch_size': 64},
    4: {'batch_size': 1024},
    5: {'batch_size': 1024},
    6: {'batch_size': 1024},
    7: {'batch_size': 2048},
    8: {'batch_size': 1024, 'accum_grad': 2},  # effective batch size of 2048, with less GPU memory
}


class Results:
    def __init__(self, train_acc, test_acc, epoch):
        self.train_acc = train_acc
        self.test_acc = test_acc
        self.epoch = epoch


if __name__ == '__main__':

    task = Task.NEIGHBORS_MATCH
    gnn_type = GNN_TYPE.GAT
    stopping_criterion = STOP.TRAIN
    min_depth = 2
    max_depth = 8

    results_all_depths = {}
    for depth in range(min_depth, max_depth + 1):
        num_layers = depth + 1
        args = main.get_fake_args(task=task, depth=depth, num_layers=num_layers, loader_workers=7,
                                  type=gnn_type, stop=stopping_criterion,
                                  no_activation=True, no_residual=False)
        if depth in override_params:
            for key, value in AttrDict(override_params[depth]).items():
                args[key] = value
        train_acc, test_acc, epoch = Experiment(args).run()
        torch.cuda.empty_cache()
        results_all_depths[depth] = Results(train_acc=train_acc, test_acc=test_acc, epoch=epoch)
        print()

    print(f'Task: {task}')
    print('depth, train_acc, test_acc, epoch, train_acc, test_acc, epoch,')
    for depth in range(min_depth, max_depth + 1):
        res = results_all_depths[depth]
        print(f'{depth}, {res.train_acc}, {res.test_acc}, {res.epoch}')
