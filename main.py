from argparse import ArgumentParser
from attrdict import AttrDict

from experiment import Experiment
from common import Task, GNN_TYPE, STOP

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--task", dest="task", default=Task.NEIGHBORS_MATCH, type=Task.from_string, choices=list(Task),
                        required=False)
    parser.add_argument("--type", dest="type", default=GNN_TYPE.GCN, type=GNN_TYPE.from_string, choices=list(GNN_TYPE),
                        required=False)
    parser.add_argument("--dim", dest="dim", default=32, type=int, required=False)
    parser.add_argument("--depth", dest="depth", default=3, type=int, required=False)
    parser.add_argument("--num_layers", dest="num_layers", default=None, type=int, required=False)
    parser.add_argument("--train_fraction", dest="train_fraction", default=0.8, type=float, required=False)
    parser.add_argument("--max_epochs", dest="max_epochs", default=50000, type=int, required=False)
    parser.add_argument("--eval_every", dest="eval_every", default=100, type=int, required=False)
    parser.add_argument("--batch_size", dest="batch_size", default=1024, type=int, required=False)
    parser.add_argument("--accum_grad", dest="accum_grad", default=1, type=int, required=False)
    parser.add_argument("--stop", dest="stop", default=STOP.TRAIN, type=STOP.from_string, choices=list(STOP),
                        required=False)
    parser.add_argument("--patience", dest="patience", default=20, type=int, required=False)
    parser.add_argument("--loader_workers", dest="loader_workers", default=0, type=int, required=False)
    parser.add_argument('--last_layer_fully_adjacent', action='store_true')
    parser.add_argument('--no_layer_norm', action='store_true')
    parser.add_argument('--no_activation', action='store_true')
    parser.add_argument('--no_residual', action='store_true')
    parser.add_argument('--unroll', action='store_true', help='use the same weights across GNN layers')

    args = parser.parse_args()
    Experiment(args).run()


def get_fake_args(
        task=Task.NEIGHBORS_MATCH,
        type=GNN_TYPE.GCN,
        dim=32,
        depth=3,
        num_layers=None,
        train_fraction=0.8,
        max_epochs=50000,
        eval_every=100,
        batch_size=1024,
        accum_grad=1,
        patience=20,
        stop=STOP.TRAIN,
        loader_workers=0,
        last_layer_fully_adjacent=False,
        no_layer_norm=False,
        no_activation=False,
        no_residual=False,
        unroll=False,
):
    return AttrDict({
        'task': task,
        'type': type,
        'dim': dim,
        'depth': depth,
        'num_layers': num_layers,
        'train_fraction': train_fraction,
        'max_epochs': max_epochs,
        'eval_every': eval_every,
        'batch_size': batch_size,
        'accum_grad': accum_grad,
        'stop': stop,
        'patience': patience,
        'loader_workers': loader_workers,
        'last_layer_fully_adjacent': last_layer_fully_adjacent,
        'no_layer_norm': no_layer_norm,
        'no_activation': no_activation,
        'no_residual': no_residual,
        'unroll': unroll,
    })
