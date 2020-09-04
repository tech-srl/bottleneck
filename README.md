# On the Bottleneck of Graph Neural Networks and its Practical Implications

This is the official implementation of the paper: [On the Bottleneck of Graph Neural Networks and its Practical Implications](https://arxiv.org/pdf/2006.05205) 

By [Uri Alon](http://urialon.cswp.cs.technion.ac.il/) and [Eran Yahav](http://www.cs.technion.ac.il/~yahave/).
See also the [[video]](https://youtu.be/vrLsEwzZTCQ) and the [[slides]](https://urialon.cswp.cs.technion.ac.il/wp-content/uploads/sites/83/2020/07/bottleneck_slides.pdf). 

This repository can be used to reproduce the experiments of 
Section 4.1 in the paper, for the "Tree-NeighborsMatch" problem. 

This project was designed to be useful in experimenting with new GNN architectures and new solutions for the bottleneck problem. 

Feel free to open an issue with any questions.


# The Tree-NeighborsMatch problem
![alt text](images/fig5.png "Figure 5 from the paper")

## Requirements

### Dependencies
This project is based on PyTorch 1.4.0 and the [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) library.
* First, install PyTorch from the official website: [https://pytorch.org/](https://pytorch.org/).
* Then install PyTorch Geometric: [https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
* Eventually, run the following to verify that all dependencies are satisfied:
```setup
pip install -r requirements.txt
```

The `requirements.txt` file lists the additional requirements.
 However, PyTorch Geometric might requires manual installation, and we thus recommend to use the 
`requirements.txt` file only afterward.


Verify that importing the dependencies goes without errors:
```
python -c 'import torch; import torch_geometric'
```



### Hardware
Training on large trees (depth=8) might require ~60GB of RAM and about 10GB of GPU memory.
GPU memory can be compromised by using a smaller batch size and using the `--accum_grad` flag.

For example, instead of running:
```
python main.py --batch_size 1024 --type GGNN
```

The following uses gradient accumulation, and takes less GPU memory:
```
python main.py --batch_size 512 --accum_grad 2 --type GGNN
```

## Reproducing Experiments

To run a single experiment from the paper, run:

```
python main.py --help
```
And see the available flags.
For example, to train a GGNN with depth=4, run:
```
python main.py --task DICTIONARY --eval_every 1000 --depth 4 --num_layers 5 --batch_size 1024 --type GGNN
```  

To train a GNN across all depths, run one of the following:
```
python run-gcn-2-8.py
python run-gat-2-8.py
python run-ggnn-2-8.py
python run-gin-2-8.py
```

## Results

The results of running the above scripts are (Section 4.1 in the paper):


![alt text](images/fig3.png "Figure 3 from the paper")


Depth:   | 2   	| 3   	| 4    	| 5    	| 6    	| 7    	| 8    	|
------	|-----	|-----	|------	|------	|------	|------	|------	|
 **GGNN** 	| 1.0 	| 1.0 	| 1.0  	| 0.60 	| 0.38 	| 0.21 	| 0.16 	|
 **GAT**  	| 1.0 	| 1.0 	| 1.0  	| 0.41 	| 0.21 	| 0.15 	| 0.11 	|
 **GIN**  	| 1.0 	| 1.0 	| 0.77 	| 0.29 	| 0.20 	|      	|      	|
 **GCN**  	| 1.0 	| 1.0 	| 0.70 	| 0.19 	| 0.14 	| 0.09 	| 0.08 	|

## Experiment with other GNN types
To experiment with other GNN types:
* Add the new GNN type to the `GNN_TYPE` enum [here](common.py#L34), for example: `MY_NEW_TYPE = auto()`
* Add another `elif self is GNN_TYPE.MY_NEW_TYPE:` to instantiate the new GNN type object [here](common.py#L47)
* Use the new type as a flag for the `main.py` file:
```
python main.py --type MY_NEW_TYPE ...
```

## Citations
If you want to cite this work, please use this bibtex entry:
```
@article{alon2020bottleneck,
  title={On the Bottleneck of Graph Neural Networks and its Practical Implications},
  author={Alon, Uri and Yahav, Eran},
  journal={arXiv preprint arXiv:2006.05205},
  year={2020}
}
```
