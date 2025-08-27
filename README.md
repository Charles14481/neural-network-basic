# Neural Network Basic
Basic multi-layer perceptron model for regression. Nice explanation of MLPs by 3Blue1Brown under resources. I wanted to stop treating deep learning as a black box, so I did this to better understand the calculus and architecture behind similar models. Very fun experience, 10/10 recommend.

**Files:**
- `model` is the main class that creates the network of neurons (nodes) and orchestrates training/testing. 
- `node` is the object class with basic operations; each node in the graph is an instance of this class
- `datahandler` handles retrieval of data and does z-score normalization among other things
- `ProgressDisplay` instance can be passed to display, showing a graph of the model loss over time
- `graph` displays the model with labeled nodes and edges. Originally meant for debugging purposes but it ended up just being a nice warmup and visual.

**Customization options:**
- There are self-explanatory parameters (ex. width, layers, target)
- Note that normalization currently only supports `None` and `"zscore"`
- Change type strings in the node activation method to use a different activation method. `leaky_relu` for hidden layers and `linear` for output layers by default, but there are some other pre-existing choices.
- The learning rate scheduler can be changed (or set to `None`), but switching scheduler will take a bit of work. Currently using step, `weird` is an updater I saw in a proof but do not know much about.

Datasets:
- [House price prediction](https://www.kaggle.com/datasets/shree1992/housedata): data.csv in files with basic info of homes. data-modified is the improved version where homes with price=0 were removed.
- [California Housing Prices](https://www.kaggle.com/code/ahmedmahmoud16/california-housing-prices): housing.csv, gives location and summary statistics of nearby homes. Model performed worse on this dataset (with same hyper-parameters)
## Next Steps
I would like to implement drop-out regularization and more advanced LR schedulers like Adam. Currently quite slow even for small models, could benefit from multi-core/GPU utilization. 
Using `numpy` linear algebra would probably make the code much faster, but I wanted to use as much pure python as possible so I could learn more about MLPs. Mainly used `numpy` for arrays because I come from `Java` and they are faster (I think).
## Resources
http://neuralnetworksanddeeplearning.com/  
https://youtu.be/aircAruvnKk?si=wWsKyu0jc64QYHLr  
Wikipedia and G4G
