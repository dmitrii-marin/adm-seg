# Beyond Gradient Descent for Regularized Segmentation Losses
[Dmitrii Marin](http://maryin.net), [Meng Tang](https://cs.uwaterloo.ca/~m62tang/), [Ismail Ben Ayed](https://profs.etsmtl.ca/ibenayed/) and [Yuri Boykov](https://cs.uwaterloo.ca/~yboykov/)

Appears in *IEEE conference on Computer Vision and Pattern Recognition (CVPR), Long Beach, CA, 2019*

If you find our work useful in your research please consider citing our paper:
```
@InProceedings{ADM:cvpr19,
  author = {Dmitrii Marin and Meng Tang and Ismail Ben Ayed and Yuri Boykov},
  title = {Beyond Gradient Descent for Regularized Segmentation Losses},
  booktitle = {IEEE conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2019},
  address = {Long Beach, California}
}
```
[[pdf](https://cs.uwaterloo.ca/~yboykov/Papers/cvpr19_ADM.pdf)]

## ScribbleSup Dataset

Download original PASCAL VOC 2012 dataset:
http://host.robots.ox.ac.uk/pascal/VOC/

Download Scribble annotations:
https://jifengdai.org/downloads/scribble_sup/

## Compilation

In ```deeplab/code/``` rename ```Makefile.config.example``` into ```Makefile.config```. Edit ```Makefile.config ``` to set up the compilation. In particular, set ```USE_CUDNN := 1``` to use CUDA and set ```CUDA_DIR``` to point to your CUDA instalation; adjust ```INCLUDE_DIRS``` and ```LIBRARY_DIRS``` to include libraries BOOST, BLAS, etc. See the dependecy list [here](deeplab/code/cmake/Dependencies.cmake). Run ```make```.

## Training and Testing

Update variable ```ROOT``` in ```deeplab/exper/run_pascal_scribble.sh``` to point to the ScribbleSup dataset.

First, train and test a base model (with partial cross entropy only):

```bash
cd deeplab/exper
DEV_ID=0 bash -x ./run_pascal_scribble.sh
```
The model will be saved in ```pascal_scribble/model/deeplab_largeFOV/train_iter_9000.caffemodel```. The accuracy  should be approximately 55.8%.

Then, add regularization to the loss and train/test the model using ADM:
```bash
TRAIN_CONFIG=GRID-ADM MODEL=pascal_scribble/model/deeplab_largeFOV/train_iter_9000.caffemodel DEV_ID=0 ./run_pascal_scribble.sh
```
This should give accuracy of approximately 61.7%
