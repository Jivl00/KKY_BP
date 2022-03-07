# Vytvoření conda virtuálního prostředí
- install: https://docs.conda.io/en/latest/miniconda.html + přidat do path 
```
conda create –n ‘nazev‘ python=xx (verze)
```
# HelloWorld v JupyterLab / IDE
- JupyterLab:
```
conda install – c conda–forge jupyterlab
```
- PyCharm: https://www.jetbrains.com/pycharm/download/ 

```
print('Hello, world!')
```
# Instalace Keras/ Tensorflow
```
conda create ––name tf
conda activate tf
conda install tensorflow–gpu
```
# Instalace Scikit – Learn
```
conda install scikit-learn
```
## End-to-end MNIST úkol v Kerasu
https://keras.io/examples/vision/mnist_convnet/

---