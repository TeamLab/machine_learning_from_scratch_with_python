## XGBoost installation guide

```bash
conda install -c mndrake xgboost
```

### Install from source code
```bash
git clone --recursive https://github.com/dmlc/xgboost

git submodule init
git submodule update

mkdir build
cd build
cmake .. -G"Visual Studio 14 2015 Win64"
cd..
cd python-package; python setup.py install
```


## lightgbm installation guide

conda install -c conda-forge lightgbm
