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

```bash
conda install -c conda-forge lightgbm
```

```bash
git clone --recursive https://github.com/Microsoft/LightGBM
cd LightGBM
mkdir build
cd build
cmake -DCMAKE_GENERATOR_PLATFORM=x64 -DUSE_MPI=ON ..
cmake --build . --target ALL_BUILD --config Release
```

```
git clone --recursive https://github.com/Microsoft/LightGBM
cd LightGBM
mkdir build
cd build
cmake -DCMAKE_GENERATOR_PLATFORM=x64 ..
# cmake -DCMAKE_GENERATOR_PLATFORM=x64 -DUSE_MPI=ON ..
cmake --build . --target ALL_BUILD --config Release
```
