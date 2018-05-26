# Gradient Boosting Package Installation Guide for windows

본 문서는 대표적인 Gradient Boosting Package인 XGBoost와 LightGBM의 Windows 설치를 안내합니다. 두 패키지의 설치를 위해서는 conda를 이용한 설치와 컴파일후 pip로 설치하는 두 가지 방법이 있습니다.

## prerequiste
패키지 설치를 위해서는 아래와 같은 도구들의 준비가 필요합니다.

- git(https://git-scm.com/)
- cmake(https://cmake.org/download/)
- .Net Core SDK(https://www.microsoft.com/net/download/windows)
- .NET Framework Develop Pack
(https://www.microsoft.com/net/download/windows)


## XGBoost installation guide
### conda
conda 설치는 아래와 같이 간단한 명령어 설치 됩니다. 단 컴퓨터 사항에 따라 설치가 되지 않을 수 도 있습니다.
```bash
activate ml #가상환경 호출
conda install -c mndrake xgboost
```
### Install from source code
source code를 사용해서 설치를 할 경우 `cmd창에서` 아래와 같은 명령어를 입력합니다.

#### git clone
```bash
git clone --recursive https://github.com/dmlc/xgboost

cd xgboost
git submodule init
git submodule update
```

#### build
```bash
mkdir build
cd build
cmake .. -G"Visual Studio 15 2017 Win64"
cmake --build . --target xgboost --config Release
cd..
```

#### python 설치
파이썬 설치전 반드시 가상환경 호출 필요
```bash
activae ml #가상환경 호출
cd python-package
python setup.py install
```


## lightgbm installation guide
### conda
conda 설치는 아래와 같이 간단한 명령어 설치 됩니다. 단 컴퓨터 사항에 따라 설치가 되지 않을 수 도 있습니다.

```bash
activate ml #가상환경 호출
conda install -c conda-forge lightgbm
```

### Install from source code
source code를 사용해서 설치를 할 경우 `cmd창에서` 아래와 같은 명령어를 입력합니다.

#### git clone
```bash
git clone --recursive https://github.com/Microsoft/LightGBM
```

#### build
```bash
cd LightGBM
mkdir build
cd build
cmake -DCMAKE_GENERATOR_PLATFORM=x64 ..
cmake --build . --target ALL_BUILD --config Release
```

#### python 설치
파이썬 설치전 반드시 가상환경 호출 필요
```bash
cd ..
activae ml #가상환경 호출
cd python-package
python setup.py install
```
