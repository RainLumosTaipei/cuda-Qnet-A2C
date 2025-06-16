```sh
nvcc -m64 -shared -o qnet.dll qnet.cu -Xcompiler "/MD"
nvcc -m64 -arch=sm_120 -o qnet.dll -Xcompiler "/LD /D_USRDLL /D_WINDLL /utf-8" -lcublas qnet.cu
nvcc -m64 -arch=sm_120 -o a2c.dll -Xcompiler "/LD /D_USRDLL /D_WINDLL /utf-8" -lcublas a2c.cu
```

x64 Native Tools Command Prompt for VS 2022

```sh
conda env list
conda create -n myenv python=3.9
conda env remove -n myenv
```