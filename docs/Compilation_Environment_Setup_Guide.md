# Compilation Environment Setup Guide
It`s needed to set up a cross-compilation environment before compiling the C/C++ Demo of examples in this project on the x86 Linux system.
## Linux Platform

### Download cross-compilation tools
*(If the cross-compilation tool is already installed on your system, please ignore this step)*
1. Different system architectures rely on different cross-compilation tools.The following are download links for cross-compilation tools recommended for different system architectures.：
   - armhf:https://releases.linaro.org/components/toolchain/binaries/7.5-2019.12/arm-linux-gnueabihf/gcc-linaro-7.5.0-2019.12-x86_64_arm-linux-gnueabihf.tar.xz
2. Decompress the downloaded cross-compilation tool and remember the specific path, which will be used later during compilation.
### Compile C/C++ Demo
The command reference for compiling C/C++ Demo is as follows：
```shell
# go to the rknn_model_zoo root directory
cd <orbita_model_zoo_root_path>

# if GCC_COMPILER not found while building, please set GCC_COMPILER path
export GCC_COMPILER=<GCC_COMPILER_PATH>

# for Yulong810A
make
```

*Description:*
- `<GCC_COMPILER_PATH>`: Specify the cross-compilation path. Different system architectures use different cross-compilation tools.
    - `GCC_COMPILE_PATH` examples:
        - armhf: /opt/gcc-linaro-7.5.0-2019.12-x86_64_arm-linux-gnueabihf/bin