Files layout
============
sdk
|
+---drivers
    |  galcore.ko
    |  libCLC.so
    |  libEGL.so -> libEGL.so.1.5.0
    |  libEGL.so.1 -> libEGL.so.1.5.0
    |  libEGL.so.1.5.0
    |  libGAL.so
    |  libgbm.so -> libgbm.so.1.0.0
    |  libgbm.so.1 -> libgbm.so.1.0.0
    |  libgbm.so.1.0.0
    |  libgbm_viv.so
    |  libGLES_CL.so -> libGLES_CL.so.1.1.0
    |  libGLES_CL.so.1 -> libGLES_CL.so.1.1.0
    |  libGLES_CL.so.1.1.0
    |  libGLES_CM.so -> libGLES_CM.so.1.1.0
    |  libGLES_CM.so.1 -> libGLES_CM.so.1.1.0
    |  libGLES_CM.so.1.1.0
    |  libGLESv1_CL.so -> libGLESv1_CL.so.1.1.0
    |  libGLESv1_CL.so.1 -> libGLESv1_CL.so.1.1.0
    |  libGLESv1_CL.so.1.1.0
    |  libGLESv1_CM.so -> libGLESv1_CM.so.1.1.0
    |  libGLESv1_CM.so.1 -> libGLESv1_CM.so.1.1.0
    |  libGLESv1_CM.so.1.1.0
    |  libGLESv2.so -> libGLESv2.so.2.0.0
    |  libGLESv2.so.2 -> libGLESv2.so.2.0.0
    |  libGLESv2.so.2.0.0
    |  libGLSLC.so
    |  libGL.so -> libGL.so.1.2.0
    |  libGL.so.1 -> libGL.so.1.2.0
    |  libGL.so.1.2.0
    |  libOpenCL.so -> libOpenCL.so.3.0.0
    |  libOpenCL.so.1 -> libOpenCL.so.3.0.0
    |  libOpenCL.so.3 -> libOpenCL.so.3.0.0
    |  libOpenCL.so.3.0.0
    |  libOpenVG.3d.so.1.1.0
    |  libOpenVG.so -> libOpenVG.3d.so.1.1.0
    |  libOpenVG.so.1 -> libOpenVG.3d.so.1.1.0
    |  libSPIRV_viv.so
    |  libVDK.so -> libVDK.so.1.2.0
    |  libVDK.so.1 -> libVDK.so.1.2.0
    |  libVDK.so.1.2.0
    |  libVSC.so
    |  libvulkan.so -> libvulkan.so.1.1.6
    |  libvulkan.so.1 -> libvulkan.so.1.1.6
    |  libvulkan.so.1.1.6
    |  libOpenVX.so
|
\---include
    |  gc_vdk_types.h
    |  gc_vdk.h
    |
    +---HAL
    +---CL
    +---EGL
    +---GL
    +---GLES
    +---GLES2
    +---GLES3
    +---KHR
    +---SPIRV
    +---VG
    +---vulkan
    +---VX

Running applications on the target machine
==========================================

1. Install the libraries and symbolic links to the target system.

2. Install the kernel driver
    insmod galcore.ko registerMemBase=<REG_MEM_BASE> irqLine=<IRQ> contiguousSize=<CONTIGUOUS_MEM_SIZE>

    Ex. On an ARM development board:
    insmod galcore.ko registerMemBase=0x80000000 irqLine=104 contiguousSize=0x400000

3. Run the application

    cd $SDK_DIR/samples/vdk
    ./tutorial1


