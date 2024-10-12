#!/bin/bash

#export CNN_PERF=0
#export NN_EXT_SHOW_PERF=1
#export NN_LAYER_DUMP=1
export VIV_VX_ENABLE_TP=1
#export VIV_VX_PROFILE=1
if [ ! -f /dev/galcore ] ;then
  rmmod galcore
  echo "nothing"
fi
sleep 1
insmod ../vip_driver/sdk/drivers/galcore.ko showArgs=1 irqLine=21 registerMemBase=0xf8800000 contiguousSize=0x200000
export VIVANTE_SDK_DIR=../vip_driver/sdk
#export VIV_VX_ENABLE_SAVE_NETWORK_BINARY=0
export LD_LIBRARY_PATH=/usr/lib/arm-linux-gnueabihf/lapack:/usr/lib/arm-linux-gnueabihf:/lib/arm-linux-gnueabihf:/usr/lib:/lib:../vip_driver/sdk/drivers:/usr/lib/arm-linux-gnueabihf/blas/
#export VSI_USE_IMAGE_PROCESS=0
#export VNN_LOOP_TIME=10
export VIVANTE_SDK_DIR=../vip_driver/sdk
./yolov5s yolov5s_uint8.export.data dog_640.jpg
