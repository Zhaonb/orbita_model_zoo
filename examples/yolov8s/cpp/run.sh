#!/bin/bash

export VNN_LOOP_TIME=2
export VIV_VX_ENABLE_TP=1
export VSI_USE_IMAGE_PROCESS=1
#export VIV_VX_ENABLE_SAVE_NETWORK_BINARY=1

if [ ! -b /dev/galcore ] ;then
  echo "The driver not loaded. Load the driver......"
  sleep 3
  insmod  ../vip_driver/sdk/drivers/galcore.ko  registerMemBase=0xf8800000 contiguousSize=0x200000 irqLine=21 showArgs=1 recovery=1 stuckDump=2 powerManagement=1
  export VIVANTE_SDK_DIR=../vip_driver/sdk
  export LD_LIBRARY_PATH=/usr/lib/arm-linux-gnueabihf/lapack:/usr/lib/arm-linux-gnueabihf:/lib/arm-linux-gnueabihf:/usr/lib:/lib:../vip_driver/sdk/drivers:/usr/lib/arm-linux-gnueabihf/blas/
  ./yolov8s_shape_slim_c yolov8s_shape_slim_c_uint8.export.data ./data/person_test2.jpg
fi
rmmod galcore
