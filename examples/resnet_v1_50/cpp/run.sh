#!/bin/bash
BASE_PATH=../../vip_driver
OPENCV_HOME=$BASE_PATH/opencv
VIVANTE_SDK_DIR=$BASE_PATH/sdk
VIP_SDK_DRIVERS_PATH=$VIVANTE_SDK_DIR/drivers
GALCORE_DRIVER_NAME="galcore"

export VNN_LOOP_TIME=10
export VIV_VX_ENABLE_TP=1
export VSI_USE_IMAGE_PROCESS=1
#export VIV_VX_ENABLE_SAVE_NETWORK_BINARY=1

if lsmod | grep -q $GALCORE_DRIVER_NAME; then
	echo "Driver module $GALCORE_DRIVER_NAME is loaded, uninstalling..."
    	rmmod galcore
    	echo "old galcore rmmod"
fi
sleep 3
insmod  $VIP_SDK_DRIVERS_PATH/galcore.ko  registerMemBase=0xf8800000 contiguousSize=0x200000 irqLine=21 showArgs=1 recovery=1 stuckDump=2 powerManagement=1
export VIVANTE_SDK_DIR=$VIVANTE_SDK_DIR
export LD_LIBRARY_PATH=$VIP_SDK_DRIVERS_PATH:$OPENCV_HOME/lib
./resnetv150uint8 resnet_v1_50_uint8.export.data space_shuttle_224x224.jpg
if [ $? -eq 0 ];then
	echo "test counts:"
	echo $VNN_LOOP_TIME
	echo "test successful!"
else
	echo "test counts:"
	echo $VNN_LOOP_TIME
	echo "test failed!"
fi


