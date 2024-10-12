BASE_PATH=../../vip_driver
OPENCV_HOME = $(BASE_PATH)/opencv
VIVANTE_SDK_DIR=$(BASE_PATH)/sdk

TARGET_ARCH1 := arm
ifneq ($(shell uname -m | grep -c 'x86'), 0)
	TARGET_ARCH1 := x86
endif

ifeq ($(TARGET_ARCH1),arm)
	CXX = g++
	CC = gcc
else
	CXX = arm-linux-gnueabihf-g++
	CC = arm-linux-gnueabihf-gcc
endif

SRC = $(wildcard ./*.c)                                     #wildcard把指定目录下的所有后缀是cpp的文件全部展开
OBJS=$(patsubst %.c,%.o,$(SRC))                              #所有.o的变量

TARGET   = resnetv150uint8                                    #生成的目标变量
CFLAGS   = -Wall -O3 -mfloat-abi=hard -mfpu=neon -mtune=cortex-a9
VIVANTE_SDK_INCLUDE  = $(VIVANTE_SDK_DIR)/include
VIVANTE_SDK_LIB_PATH = $(VIVANTE_SDK_DIR)/drivers
LIBS     = -L$(VIVANTE_SDK_DIR)/drivers -lOpenVX -lOpenVXU -lCLC -lVSC -lGAL -lovxlib -lNNArchPerf -lArchModelSw -ljpeg

OPENCV_INCLUDE  = $(OPENCV_HOME)/include
OPENCV_LIBS     = -L$(OPENCV_HOME)/lib -lopencv_stitching -lopencv_video -lopencv_videostab -lopencv_photo -lopencv_flann -lopencv_ml -lopencv_features2d -lopencv_superres -lopencv_objdetect -lopencv_calib3d -lopencv_imgproc -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lpthread -lopencv_videoio


INCLUDE += -I$(VIVANTE_SDK_INCLUDE) 
INCLUDE += -I$(OPENCV_INCLUDE)
CFLAGS  += $(INCLUDE)

LIBVAR   += $(VIVANTE_SDK_LIBS) $(OPENCV_LIBS) $(LIBS)              #指明需要链接静态库.a、动态库.so名称

all:$(TARGET) permission
permission:
	@chmod 777 run.sh

$(TARGET):$(OBJS)
	@echo Linking $(notdir $@)
	@$(CXX) -rdynamic -o $@ $^  $(LIBPATH) $(LIBVAR)    #rdynamic
	
%.o:%.cpp 
	@echo Compiling $<
	@$(CXX)  $(CFLAGS) -c $< -o $@ -fpermissive

.PHONY:clean
clean:
	rm -rf ./$(TARGET)
	rm -rf ./*.o
