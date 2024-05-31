SRC = $(wildcard ./*.cpp)                                     #wildcard把指定目录下的所有后缀是cpp的文件全部展开
#SRC = mymain.c
OBJS=$(patsubst %.cpp,%.o,$(SRC))                              #所有.o的变量

CXX = g++ 
TARGET   = yolov8s_test                            #生成的目标变量
CFLAGS   = -Wall -O3 -std=c++0x -fpermissive -mfloat-abi=hard -mfpu=neon -mtune=cortex-a9
#CFLAGS  += -g
VIVANTE_SDK_INCLUDE  = ../vip_driver/sdk/include
VIVANTE_SDK_LIB_PATH = ../vip_driver/sdk/drivers
LIBS     = -L../vip_driver/sdk/drivers -lOpenVX -lOpenVXU -lCLC -lVSC -lGAL -lovxlib -lNNArchPerf -lArchModelSw -L/usr/lib/arm-linux-gnueabihf/blas -L/usr/lib/arm-linux-gnueabihf/lapack -L/usr/lib -L/lib/arm-linux-gnueabihf
LIBS += -ljpeg8

OPENCV_INCLUDE  = /usr/include
OPENCV_LIB_PATH = /usr/lib/arm-linux-gnueabihf
OPENCV_LIBS     = -lopencv_stitching -lopencv_video -lopencv_videostab -lopencv_photo -lopencv_flann -lopencv_ml -lopencv_features2d -lopencv_superres -lopencv_objdetect -lopencv_calib3d -lopencv_imgproc -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lblas -larmadillo -lpthread -lopencv_videoio -lviv_dc
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
