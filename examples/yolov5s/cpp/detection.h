#ifndef __YOLO_LAYER_H_
#define __YOLO_LAYER_H_

#include <string>
#include <vector>
#include "vsi_nn_pub.h"
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

#define MAX_NUM 100
typedef struct stDetectResult {
  char name[MAX_NUM][32];
  int total_num;
  CvPoint pt1[MAX_NUM];
  CvPoint pt2[MAX_NUM];
}DetectResult;

typedef struct{
    float x,y,w,h;
}box;

typedef struct{
    box bbox;
    int classes;
    float* prob;
    float* mask;
    float objectness;
    int sort_class;
}detection;

typedef struct layer{
    int batch;
    int total;
    int n,c,h,w;
    int out_n,out_c,out_h,out_w;
    int classes;
    int inputs,outputs;
    int *mask;
    float* biases;
    float* output;
    float* output_gpu;
}layer;

typedef struct blob{
	int w;
	int h;
	float *data;
}blob;

typedef struct{
    float dx, dy, dw, dh;
} dbox;

typedef struct yolo_type{
	char *image_name;
	char *class_names;
	int classes_number;
	float thresh;
	float nms;
	int img_w;
	int img_h;
    int *mask;
    float* biases;
}yolo_type;

vsi_status show_result(vsi_nn_graph_t *graph,vsi_nn_tensor_t *tensor);

#endif
