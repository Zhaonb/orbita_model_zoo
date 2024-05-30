//
// Created by OBT-AI15 on 2024/5/30.
//

#ifndef YOLOV8S_TEST_UINT8_DETECTION_OBB_H
#define YOLOV8S_TEST_UINT8_DETECTION_OBB_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <limits>
#include <iomanip>
#include <locale>
#include <iostream>
#include <chrono>

#include "vsi_nn_pub.h"

#include "vnn_global.h"
#include "vnn_post_process.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define ZQ_MAX(a, b) ((a) > (b) ? (a) : (b))
#define ZQ_MIN(a, b) ((a) < (b) ? (a) : (b))

typedef struct
{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float score;
    int classId;
} DetectRect;

vsi_status show_result
        (
                vsi_nn_graph_t *graph,
                vsi_nn_tensor_t *tensor,
                char* image_path
        );

#endif //YOLOV8S_TEST_UINT8_DETECTION_OBB_H
