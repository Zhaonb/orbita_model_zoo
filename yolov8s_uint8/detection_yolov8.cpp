//
// Created by OBT-AI15 on 2024/5/30.
//
/*-------------------------------------------
                Includes
-------------------------------------------*/
#include "detection_yolov8.h"
/*-------------------------------------------
                  Variable definitions
-------------------------------------------*/
std::vector<float> meshgrid;
std::vector<float> regdfl;
float regdeq[16] = {0};

const int class_num = 80;
int headNum = 3;

int input_w = 640;
int input_h = 640;
int strides[3] = {8, 16, 32};
int mapSize[3][2] = {{80, 80}, {40, 40}, {20, 20}};

float nmsThresh = 0.45;
float objectThresh = 0.1;

char *class_names[] = {"person","bicycle", "car", "motorcycle", "airplane", "bus",
                       "train", "truck", "boat", "traffic light", "fire hydrant",
                       "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                       "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                       "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                       "skis", "snowboard", "sports ball", "kite", "baseball bat",
                       "baseball glove", "skateboard", "surfboard", "tennis racket",
                       "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                       "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                       "hot dog", "pizza", "donut", "cake", "chair", "couch",
                       "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                       "mouse", "remote", "keyboard", "cell phone", "microwave",
                       "oven", "toaster", "sink", "refrigerator", "book", "clock",
                       "vase", "scissors", "teddy bear","hair drier", "toothbrush"};

// 定义类别颜色
vector<Scalar> CLASS_COLORS = {
        Scalar(220, 20, 60), Scalar(119, 11, 32), Scalar(0, 0, 142), Scalar(0, 0, 230),
        Scalar(106, 0, 228), Scalar(0, 60, 100), Scalar(0, 80, 100), Scalar(0, 0, 70),
        Scalar(0, 0, 192), Scalar(250, 170, 30), Scalar(100, 170, 30), Scalar(220, 220, 0),
        Scalar(175, 116, 175), Scalar(250, 0, 30), Scalar(165, 42, 42), Scalar(255, 77, 255),
        Scalar(0, 226, 252), Scalar(182, 182, 255), Scalar(0, 82, 0), Scalar(120, 166, 157),
        Scalar(110, 76, 0), Scalar(174, 57, 255), Scalar(199, 100, 0), Scalar(72, 0, 118),
        Scalar(255, 179, 240), Scalar(0, 125, 92), Scalar(209, 0, 151), Scalar(188, 208, 182),
        Scalar(0, 220, 176), Scalar(255, 99, 164), Scalar(92, 0, 73), Scalar(133, 129, 255),
        Scalar(78, 180, 255), Scalar(0, 228, 0), Scalar(174, 255, 243), Scalar(45, 89, 255),
        Scalar(134, 134, 103), Scalar(145, 148, 174), Scalar(255, 208, 186),
        Scalar(197, 226, 255), Scalar(171, 134, 1), Scalar(109, 63, 54), Scalar(207, 138, 255),
        Scalar(151, 0, 95), Scalar(9, 80, 61), Scalar(84, 105, 51), Scalar(74, 65, 105),
        Scalar(166, 196, 102), Scalar(208, 195, 210), Scalar(255, 109, 65),
        Scalar(0, 143, 149), Scalar(179, 0, 194), Scalar(209, 99, 106), Scalar(5, 121, 0),
        Scalar(227, 255, 205), Scalar(147, 186, 208), Scalar(153, 69, 1), Scalar(3, 95, 161),
        Scalar(163, 255, 0), Scalar(119, 0, 170), Scalar(0, 182, 199), Scalar(0, 165, 120),
        Scalar(183, 130, 88), Scalar(95, 32, 0), Scalar(130, 114, 135), Scalar(110, 129, 133),
        Scalar(166, 74, 118), Scalar(219, 142, 185), Scalar(79, 210, 114), Scalar(178, 90, 62),
        Scalar(65, 70, 15), Scalar(127, 167, 115), Scalar(59, 105, 106), Scalar(142, 108, 45),
        Scalar(196, 172, 0), Scalar(95, 54, 80), Scalar(128, 76, 255), Scalar(201, 57, 1),
        Scalar(246, 0, 122), Scalar(191, 162, 208)
};
/*-------------------------------------------
                  Functions
-------------------------------------------*/
// 将数据保存到指定文件中
void save_to_file(const std::string& data_path, float* data, int size) {
    // 创建并打开 .txt 文件
    std::ofstream file(data_path);

    // 检查文件是否成功打开
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open file: " << data_path << std::endl;
        return;
    }

    // 将数据逐行写入文件
    for (int i = 0; i < size; ++i) {
        file << data[i] << std::endl;
    }

    // 关闭文件
    file.close();
}

static inline float fast_exp(float x)
{
    // return exp(x);
    union
    {
        uint32_t i;
        float f;
    } v;
    v.i = (12102203.1616540672 * x + 1064807160.56887296);
    return v.f;
}

static float sigmoid(float x)
{
    return 1 / (1 + fast_exp(-x));
//    return 1 / (1 + exp(-x));
}

static float DeQnt2F32(float qnt, int zp, float scale)
{
    return ((float)qnt - (float)zp) * scale;
}

static inline float IOU(float XMin1, float YMin1, float XMax1, float YMax1, float XMin2, float YMin2, float XMax2, float YMax2)
{
    float Inter = 0;
    float Total = 0;
    float XMin = 0;
    float YMin = 0;
    float XMax = 0;
    float YMax = 0;
    float Area1 = 0;
    float Area2 = 0;
    float InterWidth = 0;
    float InterHeight = 0;

    XMin = ZQ_MAX(XMin1, XMin2);
    YMin = ZQ_MAX(YMin1, YMin2);
    XMax = ZQ_MIN(XMax1, XMax2);
    YMax = ZQ_MIN(YMax1, YMax2);

    InterWidth = XMax - XMin;
    InterHeight = YMax - YMin;

    InterWidth = (InterWidth >= 0) ? InterWidth : 0;
    InterHeight = (InterHeight >= 0) ? InterHeight : 0;

    Inter = InterWidth * InterHeight;

    Area1 = (XMax1 - XMin1) * (YMax1 - YMin1);
    Area2 = (XMax2 - XMin2) * (YMax2 - YMin2);

    Total = Area1 + Area2 - Inter;

    return float(Inter) / float(Total);
}

int GenerateMeshgrid()
{
    int ret = 0;
    if (headNum == 0)
    {
        printf("=== yolov8 Meshgrid  Generate failed! \n");
    }

    for (int index = 0; index < headNum; index++)
    {
        for (int i = 0; i < mapSize[index][0]; i++)
        {
            for (int j = 0; j < mapSize[index][1]; j++)
            {
                meshgrid.push_back(float(j + 0.5));
                meshgrid.push_back(float(i + 0.5));
            }
        }
    }

    printf("=== yolov8 Meshgrid  Generate success! \n");

    return ret;
}

int GetConvDetectionResult(float **pBlob, std::vector<float> &DetectiontRects)
{
    int ret = 0;
    if (meshgrid.empty())
    {
        ret = GenerateMeshgrid();
    }
    int gridIndex = -2;
    float xmin = 0, ymin = 0, xmax = 0, ymax = 0;
    float cls_val = 0;
    float cls_max = 0;
    int cls_index = 0;

    int quant_zp_cls = 0, quant_zp_reg = 0;
    float quant_scale_cls = 1, quant_scale_reg = 1;

    float sfsum = 0;
    float locval = 0;
    float locvaltemp = 0;

    DetectRect temp;
    std::vector<DetectRect> detectRects;

    for (int index = 0; index < headNum; index++)
    {
        float *reg = pBlob[index * 2 + 0];  // 获取回归输出
        float *cls = pBlob[index * 2 + 1];  // 获取分类输出

        for (int h = 0; h < mapSize[index][0]; h++)
        {
            for (int w = 0; w < mapSize[index][1]; w++)
            {
                gridIndex += 2;

                if (1 == class_num)
                {
                    cls_max = sigmoid(cls[0 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]);
                    cls_index = 0;
                }
                else
                {
                    for (int cl = 0; cl < class_num; cl++)
                    {
                        cls_val = cls[cl * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w];

                        if (0 == cl)
                        {
                            cls_max = cls_val;
                            cls_index = cl;
                        }
                        else
                        {
                            if (cls_val > cls_max)
                            {
                                cls_max = cls_val;
                                cls_index = cl;
                            }
                        }
                    }
                    cls_max = sigmoid(DeQnt2F32(cls_max, quant_zp_cls, quant_scale_cls));
                }

                if (cls_max > objectThresh)  // 如果分类分数大于阈值
                {
                    regdfl.clear();  // 清空回归偏移量向量
                    for (int lc = 0; lc < 4; lc++)  // 遍历每一个边界框坐标
                    {
                        sfsum = 0;  // Softmax和初始化
                        locval = 0;  // 位置值初始化
                        for (int df = 0; df < 16; df++)  // 遍历每一个Dfl（边界框回归量化值）
                        {
                            // 计算当前量化值的反量化结果并取指数
                            locvaltemp = exp(reg[((lc * 16) + df) * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]);
                            regdeq[df] = locvaltemp;  // 存储反量化结果
                            sfsum += locvaltemp;  // 累加Softmax和
                        }
                        for (int df = 0; df < 16; df++)  // 计算Softmax结果
                        {
                            locvaltemp = regdeq[df] / sfsum;  // Softmax归一化
                            locval += locvaltemp * df;  // 计算边界框坐标
                        }
                        //outputs 1*1*4* X

                        regdfl.push_back(locval);  // 存储边界框坐标
                    }

                    // 计算边界框的左上角和右下角坐标
                    xmin = (meshgrid[gridIndex + 0] - regdfl[0]) * strides[index];
                    ymin = (meshgrid[gridIndex + 1] - regdfl[1]) * strides[index];
                    xmax = (meshgrid[gridIndex + 0] + regdfl[2]) * strides[index];
                    ymax = (meshgrid[gridIndex + 1] + regdfl[3]) * strides[index];

                    // 限制边界框坐标在图像范围内
                    xmin = xmin > 0 ? xmin : 0;
                    ymin = ymin > 0 ? ymin : 0;
                    xmax = xmax < input_w ? xmax : input_w;
                    ymax = ymax < input_h ? ymax : input_h;

                    // 检查边界框是否有效，并将结果存储在检测结果向量中
                    if (xmin >= 0 && ymin >= 0 && xmax <= input_w && ymax <= input_h)
                    {
                        temp.xmin = xmin / input_w;  // 归一化左上角x坐标
                        temp.ymin = ymin / input_h;  // 归一化左上角y坐标
                        temp.xmax = xmax / input_w;  // 归一化右下角x坐标
                        temp.ymax = ymax / input_h;  // 归一化右下角y坐标
                        temp.classId = cls_index;  // 设置类别ID
                        temp.score = cls_max;  // 设置分类分数
                        detectRects.push_back(temp);  // 将检测结果添加到结果向量中
                    }
                }
            }
        }
    }
    std::sort(detectRects.begin(), detectRects.end(), [](DetectRect &Rect1, DetectRect &Rect2) -> bool
    { return (Rect1.score > Rect2.score); });

    std::cout << "NMS Before num :" << detectRects.size() << std::endl;
    for (int i = 0; i < detectRects.size(); ++i)
    {
        float xmin1 = detectRects[i].xmin;
        float ymin1 = detectRects[i].ymin;
        float xmax1 = detectRects[i].xmax;
        float ymax1 = detectRects[i].ymax;
        int classId = detectRects[i].classId;
        float score = detectRects[i].score;

        if (classId != -1)
        {
            // 将检测结果按照classId、score、xmin1、ymin1、xmax1、ymax1 的格式存放在vector<float>中
            DetectiontRects.push_back(float(classId));
            DetectiontRects.push_back(float(score));
            DetectiontRects.push_back(float(xmin1));
            DetectiontRects.push_back(float(ymin1));
            DetectiontRects.push_back(float(xmax1));
            DetectiontRects.push_back(float(ymax1));

            for (int j = i + 1; j < detectRects.size(); ++j)
            {
                float xmin2 = detectRects[j].xmin;
                float ymin2 = detectRects[j].ymin;
                float xmax2 = detectRects[j].xmax;
                float ymax2 = detectRects[j].ymax;
                float iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2);
                if (iou > nmsThresh)
                {
                    detectRects[j].classId = -1;
                }
            }
        }
    }
    return ret;
}

vsi_status show_result
        (
                vsi_nn_graph_t *graph,
                vsi_nn_tensor_t *tensor,
                char* image_path

        )
{
    vsi_status status = VSI_FAILURE;
    uint32_t i, sz1, sz2, sz3, sz4, sz5, sz6, stride;

    vsi_nn_tensor_t *output1_tensor;
    float *output1_buffer = NULL;
    uint8_t *output1_tensor_data = NULL;

    vsi_nn_tensor_t *output2_tensor;
    float *output2_buffer = NULL;
    uint8_t *output2_tensor_data = NULL;

    vsi_nn_tensor_t *output3_tensor;
    float *output3_buffer = NULL;
    uint8_t *output3_tensor_data = NULL;

    vsi_nn_tensor_t *output4_tensor;
    float *output4_buffer = NULL;
    uint8_t *output4_tensor_data = NULL;

    vsi_nn_tensor_t *output5_tensor;
    float *output5_buffer = NULL;
    uint8_t *output5_tensor_data = NULL;

    vsi_nn_tensor_t *output6_tensor;
    float *output6_buffer = NULL;
    uint8_t *output6_tensor_data = NULL;

    vsi_nn_tensor_t *input_tensor = NULL;
    input_tensor = vsi_nn_GetTensor(graph, graph->input.tensors[0]);

    Mat image;
    image = imread(image_path);
    int img_width = image.cols;
    int img_height = image.rows;

    // output1 tensor
    sz1 = 1;
    output1_tensor = vsi_nn_GetTensor(graph, graph->output.tensors[0]);
    for (i = 0; i < output1_tensor->attr.dim_num; i++) {
        sz1 *= output1_tensor->attr.size[i];
    }
    stride = vsi_nn_TypeGetBytes(output1_tensor->attr.dtype.vx_type);
    output1_tensor_data = (uint8_t *) vsi_nn_ConvertTensorToData(graph, output1_tensor);
    output1_buffer = (float *) malloc(sizeof(float) * sz1);
    for (i = 0; i < sz1; i++) {
        status = vsi_nn_DtypeToFloat32(&output1_tensor_data[stride * i], &output1_buffer[i],
                                       &output1_tensor->attr.dtype);
    }
    printf("sz1 = %d\n", sz1);

    // output2 tensor
    sz2 = 1;
    output2_tensor = vsi_nn_GetTensor(graph, graph->output.tensors[1]);
    for (i = 0; i < output2_tensor->attr.dim_num; i++) {
        sz2 *= output2_tensor->attr.size[i];
    }
    stride = vsi_nn_TypeGetBytes(output2_tensor->attr.dtype.vx_type);
    output2_tensor_data = (uint8_t *) vsi_nn_ConvertTensorToData(graph, output2_tensor);
    output2_buffer = (float *) malloc(sizeof(float) * sz2);
    for (i = 0; i < sz2; i++) {
        status = vsi_nn_DtypeToFloat32(&output2_tensor_data[stride * i], &output2_buffer[i],
                                       &output2_tensor->attr.dtype);
    }
    printf("sz2 = %d\n", sz2);

    // output3 tensor
    sz3 = 1;
    output3_tensor = vsi_nn_GetTensor(graph, graph->output.tensors[2]);
    for (i = 0; i < output3_tensor->attr.dim_num; i++) {
        sz3 *= output3_tensor->attr.size[i];
    }
    stride = vsi_nn_TypeGetBytes(output3_tensor->attr.dtype.vx_type);
    output3_tensor_data = (uint8_t *) vsi_nn_ConvertTensorToData(graph, output3_tensor);
    output3_buffer = (float *) malloc(sizeof(float) * sz3);
    for (i = 0; i < sz3; i++) {
        status = vsi_nn_DtypeToFloat32(&output3_tensor_data[stride * i], &output3_buffer[i],
                                       &output3_tensor->attr.dtype);
    }
    printf("sz3 = %d\n", sz3);

    // output4 tensor
    sz4 = 1;
    output4_tensor = vsi_nn_GetTensor(graph, graph->output.tensors[3]);
    for (i = 0; i < output4_tensor->attr.dim_num; i++) {
        sz4 *= output4_tensor->attr.size[i];
    }
    stride = vsi_nn_TypeGetBytes(output4_tensor->attr.dtype.vx_type);
    output4_tensor_data = (uint8_t *) vsi_nn_ConvertTensorToData(graph, output4_tensor);
    output4_buffer = (float *) malloc(sizeof(float) * sz4);
    for (i = 0; i < sz4; i++) {
        status = vsi_nn_DtypeToFloat32(&output4_tensor_data[stride * i], &output4_buffer[i],
                                       &output4_tensor->attr.dtype);
    }
    printf("sz4 = %d\n", sz4);

    // output5 tensor
    sz5 = 1;
    output5_tensor = vsi_nn_GetTensor(graph, graph->output.tensors[4]);
    for (i = 0; i < output5_tensor->attr.dim_num; i++) {
        sz5 *= output5_tensor->attr.size[i];
    }
    stride = vsi_nn_TypeGetBytes(output5_tensor->attr.dtype.vx_type);
    output5_tensor_data = (uint8_t *) vsi_nn_ConvertTensorToData(graph, output5_tensor);
    output5_buffer = (float *) malloc(sizeof(float) * sz5);
    for (i = 0; i < sz5; i++) {
        status = vsi_nn_DtypeToFloat32(&output5_tensor_data[stride * i], &output5_buffer[i],
                                       &output5_tensor->attr.dtype);
    }
    printf("sz5 = %d\n", sz5);

    // output6 tensor
    sz6 = 1;
    output6_tensor = vsi_nn_GetTensor(graph, graph->output.tensors[5]);
    for (i = 0; i < output6_tensor->attr.dim_num; i++) {
        sz6 *= output6_tensor->attr.size[i];
    }
    stride = vsi_nn_TypeGetBytes(output6_tensor->attr.dtype.vx_type);
    output6_tensor_data = (uint8_t *) vsi_nn_ConvertTensorToData(graph, output6_tensor);
    output6_buffer = (float *) malloc(sizeof(float) * sz6);
    for (i = 0; i < sz6; i++) {
        status = vsi_nn_DtypeToFloat32(&output6_tensor_data[stride * i], &output6_buffer[i],
                                       &output6_tensor->attr.dtype);
    }
    printf("sz6 = %d\n", sz6);

//    // 设置保存文件的路径
//    std::string data_path = "output6_data_image_rgb_2.txt"; // 指定文件路径
//    // 调用保存函数
//    save_to_file(data_path, output6_buffer, sz6);
//
//    // 设置保存文件的路径
//    std::string data_path_2 = "output5_data_image_rgb_2.txt"; // 指定文件路径
//    // 调用保存函数
//    save_to_file(data_path_2, output5_buffer, sz5);

    float *pblob[6];
    pblob[0] = output1_buffer;
    pblob[1] = output2_buffer;
    pblob[2] = output3_buffer;
    pblob[3] = output4_buffer;
    pblob[4] = output5_buffer;
    pblob[5] = output6_buffer;

//    test(pblob);

    std::vector<float> DetectiontRects;
    GetConvDetectionResult(pblob, DetectiontRects);

    string save_image_path = "./result/test.jpg";
    Mat src_image;
    src_image = imread(image_path);
    for (int i = 0; i < DetectiontRects.size(); i += 6)
    {
        int classId = int(DetectiontRects[i + 0]);
        string class_name = class_names[classId];
        // 使用不同类别的颜色
        Scalar color = CLASS_COLORS[classId];

        float conf = DetectiontRects[i + 1];
        int xmin = int(DetectiontRects[i + 2] * float(img_width) + 0.5);
        int ymin = int(DetectiontRects[i + 3] * float(img_height) + 0.5);
        int xmax = int(DetectiontRects[i + 4] * float(img_width) + 0.5);
        int ymax = int(DetectiontRects[i + 5] * float(img_height) + 0.5);

//        char text1[256];
//        sprintf(text1, "%d:%.2f", classId, conf);
        string text1 = class_name + ": "+to_string(conf);
//        rectangle(src_image, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(255, 0, 0), 2);
        rectangle(src_image, cv::Point(xmin, ymin), cv::Point(xmax, ymax), color, 2);
        putText(src_image, text1, cv::Point(xmin, ymin + 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, CV_AA);
    }
    imwrite(save_image_path, src_image);
    printf("== obj: %d \n", int(float(DetectiontRects.size()) / 6.0));

    final:
    if(output1_tensor_data)vsi_nn_Free(output1_tensor_data);
    if(output1_buffer)free(output1_buffer);

    if(output2_tensor_data)vsi_nn_Free(output2_tensor_data);
    if(output2_buffer)free(output2_buffer);

    if(output3_tensor_data)vsi_nn_Free(output3_tensor_data);
    if(output3_buffer)free(output3_buffer);

    if(output4_tensor_data)vsi_nn_Free(output4_tensor_data);
    if(output4_buffer)free(output4_buffer);

    if(output5_tensor_data)vsi_nn_Free(output5_tensor_data);
    if(output5_buffer)free(output5_buffer);

    if(output6_tensor_data)vsi_nn_Free(output6_tensor_data);
    if(output6_buffer)free(output6_buffer);

//    free(pblob);
    return VSI_SUCCESS;
}
