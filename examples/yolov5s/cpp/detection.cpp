#include "detection.h"
#include "vsi_nn_pub.h"
#include "vnn_global.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <cstring>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <iomanip>

using namespace std;
using namespace cv;
DetectResult g_detect_objs;

/*-------------------------------------------
                  Variable definitions
-------------------------------------------*/

//#define YOLOV3_TINY_SUPPORT
//#define YOLOV4_TINY_SUPPORT
//#define YOLOV3_SUPPORT
//#define YOLOV4_SUPPORT
#define YOLOV5_SUPPORT

char *image_name="dog_640.jpg";//image name
char *class_names[] = {"person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
};//class names
float thresh = 0.2;    //thresh
float nms = 0.45;      //non-maximum suppression

/*-------------------------------------------
                  Yolo type definitions
-------------------------------------------*/

#ifdef YOLOV3_TINY_SUPPORT
int network=1;
float biases[12] = {10,14,23,27,37,58,81,82,135,169,344,319};
int mask[2][3] = {{3,4,5},{0,1,2}};

#elif defined(YOLOV4_TINY_SUPPORT)
int network=2;
float biases[12] = {10,14,23,27,37,58,81,82,135,169,344,319};
int mask[2][3] = {{3,4,5},{0,1,2}};

#elif defined(YOLOV3_SUPPORT)
int network=3;
float biases[18] = {10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326};
int mask[3][3] = {{6,7,8},{3,4,5},{0,1,2}};

#elif defined(YOLOV4_SUPPORT)
int network=4;
float biases[18] = {10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326};
int mask[3][3] = {{6,7,8},{3,4,5},{0,1,2}};

#elif defined(YOLOV5_SUPPORT)
int network=5;
float biases[18] = {10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326};
int mask[3][3] = {{6,7,8},{3,4,5},{0,1,2}};
#else
#error "unknown yolo type"
#endif

/*-------------------------------------------
                  Functions
-------------------------------------------*/

int nms_comparator(const void *pa, const void *pb)
{
    detection a = *(detection *)pa;
    detection b = *(detection *)pb;
    float diff = 0;
    if(b.sort_class >= 0){
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
    } else {
        diff = a.objectness - b.objectness;
    }
    if(diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}

float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float box_iou(box a, box b)
{
    return box_intersection(a, b)/box_union(a, b);
}

void do_nms_sort(detection *dets, int total, int classes, float thresh)
{
    int i, j, k;
    k = total-1;
    for(i = 0; i <= k; ++i){
        if(dets[i].objectness == 0){
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k+1;

    for(k = 0; k < classes; ++k){
        for(i = 0; i < total; ++i){
            dets[i].sort_class = k;
        }
        qsort(dets, total, sizeof(detection), nms_comparator);
        for(i = 0; i < total; ++i){
            if(dets[i].prob[k] == 0) continue;
            box a = dets[i].bbox;
            for(j = i+1; j < total; ++j){
                box b = dets[j].bbox;
                if (box_iou(a, b) > thresh){
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}

layer make_yolo_layer(int batch,int w,int h,int n,int total,int *mask,int classes)
{
    layer l = {0};
    l.n = n;
    l.total = total;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n*(classes+ 4 + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.inputs = l.w*l.h*l.c;

    l.biases = (float*)calloc(total*2,sizeof(float));
    for(int i =0;i<total*2;++i){
        l.biases[i] = biases[i];
    }
    if(mask) l.mask = mask;
    else{
    	l.mask = (int*)calloc(n,sizeof(int));
    	for(int i = 0; i < n; ++i){
    		l.mask[i] = i;
    	}
    }

    l.outputs = l.inputs;
    l.output = (float*)calloc(batch*l.outputs,sizeof(float));
    return l;
}

void free_yolo_layer(layer l)
{
    if(NULL != l.biases){
        free(l.biases);
        l.biases = NULL;
    }

    if(NULL != l.mask){
        l.mask = NULL;
    }
    if(NULL != l.output){
        free(l.output);
        l.output = NULL;
    }
}

static int entry_index(layer l,int batch,int location,int entry)
{
    int n = location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(4 + l.classes + 1) + entry*l.w*l.h + loc;
}

float logistic_activate_kernel(float x){return 1.f/(1.f + expf(-x));}

void activate_array(float *x, int n) {
  for(int i=0; i < n; i++) x[i] = logistic_activate_kernel(x[i]);
}

void forward_yolo_layer_cpu(const float* input,layer l)
{
    std::memcpy(l.output, (float*)input, l.batch*l.inputs*sizeof(float));
    int b,n;
    for(b = 0;b < l.batch;++b){
  	    for(n =0;n< l.n;++n){
  	    	int index = entry_index(l,b,n*l.w*l.h,0);
        	if (network==5)
        	{
      	    	activate_array(l.output + index,(4 + 1 + l.classes)*l.w*l.h);
        	}
        	else
        	{
        		activate_array(l.output + index, 2*l.w*l.h);
        		index = entry_index(l,b,n*l.w*l.h,4);
        		activate_array(l.output + index,(1 + l.classes)*l.w*l.h);
        	}
	    }
    }
}

int yolo_num_detections(layer l,float thresh)
{
    int i,n;
    int count = 0;

    for(i=0;i<l.w*l.h;++i){
        for(n=0;n<l.n;++n){
            int obj_index = entry_index(l,0,n*l.w*l.h+i,4);
            if(l.output[obj_index] > thresh && l.output[obj_index] < 1.0) {
            	++count;
            }
        }
    }
    return count;
}

int num_detections(vector<layer> layers_params,float thresh)
{
    unsigned int i;
    int s=0;
    for(i=0; i<layers_params.size(); ++i){
        layer l  = layers_params[i];
        s += yolo_num_detections(l,thresh);
    }
    return s;

}

detection* make_network_boxes(vector<layer> layers_params,float thresh,int* num)
{
    layer l = layers_params[0];
    int i;

    int nboxes = num_detections(layers_params,thresh);

    if(num) *num = nboxes;
    detection *dets = (detection*)calloc(nboxes,sizeof(detection));
    for(i=0;i<nboxes;++i){
        dets[i].prob = (float*)calloc(l.classes,sizeof(float));
    }
    return dets;
}

void correct_yolo_boxes(detection* dets,int n,int w,int h,int netw,int neth,int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)){
        new_w = netw;
        new_h = (h * netw)/w;
    }
    else{
        new_h = neth;
        new_w = (w * neth)/h;
    }

    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw);
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth);
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

box get_yolo_box_v5(float* x,float* biases,int n,int index,int i,int j,int lw, int lh,int w, int h,int stride)
{
    box b;
    b.x = (i + x[index + 0*stride] * 2 - 0.5) / lw;
    b.y = (j + x[index + 1*stride] * 2 - 0.5) / lh;
    b.w = pow(x[index + 2*stride] * 2, 2) * biases[2*n] / w;
    b.h = pow(x[index + 3*stride] * 2, 2) * biases[2*n + 1] / h;
    return b;
}

box get_yolo_box_v3(float* x,float* biases,int n,int index,int i,int j,int lw, int lh,int w, int h,int stride)
{
	//printf("biases[2*%d] = %f\n",n,biases[2*n]);
	//printf("biases[2*%d + 1] = %f\n",n,biases[2*n + 1]);
    box b;
    b.x = (i + x[index + 0*stride]) / lw;
    b.y = (j + x[index + 1*stride]) / lh;
    b.w = exp(x[index + 2*stride]) * biases[2*n] / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n + 1] / h;
    return b;
}

int get_yolo_detections(layer l,int w, int h, int netw,int neth,float thresh,int *map,int relative,detection *dets)
{

    int i,j,n;
    float* predictions = l.output;
    int count = 0;
    for(i=0;i<l.w*l.h;++i){
        int row = i/l.w;
        int col = i%l.w;
        for(n = 0;n<l.n;++n){           

            int obj_index = entry_index(l,0,n*l.w*l.h + i,4);
            float objectness = predictions[obj_index];
            if(objectness <= thresh) continue;
            int box_index = entry_index(l,0,n*l.w*l.h + i,0);

        	if (network==5)
        	{
        		dets[count].bbox = get_yolo_box_v5(predictions,l.biases,l.mask[n],box_index,col,row,l.w,l.h,netw,neth,l.w*l.h);
        	}
        	else
        	{
        		dets[count].bbox = get_yolo_box_v3(predictions,l.biases,l.mask[n],box_index,col,row,l.w,l.h,netw,neth,l.w*l.h);
        	}

            dets[count].objectness = objectness;
            dets[count].classes = l.classes;

            for(j=0;j<l.classes;++j){

                int class_index = entry_index(l,0,n*l.w*l.h+i,4+1+j);

                float prob = objectness*predictions[class_index];

                dets[count].prob[j] = (prob > thresh) ? prob : 0;
            }
            ++count;
        }
    }

    correct_yolo_boxes(dets,count,w,h,netw,neth,relative);
    return count;
}

void fill_network_boxes(vector<layer> layers_params,int w,int h,int netw, int neth,float thresh, float hier, int *map,int relative,detection *dets)
{
    int j;
    for(j=0;j<layers_params.size();++j){
        layer l = layers_params[j];
        int count = get_yolo_detections(l,w,h,netw,neth,thresh,map,relative,dets);
        dets += count;
    }
}

detection* get_network_boxes(vector<layer> layers_params,
                             int img_w,int img_h,int netw, int neth,float thresh,float hier,int* map,int relative,int *num)
{
    detection *dets = make_network_boxes(layers_params,thresh,num);
    fill_network_boxes(layers_params,img_w,img_h,netw,neth,thresh,hier,map,relative,dets);
    return dets;
}

//get detection result
detection* get_detections(vector<blob> blobs,int img_w,int img_h, int netw, int neth, int *nboxes, int classes, float thresh, float nms)
{

    float hier_thresh = 0.5;
    vector<layer> layers_params;
    layers_params.clear();
    for(int i=0;i<blobs.size();++i){
        layer l_params = make_yolo_layer(1, blobs[i].w, blobs[i].h, 3, sizeof(biases)/sizeof(biases[0])/2, *(mask + i), classes);
        layers_params.push_back(l_params);
        forward_yolo_layer_cpu(blobs[i].data,l_params);
    }
    detection* dets = get_network_boxes(layers_params,img_w,img_h,netw,neth,thresh,hier_thresh,0,1,nboxes);
    for(int index =0;index < layers_params.size();++index){
        free_yolo_layer(layers_params[index]);
    }
    if(nms) do_nms_sort(dets,(*nboxes),classes,nms);
    return dets;
}

void free_detections(detection *dets,int nboxes)
{
    int i;
    for(i = 0;i<nboxes;++i){
        free(dets[i].prob);
    }
    free(dets);
}

void draw_detections(Mat img)
{

int total_num = 0;
total_num = g_detect_objs.total_num;
for(int i=0; i<total_num; ++i)
{
	string txt = g_detect_objs.name[i];
	cv::rectangle(img, g_detect_objs.pt1[i], g_detect_objs.pt2[i], cv::Scalar(10,255,10), 2);
	cv::Point txtOrg(g_detect_objs.pt1[i].x,g_detect_objs.pt1[i].y-5);
	cv::putText(img, txt, txtOrg, CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,255), 1, CV_AA);
}
char *save_path = "result.png";
cv::imwrite(save_path, img);
printf("save result to [%s].\n", save_path);
}

vsi_status show_result_3
    (
    vsi_nn_graph_t *graph,
    vsi_nn_tensor_t *tensor,
    int img_w,
    int img_h,
    int classes
    )
{
	    vsi_status status = VSI_FAILURE;
		uint32_t i, sz1,sz2,sz3,stride;

		vsi_nn_tensor_t *output1_tensor;
		float *output1_buffer = NULL;
		uint8_t *output1_tensor_data = NULL;

		vsi_nn_tensor_t *output2_tensor;
		float *output2_buffer = NULL;
		uint8_t *output2_tensor_data = NULL;

		vsi_nn_tensor_t *output3_tensor;
		float *output3_buffer = NULL;
		uint8_t *output3_tensor_data = NULL;

		vsi_nn_tensor_t *input_tensor = NULL;
		input_tensor = vsi_nn_GetTensor( graph, graph->input.tensors[0]);
		int max_side = input_tensor->attr.size[0];    // image width
		int min_side = input_tensor->attr.size[1];    // image height

			sz1 = 1;
			output1_tensor = vsi_nn_GetTensor(graph, graph->output.tensors[0]);
			for(i = 0; i < output1_tensor->attr.dim_num; i++)
			{
				sz1 *= output1_tensor->attr.size[i];
			}
			stride = vsi_nn_TypeGetBytes(output1_tensor->attr.dtype.vx_type);
			output1_tensor_data = (uint8_t *)vsi_nn_ConvertTensorToData(graph, output1_tensor);
			output1_buffer = (float *)malloc(sizeof(float) * sz1);
			for(i = 0; i < sz1; i++)
			{
				status = vsi_nn_DtypeToFloat32(&output1_tensor_data[stride * i], &output1_buffer[i], &output1_tensor->attr.dtype);
			}

			// output2 tensor
			sz2 = 1;
			output2_tensor = vsi_nn_GetTensor(graph, graph->output.tensors[1]);
			for(i = 0; i < output2_tensor->attr.dim_num; i++)
			{
			   sz2 *= output2_tensor->attr.size[i];
			}
			stride = vsi_nn_TypeGetBytes(output2_tensor->attr.dtype.vx_type);
			output2_tensor_data = (uint8_t *)vsi_nn_ConvertTensorToData(graph, output2_tensor);
			output2_buffer = (float *)malloc(sizeof(float) * sz2);
			for(i = 0; i < sz2; i++)
			{
			   status = vsi_nn_DtypeToFloat32(&output2_tensor_data[stride * i], &output2_buffer[i], &output2_tensor->attr.dtype);
			}

			// output3 tensor
			sz3 = 1;
			output3_tensor = vsi_nn_GetTensor(graph, graph->output.tensors[2]);
			for(i = 0; i < output3_tensor->attr.dim_num; i++)
			{
			   sz3 *= output3_tensor->attr.size[i];
			}
			stride = vsi_nn_TypeGetBytes(output3_tensor->attr.dtype.vx_type);
			output3_tensor_data = (uint8_t *)vsi_nn_ConvertTensorToData(graph, output3_tensor);
			output3_buffer = (float *)malloc(sizeof(float) * sz3);
			for(i = 0; i < sz3; i++)
			{
			   status = vsi_nn_DtypeToFloat32(&output3_tensor_data[stride * i], &output3_buffer[i], &output3_tensor->attr.dtype);
			}

			vector<blob> blobs;
                        for (int i=0; i<3; i++)
			{
			if(network==4||network==5)
				{
				blob temp_blob = {0, NULL};
				if( 0 == i ) temp_blob.data = output3_buffer;
				else if( 1 == i ) temp_blob.data = output2_buffer;
				else if( 2 == i ) temp_blob.data = output1_buffer;
				blobs.push_back(temp_blob);
				}

			else if(network==3)
			{
				blob temp_blob = {0, NULL};
				if( 0 == i ) temp_blob.data = output1_buffer;
				else if( 1 == i ) temp_blob.data = output2_buffer;
				else if( 2 == i ) temp_blob.data = output3_buffer;

				blobs.push_back(temp_blob);
			}
			}

		int netw = img_w;
		int neth = img_h;
		int nboxes = 0;

		detection *dets;

			blobs[0].w = netw/32;
			blobs[1].w = netw/16;
			blobs[2].w = netw/8;
			blobs[0].h = neth/32;
			blobs[1].h = neth/16;
			blobs[2].h = neth/8;

			dets = get_detections(blobs, img_w, img_h,
									netw, /*network_w*/
									 neth, /*network_h*/
									 &nboxes, classes, thresh, nms);

		//printf("nboxes = %d\n", nboxes);

		int detect_num = 0;
		for(int i=0;i< nboxes;++i)
		{
			int cls = -1;
			float best_thresh=thresh;
			for(int j=0; j<classes; ++j)
			{
				if(dets[i].prob[j] > best_thresh)
				{
						cls = j;
						best_thresh=dets[i].prob[j];
				}
			}

			if(cls >= 0){
				box b = dets[i].bbox;

				int left  = (b.x-b.w/2.)*img_w;
				int right = (b.x+b.w/2.)*img_w;
				int top   = (b.y-b.h/2.)*img_h;
				int bot   = (b.y+b.h/2.)*img_h;

				if( left < 0) left = 0;
				if( right > img_w -1) right = img_w - 1;
				if( top < 0 ) top = 0;
				if( bot > img_h - 1) bot = img_h - 1;

				sprintf(g_detect_objs.name[detect_num], "%s: %.0f%%", class_names[cls], dets[i].prob[cls]*100);
				printf("%s %f %f %f %f\n",g_detect_objs.name[detect_num],b.x*img_w,b.y*img_h,b.w*img_w,b.h*img_h);
				g_detect_objs.pt1[detect_num] = cvPoint(left, top);
				g_detect_objs.pt2[detect_num] = cvPoint(right, bot);
				detect_num++;
			}
		}

		g_detect_objs.total_num = detect_num;

		status = VSI_SUCCESS;

	final:
		if(output1_tensor_data)vsi_nn_Free(output1_tensor_data);
		if(output1_buffer)free(output1_buffer);

		if(output2_tensor_data)vsi_nn_Free(output2_tensor_data);
		if(output2_buffer)free(output2_buffer);

		if(output3_tensor_data)vsi_nn_Free(output3_tensor_data);
		if(output3_buffer)free(output3_buffer);

		free_detections(dets, nboxes);
		return status;
}

vsi_status show_result_2
    (
    vsi_nn_graph_t *graph,
    vsi_nn_tensor_t *tensor,
    int img_w,
    int img_h,
    int classes
    )
{
	printf("show_result_2\n");

	    vsi_status status = VSI_FAILURE;
		uint32_t i, sz1,sz2,stride;

		vsi_nn_tensor_t *output1_tensor;
		float *output1_buffer = NULL;
		uint8_t *output1_tensor_data = NULL;

		vsi_nn_tensor_t *output2_tensor;
		float *output2_buffer = NULL;
		uint8_t *output2_tensor_data = NULL;

		vsi_nn_tensor_t *input_tensor = NULL;
		input_tensor = vsi_nn_GetTensor( graph, graph->input.tensors[0]);
		int max_side = input_tensor->attr.size[0];    // image width
		int min_side = input_tensor->attr.size[1];    // image height

		//if (output_tensor_number==3)
		//{
			sz1 = 1;
			output1_tensor = vsi_nn_GetTensor(graph, graph->output.tensors[0]);
			for(i = 0; i < output1_tensor->attr.dim_num; i++)
			{
				sz1 *= output1_tensor->attr.size[i];
			}
			stride = vsi_nn_TypeGetBytes(output1_tensor->attr.dtype.vx_type);
			output1_tensor_data = (uint8_t *)vsi_nn_ConvertTensorToData(graph, output1_tensor);
			output1_buffer = (float *)malloc(sizeof(float) * sz1);
			for(i = 0; i < sz1; i++)
			{
				status = vsi_nn_DtypeToFloat32(&output1_tensor_data[stride * i], &output1_buffer[i], &output1_tensor->attr.dtype);
			}

			// output2 tensor
			sz2 = 1;
			output2_tensor = vsi_nn_GetTensor(graph, graph->output.tensors[1]);
			for(i = 0; i < output2_tensor->attr.dim_num; i++)
			{
			   sz2 *= output2_tensor->attr.size[i];
			}
			stride = vsi_nn_TypeGetBytes(output2_tensor->attr.dtype.vx_type);
			output2_tensor_data = (uint8_t *)vsi_nn_ConvertTensorToData(graph, output2_tensor);
			output2_buffer = (float *)malloc(sizeof(float) * sz2);
			for(i = 0; i < sz2; i++)
			{
			   status = vsi_nn_DtypeToFloat32(&output2_tensor_data[stride * i], &output2_buffer[i], &output2_tensor->attr.dtype);
			}

			vector<blob> blobs;
			for (int i=0; i<2; i++)
			{
				blob temp_blob = {0, NULL};

				if( 0 == i ) temp_blob.data = output1_buffer;
				else if( 1 == i ) temp_blob.data = output2_buffer;

				blobs.push_back(temp_blob);
			}

		int netw = img_w;
		int neth = img_h;
		int nboxes = 0;

		detection *dets;

			blobs[0].w = netw/32;
			blobs[1].w = netw/16;
			blobs[0].h = neth/32;
			blobs[1].h = neth/16;

			dets = get_detections(blobs, img_w, img_h,
									netw, /*network_w*/
									 neth, /*network_h*/
									 &nboxes, classes, thresh, nms);
			//}

		//printf("nboxes = %d\n", nboxes);

		int detect_num = 0;
		for(int i=0;i< nboxes;++i)
		{
			int cls = -1;
			float best_thresh=thresh;
			for(int j=0; j<classes; ++j)
			{
				if(dets[i].prob[j] > best_thresh)
				{
						cls = j;
						best_thresh=dets[i].prob[j];
				}
			}

			if(cls >= 0){
				box b = dets[i].bbox;

				int left  = (b.x-b.w/2.)*img_w;
				int right = (b.x+b.w/2.)*img_w;
				int top   = (b.y-b.h/2.)*img_h;
				int bot   = (b.y+b.h/2.)*img_h;

				if( left < 0) left = 0;
				if( right > img_w -1) right = img_w - 1;
				if( top < 0 ) top = 0;
				if( bot > img_h - 1) bot = img_h - 1;

				sprintf(g_detect_objs.name[detect_num], "%s: %.0f%%", class_names[cls], dets[i].prob[cls]*100);
				printf("%s %f %f %f %f\n",g_detect_objs.name[detect_num],b.x*img_w,b.y*img_h,b.w*img_w,b.h*img_h);
				g_detect_objs.pt1[detect_num] = cvPoint(left, top);
				g_detect_objs.pt2[detect_num] = cvPoint(right, bot);
				detect_num++;
			}
		}

		g_detect_objs.total_num = detect_num;
		status = VSI_SUCCESS;

	final:
		if(output1_tensor_data)vsi_nn_Free(output1_tensor_data);
		if(output1_buffer)free(output1_buffer);

		if(output2_tensor_data)vsi_nn_Free(output2_tensor_data);
		if(output2_buffer)free(output2_buffer);

		free_detections(dets, nboxes);
		return status;
}

vsi_status show_result
    (
    vsi_nn_graph_t *graph,
    vsi_nn_tensor_t *tensor
    )
{
	Mat img = cv::imread(image_name, 1);

	int img_w=img.rows;
	int img_h=img.cols;

	int classes=sizeof(class_names)/sizeof(class_names[0]);
	//printf("class_number=%d",classes);

	vsi_status status = VSI_FAILURE;

	if (network==3||network==4||network==5)
	{
		status = show_result_3(graph, tensor,img_w,img_h,classes);
	}
	else if (network==1||network==2)
	{
		status = show_result_2(graph, tensor,img_w,img_h,classes);
	}

	draw_detections(img);
	return status;
}
