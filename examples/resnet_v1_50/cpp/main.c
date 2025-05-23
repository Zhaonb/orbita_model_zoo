/****************************************************************************
*   Generated by ACUITY 5.24.6
*   Match ovxlib 1.1.31
*
*   Neural Network application project entry file
****************************************************************************/
/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __linux__
#include <time.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

#define _BASETSD_H

#include "vsi_nn_pub.h"

#include "vnn_global.h"
#include "vnn_pre_process.h"
#include "vnn_post_process.h"
#include "vnn_resnetv150uint8.h"

/*-------------------------------------------
        Macros and Variables
-------------------------------------------*/

/*-------------------------------------------
                  Functions
-------------------------------------------*/
static void vnn_ReleaseNeuralNetwork
    (
    vsi_nn_graph_t *graph
    )
{
    vnn_ReleaseResnetV150Uint8( graph, TRUE );
    if (vnn_UseImagePreprocessNode())
    {
        vnn_ReleaseBufferImage();
    }
}

static vsi_status vnn_PostProcessNeuralNetwork
    (
    vsi_nn_graph_t *graph
    )
{
    return vnn_PostProcessResnetV150Uint8( graph );
}

#define BILLION                                 1000000000
static uint64_t get_perf_count()
{
#if defined(__linux__) || defined(__ANDROID__) || defined(__QNX__) || defined(__CYGWIN__)
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);

    return (uint64_t)((uint64_t)ts.tv_nsec + (uint64_t)ts.tv_sec * BILLION);
#elif defined(_WIN32) || defined(UNDER_CE)
    LARGE_INTEGER ln;

    QueryPerformanceCounter(&ln);

    return (uint64_t)ln.QuadPart;
#endif
}

static vsi_status vnn_VerifyGraph
    (
    vsi_nn_graph_t *graph
    )
{
    vsi_status status = VSI_FAILURE;
    uint64_t tmsStart, tmsEnd, msVal, usVal;

    /* Verify graph */
    printf("Verify...\n");
    tmsStart = get_perf_count();
    status = vsi_nn_VerifyGraph( graph );
    TEST_CHECK_STATUS(status, final);
    tmsEnd = get_perf_count();
    msVal = (tmsEnd - tmsStart)/1000000;
    usVal = (tmsEnd - tmsStart)/1000;
    printf("Verify Graph: %ldms or %ldus\n", msVal, usVal);

final:
    return status;
}

static vsi_status vnn_ProcessGraph
    (
    vsi_nn_graph_t *graph
    )
{
    vsi_status status = VSI_FAILURE;
    int32_t i,loop;
    char *loop_s;
    uint64_t tmsStart, tmsEnd, sigStart, sigEnd;
    float msVal, usVal;

    status = VSI_FAILURE;
    loop = 1; /* default loop time is 1 */
    loop_s = getenv("VNN_LOOP_TIME");
    if(loop_s)
    {
        loop = atoi(loop_s);
    }

    /* Run graph */
    tmsStart = get_perf_count();
    printf("Start run graph [%d] times...\n", loop);
    for(i = 0; i < loop; i++)
    {
        sigStart = get_perf_count();
#ifdef VNN_APP_ASYNC_RUN
        status = vsi_nn_AsyncRunGraph( graph );
        if(status != VSI_SUCCESS)
        {
            printf("Async Run graph the %d time fail\n", i);
        }
        TEST_CHECK_STATUS( status, final );

        //do something here...

        status = vsi_nn_AsyncRunWait( graph );
        if(status != VSI_SUCCESS)
        {
            printf("Wait graph the %d time fail\n", i);
        }
#else
        status = vsi_nn_RunGraph( graph );
        if(status != VSI_SUCCESS)
        {
            printf("Run graph the %d time fail\n", i);
        }
#endif
        TEST_CHECK_STATUS( status, final );

        sigEnd = get_perf_count();
        msVal = (sigEnd - sigStart)/1000000;
        usVal = (sigEnd - sigStart)/1000;
        printf("Run the %u time: %.2fms or %.2fus\n", (i + 1), msVal, usVal);
    }
    tmsEnd = get_perf_count();
    msVal = (tmsEnd - tmsStart)/1000000;
    usVal = (tmsEnd - tmsStart)/1000;
    printf("vxProcessGraph execution time:\n");
    printf("Total   %.2fms or %.2fus\n", msVal, usVal);
    printf("Average %.2fms or %.2fus\n", ((float)usVal)/1000/loop, ((float)usVal)/loop);

final:
    return status;
}

static vsi_status vnn_PreProcessNeuralNetwork
    (
    vsi_nn_graph_t *graph,
    int argc,
    char **argv
    )
{
    /*
     * argv0:   execute file
     * argv1:   data file
     * argv2~n: inputs n file
     */
    const char **inputs = (const char **)argv + 2;
    uint32_t input_num = argc - 2;

    /*
    if(vnn_UseImagePreprocessNode())
    {
        return vnn_PreProcessResnetV150Uint8_ImageProcess(graph, inputs, input_num);
    }
    */
    return vnn_PreProcessResnetV150Uint8( graph, inputs, input_num );
}

static vsi_nn_graph_t *vnn_CreateNeuralNetwork
    (
    const char *data_file_name
    )
{
    vsi_nn_graph_t *graph = NULL;
    uint64_t tmsStart, tmsEnd, msVal, usVal;

    tmsStart = get_perf_count();
    graph = vnn_CreateResnetV150Uint8( data_file_name, NULL,
                      vnn_GetPreProcessMap(), vnn_GetPreProcessMapCount(),
                      vnn_GetPostProcessMap(), vnn_GetPostProcessMapCount() );
    TEST_CHECK_PTR(graph, final);
    tmsEnd = get_perf_count();
    msVal = (tmsEnd - tmsStart)/1000000;
    usVal = (tmsEnd - tmsStart)/1000;
    printf("Create Neural Network: %ldms or %ldus\n", msVal, usVal);

final:
    return graph;
}

/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main
    (
    int argc,
    char **argv
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_graph_t *graph;
    const char *data_name = NULL;

    if(argc < 3)
    {
        printf("Usage: %s data_file inputs...\n", argv[0]);
        return -1;
    }

    data_name = (const char *)argv[1];

    /* Create the neural network */
    graph = vnn_CreateNeuralNetwork( data_name );
    TEST_CHECK_PTR( graph, final );

    /* Pre process the image data */
    status = vnn_PreProcessNeuralNetwork( graph, argc, argv );
    TEST_CHECK_STATUS( status, final );

    /* Verify graph */
    status = vnn_VerifyGraph( graph );
    TEST_CHECK_STATUS( status, final);

    /* Process graph */
    status = vnn_ProcessGraph( graph );
    TEST_CHECK_STATUS( status, final );

    if(VNN_APP_DEBUG)
    {
        /* Dump all node outputs */
        vsi_nn_DumpGraphNodeOutputs(graph, "./network_dump", NULL, 0, TRUE, 0);
    }

    /* Post process output data */
    status = vnn_PostProcessNeuralNetwork( graph );
    TEST_CHECK_STATUS( status, final );

final:
    vnn_ReleaseNeuralNetwork( graph );
    fflush(stdout);
    fflush(stderr);
    return status;
}

