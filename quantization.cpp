#include <sys/time.h>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include "quantization.h"

int m = 4, n = 4608, k = 12288;



int main(){
    std::srand(std::time(nullptr));
    int8_t* weight;
    __half* weight16;
    __half* input;
    __half* output;
    __half* output2;
    __half* scale_list;

    cudaMallocManaged(&weight, n * k * sizeof(int8_t));
    cudaMallocManaged(&weight16, n * k * sizeof(__half));
    cudaMallocManaged(&scale_list, n * m * sizeof(__half));
    cudaMallocManaged(&input, k * m * sizeof(__half));
    cudaMallocManaged(&output, n * m * sizeof(__half));
    cudaMallocManaged(&output2, n * m * sizeof(__half));

    for (int i = 0; i < n * k; ++i){
        weight[i] = (std::rand() % 255 - 127);
    }
    for (int i = 0; i < n; ++i) {
        scale_list[i] = __half(float(std::rand())/(RAND_MAX + 1u));
    }
    for (int i = 0; i < k * m; ++i) {
        input[i] = __half(float(std::rand())/(RAND_MAX + 1u));
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cublasHandle_t handle;
    cublasCreate(&handle);
            
    // invokeInt4WeightExtractionNoTrans(weight, scale_list, weight16, n, k, stream);
    // cudaStreamSynchronize(stream);

    __half alpha = __half(1.0f), beta = __half(0.0f);

    int start_algo = CUBLAS_GEMM_DEFAULT;
    int end_algo = CUBLAS_GEMM_ALGO23;
    float total_time = 0;

    int iteration = 20;

    total_time = 0;

    int algo = -1;
    for (int i = 0; i < iteration; ++i) {
        struct timeval start, end;
        cudaDeviceSynchronize();
        cudaProfilerStart();
        gettimeofday(&start, NULL);

        cublasGemmEx(handle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        n,
                        m,
                        k,
                        &alpha,
                        weight16,
                        CUDA_R_16F,
                        n,
                        input,
                        CUDA_R_16F,
                        k,
                        &beta,
                        output,
                        CUDA_R_16F,
                        n,
                        CUDA_R_16F,
                        static_cast<cublasGemmAlgo_t>(-1));
        cudaDeviceSynchronize();

        gettimeofday(&end, NULL);
        cudaProfilerStop();
        if (i > 0)
            total_time += (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
    }
    if (total_time > 0)
        printf("fp16 gemm with algo %d time: %.3f ms\n", algo, total_time / (iteration - 1));


    // for (int i = 0; i < 100; ++i) {
    //     // if (float(output[i]) - 12288 > 1e-8) {
    //         printf("%f ", float(input[i]));
    //     // }
    // }

    // total_time = 0;

    for (int i = 0; i < iteration; ++i) {
        struct timeval start, end;
        cudaDeviceSynchronize();
        cudaProfilerStart();
        gettimeofday(&start, NULL);

        int8WeightPerChannelLdkMultiplicationLauncher(weight, input, scale_list, output, m, n, k, stream);
        cudaStreamSynchronize(stream);

        gettimeofday(&end, NULL);
        cudaProfilerStop();
        if (i > 0)
            total_time += (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
    }
    if (total_time > 0)
        printf("fp16(int8, fp16) time: %.3f ms\n", total_time / (iteration - 1));

    total_time = 0;

    for (int i = 0; i < iteration; ++i) {
        struct timeval start, end;
        cudaDeviceSynchronize();
        cudaProfilerStart();
        gettimeofday(&start, NULL);

        invokeInt4WeightExtractionNoTrans(weight, scale_list, weight16, n, k, stream);
        cudaStreamSynchronize(stream);

        cublasGemmEx(handle,
                        CUBLAS_OP_T,
                        CUBLAS_OP_N,
                        n,
                        m,
                        k,
                        &alpha,
                        weight16,
                        CUDA_R_16F,
                        k,
                        input,
                        CUDA_R_16F,
                        k,
                        &beta,
                        output2,
                        CUDA_R_16F,
                        n,
                        CUDA_R_16F,
                        static_cast<cublasGemmAlgo_t>(-1));
        cudaDeviceSynchronize();

        gettimeofday(&end, NULL);
        cudaProfilerStop();
        if (i > 0)
            total_time += (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
    }
    if (total_time > 0)
        printf("gemm(int4 -> fp16, fp16) time: %.3f ms\n", total_time / (iteration - 1));

    // for (int i = 0; i < 100; ++i) {
    //     if (float(output[i]) - 12288 > 1e-8) {
    //         printf("%f ", float(output2[i]));
    //     }
    // }



    total_time = 0;

    for (int i = 0; i < iteration; ++i) {
        struct timeval start, end;
        cudaDeviceSynchronize();
        cudaProfilerStart();
        gettimeofday(&start, NULL);

        int4WeightPerChannelLdkMultiplicationLauncher(weight, input, scale_list, output, m, n, k, stream);
        cudaStreamSynchronize(stream);

        gettimeofday(&end, NULL);
        cudaProfilerStop();
        if (i > 0)
            total_time += (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
    }
    if (total_time > 0)
        printf("gemm(int4, fp16) time: %.3f ms\n", total_time / (iteration - 1));

    float max_cal_err = -1.0f;
    for (int i = 0; i < n * m; ++i) {
        max_cal_err = std::max(max_cal_err, float(output[i]) - float(output2[i]));
    }
    printf("max_cal_err: %f\n",max_cal_err);

    // for (algo = start_algo; algo <= end_algo; ++algo){

    //     total_time = 0;

    //     for (int i = 0; i < iteration; ++i) {
    //         struct timeval start, end;
    //         cudaDeviceSynchronize();
    //         cudaProfilerStart();
    //         gettimeofday(&start, NULL);

    //         cublasGemmEx(handle,
    //                         CUBLAS_OP_N,
    //                         CUBLAS_OP_N,
    //                         n,
    //                         m,
    //                         k,
    //                         &alpha,
    //                         weight16,
    //                         CUDA_R_16F,
    //                         n,
    //                         input,
    //                         CUDA_R_16F,
    //                         k,
    //                         &beta,
    //                         output,
    //                         CUDA_R_16F,
    //                         n,
    //                         CUDA_R_16F,
    //                         static_cast<cublasGemmAlgo_t>(-1));
    //         cudaDeviceSynchronize();

    //         gettimeofday(&end, NULL);
    //         cudaProfilerStop();
    //         if (i > 0)
    //             total_time += (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
    //     }
    //     if (total_time > 0)
    //         printf("fp16 gemm with algo %d time: %.3f ms\n", algo, total_time / (iteration - 1));
    // }

    // int start_algo_t_op = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    // int end_algo_t_op = CUBLAS_GEMM_ALGO15_TENSOR_OP;


    // for (algo = start_algo_t_op; algo <= end_algo_t_op; ++algo){

    //     total_time = 0;

    //     for (int i = 0; i < iteration; ++i) {
    //         struct timeval start, end;
    //         cudaDeviceSynchronize();
    //         cudaProfilerStart();
    //         gettimeofday(&start, NULL);

    //         cublasGemmEx(handle,
    //                         CUBLAS_OP_N,
    //                         CUBLAS_OP_N,
    //                         n,
    //                         m,
    //                         k,
    //                         &alpha,
    //                         weight16,
    //                         CUDA_R_16F,
    //                         n,
    //                         input,
    //                         CUDA_R_16F,
    //                         k,
    //                         &beta,
    //                         output,
    //                         CUDA_R_16F,
    //                         n,
    //                         CUDA_R_16F,
    //                         static_cast<cublasGemmAlgo_t>(-1));
    //         cudaDeviceSynchronize();

    //         gettimeofday(&end, NULL);
    //         cudaProfilerStop();
    //         if (i > 0)
    //             total_time += (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
    //     }
    //     if (total_time > 0)
    //         printf("fp16 gemm with algo %d time: %.3f ms\n", algo, total_time / (iteration - 1));
    // }

    return 0;
}
