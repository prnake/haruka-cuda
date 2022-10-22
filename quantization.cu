#include <cassert>
#include <cmath>
#include <cuda_fp16.h>
#include "reduce_kernel_utils.cuh"

typedef struct half4 {
    half x, y, z, w;
} half4;


template<int NUM, typename T>
struct ARRAY {
    T data[NUM];
};

extern __shared__ float cgBlockReduceSumElements_shm[];

// T = float4
// weight is int8 [n, k] row-major
// input is [k]
// scale_list is [n] for per_channel quantization.
// output is [n]
// each thread deals with at least 4 cols (k)
// each block deals with nPerThread rows (n)
// assume n % nPerThread == 0 && k % 4 == 0
// grid(n/nPerThread)
template<int m, int nPerThread>
__global__ void int8WeightPerChannelLdkMultiplication(
    const char4* weight, const float4* input, const float* scale_list, void* output, const int k_4)
{

    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int row_idx = bidx * nPerThread;
    const size_t b_offset = row_idx * k_4;

    using array = struct ARRAY<nPerThread, float>;
    const array scale = *((const array*)scale_list + bidx);
    array sum_list[m];
#pragma unroll
    for (int m_i = 0; m_i < m; m_i++) {
#pragma unroll
        for (int i = 0; i < nPerThread; i++) {
            sum_list[m_i].data[i] = 0.0f;
        }
    }

    for (int k_idx = tidx; k_idx < k_4; k_idx += blockDim.x) {
        float4 input_val[m];
#pragma unroll
        for (int m_i = 0; m_i < m; m_i++) {
            input_val[m_i] = input[k_idx + m_i * k_4];
        }
#pragma unroll
        for (int i = 0; i < nPerThread; i++) {
            const char4 weight_val = weight[b_offset + i * k_4 + k_idx];
#pragma unroll
            for (int m_i = 0; m_i < m; m_i++) {
                sum_list[m_i].data[i] += ((static_cast<float>(weight_val.x) * input_val[m_i].x)
                                          + (static_cast<float>(weight_val.y) * input_val[m_i].y)
                                          + (static_cast<float>(weight_val.z) * input_val[m_i].z)
                                          + (static_cast<float>(weight_val.w) * input_val[m_i].w))
                                         * scale.data[i];
            }
        }
    }
#pragma unroll
    for (int m_i = 0; m_i < m; m_i++) {
        cgBlockReduceSumElements<nPerThread>(sum_list[m_i].data, cgBlockReduceSumElements_shm);
        __syncthreads();
    }
    if (tidx == 0) {
        for (int m_i = 0; m_i < m; m_i++) {
            *((array*)output + bidx + m_i * gridDim.x) = sum_list[m_i];
        }
    }
}

///////////////////////////////////////////////////////////////////////
// FP16 & FP32 accumulators
// for T = half4
// weight is int8 [n, k] row-major
// input is [m, k]
// scale_list is [n] for per_channel quantization.
// output is [m, n]
// each thread deals with at least m * 4 cols (k)
// each block deals with nPerThread m * rows (n)
// assume n % nPerThread == 0 && k % 4 == 0
// grid(n/nPerThread)
template<int m, int nPerThread>
__global__ void int8WeightPerChannelLdkMultiplication(
    const char4* weight, const half4* input, const half* scale_list, void* output, const int k_4)
{

    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int row_idx = bidx * nPerThread;
    const size_t b_offset = row_idx * k_4;

    using array = struct ARRAY<nPerThread, float>;
    array sum_list[m];
#pragma unroll
    for (int m_i = 0; m_i < m; m_i++) {
#pragma unroll
        for (int i = 0; i < nPerThread; i++) {
            sum_list[m_i].data[i] = 0.0f;
        }
    }

    for (int k_idx = tidx; k_idx < k_4; k_idx += blockDim.x) {
        half4 input_val[m];
        // half2 input_val_0[m];
        // half2 input_val_1[m];
#pragma unroll
        for (int m_i = 0; m_i < m; m_i++) {
            input_val[m_i] = input[k_idx + m_i * k_4];
            // const half4 input_val = input[k_idx + m_i * k_4];
            // input_val_0[m_i] = {input_val.x, input_val.y};
            // input_val_1[m_i] = {input_val.z, input_val.w};
        }
#pragma unroll
        for (int i = 0; i < nPerThread; i++) {
            const char4 weight_val = weight[b_offset + i * k_4 + k_idx];
            // const half2 weight_val_0 = {static_cast<half>(weight_val.x), static_cast<half>(weight_val.y)};
            // const half2 weight_val_1 = {static_cast<half>(weight_val.z), static_cast<half>(weight_val.w)};
#pragma unroll
            for (int m_i = 0; m_i < m; m_i++) {
                // const half2 weight_val_2 =
                //     __hadd2(__hmul2(input_val_0[m_i], weight_val_0), __hmul2(input_val_1[m_i], weight_val_1));
                // sum_list[m_i].data[i] += static_cast<float>(weight_val_2.x + weight_val_2.y);
                sum_list[m_i].data[i] += ((static_cast<float>(weight_val.x) * static_cast<float>(input_val[m_i].x))
                                          + (static_cast<float>(weight_val.y) * static_cast<float>(input_val[m_i].y))
                                          + (static_cast<float>(weight_val.z) * static_cast<float>(input_val[m_i].z))
                                          + (static_cast<float>(weight_val.w) * static_cast<float>(input_val[m_i].w)));
            }
        }
    }
#pragma unroll
    for (int m_i = 0; m_i < m; m_i++) {
        cgBlockReduceSumElements<nPerThread>(sum_list[m_i].data, cgBlockReduceSumElements_shm);
        __syncthreads();
    }
    if (tidx == 0) {
        using array_half = struct ARRAY<nPerThread, half>;
        const array_half scale = *((const array_half*)scale_list + bidx);
#pragma unroll
        for (int m_i = 0; m_i < m; m_i++) {
            array_half sum_list_half;
#pragma unroll
            for (int i = 0; i < nPerThread; i++) {
                sum_list_half.data[i] = __float2half_rn(sum_list[m_i].data[i] * float(scale.data[i]));
            }
            *((array_half*)output + bidx + m_i * gridDim.x) = sum_list_half;
        }
    }
}

///////////////////////////////////////////////////////////////////////

#define RUN(M, TYPE, TYPE1)                                                                                                   \
    int8WeightPerChannelLdkMultiplication<M, nPerThread><<<grid, block, shm_size, stream>>>(                           \
        (const char4*)weight, (const TYPE*)input, (const TYPE1*)scale_list, (void*)output, k / 4);

template<typename T>
void int8WeightPerChannelLdkMultiplicationLauncher(const int8_t* weight,
                                                   const T* input,
                                                   const T* scale_list,
                                                   T* output,
                                                   const int m,
                                                   const int n,
                                                   const int k,
                                                   cudaStream_t stream)
{
    const int nPerThread = 2;
    if ((n % nPerThread != 0) || (k % 4 != 0)) {
        printf("[ERROR][int8WeightPerChannelLdkMultiplicationLauncher] (%d % %d != 0) || (%d % 4 != 0).\n",
               n,
               nPerThread,
               k);
        exit(-1);
    }

    dim3 grid(n / nPerThread);
    dim3 block;
    // block size tuned with gpt-3 parameter
    if (k > 10000) {
        block.x = 256;
    }
    else if (k > 2000) {
        block.x = 128;
    }
    else {
        block.x = 64;
    }
    while (block.x * 4 > k) {
        block.x /= 2;
    }
    block.x = (block.x + 31) / 32 * 32;
    const size_t shm_size = block.x * nPerThread * sizeof(float);
    if (m == 1) {
        if (std::is_same<T, half>::value) {
            RUN(1, half4, half);
        }
        else {
            RUN(1, float4, float);
        }
    }
    else if (m == 2) {
        if (std::is_same<T, half>::value) {
            RUN(2, half4, half);
        }
        else {
            RUN(2, float4, float);
        }
    }
    else if (m == 3) {
        if (std::is_same<T, half>::value) {
            RUN(3, half4, half);
        }
        else {
            RUN(3, float4, float);
        }
    }
    else if (m == 4) {
        if (std::is_same<T, half>::value) {
            RUN(4, half4, half);
        }
        else {
            RUN(4, float4, float);
        }
    }
    else {
        printf("[ERROR][int8WeightPerChannelLdkMultiplicationLauncher] not support m == %d.\n", m);
        exit(-1);
    }
}

template void int8WeightPerChannelLdkMultiplicationLauncher(const int8_t* matrix,
                                                            const float* vector,
                                                            const float* scale_list,
                                                            float* output,
                                                            const int m,
                                                            const int n,
                                                            const int k,
                                                            cudaStream_t stream);

template void int8WeightPerChannelLdkMultiplicationLauncher(const int8_t* matrix,
                                                            const half* vector,
                                                            const half* scale_list,
                                                            half* output,
                                                            const int m,
                                                            const int n,
                                                            const int k,
                                                            cudaStream_t stream);

/////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
// FP16 & FP32 accumulators
// for T = half4
// weight is int4 [n, k / 2] row-major
// input is [m, k]
// scale_list is [n] for per_channel quantization.
// output is [m, n]
// each thread deals with at least m * 4 cols (k)
// each block deals with nPerThread m * rows (n)
// assume n % nPerThread == 0 && k % 4 == 0
// grid(n/nPerThread)
template<int m, int nPerThread>
__global__ void int4WeightPerChannelLdkMultiplication(
    const char2* weight, const half4* input, const half* scale_list, void* output, const int k_4)
{
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int row_idx = bidx * nPerThread;
    const size_t b_offset = row_idx * k_4;

    using array = struct ARRAY<nPerThread, float>;
    array sum_list[m];
#pragma unroll
    for (int m_i = 0; m_i < m; m_i++) {
#pragma unroll
        for (int i = 0; i < nPerThread; i++) {
            sum_list[m_i].data[i] = 0.0f;
        }
    }

    for (int k_idx = tidx; k_idx < k_4; k_idx += blockDim.x) {
        // half2 input_val_0[m];
        // half2 input_val_1[m];
        half4 input_val[m];
#pragma unroll
        for (int m_i = 0; m_i < m; m_i++) {
            // const half4 input_val = input[k_idx + m_i * k_4];
            // input_val_0[m_i] = {input_val.x, input_val.y};
            // input_val_1[m_i] = {input_val.z, input_val.w};
            input_val[m_i] = input[k_idx + m_i * k_4];
        }
#pragma unroll
        for (int i = 0; i < nPerThread; i++) {
            const char2 weight_val = weight[b_offset + i * k_4 + k_idx];
            int8_t original, high1, high2, low1, low2;
            original = weight_val.x;
            high1 = original >> 4;
            low1 = original << 4;
            low1 = low1 >> 4;
            // const half2 weight_val_0 = {static_cast<half>(high), static_cast<half>(low)};
            original = weight_val.y;
            high2 = original >> 4;
            low2 = original << 4;
            low2 = low2 >> 4;
            // const half2 weight_val_1 = {static_cast<half>(high), static_cast<half>(low)};
#pragma unroll
            for (int m_i = 0; m_i < m; m_i++) {
                // const half2 weight_val_2 =
                //     __hadd2(__hmul2(input_val_0[m_i], weight_val_0), __hmul2(input_val_1[m_i], weight_val_1));
                // sum_list[m_i].data[i] += static_cast<float>(weight_val_2.x + weight_val_2.y);
                sum_list[m_i].data[i] += ((static_cast<float>(high1) * static_cast<float>(input_val[m_i].x))
                                          + (static_cast<float>(low1) * static_cast<float>(input_val[m_i].y))
                                          + (static_cast<float>(high2) * static_cast<float>(input_val[m_i].z))
                                          + (static_cast<float>(low2) * static_cast<float>(input_val[m_i].w)));
            }
        }
    }
#pragma unroll
    for (int m_i = 0; m_i < m; m_i++) {
        cgBlockReduceSumElements<nPerThread>(sum_list[m_i].data, cgBlockReduceSumElements_shm);
        __syncthreads();
    }
    if (tidx == 0) {
        using array_half = struct ARRAY<nPerThread, half>;
        const array_half scale = *((const array_half*)scale_list + bidx);
#pragma unroll
        for (int m_i = 0; m_i < m; m_i++) {
            array_half sum_list_half;
#pragma unroll
            for (int i = 0; i < nPerThread; i++) {
                sum_list_half.data[i] = __float2half_rn(sum_list[m_i].data[i] * float(scale.data[i]));
            }
            *((array_half*)output + bidx + m_i * gridDim.x) = sum_list_half;
        }
    }
}

///////////////////////////////////////////////////////////////////////

#define RUN4(M, TYPE, TYPE1)                                                                                                  \
    int4WeightPerChannelLdkMultiplication<M, nPerThread><<<grid, block, shm_size, stream>>>(                           \
        (const char2*)weight, (const TYPE*)input, (const TYPE1*)scale_list, (void*)output, k / 4);

template<typename T>
void int4WeightPerChannelLdkMultiplicationLauncher(const int8_t* weight,
                                                   const T* input,
                                                   const T* scale_list,
                                                   T* output,
                                                   const int m,
                                                   const int n,
                                                   const int k,
                                                   cudaStream_t stream)
{
    const int nPerThread = 2;
    if ((n % nPerThread != 0) || (k % 4 != 0)) {
        printf("[ERROR][int4WeightPerChannelLdkMultiplicationLauncher] (%d % %d != 0) || (%d % 4 != 0).\n",
               n,
               nPerThread,
               k);
        exit(-1);
    }

    dim3 grid(n / nPerThread);
    dim3 block;
    // block size tuned with gpt-3 parameter
    if (k > 10000) {
        block.x = 256;
    }
    else if (k > 2000) {
        block.x = 128;
    }
    else {
        block.x = 64;
    }

    while (block.x * 4 > k) {
        block.x /= 2;
    }
    block.x = (block.x + 31) / 32 * 32;
    const size_t shm_size = block.x * nPerThread * sizeof(float);
    if (m == 1) {
        if (std::is_same<T, half>::value) {
            RUN4(1, half4, half);
        }
    }
    else if (m == 2) {
        if (std::is_same<T, half>::value) {
            RUN4(2, half4, half);
        }
    }
    else if (m == 3) {
        if (std::is_same<T, half>::value) {
            RUN4(3, half4, half);
        }
    }
    else if (m == 4) {
        if (std::is_same<T, half>::value) {
            RUN4(4, half4, half);
        }
    }
    else {
        printf("[ERROR][int4WeightPerChannelLdkMultiplicationLauncher] not support m == %d.\n", m);
        exit(-1);
    }
}

template void int4WeightPerChannelLdkMultiplicationLauncher(const int8_t* matrix,
                                                            const float* vector,
                                                            const float* scale_list,
                                                            float* output,
                                                            const int m,
                                                            const int n,
                                                            const int k,
                                                            cudaStream_t stream);

template void int4WeightPerChannelLdkMultiplicationLauncher(const int8_t* matrix,
                                                            const half* vector,
                                                            const half* scale_list,
                                                            half* output,
                                                            const int m,
                                                            const int n,
                                                            const int k,
                                                            cudaStream_t stream);


// template<typename T>
// __global__ void
// int4WeightExtractionDevice(const int8_t* weight, const T* scale_list, T* output, const int n, const int k)
// {
//     for (int i = blockIdx.x * k + threadIdx.x, j = threadIdx.x; i < blockIdx.x * k + k; i += blockDim.x, j += blockDim.x) {
//         int idx = j * n + blockIdx.x;
//         int8_t original = weight[idx/2];
//         int8_t high = original >> 4;
//         int8_t low = original << 4;
//         low = low >> 4;
//         if(idx % 2 == 0)
//             output[i] = T(high) * scale_list[j];
//         else
//             output[i] = T(low) * scale_list[j];
//     }
// }

template<typename T>
__global__ void
int4WeightExtractionDevice(const int8_t* weight, const T* scale_list, T* output, const int n, const int k)
{
    for (int i = blockIdx.x * k + threadIdx.x, j = threadIdx.x; i < blockIdx.x * k + k; i += blockDim.x, j += blockDim.x) {
        int8_t original = weight[j * n + blockIdx.x];
        int8_t high = original >> 4;
        int8_t low = original << 4;
        low = low >> 4;
        output[blockIdx.x * k * 2 + j] = T(high) * T(scale_list[j]);
        output[blockIdx.x * k * 2 + k + j] = T(low) * T(scale_list[j]);
    }
}

template<typename T>
void invokeInt4WeightExtraction(
    const int8_t* weight, const T* scale_list, T* output, const int n, const int k, cudaStream_t stream)
{
    const int n_div_2 = n / 2;
    dim3 grid(n_div_2);
    dim3 block(1024);

    int4WeightExtractionDevice<T><<<grid, block, 0, stream>>>(weight, scale_list, output, n_div_2, k);
}

template void invokeInt4WeightExtraction(
    const int8_t* weight, const float* scale_list, float* output, const int n, const int k, cudaStream_t stream);

template void invokeInt4WeightExtraction(
    const int8_t* weight, const half* scale_list, half* output, const int n, const int k, cudaStream_t stream);


template<typename T>
__global__ void
int4WeightExtractionNoTransDevice(const int8_t* weight, const T* scale_list, T* output, const int n, const int k)
{
    for(int i = blockIdx.x * k + threadIdx.x; i < blockIdx.x * k + k; i += blockDim.x){
        int8_t original = weight[i];
        int8_t high = original >> 4;
        int8_t low = original << 4; low = low >> 4;
        output[i * 2] = T(high) * scale_list[blockIdx.x];
        output[i * 2 + 1] = T(low) * scale_list[blockIdx.x];
    }
}

template<typename T>
void invokeInt4WeightExtractionNoTrans(
    const int8_t* weight, const T* scale_list, T* output, const int n, const int k, cudaStream_t stream)
{
    dim3 grid(n);
    dim3 block(1024);

    int4WeightExtractionNoTransDevice<T><<<grid, block, 0, stream>>>(weight, scale_list, output, n, k / 2);
}

template void invokeInt4WeightExtractionNoTrans(
    const int8_t* weight, const float* scale_list, float* output, const int n, const int k, cudaStream_t stream);

template void invokeInt4WeightExtractionNoTrans(
    const int8_t* weight, const half* scale_list, half* output, const int n, const int k, cudaStream_t stream);


// __global__ void
// int4WeightCompressionDevice(const int8_t* input,
//                                 int8_t* output,
//                                 const int n,
//                                 const int k)
// {
//     for(int i = blockIdx.x * k + threadIdx.x; i < blockIdx.x * k + k; i += blockDim.x){
//         output[i] = (input[i * 2] << 4) | (input[i * 2 + 1] & 0b00001111);
//     }
// }

// super super slow
template<typename T>
__global__ void
int8WeightExtractionDevice(const int8_t* weight, const T* scale_list, T* output, const int n, const int k)
{
    // transpose and extract
    for (int i = blockIdx.x * k + threadIdx.x, j = threadIdx.x; i < blockIdx.x * k + k; i += blockDim.x, j += blockDim.x) {
        output[i] = T(weight[j * n + blockIdx.x]) * scale_list[j];
    }
}

template<typename T>
void invokeInt8WeightExtraction(
    const int8_t* weight, const T* scale_list, T* output, const int n, const int k, cudaStream_t stream)
{
    dim3 grid(n);
    dim3 block(1024);

    int8WeightExtractionDevice<<<grid, block, 0, stream>>>(weight, scale_list, output, n, k);
}


template void invokeInt8WeightExtraction(
    const int8_t* weight, const float* scale_list, float* output, const int n, const int k, cudaStream_t stream);

template void invokeInt8WeightExtraction(
    const int8_t* weight, const half* scale_list, half* output, const int n, const int k, cudaStream_t stream);
