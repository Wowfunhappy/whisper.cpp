#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#if CUDART_VERSION >= 12050
#include <cuda_fp8.h>
#endif // CUDART_VERSION >= 12050

#if CUDART_VERSION >= 12080
#include <cuda_fp4.h>
#endif // CUDART_VERSION >= 12080

#if CUDART_VERSION < 9000
/* CUDA 7.5/8.0: __shfl_*_sync don't exist, use non-sync versions (implicit warp sync on Kepler/Maxwell) */
#define __shfl_sync(mask, val, srcLane, width)       __shfl((val), (srcLane), (width))
#define __shfl_up_sync(mask, val, delta, width)      __shfl_up((val), (delta), (width))
#define __shfl_down_sync(mask, val, delta, width)    __shfl_down((val), (delta), (width))
#define __shfl_xor_sync(mask, val, laneMask, width)  __shfl_xor((val), (laneMask), (width))
#define __all_sync(mask, predicate) __all((predicate))
#define __any_sync(mask, predicate) __any((predicate))
#define __ballot_sync(mask, predicate) __ballot((predicate))

/* CUBLAS_TENSOR_OP_MATH and cublasSetMathMode don't exist before CUDA 9.0 */
#define CUBLAS_TENSOR_OP_MATH 0
#define CUBLAS_TF32_TENSOR_OP_MATH 0
#define CUBLAS_COMPUTE_16F CUDA_R_16F
#define CUBLAS_COMPUTE_32F CUDA_R_32F
#define cublasComputeType_t cudaDataType_t
static inline cublasStatus_t cublasSetMathMode(cublasHandle_t, int) { return CUBLAS_STATUS_SUCCESS; }

/* cudaFuncSetAttribute doesn't exist before CUDA 9.0 */
#define cudaFuncAttributeMaxDynamicSharedMemorySize 0
template<typename T>
static inline cudaError_t cudaFuncSetAttribute(T, int, size_t) { return cudaSuccess; }

/* CUBLAS_DEFAULT_MATH doesn't exist before CUDA 9.0 */
#define CUBLAS_DEFAULT_MATH 0

/* cudaDevAttrCooperativeLaunch doesn't exist before CUDA 9.0 */
#define cudaDevAttrCooperativeLaunch ((cudaDeviceAttr)95)

/* CUBLAS_GEMM_DEFAULT_TENSOR_OP doesn't exist before CUDA 9.0 */
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP 99

/* CUDA_R_16BF doesn't exist before CUDA 11.0 */
#ifndef CUDA_R_16BF
#define CUDA_R_16BF ((cudaDataType_t)14)
#endif

/* cublasGemmEx: stub for CUDA 7.5 (CUDA 8.0+ has the real version) */
static inline cublasStatus_t cublasGemmEx(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int, const void*, const void*, cudaDataType_t, int,
    const void*, cudaDataType_t, int, const void*, void*, cudaDataType_t, int,
    cudaDataType_t, int) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
}
static inline cublasStatus_t cublasGemmBatchedEx(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int, const void*,
    const void* const*, cudaDataType_t, int,
    const void* const*, cudaDataType_t, int,
    const void*, void* const*, cudaDataType_t, int,
    int, cudaDataType_t, int) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
}
static inline cublasStatus_t cublasGemmStridedBatchedEx(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int, const void*,
    const void*, cudaDataType_t, int, long long,
    const void*, cudaDataType_t, int, long long,
    const void*, void*, cudaDataType_t, int, long long,
    int, cudaDataType_t, int) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

#elif CUDART_VERSION < 11020
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED
#define CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#define CUBLAS_COMPUTE_16F CUDA_R_16F
#define CUBLAS_COMPUTE_32F CUDA_R_32F
#define cublasComputeType_t cudaDataType_t
#endif // CUDART_VERSION
