#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdint.h>
#include <stdio.h>
#include <atomic>
#include <assert.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include "ggml-cuda.h"
#include "ggml.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>      
#include <string>       


#define CUDA_RELU_BLOCK_SIZE 256

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

#define CUDA_CHECK(err)                                                                 \
    do {                                                                                \
        cudaError_t err_ = (err);                                                       \
        if (err_ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %d at %s:%d: %s\n", err_, __FILE__, __LINE__,   \
                cudaGetErrorString(err_));                                              \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)

#if CUDART_VERSION >= 12000
#define CUBLAS_CHECK(err)                                                               \
    do {                                                                                \
        cublasStatus_t err_ = (err);                                                    \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                            \
            fprintf(stderr, "\ncuBLAS error %d at %s:%d: %s\n",                         \
                    err_, __FILE__, __LINE__, cublasGetStatusString(err_));             \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)
#else
#define CUBLAS_CHECK(err)                                                               \
    do {                                                                                \
        cublasStatus_t err_ = (err);                                                    \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                            \
            fprintf(stderr, "\ncuBLAS error %d at %s:%d\n", err_, __FILE__, __LINE__);  \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)
#endif // CUDART_VERSION >= 11

#ifdef GGML_CUDA_F16
typedef half dfloat; // dequantize float
typedef half2 dfloat2;
#else
typedef float dfloat; // dequantize float
typedef float2 dfloat2;
#endif //GGML_CUDA_F16

typedef void (*cpy_kernel_t)(const char * cx, char * cdst);

struct ggml_tensor_extra_gpu_ {
    void * data_device[GGML_CUDA_MAX_DEVICES]; // 1 pointer for each device for split tensors
    cudaEvent_t events[GGML_CUDA_MAX_DEVICES]; // events for synchronizing multiple GPUs
};

static   cudaStream_t transfer_w_stream_;


static int g_main_device_ = 0;

template <cpy_kernel_t cpy_1>
static __global__ void transfer_matrix_kernel(half * cx, half * cdst, size_t height, size_t trans_width, size_t total_width) {
    // row-order
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // check index
    if (col < trans_width && row < height) {
        // row-order on cpu
        int cpu_index = row * total_width + col;
    } 
}

void ggml_cuda_cpy_tensor_cpu2gpu(const struct ggml_tensor * tensor, size_t il, size_t n_ctx, size_t N, size_t n_past, size_t n_embd_gqa) {
    char * dst_ptr;
    const char * src_ptr;
    int64_t count;
    cudaMemcpyKind kind;
    kind = cudaMemcpyHostToDevice;
    int id;
    CUDA_CHECK(cudaGetDevice(&id));

    struct ggml_tensor_extra_gpu_ * extra = (ggml_tensor_extra_gpu_ * ) tensor->extra;
    dst_ptr = (char *) extra->data_device[0]+(tensor->nb[0]*n_embd_gqa)*il*n_ctx;
    src_ptr = (char *) tensor->data+(tensor->nb[0]*n_embd_gqa)*il*n_ctx;
    count = (tensor->nb[0]*n_embd_gqa)*(n_past+N);

    cudaError_t r = cudaMemcpy(dst_ptr, src_ptr, count, kind);

    // if (r != cudaSuccess) return r;
    // return cudaSuccess;
}

void ggml_cuda_cpy_tensor_gpu2cpu(const struct ggml_tensor * tensor, size_t il, size_t n_ctx, size_t N, size_t n_past, size_t n_embd_gqa) {
    char * dst_ptr;
    const char * src_ptr;
    int64_t count;
    cudaMemcpyKind kind;
    kind = cudaMemcpyDeviceToHost;
    int id;
    CUDA_CHECK(cudaGetDevice(&id));

    struct ggml_tensor_extra_gpu_ * extra = (ggml_tensor_extra_gpu_ * ) tensor->extra;
    src_ptr = (char *) extra->data_device[0]+(tensor->nb[0]*n_embd_gqa)*il*n_ctx;
    dst_ptr = (char *) tensor->data+(tensor->nb[0]*n_embd_gqa)*il*n_ctx;
    count = (tensor->nb[0]*n_embd_gqa)*(n_past+N);

    cudaError_t r = cudaMemcpy(dst_ptr, src_ptr, count, kind);

    // if (r != cudaSuccess) return r;
    // return cudaSuccess;
}

static __device__ void cpy_1_f16_f16_(const char * cxi, char * cdsti) {
    half * xi = (half *) cxi;
    half * dsti = (half *) cdsti;

    *dsti = *xi;
}

void ggml_cuda_cpy_tensor_v(const struct ggml_tensor * tensor, size_t il, size_t n_ctx, size_t N, size_t n_past, size_t n_embd_gqa, bool cpu2gpu) {


    size_t total_col = n_ctx;

    dim3 threads_per_block(16, 16); 
    dim3 num_blocks((total_col + threads_per_block.x - 1) / threads_per_block.x,
                    (n_embd_gqa + threads_per_block.y - 1) / threads_per_block.y); 

    half * dst_ptr;
    half * src_ptr;
    struct ggml_tensor_extra_gpu_ * extra = (ggml_tensor_extra_gpu_ * ) tensor->extra;
    if(cpu2gpu){
        dst_ptr = (half *) extra->data_device[0]+(tensor->nb[0]*n_embd_gqa)*il*n_ctx;
        src_ptr = (half *) tensor->data+(tensor->nb[0]*n_embd_gqa)*il*n_ctx;
    } else{
        src_ptr = (half *) extra->data_device[0]+(tensor->nb[0]*n_embd_gqa)*il*n_ctx;
        dst_ptr = (half *) tensor->data+(tensor->nb[0]*n_embd_gqa)*il*n_ctx;
    }
    
    transfer_matrix_kernel<cpy_1_f16_f16_><<<num_blocks, threads_per_block>>>(src_ptr, dst_ptr, n_embd_gqa, N+n_past, total_col);

    // std::cout << "n_past: " << n_past << std::endl;
    // std::cout << "layer: " << il << std::endl;
}



void ggml_cuda_cpy_tensor_all_v(const struct ggml_tensor * tensor, size_t il, size_t n_ctx, size_t N, size_t n_past, size_t n_embd_gqa, bool cpu2gpu) {
    char * dst_ptr;
    const char * src_ptr;
    int64_t count;
    cudaMemcpyKind kind;
    // kind = cudaMemcpyDeviceToHost;
    int id;
    CUDA_CHECK(cudaGetDevice(&id));

    struct ggml_tensor_extra_gpu_ * extra = (ggml_tensor_extra_gpu_ * ) tensor->extra;
    if(cpu2gpu){
        kind = cudaMemcpyHostToDevice;
        dst_ptr = (char *) extra->data_device[g_main_device_]+(tensor->nb[0]*n_embd_gqa)*il*n_ctx;
        src_ptr = (char *) tensor->data+(tensor->nb[0]*n_embd_gqa)*il*n_ctx;
    }
    else{
        kind = cudaMemcpyDeviceToHost;
        src_ptr = (char *) extra->data_device[g_main_device_]+(tensor->nb[0]*n_embd_gqa)*il*n_ctx;
        dst_ptr = (char *) tensor->data+(tensor->nb[0]*n_embd_gqa)*il*n_ctx;
    }
    
    count = (tensor->nb[0]*n_embd_gqa)*(n_ctx);

    CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, count, kind, 0));
    // std::cout << "exe offload: " << n_past << std::endl;

    // if (r != cudaSuccess) return r;
    // return cudaSuccess;
}

void ggml_cuda_cpy_cpu_gpu(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst){
    int32_t il = dst->op_params[0];
    int32_t n_ctx = dst->op_params[1];
    int32_t N = dst->op_params[2];
    int32_t n_past = dst->op_params[3];
    int32_t n_embd_gqa = dst->op_params[4];
    int32_t cpu2gpu = dst->op_params[5];
    int32_t is_v = dst->op_params[6];
    cudaMemcpyKind kind;
    void * dst_ptr;
    const void * src_ptr;

    
    if(cpu2gpu==1){
        struct ggml_tensor_extra_gpu_ * extra = (ggml_tensor_extra_gpu_ * ) dst->extra;
        kind = cudaMemcpyHostToDevice;
        dst_ptr = (void *) extra->data_device[g_main_device_]+(2*n_embd_gqa)*il*n_ctx;
        src_ptr = (void *) dst->data+(2*n_embd_gqa)*il*n_ctx;
    }
    else{
        struct ggml_tensor_extra_gpu_ * extra = (ggml_tensor_extra_gpu_ * ) dst->extra;
        kind = cudaMemcpyDeviceToHost;
        src_ptr = (void *) extra->data_device[g_main_device_]+(2*n_embd_gqa)*il*n_ctx;
        dst_ptr = (void *) dst->data+(2*n_embd_gqa)*il*n_ctx;
    }
    int64_t count;
    if(is_v==0){
        count = (2*n_embd_gqa)*(N+n_past);
        CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, count, kind, 0));
    } 
    else{
        
        count = (2*n_embd_gqa)*(N+n_past);
        CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, count, kind, 0));
    }
    // count = (2*n_embd_gqa)*(N+n_past);


    // CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, count, kind, transfer_stream));
    // CUDA_CHECK(cudaMemcpy(dst_ptr, src_ptr, count, kind));
    CUDA_CHECK(cudaDeviceSynchronize());
    // std::cout << "exe offload: " << n_past << std::endl;
}

void ggml_cuda_cpy_w_cpu_gpu(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst){
    cudaMemcpyKind kind;
    void * dst_ptr;
    const void * src_ptr;

    kind = cudaMemcpyHostToDevice;

    int64_t count = (2*dst->ne[0]*dst->ne[1]);

    struct ggml_tensor_extra_gpu_ * extra = (ggml_tensor_extra_gpu_ * ) dst->extra;

    dst_ptr = (void *) extra->data_device[g_main_device_];
    src_ptr = (void *) dst->data;

    CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, count, kind, transfer_w_stream_));
    CUDA_CHECK(cudaDeviceSynchronize());
    // std::cout << "exe offload: " << n_past << std::endl;
}

void ggml_print_cal_data_fp32(float * data_ptr, int ne0, int ne1, int ne2, int data_len, int offset,
                                int token_id, int il){
    std::string output_directory = "kqv_result_re/";
    std::string filename = output_directory+ std::to_string(token_id) + "_token_" + std::to_string(il) + "_layer.txt";
    std::vector<float> data_cpu_ptr(ne0 * ne1 * ne2);
    cudaMemcpy(data_cpu_ptr.data(), data_ptr, ne0 * ne1 * ne2*sizeof(float), cudaMemcpyDeviceToHost);
    std::ostringstream oss;
    std::ofstream result_file(filename);
    if (result_file.is_open()) {
        for (size_t i = 0; i < data_cpu_ptr.size(); ++i) {
            result_file << "Index " << i << ": " << data_cpu_ptr[i] << "\n";
        }
        result_file.close();
        // std::cout << "Results successfully written to matrix_result.txt" << std::endl;
        // oss << "data[" << i << "] = " << data_cpu_ptr[offset+i] << " ";
    }
    // std::cout << oss.str() << std::endl;
}

void ggml_print_cal_data_fp16(half * data_ptr, int ne0, int ne1, int ne2, int data_len,
        int token_id,
        int head_id,
        int n_head_embd,
        int n_head
        ){
    std::vector<__half> data_cpu_ptr(ne0 * ne1 * ne2);
    cudaMemcpy(data_cpu_ptr.data(), data_ptr, ne0 * ne1 * ne2*sizeof(__half), cudaMemcpyDeviceToHost);
    std::ostringstream oss;
    int offset = token_id*n_head*n_head_embd + head_id*n_head_embd;
    for (int i = 0; i < data_len; ++i) {
        // if(i==12){
        //     std::cout << "\n" << " ";
        // }
        oss << "data[" << i << "] = " << __half2float(data_cpu_ptr[offset+i]) << " ";
        
    }
    std::cout << oss.str() << std::endl;
}

static __global__ void mat_mul_contiguous_cuda(float * s_mat, half * v_mat, 
                                                float * dst_mat,
                                               size_t src_token_num, 
                                               size_t tgt_token_num, 
                                               size_t n_head_embed, 
                                               size_t n_head) {
    // Computing matrix multiplication with contiguous memory focuses on computing S*V so that V doesn't have to be transposed
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    // int tgt_token_id = row/n_head;

    // int head_id = row % n_head;

    int tgt_token_id = row % tgt_token_num;

    int head_id = row / tgt_token_num;

    int feat_id = col;

    extern __shared__ float shared_space[];
    // check index
    if (feat_id < n_head_embed && head_id < n_head && tgt_token_id < tgt_token_num) {
        //init shared mem
        #pragma unroll
        for (int src = 0; src < src_token_num; src += 1){
            shared_space[src] = *(s_mat+head_id*tgt_token_num*src_token_num+tgt_token_id*src_token_num+src);
            // output[current_dst_id][src] = nid;
        }
        __syncthreads();
        #pragma unroll
        for (int src = 0; src < src_token_num; src += 1){
            *(dst_mat+head_id*tgt_token_num*n_head_embed+tgt_token_id*n_head_embed + feat_id)  
            += __fmul_rn(shared_space[src], 
            	__half2float(*(v_mat+src*n_head*n_head_embed+head_id*n_head_embed+feat_id)));
            // output[current_dst_id][src] = nid;
        }
    } 
}

inline void ggml_cuda_mul_mat_cont(
        const ggml_tensor * s_mat, const ggml_tensor * v_mat, ggml_tensor * dst
){
    int32_t il = dst->op_params[0];
    int32_t n_ctx = dst->op_params[1];
    int32_t src_token_num = dst->op_params[2];
    int32_t tgt_token_num = dst->op_params[3];
    int32_t n_embd_gqa = dst->op_params[4];
    int32_t n_head = dst->op_params[5];
    int32_t n_head_embed = dst->op_params[6];
    cudaMemcpyKind kind;
    // void * dst_ptr;
    // const void * src_ptr;

    struct ggml_tensor_extra_gpu_ * extra_v = (ggml_tensor_extra_gpu_ * ) v_mat->extra;
    struct ggml_tensor_extra_gpu_ * extra_s = (ggml_tensor_extra_gpu_ * ) s_mat->extra;
    struct ggml_tensor_extra_gpu_ * extra_dst = (ggml_tensor_extra_gpu_ * ) dst->extra;

    half * v_mat_ptr = (half *) extra_v->data_device[g_main_device_];

    float * s_mat_ptr = (float *) extra_s->data_device[g_main_device_];

    // float * dst_tmp_ptr;
    // cudaMalloc(&dst_tmp_ptr, tgt_token_num*n_embd_gqa*sizeof(float));
    float * dst_ptr = (float *) extra_dst->data_device[g_main_device_];
    // extra_dst->data_device[g_main_device_] = dst_tmp_ptr;


    // ggml_print_cal_data_fp16(v_mat_ptr,32,3,128,100,0,0,128,32);

    int shared_memory = src_token_num * sizeof(float);

    const dim3 threads(1, n_head_embed);
    const dim3 blocks((n_head*tgt_token_num+threads.x-1)/threads.x, 
    (n_head_embed+threads.y-1)/threads.y);

    mat_mul_contiguous_cuda<<<blocks, threads, shared_memory>>>(s_mat_ptr, 
    v_mat_ptr, dst_ptr, src_token_num, tgt_token_num, n_head_embed, n_head);

    CUDA_CHECK(cudaDeviceSynchronize());

}

static __global__ void mat_mul_contiguous_out_cuda(float * s_mat, half * v_mat, 
                                                float * dst_mat,
                                               size_t src_token_num, 
                                               size_t tgt_token_num, 
                                               size_t n_head_embed, 
                                               size_t n_head) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    // int tgt_token_id = row/n_head;

    // int head_id = row % n_head;

    int src_token_id = row % src_token_num;

    int head_id = row / src_token_num;

    int feat_id = col;

    extern __shared__ float shared_space[];
    if (feat_id < n_head_embed && head_id < n_head && src_token_id < src_token_num) {
        #pragma unroll
        for (int tgt = 0; tgt < tgt_token_num; tgt += 1){
            shared_space[tgt] = *(s_mat+head_id*tgt_token_num*src_token_num+tgt*src_token_num+src_token_id);
            // output[current_dst_id][src] = nid;
        }
        __syncthreads();
        #pragma unroll
        for (int tgt = 0; tgt < tgt_token_num; tgt += 1){
            float partial_sum = __fmul_rn(shared_space[tgt], 
            	__half2float(*(v_mat+src_token_id*n_head*n_head_embed+head_id*n_head_embed+feat_id)));
            atomicAdd(dst_mat+head_id*tgt_token_num*n_head_embed+tgt*n_head_embed + feat_id, partial_sum);
            // *(dst_mat+head_id*tgt_token_num*n_head_embed+tgt*n_head_embed + feat_id)  
            // += __fmul_rn(shared_space[tgt], 
            // 	__half2float(*(v_mat+src_token_id*n_head*n_head_embed+head_id*n_head_embed+feat_id)));
            // output[current_dst_id][src] = nid;
        }
    } 
}

inline void ggml_cuda_mul_mat_cont_out(
        const ggml_tensor * s_mat, const ggml_tensor * v_mat, ggml_tensor * dst
){
    int32_t il = dst->op_params[0];
    int32_t n_ctx = dst->op_params[1];
    int32_t src_token_num = dst->op_params[2];
    int32_t tgt_token_num = dst->op_params[3];
    int32_t n_embd_gqa = dst->op_params[4];
    int32_t n_head = dst->op_params[5];
    int32_t n_head_embed = dst->op_params[6];
    cudaMemcpyKind kind;
    // void * dst_ptr;
    // const void * src_ptr;

    struct ggml_tensor_extra_gpu_ * extra_v = (ggml_tensor_extra_gpu_ * ) v_mat->extra;
    struct ggml_tensor_extra_gpu_ * extra_s = (ggml_tensor_extra_gpu_ * ) s_mat->extra;
    struct ggml_tensor_extra_gpu_ * extra_dst = (ggml_tensor_extra_gpu_ * ) dst->extra;

    half * v_mat_ptr = (half *) extra_v->data_device[g_main_device_];

    float * s_mat_ptr = (float *) extra_s->data_device[g_main_device_];

    // float * dst_tmp_ptr;
    // cudaMalloc(&dst_tmp_ptr, tgt_token_num*n_embd_gqa*sizeof(float));
    float * dst_ptr = (float *) extra_dst->data_device[g_main_device_];
    // extra_dst->data_device[g_main_device_] = dst_tmp_ptr;


    // ggml_print_cal_data_fp16(v_mat_ptr,32,3,128,100,0,0,128,32);

    int shared_memory = tgt_token_num * sizeof(float);

    const dim3 threads(1, n_head_embed);
    const dim3 blocks((n_head*src_token_num+threads.x-1)/threads.x, 
    (n_head_embed+threads.y-1)/threads.y);

    mat_mul_contiguous_out_cuda<<<blocks, threads, shared_memory>>>(s_mat_ptr, 
    v_mat_ptr, dst_ptr, src_token_num, tgt_token_num, n_head_embed, n_head);

    CUDA_CHECK(cudaDeviceSynchronize());

    // if(src_token_num>1){
    //     ggml_print_cal_data_fp32(dst_ptr,dst->ne[0],dst->ne[1],dst->ne[2],
    //     100,0,(src_token_num-tgt_token_num),il);
    // }
}

inline void ggml_cuda_free_tmp_data(
        const ggml_tensor * s_mat, const ggml_tensor * v_mat, ggml_tensor * dst
){
    ggml_cuda_free_data(dst);
}

static __global__ void relu_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = fmaxf(x[i], 0);
}

static void relu_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_RELU_BLOCK_SIZE - 1) / CUDA_RELU_BLOCK_SIZE;
    relu_f32<<<num_blocks, CUDA_RELU_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

inline void ggml_cuda_op_relu(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    struct ggml_tensor_extra_gpu_ * extra_src0 = (ggml_tensor_extra_gpu_ * ) src0->extra;
    struct ggml_tensor_extra_gpu_ * extra_dst = (ggml_tensor_extra_gpu_ * ) dst->extra;

    float * src0_dd = (float *) extra_src0->data_device[g_main_device_];
    float * dst_dd = (float *) extra_dst->data_device[g_main_device_];


    relu_f32_cuda(src0_dd, dst_dd, ggml_nelements(src0), 0);

    // (void) src1;
    // (void) dst;
    // (void) src0_dd;
}