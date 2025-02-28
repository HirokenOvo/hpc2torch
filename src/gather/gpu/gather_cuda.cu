#include <cuda_fp16.h>
#include <cstdio>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                __FILE__, __LINE__, static_cast<int>(err), cudaGetErrorName(err), cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

using ll = long long;

static int indices_size, inputR_stride, axis_size, output_size;
const int block_size = 128;

template<typename T>
__global__ void gather(const T *input, const ll*indices, T *output, int N, int axis_size, int inputR_stride)
{
    /* 
        blockIdx.x : indices
        blockIdx.y : inputL
        threadIdx.x : inputR
        gridDim.x : indices_size
        gridDim.y : inputL
    */
    int output_offset = (blockIdx.x + blockIdx.y * gridDim.x) * inputR_stride;
    int input_offset = (blockIdx.y * axis_size + indices[blockIdx.x]) * inputR_stride;
    // int output_offset = blockIdx.y * inputR_stride + blockIdx.x * d_indices_size * inputR_stride;
    // int input_offset = blockIdx.x * axis_size * inputR_stride + indices[blockIdx.y] * inputR_stride;
    for(int tid = threadIdx.x; tid < inputR_stride; tid += block_size)
        if(output_offset + tid < N)
            output[output_offset + tid] = input[input_offset + tid];
}

template<typename T>
void gatherLaunch(const void *input, const void *indices, void *output, int output_size)
{
    int base = indices_size * inputR_stride;
    int inputL = (output_size + base - 1) / base;
    dim3 grid_size(indices_size, inputL);
    // dim3 grid_size(inputL, indices_size);
    // printf("%d %d %d\n",inputL, indices_size, block_size.x);
    gather<T><<<grid_size, block_size>>>(static_cast<const T*>(input), 
                                         static_cast<const ll*>(indices), 
                                         static_cast<T*>(output), 
                                         output_size, axis_size, inputR_stride);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void dataPreprocess(const int *input_shape, const int input_shape_len, 
                           const int *indices_shape, const int indices_shape_len, 
                           const int axis)
{
    axis_size = input_shape[axis];
    indices_size = inputR_stride = output_size = 1;
    for(int i = 0; i < indices_shape_len; i++)
        indices_size *= indices_shape[i];
    
    for(int i = input_shape_len - 1; i >= 0; i--)
    {
        output_size *= input_shape[i];
        if(i > axis)
            inputR_stride *= input_shape[i];
    }
    output_size = output_size / axis_size * indices_size;
}

extern "C" void gather_nv_f32(const void *input, const void *indices, void *output, const int axis, 
                              const int *input_shape, const int input_shape_len, 
                              const int *indices_shape, const int indices_shape_len)
{
    dataPreprocess(input_shape, input_shape_len, indices_shape, indices_shape_len, axis);
    gatherLaunch<float>(input, indices, output, output_size);
}

extern "C" void gather_nv_f16(const void *input, const void *indices, void *output, const int axis, 
                              const int *input_shape, const int input_shape_len, 
                              const int *indices_shape, const int indices_shape_len)
{
    dataPreprocess(input_shape, input_shape_len, indices_shape, indices_shape_len, axis);
    gatherLaunch<half>(input, indices, output, output_size);
}
