#include <cuda_fp16.h>

using ull = unsigned long long;

static int indices_size, inputR_stride, axis_size, output_size;
constexpr int block_size = 128;

template<typename T>
__global__ void gather(const T *input, const ull*indices, T *output, int N, int axis_size, int inputR_stride)
{
    /* 
        gridDim.x : indices_size
        gridDim.y : inputL
        blockIdx.x : indices
        blockIdx.y : inputL
        threadIdx.x : inputR
    */
    int output_offset = (blockIdx.x + blockIdx.y * gridDim.x) * inputR_stride;
    int input_offset = (blockIdx.y * axis_size + indices[blockIdx.x]) * inputR_stride;
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
    gather<T><<<grid_size, block_size>>>(static_cast<const T*>(input), 
                                         static_cast<const ull*>(indices), 
                                         static_cast<T*>(output), 
                                         output_size, axis_size, inputR_stride);
}

extern "C" void gather_nv_f32(const void *input, const void *indices, void *output, 
                              const int axis, 
                              const int indices_size_, const int inputR_stride_,
                              const int axis_size_, const int output_size_)
{
    indices_size = indices_size_;
    inputR_stride = inputR_stride_;
    axis_size = axis_size_;
    output_size = output_size_;
    gatherLaunch<float>(input, indices, output, output_size);
}

extern "C" void gather_nv_f16(const void *input, const void *indices, void *output, 
                              const int axis, 
                              const int indices_size_, const int inputR_stride_,
                              const int axis_size_, const int output_size_)
{
    indices_size = indices_size_;
    inputR_stride = inputR_stride_;
    axis_size = axis_size_;
    output_size = output_size_;
    gatherLaunch<half>(input, indices, output, output_size);
}
