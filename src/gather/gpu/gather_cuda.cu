#include <cuda_fp16.h>

using ull = unsigned long long;
using uint = unsigned int;

static uint indices_size, inputR_stride, axis_size, output_size, grid_max_size;
constexpr uint block_max_size = 128;

template<typename T>
__global__ void gather(const T *input, const ull*indices, T *output, 
                       const uint grid_max_size, 
                       const uint N, const uint axis_size, const uint inputR_stride, const uint indices_size)
{
    /* 
        blockIdx.x : inputL | index
        threadIdx.x : inputR
    */
    for(uint bid = blockIdx.x; bid < grid_max_size; bid += gridDim.x)
    {
        const uint inputL = bid / indices_size;
        const uint index = bid % indices_size;
        const uint output_offset = inputL * indices_size * inputR_stride + index * inputR_stride;
        const uint input_offset = inputL * axis_size * inputR_stride + indices[index] * inputR_stride;

        for(uint tid = threadIdx.x; tid < inputR_stride; tid += blockDim.x)
            if(output_offset + tid < N)
                output[output_offset + tid] = input[input_offset + tid];
    }
}

template<typename T>
void gatherLaunch(const void *input, const void *indices, void *output)
{
    grid_max_size = (output_size + inputR_stride - 1) / inputR_stride;
    const uint grid_size = std::min(grid_max_size, static_cast<uint>(INT_MAX));
    const uint block_size = std::max(static_cast<uint>(32), std::min(inputR_stride, block_max_size));
    gather<T><<<grid_size, block_size>>>(static_cast<const T*>(input), 
                                         static_cast<const ull*>(indices), 
                                         static_cast<T*>(output), 
                                         grid_max_size,
                                         output_size, axis_size, inputR_stride, indices_size);
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
    gatherLaunch<float>(input, indices, output);
}

extern "C" void gather_nv_f16(const void *input, const void *indices, void *output, const int axis, 
                              const int *input_shape, const int input_shape_len, 
                              const int *indices_shape, const int indices_shape_len)
{
    dataPreprocess(input_shape, input_shape_len, indices_shape, indices_shape_len, axis);
    gatherLaunch<half>(input, indices, output);
}
