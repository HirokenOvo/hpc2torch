## Gather

不管多少维度，将其视为一维处理。对于 outTensor ，其总元素大小为`outTensor_size = inputTensor_size / axis_size * Indices_size`。对于每个 outTensor 中的元素，可将其下标 output_idx 分解为

```
---------------------------------
|  inputL  |  index  |  inputR  |
---------------------------------
```

通过 index 在 indices 数组中找到相应的下标 indices[ index ]，即为 inputTensor 中 axis 维度的值。然后将 inputL、indices、inputR 按照相应的权值重新组合为 inputIdx 后 在 inputTensor 中找到相应的元素，将其赋值给 outTensor[ output_idx ]。

~~将 inputL、 index 作为二维 gridDim 减少取模和除法开销，~~ block_size 设置为 128 线程并按照 block_size 进行 block-stride loop。

🤡:

-   测试框架中，tensor 本来就生成在 GPU 端，无需`cudaDeviceSynchronize()`与主机端同步。
-   gridDim 在 x、y、z 三个方向的最大允许值分别为 INT_MAX、65535、65535，因此当测试数据为`((512, 128, 4, 4), (1, 1, 1), 2, torch.float16, "cuda")`时 gridDim.y 会炸，需要 grid-stride loop 处理所有数据。gridDim 用一维，INT_MAX 基本可以满足数据大小，实在不够也能用 grid-stride loop 覆盖到所有数据。
-   想用`cudaOccupancyMaxPotentialBlockSize()`和`cudaOccupancyMaxActiveBlocksPerMultiprocessor`针对设备动态设置线程块数目的，但实在太慢了，不适合放在算子里面。

    ```cpp
    int blockSize = 0;
    int minGridSize = 0;
    // 获取最优块大小
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        gather_kernel<T>, 0, 0   
    );

    int gridSize = (total_elements + blockSize - 1) / blockSize;

    // 查询设备属性优化配置
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // 计算最大可用网格大小
    int numSMs = prop.multiProcessorCount;
    int maxBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM,
        gather_kernel<T>,
        blockSize,
        0
    );

    // 优化网格大小以最大化SM利用率
    int optimalGridSize = numSMs * maxBlocksPerSM;
    gridSize = std::min(optimalGridSize, gridSize);
    gridSize = std::min(gridSize, prop.maxGridSize[0]);
    ```
