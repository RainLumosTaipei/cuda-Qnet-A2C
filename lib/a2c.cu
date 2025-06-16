
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>


__global__ void linear_forward(
    const float *input,   // 输入  n x a
    float *output,        // 输出  n x b
    const float *weights, // b x a
    const float *bias,    // b
    int input_dim,        // a
    int output_dim,       // b
    int batch_size)       // n
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // n x b的索引
    if (idx >= batch_size * output_dim)              // n x b
        return;

    int line_idx = idx / output_dim; // 行数 x
    int row_idx = idx % output_dim;  // 列数 y : [0, b-1]

    float sum = 0.0f;
    // n x b = (n x a) * (b x a)^T + (1 x b)
    //            x行         y行
    //        n * x + i     b * y + i
    for (int i = 0; i < input_dim; ++i) // a
    {
        sum +=
            input[line_idx * input_dim + i] * weights[row_idx * output_dim + i];
    }
    sum += bias[row_idx]; // bias
    output[idx] = sum;
}

__global__ void relu(float *input, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;
    float val = input[idx];
    input[idx] = fmaxf(0.0f, val);
}

__global__ void softmax(float *input, int input_dim, int batch_size)
{
    __shared__ float sum[1024];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= input_dim * batch_size)
        return;

    int line_idx = idx / input_dim; // 行数 x
    int row_idx = idx % input_dim;  // 列数 y : [0, c-1]

    input[idx] = expf(input[idx]);

    if (row_idx == 0)
        sum[line_idx] = 0.0f;
    __syncthreads();
    atomicAdd(&sum[line_idx], input[idx]);
    __syncthreads();

    input[idx] /= sum[line_idx];
}

__global__ void linear_fc2_backward(
    const float *__restrict__ hidden,      // n x b
    float *__restrict__ grad_output, // n x c
    float *__restrict__ grad_fc2_w,        // b x c
    float *__restrict__ grad_fc2_b,        // 1 x c
    int hidden_dim,                        // b
    int output_dim,                        // c
    int batch_size)                        // n
{

    // i：输出层神经元索引（范围 0~c-1）
    // j：隐藏层神经元索引（范围 0~b-1）
    // b：批次样本索引（范围 0~n-1）

    int i = blockIdx.x * blockDim.x + threadIdx.x; // c
    int j = blockIdx.y * blockDim.y + threadIdx.y; // b
    int k = blockIdx.z * blockDim.z + threadIdx.z; // n

    if (i >= output_dim || j >= hidden_dim || k >= batch_size)
        return;

    // (n x b)^T * (n x c) = b x c
    //  第 j 列       第 i 列
    // k x b + j    k x c + i
    float partial =
        hidden[k * hidden_dim + j] * grad_output[k * output_dim + i];
    // j , i
    atomicAdd(&grad_fc2_w[j * output_dim + i], partial);
    

    // j = 0 只计算一次
    if (j == 0)
    {
        // 对梯度矩阵按列求和
        // 第 i 列求和
        // n x c --> 1 x c
        atomicAdd(&grad_fc2_b[i], grad_output[k * output_dim + i]);
    }
}


__global__ void linear_fc1_backward(
    const float *__restrict__ input,       // n x a
    const float *__restrict__ hidden,      // n x b
    const float *__restrict__ grad_output, // n x c
    const float *__restrict__ fc2_w,       // b x c
    float *__restrict__ grad_hidden,       // n x b
    float *__restrict__ grad_fc1_w,        // b x a
    float *__restrict__ grad_fc1_b,        // 1 x b
    int input_dim,                         // a
    int hidden_dim,                        // b
    int output_dim,                        // c
    int batch_size)                        // n
{

    int i = blockIdx.x * blockDim.x + threadIdx.x; // b
    int j = blockIdx.y * blockDim.y + threadIdx.y; // a
    int k = blockIdx.z * blockDim.z + threadIdx.z; // n

    if (i >= hidden_dim || j >= input_dim || k >= batch_size)
        return;

    // 若隐藏层输出值 ≤0，则梯度为 0，直接跳过计算
    // ReLU
    if (hidden[k * hidden_dim + i] > 0)
    {
        // 隐藏层的梯度 delta 等于输出层的梯度 grad_output 与权重 fc2_weights 的加权和
        // n x b = (n x c) * (b x c)^T
        // 第 k 行       第 i 行
        float sum = 0.0f;
        for (int l = 0; l < output_dim; ++l)
        {
            sum +=
                grad_output[k * output_dim + l] * fc2_w[i * output_dim + l];
        }
        grad_hidden[k * hidden_dim + i] = sum;
    }

    __syncthreads();

    // 计算 grad_fc1_w
    // b x a = (n x b)^T * (n x a)
    //          第 i 列     第 j 列
    if (hidden[k * hidden_dim + i] > 0)
    {
        float partial = input[k * input_dim + j] * grad_hidden[k * hidden_dim + i];
        atomicAdd(&grad_fc1_w[i * input_dim + j], partial);
    }

    // j = 0 只计算一次
    if (j == 0 && hidden[k * hidden_dim + i] > 0)
    {
        // 对梯度矩阵按列求和
        // 第 i 列求和
        // n x b --> 1 x b
        atomicAdd(&grad_fc1_b[i], grad_hidden[k * hidden_dim + i]);
    }
}


// 计算softmax的梯度
__global__ void softmax_backward(
    const float *__restrict__ output,      // n x c (softmax输出)
    float *__restrict__ grad_output, // n x c (下游梯度)
    int output_dim,                        // c
    int batch_size)                        // n
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // c
    int k = blockIdx.y * blockDim.y + threadIdx.y; // n

    if (i < output_dim && k < batch_size)
    {
        float grad = 0.0f;
        for (int l = 0; l < output_dim; l++) {
            // 当 i==l 时为1，否则为0
            float delta_il = (i == l) ? 1.0f : 0.0f;
            grad += grad_output[k * output_dim + l] * 
                   (output[k * output_dim + i] * (delta_il - output[k * output_dim + l]));
        }
        grad_output[k * output_dim + i] = grad;
    }
}


extern "C"
{
    __declspec(dllexport) void actor_forward(
        const float *input,
        float *hidden,
        float *output,
        int batch_size,     // n
        int input_dim,      // a
        int hidden_dim,     // b
        int output_dim,     // c
        const float *fc1_w, // b x a
        const float *fc2_w, // c x b
        const float *fc1_b, // 1 x b
        const float *fc2_b  // 1 x c
    )
    {
        float *d_input, *d_hidden, *d_output;
        float *d_fc1_w, *d_fc1_b, *d_fc2_w, *d_fc2_b;

        cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));   // n x a
        cudaMalloc(&d_fc1_w, hidden_dim * input_dim * sizeof(float));   // b x a
        cudaMalloc(&d_fc1_b, hidden_dim * sizeof(float));               // b
        cudaMalloc(&d_hidden, batch_size * hidden_dim * sizeof(float)); // n x b
        cudaMalloc(&d_fc2_w, output_dim * hidden_dim * sizeof(float));
        cudaMalloc(&d_fc2_b, output_dim * sizeof(float));
        cudaMalloc(&d_output, batch_size * output_dim * sizeof(float));

        cudaMemcpy(d_input, input, batch_size * input_dim * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_fc1_w, fc1_w, hidden_dim * input_dim * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_fc1_b, fc1_b, hidden_dim * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_fc2_w, fc2_w, output_dim * hidden_dim * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_fc2_b, fc2_b, output_dim * sizeof(float),
                   cudaMemcpyHostToDevice);

        int threads = 256;
        dim3 block1((batch_size * hidden_dim + threads - 1) / threads); // 均分 n x b
        linear_forward<<<block1, threads>>>(
            d_input, d_hidden,
            d_fc1_w, d_fc1_b,
            input_dim, hidden_dim, batch_size);

        relu<<<block1, threads>>>(d_hidden, batch_size * hidden_dim);

        dim3 block2((batch_size * output_dim + threads - 1) / threads);
        linear_forward<<<block2, threads>>>(
            d_hidden, d_output,
            d_fc2_w, d_fc2_b,
            hidden_dim, output_dim, batch_size);

        softmax<<<block2, threads>>>(d_output, output_dim, batch_size);

        cudaMemcpy(output, d_output, batch_size * output_dim * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(hidden, d_hidden, batch_size * hidden_dim * sizeof(float),
                   cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_hidden);
        cudaFree(d_output);
        cudaFree(d_fc1_w);
        cudaFree(d_fc1_b);
        cudaFree(d_fc2_w);
        cudaFree(d_fc2_b);
    }

    __declspec(dllexport) void critic_forward(
        const float *input,
        float *hidden,
        float *output,
        int batch_size,     // n
        int input_dim,      // a
        int hidden_dim,     // b
        int output_dim,     // c
        const float *fc1_w, // b x a
        const float *fc2_w, // c x b
        const float *fc1_b, // 1 x b
        const float *fc2_b  // 1 x c
    )
    {
        float *d_input, *d_hidden, *d_output;
        float *d_fc1_w, *d_fc1_b, *d_fc2_w, *d_fc2_b;

        cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));   // n x a
        cudaMalloc(&d_fc1_w, hidden_dim * input_dim * sizeof(float));   // b x a
        cudaMalloc(&d_fc1_b, hidden_dim * sizeof(float));               // b
        cudaMalloc(&d_hidden, batch_size * hidden_dim * sizeof(float)); // n x b
        cudaMalloc(&d_fc2_w, output_dim * hidden_dim * sizeof(float));
        cudaMalloc(&d_fc2_b, output_dim * sizeof(float));
        cudaMalloc(&d_output, batch_size * output_dim * sizeof(float));

        cudaMemcpy(d_input, input, batch_size * input_dim * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_fc1_w, fc1_w, hidden_dim * input_dim * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_fc1_b, fc1_b, hidden_dim * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_fc2_w, fc2_w, output_dim * hidden_dim * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_fc2_b, fc2_b, output_dim * sizeof(float),
                   cudaMemcpyHostToDevice);

        int threads = 256;
        dim3 block1((batch_size * hidden_dim + threads - 1) / threads); // 均分 n x b
        linear_forward<<<block1, threads>>>(
            d_input, d_hidden,
            d_fc1_w, d_fc1_b,
            input_dim, hidden_dim, batch_size);

        relu<<<block1, threads>>>(d_hidden, batch_size * hidden_dim);

        dim3 block2((batch_size * output_dim + threads - 1) / threads);
        linear_forward<<<block2, threads>>>(
            d_hidden, d_output,
            d_fc2_w, d_fc2_b,
            hidden_dim, output_dim, batch_size);

        cudaMemcpy(output, d_output, batch_size * output_dim * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(hidden, d_hidden, batch_size * hidden_dim * sizeof(float),
                   cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_hidden);
        cudaFree(d_output);
        cudaFree(d_fc1_w);
        cudaFree(d_fc1_b);
        cudaFree(d_fc2_w);
        cudaFree(d_fc2_b);
    }

    __declspec(dllexport) void critic_backward(
        const float *input,       // n x a
        const float *hidden,      // n x b
        const float *grad_output, // n x c
        const float *fc2_weights, // b x c
        int batch_size,           // n
        int input_dim,            // a
        int hidden_dim,           // b
        int output_dim,           // c
        float *grad_fc1_w,        // a x b
        float *grad_fc2_w,        // b x c
        float *grad_fc1_b,        // 1 x b
        float *grad_fc2_b)        // 1 x c
    {
        float *d_input, *d_hidden;
        float *d_grad_output, *d_grad_hidden;
        float *d_fc2_weights;
        float *d_grad_fc1_w, *d_grad_fc1_b, *d_grad_fc2_w, *d_grad_fc2_b;

        cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
        cudaMalloc(&d_hidden, batch_size * hidden_dim * sizeof(float));
        cudaMalloc(&d_grad_output, batch_size * output_dim * sizeof(float));
        cudaMalloc(&d_fc2_weights, output_dim * hidden_dim * sizeof(float));
        cudaMalloc(&d_grad_hidden, batch_size * hidden_dim * sizeof(float));
        cudaMalloc(&d_grad_fc1_w, hidden_dim * input_dim * sizeof(float));
        cudaMalloc(&d_grad_fc1_b, hidden_dim * sizeof(float));
        cudaMalloc(&d_grad_fc2_w, output_dim * hidden_dim * sizeof(float));
        cudaMalloc(&d_grad_fc2_b, output_dim * sizeof(float));

        // input copy
        cudaMemcpy(d_input, input, batch_size * input_dim * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_hidden, hidden, batch_size * hidden_dim * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_grad_output, grad_output,
                   batch_size * output_dim * sizeof(float), 
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_fc2_weights, fc2_weights,
                   output_dim * hidden_dim * sizeof(float), 
                   cudaMemcpyHostToDevice);

        // result memset
        cudaMemset(d_grad_hidden, 0, batch_size * hidden_dim * sizeof(float));
        cudaMemset(d_grad_fc1_w, 0, hidden_dim * input_dim * sizeof(float));
        cudaMemset(d_grad_fc1_b, 0, hidden_dim * sizeof(float));
        cudaMemset(d_grad_fc2_w, 0, output_dim * hidden_dim * sizeof(float));
        cudaMemset(d_grad_fc2_b, 0, output_dim * sizeof(float));

        dim3 fc2_block(8, 8, 4); // 256 = 8 * 8 * 4  thread count
        // n x b x c
        dim3 fc2_grid((output_dim + fc2_block.x - 1) / fc2_block.x,  // c
                      (hidden_dim + fc2_block.y - 1) / fc2_block.y,  // b
                      (batch_size + fc2_block.z - 1) / fc2_block.z); // n

        linear_fc2_backward<<<fc2_grid, fc2_block>>>(
            d_hidden, d_grad_output, d_grad_fc2_w, d_grad_fc2_b,
            hidden_dim, output_dim, batch_size); // b, c , n

        dim3 fc1_block(8, 8, 4);                                     // 256 = 8 * 8 * 4
        dim3 fc1_grid((hidden_dim + fc1_block.x - 1) / fc1_block.x,  // b
                      (input_dim + fc1_block.y - 1) / fc1_block.y,   // a
                      (batch_size + fc1_block.z - 1) / fc1_block.z); // n

        linear_fc1_backward<<<fc1_grid, fc1_block>>>(
            d_input, d_hidden, d_grad_output, d_fc2_weights,
            d_grad_hidden,
            d_grad_fc1_w, d_grad_fc1_b,
            input_dim, hidden_dim, output_dim, batch_size);

        float scale = 1.0f / batch_size;
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSscal(handle, hidden_dim * input_dim, &scale, d_grad_fc1_w, 1);
        cublasSscal(handle, hidden_dim, &scale, d_grad_fc1_b, 1);
        cublasSscal(handle, output_dim * hidden_dim, &scale, d_grad_fc2_w, 1);
        cublasSscal(handle, output_dim, &scale, d_grad_fc2_b, 1);
        cublasDestroy(handle);

        cudaMemcpy(grad_fc1_w, d_grad_fc1_w,
                   input_dim * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(grad_fc1_b, d_grad_fc1_b,
                   hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(grad_fc2_w, d_grad_fc2_w,
                   output_dim * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(grad_fc2_b, d_grad_fc2_b,
                   output_dim * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_hidden);
        cudaFree(d_grad_output);
        cudaFree(d_fc2_weights);
        cudaFree(d_grad_fc1_w);
        cudaFree(d_grad_fc1_b);
        cudaFree(d_grad_fc2_w);
        cudaFree(d_grad_fc2_b);
        cudaFree(d_grad_hidden);
    }

    __declspec(dllexport) void actor_backward(
        const float *input,       // n x a
        const float *hidden,      // n x b
        float *output,      // n x c
        const float *grad_output, // n x c
        const float *fc2_weights, // b x c
        int batch_size,           // n
        int input_dim,            // a
        int hidden_dim,           // b
        int output_dim,           // c
        float *grad_fc1_w,        // a x b
        float *grad_fc2_w,        // b x c
        float *grad_fc1_b,        // 1 x b
        float *grad_fc2_b)        // 1 x c
    {
        float *d_input, *d_hidden, *d_output;
        float *d_grad_output, *d_grad_hidden;
        float *d_fc2_weights;
        float *d_grad_fc1_w, *d_grad_fc1_b, *d_grad_fc2_w, *d_grad_fc2_b;

        cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
        cudaMalloc(&d_hidden, batch_size * hidden_dim * sizeof(float));
        cudaMalloc(&d_output, batch_size * output_dim * sizeof(float));
        cudaMalloc(&d_grad_output, batch_size * output_dim * sizeof(float));
        cudaMalloc(&d_fc2_weights, output_dim * hidden_dim * sizeof(float));
        cudaMalloc(&d_grad_hidden, batch_size * hidden_dim * sizeof(float));
        cudaMalloc(&d_grad_fc1_w, hidden_dim * input_dim * sizeof(float));
        cudaMalloc(&d_grad_fc1_b, hidden_dim * sizeof(float));
        cudaMalloc(&d_grad_fc2_w, output_dim * hidden_dim * sizeof(float));
        cudaMalloc(&d_grad_fc2_b, output_dim * sizeof(float));

        // input copy
        cudaMemcpy(d_input, input, batch_size * input_dim * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_hidden, hidden, batch_size * hidden_dim * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_output, output, batch_size * output_dim * sizeof(float), 
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_grad_output, grad_output,
                   batch_size * output_dim * sizeof(float), 
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_fc2_weights, fc2_weights,
                   output_dim * hidden_dim * sizeof(float), 
                   cudaMemcpyHostToDevice);

        // result memset
        cudaMemset(d_grad_hidden, 0, batch_size * hidden_dim * sizeof(float));
        cudaMemset(d_grad_fc1_w, 0, hidden_dim * input_dim * sizeof(float));
        cudaMemset(d_grad_fc1_b, 0, hidden_dim * sizeof(float));
        cudaMemset(d_grad_fc2_w, 0, output_dim * hidden_dim * sizeof(float));
        cudaMemset(d_grad_fc2_b, 0, output_dim * sizeof(float));

        dim3 fc2_block(8, 8, 4); // 256 = 8 * 8 * 4  thread count
        // n x b x c
        dim3 fc2_grid((output_dim + fc2_block.x - 1) / fc2_block.x,  // c
                      (hidden_dim + fc2_block.y - 1) / fc2_block.y,  // b
                      (batch_size + fc2_block.z - 1) / fc2_block.z); // n

        softmax_backward<<<fc2_grid, fc2_block>>>(
            d_output, d_grad_output,
            output_dim, batch_size);

        linear_fc2_backward<<<fc2_grid, fc2_block>>>(
            d_hidden, d_grad_output, d_grad_fc2_w, d_grad_fc2_b,
            hidden_dim, output_dim, batch_size); // b, c , n

        dim3 fc1_block(8, 8, 4);                                     // 256 = 8 * 8 * 4
        dim3 fc1_grid((hidden_dim + fc1_block.x - 1) / fc1_block.x,  // b
                      (input_dim + fc1_block.y - 1) / fc1_block.y,   // a
                      (batch_size + fc1_block.z - 1) / fc1_block.z); // n

        linear_fc1_backward<<<fc1_grid, fc1_block>>>(
            d_input, d_hidden, d_grad_output, d_fc2_weights,
            d_grad_hidden,
            d_grad_fc1_w, d_grad_fc1_b,
            input_dim, hidden_dim, output_dim, batch_size);

        float scale = 1.0f / batch_size;
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSscal(handle, hidden_dim * input_dim, &scale, d_grad_fc1_w, 1);
        cublasSscal(handle, hidden_dim, &scale, d_grad_fc1_b, 1);
        cublasSscal(handle, output_dim * hidden_dim, &scale, d_grad_fc2_w, 1);
        cublasSscal(handle, output_dim, &scale, d_grad_fc2_b, 1);
        cublasDestroy(handle);

        cudaMemcpy(grad_fc1_w, d_grad_fc1_w,
                   input_dim * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(grad_fc1_b, d_grad_fc1_b,
                   hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(grad_fc2_w, d_grad_fc2_w,
                   output_dim * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(grad_fc2_b, d_grad_fc2_b,
                   output_dim * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_hidden);
        cudaFree(d_output);
        cudaFree(d_grad_output);
        cudaFree(d_fc2_weights);
        cudaFree(d_grad_fc1_w);
        cudaFree(d_grad_fc1_b);
        cudaFree(d_grad_fc2_w);
        cudaFree(d_grad_fc2_b);
        cudaFree(d_grad_hidden);
    }

}
