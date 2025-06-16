
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>

#define WARP_SIZE 32


__global__ void fc1_forward_kernel(
  const float *input,   // 输入  n x a
  const float *weights,  // a x b
  const float *bias,   // b
   float *hidden,     // 输出                  
   int input_dim,     // a
   int hidden_dim,    // b       
   int batch_size)    // n
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x; // n x b的索引
  if (idx >= batch_size * hidden_dim)  // n x b
    return;

  int batch_idx = idx / hidden_dim; // 行数 x
  int neuron_idx = idx % hidden_dim;  // 列数 y

  float sum = 0.0f;
  // n x a         a x b  = n x b
  //  x行           y列
  // n * x + i     i * a + y
  for (int i = 0; i < input_dim; ++i)  // a
  {
    sum +=
        input[batch_idx * input_dim + i] * weights[neuron_idx + input_dim * i];
  }
  sum += bias[neuron_idx]; // bias
  hidden[idx] = fmaxf(0.0f, sum); // ReLU
}

__global__ void fc2_forward_kernel(
  const float *hidden, // n x b
  const float *weights,
  const float *bias, 
  float *output,  // n x c        
  int hidden_dim, // b
  int output_dim, // c                 
  int batch_size)  // n
{

  const int output_idx = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;  // n x c索引
  if (output_idx >= batch_size * output_dim) // n x c x 32
    return;

  const int batch_idx = output_idx / output_dim;  // 行数 x
  const int neuron_idx = output_idx % output_dim; // 列数 y

  // 256个线程内部分组
  const int lane_id = threadIdx.x % WARP_SIZE;  // 组内索引

  float sum = 0.0f;
  // n x b         b x c  = n x c
  //  x行              y列
  // n * x + i     i * b + y
  for (int i = lane_id; i < hidden_dim; i += WARP_SIZE) // 每32个线程同时计算一个乘积
  {
    sum += hidden[batch_idx * hidden_dim + i] * weights[neuron_idx + hidden_dim * i];
  }


  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
  {
    sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
  }


  if (lane_id == 0)
  {
    output[output_idx] = sum + bias[neuron_idx];
  }
}


__global__ void fc2_backward_kernel(
  const float *__restrict__ hidden,     // n x b                         
  const float *__restrict__ grad_output,      // n x c                     
  float *__restrict__ grad_fc2_w,          // b x c                
  float *__restrict__ grad_fc2_b,        // 1 x c          
  int hidden_dim, // b
  int output_dim,  // c    
  int batch_size)  // n
{

  // i：输出层神经元索引（范围 0~c-1）
  // j：隐藏层神经元索引（范围 0~b-1）
  // b：批次样本索引（范围 0~n-1）

  int i = blockIdx.x * blockDim.x + threadIdx.x; // c
  int j = blockIdx.y * blockDim.y + threadIdx.y;  // b
  int k = blockIdx.z * blockDim.z + threadIdx.z;  // n


  if (i < output_dim && j < hidden_dim && k < batch_size)
  {
    // (n x b)^T * (n x c) = b x c
    //  第 j 列       第 i 列
    // k x b + j    k x c + i
    float partial =
        hidden[k * hidden_dim + j] * grad_output[k * output_dim + i];
    // j , i
    atomicAdd(&grad_fc2_w[j * output_dim + i], partial);
  }

  // j = 0 只计算一次
  if (i < output_dim && k < batch_size && j == 0)
  { 
    // 对梯度矩阵按列求和
    // 第 i 列求和
    // n x c --> 1 x c
    atomicAdd(&grad_fc2_b[i], grad_output[k * output_dim + i]);
  }
}


__global__ void fc1_backward_kernel(
  const float *__restrict__ input,              // n x a              
  const float *__restrict__ hidden,            // n x b                 
  const float *__restrict__ grad_output,       // n x c                   
  const float *__restrict__ fc2_w,       //   b x c   
  float *__restrict__ grad_hidden,    // n x b           
  float *__restrict__ grad_fc1_w,         // a x b                  
  float *__restrict__ grad_fc1_b,         // 1 x b    
  int input_dim,   // a
  int hidden_dim,    // b             
  int output_dim, // c
  int batch_size)  // n
{

  int i = blockIdx.x * blockDim.x + threadIdx.x;  // b
  int j = blockIdx.y * blockDim.y + threadIdx.y;  // a
  int k = blockIdx.z * blockDim.z + threadIdx.z;  // n

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
    for (int l = 0; l < output_dim; ++l){
      sum +=
          grad_output[k * output_dim + l] * fc2_w[i * output_dim + l];
    }
    grad_hidden[k * hidden_dim + i] = sum;
  }
  
  __syncthreads();

  // 计算 grad_fc1_w
  // a x b = (n x a)^T * (n x b)
  // 第 j 列        第 i 列
  if (hidden[k * hidden_dim + i] > 0) { 
    float partial = input[k * input_dim + j] * grad_hidden[k * hidden_dim + i] ;
    atomicAdd(&grad_fc1_w[j * hidden_dim + i], partial);
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

extern "C"
{
   __declspec(dllexport) 
   void cuda_forward(
    const float *input, 
    int batch_size,   // n
    int input_dim,    // a
    int hidden_dim,   // b
    int output_dim,   // c
    const float *fc1_w,  // b x a
    const float *fc1_b, 
    const float *fc2_w, // c x b
    const float *fc2_b,
    float *output)
  {
    float *d_input, *d_fc1_w, *d_fc1_b, *d_fc2_w, *d_fc2_b;
    float *d_hidden, *d_output;

 
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float)); // n x a
    cudaMalloc(&d_fc1_w, hidden_dim * input_dim * sizeof(float)); // b x a
    cudaMalloc(&d_fc1_b, hidden_dim * sizeof(float)); // b

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
    dim3 block1((batch_size * hidden_dim + threads - 1) / threads);  // 均分 n x b
    // 全部使用一维
    // block数量 x 线程数
    fc1_forward_kernel<<<block1, threads>>>(d_input, d_fc1_w, d_fc1_b, d_hidden,
                                            input_dim, hidden_dim, batch_size);

    // 使用 warp 每32个计算
    dim3 block2((batch_size * output_dim * WARP_SIZE + threads - 1) / threads);  // n x c x 32
    fc2_forward_kernel<<<block2, threads>>>(d_hidden, d_fc2_w, d_fc2_b, d_output,
                                            hidden_dim, output_dim, batch_size);


    cudaMemcpy(output, d_output, batch_size * output_dim * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_fc1_w);
    cudaFree(d_fc1_b);
    cudaFree(d_hidden);
    cudaFree(d_fc2_w);
    cudaFree(d_fc2_b);
    cudaFree(d_output);
  }

   __declspec(dllexport) void cuda_backward(
    const float *input,       // n x a
    const float *hidden,     // n x b
    const float *grad_output, // n x c
    const float *fc2_weights, // b x c
    int input_dim, // a
    int hidden_dim, // b
    int output_dim,  // c
    int batch_size,   // n
    float *grad_fc1_w, // a x b
    float *grad_fc1_b,  // 1 x b
    float *grad_fc2_w, // b x c
    float *grad_fc2_b)  // 1 x c
  {
    float *d_input, *d_hidden, *d_grad_output, *d_fc2_weights;
    float *d_grad_fc1_w, *d_grad_fc1_b, *d_grad_fc2_w, *d_grad_fc2_b;
    float *d_grad_hidden;

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
               batch_size * output_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc2_weights, fc2_weights,
               output_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);

    // result memset
    cudaMemset(d_grad_hidden, 0, batch_size * hidden_dim * sizeof(float));
    cudaMemset(d_grad_fc1_w, 0, hidden_dim * input_dim * sizeof(float));
    cudaMemset(d_grad_fc1_b, 0, hidden_dim * sizeof(float));
    cudaMemset(d_grad_fc2_w, 0, output_dim * hidden_dim * sizeof(float));
    cudaMemset(d_grad_fc2_b, 0, output_dim * sizeof(float));

 
    dim3 fc2_block(8, 8, 4);  // 256 = 8 * 8 * 4  thread count
    // n x b x c
    dim3 fc2_grid((output_dim + fc2_block.x - 1) / fc2_block.x, // c
                  (hidden_dim + fc2_block.y - 1) / fc2_block.y,  // b
                  (batch_size + fc2_block.z - 1) / fc2_block.z);  // n

    fc2_backward_kernel<<<fc2_grid, fc2_block>>>(
      d_hidden, d_grad_output, d_grad_fc2_w, d_grad_fc2_b, 
      hidden_dim, output_dim, batch_size);  // b, c , n

 
    dim3 fc1_block(8, 8, 4); // 256 = 8 * 8 * 4
    dim3 fc1_grid((hidden_dim + fc1_block.x - 1) / fc1_block.x, // b
                  (input_dim + fc1_block.y - 1) / fc1_block.y,  // a
                  (batch_size + fc1_block.z - 1) / fc1_block.z);  // n

    fc1_backward_kernel<<<fc1_grid, fc1_block>>>(
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
}
