#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>

// 三层全连接网络结构
typedef struct {
    int input_size;
    int hidden_size;
    int output_size;
    int batch_size;

    // 权重和偏置
    float *d_W1, *d_b1;
    float *d_W2, *d_b2;
    
    // 中间结果缓存
    float *d_z1, *d_a1;
    float *d_z2, *d_a2;
    
    // 梯度缓存
    float *d_dW1, *d_db1;
    float *d_dW2, *d_db2;
    
    // cuBLAS句柄
    cublasHandle_t handle;
} FCNetwork;

// 初始化网络
void init_network(FCNetwork *net, 
    int input_size, 
    int hidden_size, 
    int output_size, 
    int batch_size,

    float* h_W1,
    float* h_W2,
    float* h_b1,
    float* h_b2
) {

    net->input_size = input_size;
    net->hidden_size = hidden_size;
    net->output_size = output_size;
    net->batch_size = batch_size;

    // 初始化cuBLAS
    cublasCreate(&net->handle);
    
    // 分配设备内存
    cudaMalloc((void**)&net->d_W1, input_size * hidden_size * sizeof(float));
    cudaMalloc((void**)&net->d_b1, hidden_size * sizeof(float));
    cudaMalloc((void**)&net->d_W2, hidden_size * output_size * sizeof(float));
    cudaMalloc((void**)&net->d_b2, output_size * sizeof(float));
    
    // 中间结果
    cudaMalloc((void**)&net->d_z1, batch_size * hidden_size * sizeof(float));
    cudaMalloc((void**)&net->d_a1, batch_size * hidden_size * sizeof(float));
    cudaMalloc((void**)&net->d_z2, batch_size * output_size * sizeof(float));
    cudaMalloc((void**)&net->d_a2, batch_size * output_size * sizeof(float));
    
    // 梯度内存
    cudaMalloc((void**)&net->d_dW1, input_size * hidden_size * sizeof(float));
    cudaMalloc((void**)&net->d_dW2, hidden_size * output_size * sizeof(float));
    cudaMalloc((void**)&net->d_db1, hidden_size * sizeof(float));
    cudaMalloc((void**)&net->d_db2, output_size * sizeof(float));
    

    // 将初始化值复制到设备
    cudaMemcpy(net->d_W1, h_W1, input_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(net->d_W2, h_W2, input_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(net->d_b1, h_b1, hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(net->d_b2, h_b2, hidden_size * sizeof(float), cudaMemcpyHostToDevice);
}

// 释放网络资源
void free_network(FCNetwork *net) {
    // 释放设备内存
    cudaFree(net->d_W1);
    cudaFree(net->d_b1);
    cudaFree(net->d_W2);
    cudaFree(net->d_b2);

    cudaFree(net->d_dW1);
    cudaFree(net->d_dW2);
    cudaFree(net->d_db1);
    cudaFree(net->d_db2);


    // 销毁cuBLAS句柄
    cublasDestroy(net->handle);
}

// 前向传播函数
void forward(FCNetwork *net, float *d_input) {
    const float alpha = 1.0f, beta = 0.0f;
    int threadsPerBlock = 256;
    
    // 第一层: z1 = input x W1 + b1, b1被广播到 z1的每一行
    //   n x b = (n x a) * (b x a)^T  + (1 x b)
    cublasSgemm(net->handle,
                CUBLAS_OP_N, CUBLAS_OP_T, 
                net->batch_size, net->hidden_size, // 结果 n x b
                net->input_size,                    // a
                &alpha, 
                d_input, net->batch_size,           // n
                net->d_W1, net->input_size, 
                &beta, 
                net->d_z1, net->hidden_size);
    
    // 添加偏置 (简化实现，实际应使用广播操作)
    
    
    // 应用Sigmoid激活函数
    int blocks1 = (net->hidden_size1 * batch_size + threadsPerBlock - 1) / threadsPerBlock;
    sigmoid_kernel<<<blocks1, threadsPerBlock>>>(net->d_z1, net->d_a1, net->hidden_size1 * batch_size);
    
    // 第二层: z2 = W2*a1 + b2
    cublasSgemm(net->handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                net->hidden_size2, batch_size, net->hidden_size1,
                &alpha, net->d_W2, net->hidden_size2, 
                net->d_a1, net->hidden_size1, &beta, net->d_z2, net->hidden_size2);
    
    // 添加偏置并应用激活函数
    int blocks2 = (net->hidden_size2 * batch_size + threadsPerBlock - 1) / threadsPerBlock;
    sigmoid_kernel<<<blocks2, threadsPerBlock>>>(net->d_z2, net->d_a2, net->hidden_size2 * batch_size);
    
    // 第三层: z3 = W3*a2 + b3
    cublasSgemm(net->handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                net->output_size, batch_size, net->hidden_size2,
                &alpha, net->d_W3, net->output_size, 
                net->d_a2, net->hidden_size2, &beta, net->d_z3, net->output_size);
    
    // 添加偏置并应用激活函数
    int blocks3 = (net->output_size * batch_size + threadsPerBlock - 1) / threadsPerBlock;
    sigmoid_kernel<<<blocks3, threadsPerBlock>>>(net->d_z3, net->d_a3, net->output_size * batch_size);
}


extern "C"
{
   __declspec(dllexport) 
   void actor_forward(
    const float *input,     // n x a
    float *output,          // n x c
    int batch_size,         // n
    int input_size,          // a
    int hidden_size,         // b
    int output_size,         // c
    const float *fc1_w,     // b x a
    const float *fc2_w,     // c x b
    const float *fc1_b,     // 1 x b
    const float *fc2_b      // 1 x c
    )
  {
  
 
  }
}
