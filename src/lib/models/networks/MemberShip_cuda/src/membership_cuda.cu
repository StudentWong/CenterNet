// #include "src/gpu.cuh"
#include "src/utils.hpp"

#define BLOCK 32
#define IDX2D(i, j, dj) (dj * i + j)
#define MINGRADCLIP 0.0000

int max_dim(int N, int C){
    if (N>C) {
        return N;
    }
    else{
        return C;
    }
}
// 获取grid
dim3 cuda_gridsize(int h, int w)
{
    int y = (h- 1) / BLOCK + 1;
    int x = (w-1) / BLOCK + 1;
    // dim3 d(x, y, 1);
    dim3 d(x, y);
    return d;
}

dim3 cuda_block(){
    dim3 B(BLOCK, BLOCK);
    return B;
}

template <typename Dtype>
__global__ void MemberShipForwardKernel(Dtype* in, Dtype* c, Dtype* la, Dtype* o, int N, int D, int C)
{
    //in: N*D
    //c: D*C
    //la: D*C
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    Dtype Ovalue = 1.0;
    if((y >= N) || (x>=C)) return;
    for (int i=0; i<D; i++){
        Ovalue *= expf(-(powf((in[IDX2D(y, i, D)] - c[IDX2D(i, x, C)]), 2)/(0.0001+2 * la[IDX2D(i, x, C)] * la[IDX2D(i, x, C)])));
      }
    o[IDX2D(y, x, C)] = Ovalue;
    // o[0] = 5.0;
}


template <typename Dtype>
void MemberShipForward(Dtype* in, Dtype* c, Dtype* la, Dtype* o, int N, int D, int C,
                  cudaStream_t stream) {
  
  int maxNC = max_dim(N, C);
  int maxDNC = max_dim(maxNC, D);
  MemberShipForwardKernel <Dtype>
            <<<cuda_gridsize(maxDNC, maxDNC), cuda_block(), 0, stream>>>(in, c, la, o, N, D, C);

    
//   cudaError_t err = cudaGetLastError();
  cudaError_t err = cudaDeviceSynchronize();
  if (cudaSuccess != err)
    throw std::runtime_error(Formatter()
                             << "CUDA kernel failed : " << cudaGetErrorString(err));
}


template <typename Dtype>
__global__ void MemberShipInputBackwardKernel(Dtype* in_g, Dtype* g_l, Dtype* in, Dtype* c, Dtype* la, Dtype* o, int N, int D, int C)
{
    //in_g: N*D
    //g_l: N*C
    //in: N*D
    //c: D*C
    //la: D*C
    //o: N*C
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    Dtype Ovalue = 0.0;
    if((y >= N) || (x>=D)) return;
    for (int i=0; i<C; i++){
        Ovalue += g_l[IDX2D(y, i, C)] * o[IDX2D(y, i, C)] * (-(in[IDX2D(y, x, D)] - c[IDX2D(x, i, C)])/(0.0001+la[IDX2D(x, i, C)]*la[IDX2D(x, i, C)]));
      }
      if (Ovalue>0.0 && Ovalue<MINGRADCLIP){
        Ovalue = MINGRADCLIP;
      }
      else if (Ovalue<0.0 && Ovalue>-MINGRADCLIP){
        Ovalue = -MINGRADCLIP;
      }
      in_g[IDX2D(y, x, D)] = Ovalue;
    // o[0] = 5.0;
}

template <typename Dtype>
void MemberShipInputBackward(Dtype* in_g, Dtype* g_l, Dtype* in, Dtype* c, Dtype* la, Dtype* o, int N, int D, int C,
                  cudaStream_t stream) {
  
  int maxNC = max_dim(N, C);
  int maxDNC = max_dim(maxNC, D);
  MemberShipInputBackwardKernel <Dtype>
            <<<cuda_gridsize(maxDNC, maxDNC), cuda_block(), 0, stream>>>(in_g, g_l, in, c, la, o, N, D, C);

    
//   cudaError_t err = cudaGetLastError();
  cudaError_t err = cudaDeviceSynchronize();
  if (cudaSuccess != err)
    throw std::runtime_error(Formatter()
                             << "CUDA kernel failed : " << cudaGetErrorString(err));
}


template <typename Dtype>
__global__ void MemberShipCenterBackwardKernel(Dtype* c_g, Dtype* g_l, Dtype* in, Dtype* c, Dtype* la, Dtype* o, int N, int D, int C)
{
    //c_g: D*C
    //g_l: N*C
    //in: N*D
    //c: D*C
    //la: D*C
    //o: N*C
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    Dtype Ovalue = 0.0;
    if((y >= D) || (x>=C)) return;
    for (int i=0; i<N; i++){
        Ovalue += g_l[IDX2D(i, x, C)] * o[IDX2D(i, x, C)] * ((in[IDX2D(i, y, D)] - c[IDX2D(y, x, C)])/(0.0001+la[IDX2D(y, x, C)]*la[IDX2D(y, x, C)]));
      }
      if (Ovalue>0.0 && Ovalue<MINGRADCLIP){
        Ovalue = MINGRADCLIP;
      }
      else if (Ovalue<0.0 && Ovalue>-MINGRADCLIP){
        Ovalue = -MINGRADCLIP;
      }
      c_g[IDX2D(y, x, C)] = Ovalue;
    // o[0] = 5.0;
}

template <typename Dtype>
void MemberShipCenterBackward(Dtype* c_g, Dtype* g_l, Dtype* in, Dtype* c, Dtype* la, Dtype* o, int N, int D, int C,
                  cudaStream_t stream) {
  
  int maxNC = max_dim(N, C);
  int maxDNC = max_dim(maxNC, D);
  MemberShipCenterBackwardKernel <Dtype>
            <<<cuda_gridsize(maxDNC, maxDNC), cuda_block(), 0, stream>>>(c_g, g_l, in, c, la, o, N, D, C);

    
//   cudaError_t err = cudaGetLastError();
  cudaError_t err = cudaDeviceSynchronize();
  if (cudaSuccess != err)
    throw std::runtime_error(Formatter()
                             << "CUDA kernel failed : " << cudaGetErrorString(err));
}


template <typename Dtype>
__global__ void MemberShipLamdaBackwardKernel(Dtype* la_g, Dtype* g_l, Dtype* in, Dtype* c, Dtype* la, Dtype* o, int N, int D, int C)
{
    //la_g: D*C
    //g_l: N*C
    //in: N*D
    //c: D*C
    //la: D*C
    //o: N*C
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    Dtype Ovalue = 0.0;
    if((y >= D) || (x>=C)) return;
    for (int i=0; i<N; i++){
      Dtype eps;
      if (la[IDX2D(y, x, C)] > 0){
        eps = 0.0001;
      }
      else if (la[IDX2D(y, x, C)] < 0){
        eps = -0.0001;
      }
      Ovalue += g_l[IDX2D(i, x, C)] * o[IDX2D(i, x, C)] * (powf((in[IDX2D(i, y, D)] - c[IDX2D(y, x, C)]), 2)/(eps+la[IDX2D(y, x, C)]*la[IDX2D(y, x, C)]*la[IDX2D(y, x, C)]));
    }
    if (Ovalue>0.0 && Ovalue<MINGRADCLIP){
      Ovalue = MINGRADCLIP;
    }
    else if (Ovalue<0.0 && Ovalue>-MINGRADCLIP){
      Ovalue = -MINGRADCLIP;
    }
    la_g[IDX2D(y, x, C)] = Ovalue;
    // o[0] = 5.0;
}

template <typename Dtype>
void MemberShipLamdaBackward(Dtype* la_g, Dtype* g_l, Dtype* in, Dtype* c, Dtype* la, Dtype* o, int N, int D, int C,
                  cudaStream_t stream) {
  
  int maxNC = max_dim(N, C);
  int maxDNC = max_dim(maxNC, D);
  MemberShipLamdaBackwardKernel <Dtype>
            <<<cuda_gridsize(maxDNC, maxDNC), cuda_block(), 0, stream>>>(la_g, g_l, in, c, la, o, N, D, C);

    
//   cudaError_t err = cudaGetLastError();
  cudaError_t err = cudaDeviceSynchronize();
  if (cudaSuccess != err)
    throw std::runtime_error(Formatter()
                             << "CUDA kernel failed : " << cudaGetErrorString(err));
}


// template <typename Dtype>
// __global__ void CenterLossForwardKernel(Dtype* in, Dtype* c, Dtype* gt, Dtype* gts, int N, int D, int C)
// {
//     //in: N*D
//     //c: D*C
//     //la: D*C
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
    
//     Dtype Ovalue = 1.0;
//     if((y >= N) || (x>=C)) return;
//     for (int i=0; i<D; i++){
//         Ovalue *= expf(-(powf((in[IDX2D(y, i, D)] - c[IDX2D(i, x, C)]), 2)/(0.0001+2 * la[IDX2D(i, x, C)] * la[IDX2D(i, x, C)])));
//       }
//     o[IDX2D(y, x, C)] = Ovalue;
//     // o[0] = 5.0;
// }


// template <typename Dtype>
// void CenterLossForward(Dtype* in, Dtype* c, Dtype* gt, Dtype* gts, Dtype* ret, int N, int D, int C,
//                   cudaStream_t stream) {
  
//   int maxNC = max_dim(N, C);
//   int maxDNC = max_dim(maxNC, D);
//   MemberShipForwardKernel <Dtype>
//             <<<cuda_gridsize(maxDNC, maxDNC), cuda_block(), 0, stream>>>(in, c, la, o, N, D, C);

    
// //   cudaError_t err = cudaGetLastError();
//   cudaError_t err = cudaDeviceSynchronize();
//   if (cudaSuccess != err)
//     throw std::runtime_error(Formatter()
//                              << "CUDA kernel failed : " << cudaGetErrorString(err));
// }

template void MemberShipForward<float>(float *in, float *c, float *la, float *o, int N, int D, int C,
                                  cudaStream_t stream);

template void MemberShipInputBackward<float>(float* in_g, float* g_l, float* in, float* c, float* la, float* o, int N, int D, int C,
                                  cudaStream_t stream);

template void MemberShipCenterBackward<float>(float* c_g, float* g_l, float* in, float* c, float* la, float* o, int N, int D, int C,
                                  cudaStream_t stream);

template void MemberShipLamdaBackward<float>(float* la_g, float* g_l, float* in, float* c, float* la, float* o, int N, int D, int C,
                                  cudaStream_t stream);
