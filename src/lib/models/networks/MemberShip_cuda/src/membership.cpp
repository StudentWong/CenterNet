#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "src/membership_cuda.cuh"
#include "src/utils.hpp"

template <typename Dtype>
void MemberShip_Forward_Wrapper(at::Tensor in, at::Tensor center, at::Tensor lamda, at::Tensor out) {

  if ( in.dim() != 2 || center.dim() != 2 || lamda.dim() != 2 || out.dim() != 2){
    throw std::invalid_argument(Formatter()
                              << "Dim error");
  }
  if ( *(in.sizes().data()) != *(out.sizes().data()) ){
    throw std::invalid_argument(Formatter()
                              << "Size mismatch input N: " << *(in.sizes().data())
                              << ", output N: " << *(out.sizes().data()));
  }
  int sizeN = *(in.sizes().data());

  if ( *(in.sizes().data()+1) != *(center.sizes().data()) || *(in.sizes().data()+1) != *(lamda.sizes().data()) ){
    throw std::invalid_argument(Formatter()
                              << "Size mismatch input D: " << *(in.sizes().data()+1)
                              << ", center D: " << *(center.sizes().data())
                              << ", lamda D: " << *(lamda.sizes().data()));
  }
  int sizeD = *(in.sizes().data()+1);

  if ( *(out.sizes().data()+1) != *(center.sizes().data()+1) || *(out.sizes().data()+1) != *(lamda.sizes().data()+1) ){
    throw std::invalid_argument(Formatter()
                              << "Size mismatch output C: " << *(out.sizes().data()+1)
                              << ", center C: " << *(center.sizes().data()+1)
                              << ", lamda C: " << *(lamda.sizes().data()+1));
  }
  int sizeC = *(out.sizes().data()+1);
  
  // out.resize_({sizeN*sizeC});

  MemberShipForward<Dtype>(in.data<Dtype>(), center.data<Dtype>(),
                      lamda.data<Dtype>(), out.data<Dtype>(), sizeN, sizeD, sizeC, at::cuda::getCurrentCUDAStream());
}




template <typename Dtype>
void MemberShip_Input_Backward_Wrapper(at::Tensor in_grad, at::Tensor grad_last, 
at::Tensor in, at::Tensor center, at::Tensor lamda, at::Tensor out) {

  if ( in.dim() != 2 || center.dim() != 2 || lamda.dim() != 2 || out.dim() != 2 || in_grad.dim() != 2 || grad_last.dim() != 2){
    throw std::invalid_argument(Formatter()
                              << "Dim error");
  }
  if ( *(in.sizes().data()) != *(out.sizes().data()) || *(in_grad.sizes().data()) != *(out.sizes().data()) || *(in.sizes().data()) != *(grad_last.sizes().data())){
    throw std::invalid_argument(Formatter()
                              << "Size mismatch input N: " << *(in.sizes().data())
                              << ", output N: " << *(out.sizes().data())
                              << ", grad_last N: " << *(grad_last.sizes().data())
                              << ", in_grad N: " << *(in_grad.sizes().data()));
  }
  int sizeN = *(in.sizes().data());

  if ( *(in.sizes().data()+1) != *(center.sizes().data()) || *(in.sizes().data()+1) != *(lamda.sizes().data()) || *(in_grad.sizes().data()+1) != *(center.sizes().data())){
    throw std::invalid_argument(Formatter()
                              << "Size mismatch input D: " << *(in.sizes().data()+1)
                              << ", center D: " << *(center.sizes().data())
                              << ", in_grad D: " << *(in_grad.sizes().data()+1)
                              << ", lamda D: " << *(lamda.sizes().data()));
  }
  int sizeD = *(in.sizes().data()+1);

  if ( *(out.sizes().data()+1) != *(center.sizes().data()+1) || *(out.sizes().data()+1) != *(lamda.sizes().data()+1) || *(grad_last.sizes().data()+1) != *(center.sizes().data()+1) ){
    throw std::invalid_argument(Formatter()
                              << "Size mismatch output C: " << *(out.sizes().data()+1)
                              << ", center C: " << *(center.sizes().data()+1)
                              << ", grad_last C: " << *(grad_last.sizes().data()+1)
                              << ", lamda C: " << *(lamda.sizes().data()+1));
  }
  int sizeC = *(out.sizes().data()+1);
  
  // out.resize_({sizeN*sizeC});

  MemberShipInputBackward<Dtype>(in_grad.data<Dtype>(), grad_last.data<Dtype>(),
                      in.data<Dtype>(), center.data<Dtype>(), lamda.data<Dtype>(), 
                      out.data<Dtype>(), sizeN, sizeD, sizeC, at::cuda::getCurrentCUDAStream());
}


template <typename Dtype>
void MemberShip_Center_Backward_Wrapper(at::Tensor c_grad, at::Tensor grad_last, 
at::Tensor in, at::Tensor center, at::Tensor lamda, at::Tensor out) {

  if ( in.dim() != 2 || center.dim() != 2 || lamda.dim() != 2 || out.dim() != 2 || c_grad.dim() != 2 || grad_last.dim() != 2){
    throw std::invalid_argument(Formatter()
                              << "Dim error");
  }
  if ( *(in.sizes().data()) != *(out.sizes().data()) || *(in.sizes().data()) != *(grad_last.sizes().data())){
    throw std::invalid_argument(Formatter()
                              << "Size mismatch input N: " << *(in.sizes().data())
                              << ", output N: " << *(out.sizes().data())
                              << ", grad_last N: " << *(grad_last.sizes().data()));
  }
  int sizeN = *(in.sizes().data());

  if ( *(in.sizes().data()+1) != *(center.sizes().data()) || *(in.sizes().data()+1) != *(lamda.sizes().data()) || *(c_grad.sizes().data()) != *(center.sizes().data())){
    throw std::invalid_argument(Formatter()
                              << "Size mismatch input D: " << *(in.sizes().data()+1)
                              << ", center D: " << *(center.sizes().data())
                              << ", c_grad D: " << *(c_grad.sizes().data())
                              << ", lamda D: " << *(lamda.sizes().data()));
  }
  int sizeD = *(in.sizes().data()+1);

  if ( *(out.sizes().data()+1) != *(center.sizes().data()+1) || *(out.sizes().data()+1) != *(lamda.sizes().data()+1) || *(grad_last.sizes().data()+1) != *(center.sizes().data()+1) || *(out.sizes().data()+1) != *(c_grad.sizes().data()+1) ){
    throw std::invalid_argument(Formatter()
                              << "Size mismatch output C: " << *(out.sizes().data()+1)
                              << ", center C: " << *(center.sizes().data()+1)
                              << ", c_grad C: " << *(c_grad.sizes().data()+1)
                              << ", grad_last C: " << *(grad_last.sizes().data()+1)
                              << ", lamda C: " << *(lamda.sizes().data()+1));
  }
  int sizeC = *(out.sizes().data()+1);
  
  // out.resize_({sizeN*sizeC});

  MemberShipCenterBackward<Dtype>(c_grad.data<Dtype>(), grad_last.data<Dtype>(),
                      in.data<Dtype>(), center.data<Dtype>(), lamda.data<Dtype>(), 
                      out.data<Dtype>(), sizeN, sizeD, sizeC, at::cuda::getCurrentCUDAStream());
}


template <typename Dtype>
void MemberShip_Lamda_Backward_Wrapper(at::Tensor la_grad, at::Tensor grad_last, 
at::Tensor in, at::Tensor center, at::Tensor lamda, at::Tensor out) {

  if ( in.dim() != 2 || center.dim() != 2 || lamda.dim() != 2 || out.dim() != 2 || la_grad.dim() != 2 || grad_last.dim() != 2){
    throw std::invalid_argument(Formatter()
                              << "Dim error");
  }
  if ( *(in.sizes().data()) != *(out.sizes().data()) || *(in.sizes().data()) != *(grad_last.sizes().data())){
    throw std::invalid_argument(Formatter()
                              << "Size mismatch input N: " << *(in.sizes().data())
                              << ", output N: " << *(out.sizes().data())
                              << ", grad_last N: " << *(grad_last.sizes().data()));
  }
  int sizeN = *(in.sizes().data());

  if ( *(in.sizes().data()+1) != *(center.sizes().data()) || *(in.sizes().data()+1) != *(lamda.sizes().data()) || *(la_grad.sizes().data()) != *(center.sizes().data())){
    throw std::invalid_argument(Formatter()
                              << "Size mismatch input D: " << *(in.sizes().data()+1)
                              << ", center D: " << *(center.sizes().data())
                              << ", la_grad D: " << *(la_grad.sizes().data())
                              << ", lamda D: " << *(lamda.sizes().data()));
  }
  int sizeD = *(in.sizes().data()+1);

  if ( *(out.sizes().data()+1) != *(center.sizes().data()+1) || *(out.sizes().data()+1) != *(lamda.sizes().data()+1) || *(grad_last.sizes().data()+1) != *(center.sizes().data()+1) || *(out.sizes().data()+1) != *(la_grad.sizes().data()+1) ){
    throw std::invalid_argument(Formatter()
                              << "Size mismatch output C: " << *(out.sizes().data()+1)
                              << ", center C: " << *(center.sizes().data()+1)
                              << ", la_grad C: " << *(la_grad.sizes().data()+1)
                              << ", grad_last C: " << *(grad_last.sizes().data()+1)
                              << ", lamda C: " << *(lamda.sizes().data()+1));
  }
  int sizeC = *(out.sizes().data()+1);
  
  // out.resize_({sizeN*sizeC});

  MemberShipLamdaBackward<Dtype>(la_grad.data<Dtype>(), grad_last.data<Dtype>(),
                      in.data<Dtype>(), center.data<Dtype>(), lamda.data<Dtype>(), 
                      out.data<Dtype>(), sizeN, sizeD, sizeC, at::cuda::getCurrentCUDAStream());
}


template <typename Dtype>
at::Tensor CenterLoss_Forward_Wrapper(at::Tensor in, at::Tensor center, at::Tensor gt, at::Tensor gt_sum) {

  if ( in.dim() != 2 || center.dim() != 2 || gt.dim() != 2 || gt_sum.dim() != 1){
    throw std::invalid_argument(Formatter()
                              << "Dim error");
  }
  if ( *(in.sizes().data()) != *(gt.sizes().data()) || *(in.sizes().data()) != *(gt_sum.sizes().data()) ){
    throw std::invalid_argument(Formatter()
                              << "Size mismatch input N: " << *(in.sizes().data())
                              << ", gt N: " << *(gt.sizes().data())
                              << ", gt_sum N: " << *(gt_sum.sizes().data()));
  }
  int sizeN = *(in.sizes().data());

  if ( *(in.sizes().data()+1) != *(center.sizes().data()) ){
    throw std::invalid_argument(Formatter()
                              << "Size mismatch input D: " << *(in.sizes().data()+1)
                              << ", center D: " << *(center.sizes().data()));
  }
  int sizeD = *(in.sizes().data()+1);

  if ( *(gt.sizes().data()+1) != *(center.sizes().data()+1) ){
    throw std::invalid_argument(Formatter()
                              << "Size mismatch gt C: " << *(gt.sizes().data()+1)
                              << ", center C: " << *(center.sizes().data()+1));
  }
  int sizeC = *(gt.sizes().data()+1);

  at::Tensor output = at::ones({sizeN, sizeD}, in.options());
  CenterLossForward<Dtype>(in.data<Dtype>(), center.data<Dtype>(),
                         gt.data<Dtype>(), gt_sum.data<Dtype>(), output.data<Dtype>(), sizeN, sizeD, sizeC, at::cuda::getCurrentCUDAStream());
  return output;
  
  // out.resize_({sizeN*sizeC});

  // CenterLossForward<Dtype>(in.data<Dtype>(), center.data<Dtype>(),
  //                     gt.data<Dtype>(), gt_sum.data<Dtype>(), sizeN, sizeD, sizeC, at::cuda::getCurrentCUDAStream());
}


template <typename Dtype>
void CenterLoss_Input_Backward_Wrapper(at::Tensor in_grad, at::Tensor grad_last, at::Tensor in, at::Tensor center, at::Tensor gt, at::Tensor gt_sum) {

  if ( in_grad.dim() != 2 || grad_last.dim() != 2 || in.dim() != 2 || center.dim() != 2 || gt.dim() != 2 || gt_sum.dim() != 1){
    throw std::invalid_argument(Formatter()
                              << "Dim error");
  }
  if ( *(in.sizes().data()) != *(gt.sizes().data()) || *(in.sizes().data()) != *(gt_sum.sizes().data()) || *(in_grad.sizes().data()) != *(gt.sizes().data()) || *(grad_last.sizes().data()) != *(gt.sizes().data()) ){
    throw std::invalid_argument(Formatter()
                              << "Size mismatch input N: " << *(in.sizes().data())
                              << "in_grad N: " << *(in_grad.sizes().data())
                              << "grad_last N: " << *(grad_last.sizes().data())
                              << ", gt N: " << *(gt.sizes().data())
                              << ", gt_sum N: " << *(gt_sum.sizes().data()));
  }
  int sizeN = *(in.sizes().data());

  if ( *(in.sizes().data()+1) != *(center.sizes().data()) || *(in_grad.sizes().data()+1) != *(center.sizes().data()) || *(grad_last.sizes().data()+1) != *(center.sizes().data()) ){
    throw std::invalid_argument(Formatter()
                              << "Size mismatch input D: " << *(in.sizes().data()+1)
                              << ", center D: " << *(center.sizes().data())
                              << ", input_grad D: " << *(in_grad.sizes().data()+1)
                              << ", grad_last D: " << *(grad_last.sizes().data()+1));
  }
  int sizeD = *(in.sizes().data()+1);

  if ( *(gt.sizes().data()+1) != *(center.sizes().data()+1) ){
    throw std::invalid_argument(Formatter()
                              << "Size mismatch gt C: " << *(gt.sizes().data()+1)
                              << ", center C: " << *(center.sizes().data()+1));
  }
  int sizeC = *(gt.sizes().data()+1);


  CenterLossInputBackward<Dtype>(in_grad.data<Dtype>(), grad_last.data<Dtype>(), in.data<Dtype>(), center.data<Dtype>(),
                         gt.data<Dtype>(), gt_sum.data<Dtype>(), sizeN, sizeD, sizeC, at::cuda::getCurrentCUDAStream());
  
  // out.resize_({sizeN*sizeC});

  // CenterLossForward<Dtype>(in.data<Dtype>(), center.data<Dtype>(),
  //                     gt.data<Dtype>(), gt_sum.data<Dtype>(), sizeN, sizeD, sizeC, at::cuda::getCurrentCUDAStream());
}


template <typename Dtype>
void CenterLoss_Center_Backward_Wrapper(at::Tensor c_grad, at::Tensor grad_last, at::Tensor in, at::Tensor center, at::Tensor gt, at::Tensor gt_sum) {

  if ( c_grad.dim() != 2 || grad_last.dim() != 2 || in.dim() != 2 || center.dim() != 2 || gt.dim() != 2 || gt_sum.dim() != 1){
    throw std::invalid_argument(Formatter()
                              << "Dim error");
  }
  if ( *(in.sizes().data()) != *(gt.sizes().data()) || *(in.sizes().data()) != *(gt_sum.sizes().data()) || *(grad_last.sizes().data()) != *(gt.sizes().data()) ){
    throw std::invalid_argument(Formatter()
                              << "Size mismatch input N: " << *(in.sizes().data())
                              << "grad_last N: " << *(grad_last.sizes().data())
                              << ", gt N: " << *(gt.sizes().data())
                              << ", gt_sum N: " << *(gt_sum.sizes().data()));
  }
  int sizeN = *(in.sizes().data());

  if ( *(in.sizes().data()+1) != *(center.sizes().data()) || *(c_grad.sizes().data()) != *(center.sizes().data()) || *(grad_last.sizes().data()+1) != *(center.sizes().data()) ){
    throw std::invalid_argument(Formatter()
                              << "Size mismatch input D: " << *(in.sizes().data()+1)
                              << ", center D: " << *(center.sizes().data())
                              << ", c_grad D: " << *(c_grad.sizes().data())
                              << ", grad_last D: " << *(grad_last.sizes().data()+1));
  }
  int sizeD = *(in.sizes().data()+1);

  if ( *(gt.sizes().data()+1) != *(center.sizes().data()+1) || *(gt.sizes().data()+1) != *(c_grad.sizes().data()+1) ){
    throw std::invalid_argument(Formatter()
                              << "Size mismatch gt C: " << *(gt.sizes().data()+1)
                              << ", center C: " << *(center.sizes().data()+1)
                              << ", c_grad C: " << *(c_grad.sizes().data()+1));
  }
  int sizeC = *(gt.sizes().data()+1);


  CenterLossCenterBackward<Dtype>(c_grad.data<Dtype>(), grad_last.data<Dtype>(), in.data<Dtype>(), center.data<Dtype>(),
                         gt.data<Dtype>(), gt_sum.data<Dtype>(), sizeN, sizeD, sizeC, at::cuda::getCurrentCUDAStream());
  
  // out.resize_({sizeN*sizeC});

  // CenterLossForward<Dtype>(in.data<Dtype>(), center.data<Dtype>(),
  //                     gt.data<Dtype>(), gt_sum.data<Dtype>(), sizeN, sizeD, sizeC, at::cuda::getCurrentCUDAStream());
}


template void MemberShip_Forward_Wrapper<float>(at::Tensor in, at::Tensor center, at::Tensor lamda, at::Tensor out);
template void MemberShip_Input_Backward_Wrapper<float>(at::Tensor in_grad, at::Tensor grad_last, 
at::Tensor in, at::Tensor center, at::Tensor lamda, at::Tensor out);
template void MemberShip_Center_Backward_Wrapper<float>(at::Tensor c_grad, at::Tensor grad_last, 
at::Tensor in, at::Tensor center, at::Tensor lamda, at::Tensor out);
template void MemberShip_Lamda_Backward_Wrapper<float>(at::Tensor la_grad, at::Tensor grad_last, 
at::Tensor in, at::Tensor center, at::Tensor lamda, at::Tensor out);

template at::Tensor CenterLoss_Forward_Wrapper<float>(at::Tensor in, at::Tensor center, at::Tensor gt, 
at::Tensor gt_sum);

template void CenterLoss_Input_Backward_Wrapper<float>(at::Tensor in_grad, at::Tensor grad_last, at::Tensor in, 
at::Tensor center, at::Tensor gt, at::Tensor gt_sum);

template void CenterLoss_Center_Backward_Wrapper<float>(at::Tensor c_grad, at::Tensor grad_last, at::Tensor in, 
at::Tensor center, at::Tensor gt, at::Tensor gt_sum);