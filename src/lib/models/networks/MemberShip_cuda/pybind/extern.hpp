template <typename Dtype>
void MemberShip_Forward_Wrapper(at::Tensor in, at::Tensor center, at::Tensor lamda, at::Tensor out);

template <typename Dtype>
 void MemberShip_Input_Backward_Wrapper(at::Tensor in_grad, at::Tensor grad_last, 
at::Tensor in, at::Tensor center, at::Tensor lamda, at::Tensor out);

template <typename Dtype>
 void MemberShip_Center_Backward_Wrapper(at::Tensor c_grad, at::Tensor grad_last, 
at::Tensor in, at::Tensor center, at::Tensor lamda, at::Tensor out);

template <typename Dtype>
 void MemberShip_Lamda_Backward_Wrapper(at::Tensor la_grad, at::Tensor grad_last, 
at::Tensor in, at::Tensor center, at::Tensor lamda, at::Tensor out);

template <typename Dtype>
at::Tensor CenterLoss_Forward_Wrapper(at::Tensor in, at::Tensor center, at::Tensor gt, at::Tensor gt_sum);

template <typename Dtype>
void CenterLoss_Input_Backward_Wrapper(at::Tensor in_grad, at::Tensor grad_last, at::Tensor in, 
at::Tensor center, at::Tensor gt, at::Tensor gt_sum) ;

template <typename Dtype>
void CenterLoss_Center_Backward_Wrapper(at::Tensor c_grad, at::Tensor grad_last, at::Tensor in, 
at::Tensor center, at::Tensor gt, at::Tensor gt_sum);