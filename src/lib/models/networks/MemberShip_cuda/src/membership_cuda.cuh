template <typename Dtype>
void MemberShipForward(Dtype* in, Dtype* c, Dtype* la, Dtype* o, int N, int D, int C,
                  cudaStream_t stream);

template <typename Dtype>
void MemberShipInputBackward(Dtype* in_g, Dtype* g_l, Dtype* in, Dtype* c, Dtype* la, Dtype* o, int N, int D, int C,
    cudaStream_t stream);

template <typename Dtype>
void MemberShipCenterBackward(Dtype* c_g, Dtype* g_l, Dtype* in, Dtype* c, Dtype* la, Dtype* o, int N, int D, int C,
    cudaStream_t stream);

template <typename Dtype>
void MemberShipLamdaBackward(Dtype* la_g, Dtype* g_l, Dtype* in, Dtype* c, Dtype* la, Dtype* o, int N, int D, int C,
    cudaStream_t stream);

template <typename Dtype>
void CenterLossForward(Dtype* in, Dtype* c, Dtype* gt, Dtype* gts, Dtype* ret, int N, int D, int C,
    cudaStream_t stream);

template <typename Dtype>
void CenterLossInputBackward(Dtype* in_g, Dtype* g_l, Dtype* in, Dtype* c, Dtype* gt, Dtype* gts, int N, int D, int C,
    cudaStream_t stream);

template <typename Dtype>
void CenterLossCenterBackward(Dtype* c_g, Dtype* g_l, Dtype* in, Dtype* c, Dtype* gt, Dtype* gts, int N, int D, int C,
                  cudaStream_t stream);