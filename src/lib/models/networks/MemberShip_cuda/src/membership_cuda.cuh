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
