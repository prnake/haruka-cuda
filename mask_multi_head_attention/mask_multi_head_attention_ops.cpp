#include <torch/extension.h>
#include "mask_multi_head_attention.h"

torch::Tensor torch_launch_mask_multi_head_attention(torch::Tensor &query,
                       const torch::Tensor &key,
                       const torch::Tensor &value,
                       const torch::Tensor &causal_mask,
                       int64_t num_heads,
                       int64_t query_slice_start = 0,
                       int64_t query_slice_end = -1,
                       int64_t key_slice_start = 0,
                       int64_t key_slice_end = -1,
                       int64_t value_slice_start = 0,
                       int64_t value_slice_end = -1) {
    const int64_t batch_size = query.size(0);
    const int64_t query_seq_len = query.size(1);
    if (query_slice_end == -1) {
        query_slice_end = query.size(2);
    }
    const int64_t query_hidden_size = query_slice_end - query_slice_start;
    const int64_t kv_seq_len = key.size(1);
    if (key_slice_end == -1) {
        key_slice_end = key.size(2);
    }
    const int64_t key_hidden_size = key_slice_end - key_slice_start;
    if (value_slice_end == -1) {
        value_slice_end = value.size(2);
    }
    const int64_t value_hidden_size = value_slice_end - value_slice_start;
    torch::Tensor out = torch::empty({batch_size, query_seq_len, value_hidden_size},
                                         torch::dtype(query.dtype()).device(torch::kCUDA).requires_grad(false));
    torch::Tensor tmp = torch::empty({batch_size, query_seq_len, value_hidden_size},
                                         torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false));
    auto data_type = query.dtype();
    CHECK_EQ(key.dtype(), data_type);
    CHECK_EQ(value.dtype(), data_type);
    CHECK_EQ(causal_mask.dtype(), torch::kInt32);
    CHECK_EQ(out.dtype(), data_type);
    CHECK_EQ(query.dim(), 3);
    CHECK_EQ(key.dim(), 3);
    CHECK_EQ(value.dim(), 3);
    CHECK_EQ(out.dim(), 3);
    CHECK_EQ(key.size(0), batch_size);
    CHECK_EQ(value.size(0), batch_size);
    CHECK_EQ(value.size(1), kv_seq_len);

    Params params{};
    params.data_type = data_type == torch::kFloat32 ? DataType::kFloat : DataType::kFloat16;
    params.num_batches = batch_size;
    params.num_heads = num_heads;
    params.query_seq_len = query_seq_len;
    params.kv_seq_len = kv_seq_len;
    params.head_size = query_hidden_size / num_heads;
    params.value_head_size = value_hidden_size / num_heads;
    params.query_hidden_stride = query.size(2);
    params.key_hidden_stride = key.size(2);
    params.value_hidden_stride = value.size(2);
    params.query_ptr = (char*)query.data_ptr() + query_slice_start;
    params.key_ptr = (char*)key.data_ptr() + key_slice_start;
    params.value_ptr = (char*)value.data_ptr() + value_slice_start;
    params.out_ptr = out.data_ptr();
    params.workspace = tmp.data_ptr();
    params.causal_mask = causal_mask.data_ptr();
    launch_mask_multi_head_attention(params);
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_mask_multi_head_attention",
          &torch_launch_mask_multi_head_attention,
          "mask_multi_head_attention kernel warpper");
}

TORCH_LIBRARY(mask_multi_head_attention, m) {
    m.def("torch_launch_mask_multi_head_attention", torch_launch_mask_multi_head_attention);
}