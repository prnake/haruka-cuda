#include <cuda_runtime.h>
#include <iostream>
#include "kernel_forward.h"
#include "utils.h"

template<typename T, typename ArchTag, bool is_aligned, int queries_per_block, int keys_per_block,
         bool single_value_iteration>
void LaunchCutlassFmha(const Params& params) {
  using Attention = AttentionKernel<T, ArchTag, is_aligned, queries_per_block, keys_per_block,
                                    single_value_iteration>;
  typename Attention::Params p;
  p.query_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.query_ptr));
  p.key_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.key_ptr));
  p.value_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.value_ptr));
  p.causal_mask_ptr = const_cast<int32_t*>(reinterpret_cast<const int32_t*>(params.causal_mask));
  p.logsumexp_ptr = nullptr;
  p.output_ptr = reinterpret_cast<T*>(params.out_ptr);
  if (Attention::kNeedsOutputAccumulatorBuffer) {
    using Acc = typename Attention::accum_t;
    p.output_accum_ptr = reinterpret_cast<Acc*>(params.workspace);
  } else {
    p.output_accum_ptr = nullptr;
  }
  p.num_heads = params.num_heads;
  p.num_batches = params.num_batches;
  p.head_dim = params.head_size;
  p.head_dim_value = params.value_head_size;
  p.num_queries = params.query_seq_len;
  p.num_keys = params.kv_seq_len;
  p.q_strideM = params.query_hidden_stride;
  p.k_strideM = params.key_hidden_stride;
  p.v_strideM = params.value_hidden_stride;
  p.o_strideM = p.num_heads * params.value_head_size;

  p.q_strideH = params.head_size;
  p.k_strideH = params.head_size;
  p.v_strideH = params.value_head_size;
  p.o_strideH = params.value_head_size;

  p.q_strideB = params.query_seq_len * p.q_strideM;
  p.k_strideB = params.kv_seq_len * p.k_strideM;
  p.v_strideB = params.kv_seq_len * p.v_strideM;
  p.o_strideB = params.query_seq_len * p.o_strideM;

  constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
  int smem_bytes = sizeof(typename Attention::SharedStorage);
  if (smem_bytes > 0xc000) {
    static bool once = [&]() {
      cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
      return true;
    }();
  }
  if (!Attention::check_supported(p)) {
    std::cerr << "Kernel does not support these inputs" << std::endl;
    return;
  }
  kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);
}

template<typename T, typename ArchTag, bool is_aligned, int queries_per_block, int keys_per_block>
void DispatchSingleValueIteration(const Params& params) {
  if (params.value_head_size <= keys_per_block) {
    LaunchCutlassFmha<T, ArchTag, is_aligned, queries_per_block, keys_per_block, true>(params);
  } else {
    LaunchCutlassFmha<T, ArchTag, is_aligned, queries_per_block, keys_per_block, false>(params);
  }
}

template<typename T, typename ArchTag, bool is_aligned>
void DispatchKeysPerBlock(const Params& params) {
  if (params.value_head_size <= 64) {
    DispatchSingleValueIteration<T, ArchTag, is_aligned, 64, 64>(params);
  } else {
    DispatchSingleValueIteration<T, ArchTag, is_aligned, 32, 128>(params);
  }
}

template<typename T, typename ArchTag>
void DispatchIsAligned(const Params& params) {
  if (reinterpret_cast<uintptr_t>(params.query_ptr) % 16 == 0
      && reinterpret_cast<uintptr_t>(params.key_ptr) % 16 == 0
      && params.query_hidden_stride % (16 / sizeof(T)) == 0
      && params.key_hidden_stride % (16 / sizeof(T)) == 0) {
    DispatchKeysPerBlock<T, ArchTag, true>(params);
  } else {
    DispatchKeysPerBlock<T, ArchTag, false>(params);
  }
}

template<typename T>
void DispatchArchTag(const Params& params) {
  int device;
  cudaGetDevice(&device);

  struct cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device);
  const int major = props.major;
  const int minor = props.minor;

  if (major == 8) {
    DispatchIsAligned<T, cutlass::arch::Sm80>(params);
  } else if (major == 7) {
    if (minor == 5) {
      DispatchIsAligned<T, cutlass::arch::Sm75>(params);
    } else {
      DispatchIsAligned<T, cutlass::arch::Sm70>(params);
    }
  } else {
    // UNIMPLEMENTED();
    printf("UNIMPLEMENTED\n");
  }
}

void launch_mask_multi_head_attention(const Params& params) {
  if (params.data_type == DataType::kFloat16) {
    DispatchArchTag<cutlass::half_t>(params);
  } else if (params.data_type == DataType::kFloat) {
    DispatchArchTag<cutlass::tfloat32_t>(params);
  } else {
    // UNIMPLEMENTED();
    printf("UNIMPLEMENTED\n");
  }
}