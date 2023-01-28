# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pathlib
import subprocess

import torch
from torch.utils import cpp_extension
import unittest
import math
import numpy as np

def load(verbose=True):

    # Setting this param to a list has a problem of generating different
    # compilation commands (with diferent order of architectures) and
    # leading to recompilation of fused kernels. Set it to empty string
    # to avoid recompilation and assign arch flags explicity in
    # extra_cuda_cflags below
    #
    # but if a user wants to set an explicit list of archs to compile to, then let that list
    # through:
    arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', None)
    if arch_list is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = ""

    # # Check if cuda 11 is installed for compute capability 8.0
    # cc_flag = []
    # _, bare_metal_major, _ = _get_cuda_bare_metal_version(
    #     cpp_extension.CUDA_HOME)
    # if int(bare_metal_major) >= 11:
    #     cc_flag.append('-gencode')
    #     cc_flag.append('arch=compute_80,code=sm_80')

    # Build path
    srcpath = pathlib.Path(__file__).parent.absolute()
    parentpath = srcpath.parent.absolute()
    buildpath = srcpath / 'build'
    buildpath.mkdir(parents=True, exist_ok=True)

    # Helper function to build the kernels.
    def _cpp_extention_load_helper(name, sources, extra_cuda_flags, extra_include_paths):
        return cpp_extension.load(
            name=name,
            sources=sources,
            build_directory=buildpath,
            extra_cflags=['-O3', '-std=c++14'],
            extra_cuda_cflags=['-O3',
                               '--use_fast_math'] + extra_cuda_flags,
            extra_include_paths=extra_include_paths,
            verbose=verbose
        )
                               # '-gencode', 'arch=compute_70,code=sm_70',

    extra_cuda_flags = ['-maxrregcount=50', '-std=c++14', '--expt-extended-lambda', '--expt-relaxed-constexpr',
              '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__']
    extra_include_paths = [parentpath / "cutlass/include", parentpath / "cutlass/tools/util/include",
    srcpath / "cutlass_fused_multi_head_attention", parentpath / "tools"]
    extra_include_paths = [str(i) for i in extra_include_paths]
    sources=[srcpath / 'mask_multi_head_attention_kernel.cu', srcpath / 'mask_multi_head_attention_ops.cpp']
    mask_multi_head_attention = _cpp_extention_load_helper(
        "mask_multi_head_attention", sources, extra_cuda_flags, extra_include_paths)
    
    return mask_multi_head_attention



def _get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                         universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor


def _create_build_dir(buildpath):
    try:
        os.mkdir(buildpath)
    except OSError:
        if not os.path.isdir(buildpath):
            print(f"Creation of the build directory {buildpath} failed")


def _ref(query, key, value, num_heads, causal_mask):
    query = query.view(query.shape[0], query.shape[1], num_heads, -1).permute(
        0, 2, 1, 3
    )
    key = key.view(key.shape[0], key.shape[1], num_heads, -1).permute(0, 2, 3, 1)
    value = value.view(value.shape[0], value.shape[1], num_heads, -1).permute(
        0, 2, 1, 3
    )
    scores = torch.matmul(query, key) / math.sqrt(query.shape[-1])
    scores = torch.masked_fill(scores, causal_mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, value)
    out = out.permute(0, 2, 1, 3)
    out = out.reshape(out.shape[0], out.shape[1], -1)
    return out



def _fused_mha(query, key, value, num_heads, causal_mask):
    return cuda_module.torch_launch_mask_multi_head_attention(
        query, key, value, causal_mask, num_heads, 0, -1, 0, -1, 0, -1
    )


def _test_fused_multi_head_attention_inference(
    test_case,
    batch_size,
    num_heads,
    query_seq_len,
    kv_seq_len,
    query_head_size,
    value_head_size,
    dtype
):

    query = torch.randn(
        (batch_size, query_seq_len, num_heads * query_head_size),
        device="cuda",
        dtype=torch.float,
    ).to(dtype)
    key = torch.randn(
        (batch_size, kv_seq_len, num_heads * query_head_size),
        device="cuda",
        dtype=torch.float,
    ).to(dtype)
    value = torch.randn(
        (batch_size, kv_seq_len, num_heads * value_head_size),
        device="cuda",
        dtype=torch.float,
    ).to(dtype)

    def generate_ref_causal_mask(fused_causal_mask):
        res = []
        for i in fused_causal_mask:
            j = min(i,kv_seq_len)
            res.append([0] * j + [1] * (kv_seq_len-j))
        return torch.Tensor(res).to(torch.bool).to("cuda")

    # causal_masks need to be incremental
    causal_masks = []
    causal_masks.append(torch.arange(1, query_seq_len+1))
    causal_masks.append(torch.ones_like(causal_masks[0]) * query_seq_len)
    causal_masks.append(torch.max(causal_masks[0], torch.ones_like(causal_masks[0]) * query_seq_len / 2))
    causal_masks.append(torch.max(causal_masks[0], torch.ones_like(causal_masks[0]) * query_seq_len / 3))
    causal_masks.append(torch.max(causal_masks[0], torch.ones_like(causal_masks[0]) * query_seq_len * 2 / 3))

    for causal_mask in causal_masks:
        causal_mask = causal_mask.to(torch.int32).to("cuda")
        ref_out = _ref(query, key, value, num_heads, generate_ref_causal_mask(causal_mask))
        fused_out = _fused_mha(query, key, value, num_heads, causal_mask)

        # print(ref_out.reshape(-1))
        # print(fused_out.reshape(-1))
        # print((ref_out-fused_out).abs().min())

        test_case.assertTrue(np.allclose(ref_out.cpu().numpy(), fused_out.cpu().numpy(), atol=1e-2, rtol=1e-2))
    print(batch_size, num_heads, query_seq_len, kv_seq_len, query_head_size, value_head_size, dtype)

class TestFusedMultiHeadAttentionInference(unittest.TestCase):
    def test_multi_head_attention_inference(test_case):
        # test_case,batch_size, num_heads,query_seq_len, kv_seq_len,query_head_size,value_head_size,dtype
        _test_fused_multi_head_attention_inference(
            test_case, 2, 8, 4096, 4096, 40, 40, torch.float16
        )
        _test_fused_multi_head_attention_inference(
            test_case, 2, 8, 4096, 77, 40, 40, torch.float16
        )
        _test_fused_multi_head_attention_inference(
            test_case, 2, 8, 1024, 1024, 80, 80, torch.float16
        )
        _test_fused_multi_head_attention_inference(
            test_case, 2, 8, 1024, 77, 80, 80, torch.float16
        )
        _test_fused_multi_head_attention_inference(
            test_case, 2, 8, 256, 256, 160, 160, torch.float16
        )
        _test_fused_multi_head_attention_inference(
            test_case, 2, 8, 256, 77, 160, 160, torch.float16
        )
        _test_fused_multi_head_attention_inference(
            test_case, 9, 12, 2048, 2048, 128, 128, torch.float16
        )

        _test_fused_multi_head_attention_inference(
            test_case, 2, 8, 4096, 4096, 40, 40, torch.float
        )
        _test_fused_multi_head_attention_inference(
            test_case, 2, 8, 4096, 77, 40, 40, torch.float
        )
        _test_fused_multi_head_attention_inference(
            test_case, 2, 8, 1024, 1024, 80, 80, torch.float
        )
        _test_fused_multi_head_attention_inference(
            test_case, 2, 8, 1024, 77, 80, 80, torch.float
        )
        _test_fused_multi_head_attention_inference(
            test_case, 2, 8, 256, 256, 160, 160, torch.float
        )
        _test_fused_multi_head_attention_inference(
            test_case, 2, 8, 256, 77, 160, 160, torch.float
        )
        _test_fused_multi_head_attention_inference(
            test_case, 9, 12, 2048, 2048, 128, 128, torch.float
        )




if __name__ == "__main__":
    cuda_module = load()
    unittest.main()
    # pass
    # n = 1024 * 1024
    # a = torch.rand(n, device="cuda:0")
    # b = torch.rand(n, device="cuda:0")
    # cuda_c = torch.rand(n, device="cuda:0")
    # print(cuda_c)
    # print(a+b)
