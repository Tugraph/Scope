
pass
import torch
from torch.utils.cpp_extension import load
import os

def build_csa_attn():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    source = os.path.join(curr_dir, 'csa_attn.cpp')

    extra_cflags = [
        '-O3',
        '-march=native',
        '-mavx512f',
        '-mavx512bw',
        '-mavx512vnni',
        '-fopenmp',
        '-ffast-math',
    ]

    extra_ldflags = ['-fopenmp']

    print(f"_")
    print(f"flags: {' '.join(extra_cflags)}")

    ext = load(
        name='csa_attn_kernel',
        sources=[source],
        extra_cflags=extra_cflags,
        extra_ldflags=extra_ldflags,
        verbose=True,
    )

    print("_")
    return ext

if __name__ == '__main__':
    ext = build_csa_attn()
    print(f"_")
    print(f"_")

