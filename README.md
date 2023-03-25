# Hardware Implemented LSS+HQ

Code for hardware implementation LSS and HQ operator.

## INSTALL

Tested with PyTorch 1.12.1 + CUDA 11.3, on an Tesla A100 GPU.

> Note: This cuda program is based on [Nvidia cutlass](https://github.com/NVIDIA/cutlass) version 2.10. You need to pull down the corresponding version library. Besides, in quantize_forward_HQ/setup_easy.py and quantize_grad_weight_LSS/setup.py, you need to change the path of `include_dirs` into the absolute path on your own computer, only in this way can it work normally.

### CutLass

```bash
git clone git@github.com:NVIDIA/cutlass.git 
#checkout branch
git checkout feature/2.10/updates_before_tagging
```



### LSS

```bash
cd quantize_grad_weight_LSS
python setup.py install
```

### HQ

```bash
cd quantize_forward_HQ
python setup_easy.py install
```

