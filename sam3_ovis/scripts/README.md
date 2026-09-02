# scripts/ — sam3_ovis 可执行入口

约定（继承 0.2 工程契约第 4 条）：全部入口先 source
`scripts/load_environment.sh` 做 env 检查，不假设与 C5-v2 环境共存。

| 脚本 | 用途 |
|---|---|
| `load_environment.sh` | 环境加载器：`environment.local.sh` → `environment.sh`，缺失变量即报错。 |

## sam3 独立环境安装记录（host lthpc, 2026-09-02）

```bash
conda create -y -p /data/wangcheng/envs/sam3 python=3.12
# torch 全家桶：阿里云 wheel 被代理拖垮/单连接中断，最终采用
# 本地 wheel 并行下载 + 零网络安装（wheel 存 /data/wangcheng/tmp_wheels/）：
#   pip install --no-index --no-deps torch-2.10.0+cu128-*.whl nvidia/*.whl ...
pip install -i https://mirrors.aliyun.com/pypi/simple/ \
    "numpy>=1.26,<2" timm tqdm ftfy==6.1.1 regex iopath huggingface_hub \
    pycocotools opencv-python-headless matplotlib pillow einops psutil \
    "setuptools<81"   # >=81 移除 pkg_resources，官方 model_builder 仍依赖
cd "$SAM3_REPO" && pip install -e .
```

官方要求：python ≥ 3.12、torch ≥ 2.7、CUDA ≥ 12.6（本机驱动 CUDA 12.8，满足）。
注意本机 shell 默认走 `127.0.0.1:7890` 代理：国内源（ModelScope / 阿里云）须
`env -u http_proxy … no_proxy='*'` 直连，否则限速至 ~0.3–2 MB/s；HF 代理时通
时断，数据集下载用 hf-mirror.com。

## 权重（S1.1）

- 路径：`$SAM3_HUB_DIR`（本机 `/data/wangcheng/pretrained-models/sam3/`），
  含 `sam3.pt` + config/tokenizer 小文件（取自 ModelScope `facebook/sam3`
  镜像；HF gated 被拒，渠道与条款见 `docs/license-note.md`）。
- `sam3.pt` sha256 = `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`
  （3,450,062,241 bytes，与 HF 官方字节数一致、与 ModelScope 元数据一致；
  获官方访问权限后须用 HF 官方 LFS oid 复核）。
- 加载方式：`build_sam3_image_model(checkpoint_path=$SAM3_CKPT, load_from_HF=False)`；
  BPE 词表随包分发（`sam3/assets/bpe_simple_vocab_16e6.txt.gz`），离线可跑。
- 网络注意：本机 shell 默认走 `127.0.0.1:7890` 代理；国内源（ModelScope /
  阿里云 wheel）必须 `--noproxy '*'` 或 `env -u` 清代理直连，否则限速约
  0.3–2 MB/s（实测）。
