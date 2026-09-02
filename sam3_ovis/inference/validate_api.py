"""S1.2 SAM3 API surface validation (implementation-v2 S1.2, four checks).

Subcommands (each writes its evidence section to stdout; conclusions land in
sam3_ovis/docs/sam3-api-notes.md):

  tokens   (c) text-encoder 32-token budget on RS class names — no weights
                or GPU needed, only the bundled BPE vocab;
  contract (a) concept-segmentation output contract (mask / scores / boxes
                dtypes and shapes);
  exemplar (b) visual exemplar input contract (pos/neg boxes; ROI path);
  multi    (d) multi-phrase single-image behaviour.

Run inside the sam3 env:
  source sam3_ovis/scripts/load_environment.sh
  "$PYTHON" sam3_ovis/inference/validate_api.py tokens --dataset SIOR
  "$PYTHON" sam3_ovis/inference/validate_api.py contract|exemplar|multi \
      --image <path>
"""

import argparse
import json
import os
import sys
from pathlib import Path

MAINLINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAINLINE))


def _load_tokenizer(context_length=32):
    """CLIP-style BPE tokenizer; SAM3 text encoder uses a 32-token budget
    (verify at runtime, DEFAULT_CONTEXT_LENGTH in tokenizer_ve is 77)."""
    from sam3.model.tokenizer_ve import SimpleTokenizer

    bpe = Path(os.environ["SAM3_REPO"]) / "sam3" / "assets" / \
        "bpe_simple_vocab_16e6.txt.gz"
    return SimpleTokenizer(bpe_path=str(bpe), context_length=context_length)


def cmd_tokens(args):
    from data.score_datasets import DATASETS, load_coco_instance_json

    if args.classes_file:
        names = [ln.strip() for ln in open(args.classes_file) if ln.strip()]
        source = args.classes_file
    else:
        spec = DATASETS[args.dataset]
        root = Path(os.environ["SAM3_OVIS_DATA_ROOT"])
        # train jsons live under the dataset subtree; val for SIOR-val etc.
        json_path = root / spec.val_json
        d = load_coco_instance_json(json_path)
        names = [c["name"] for c in d["categories"]]
        source = str(json_path)

    tok = _load_tokenizer()
    rows, over = [], []
    for name in names:
        seq = tok.encode(name)
        if hasattr(seq, "tolist"):
            seq = seq.tolist()
        if seq and isinstance(seq[0], list):  # batched [1, T] form
            seq = seq[0]
        eot = seq.index(tok.eot_token_id) if tok.eot_token_id in seq else len(seq)
        n = eot + 1  # SOT + tokens + EOT within the context window
        rows.append((name, n))
        if n > tok.context_length:
            over.append((name, n))
    print(f"tokenizer context_length = {tok.context_length}")
    print(f"source: {source}  ({len(names)} names)")
    for name, n in rows:
        flag = " OVER" if n > tok.context_length else ""
        print(f"  {n:3d}  {name}{flag}")
    print(f"truncated/overflow count: {len(over)}")
    if over:
        out = Path(os.environ.get("SAM3_OVIS_LOG_ROOT", "/tmp"))
        (out / "token_overflow.json").write_text(
            json.dumps([{"name": n, "tokens": t} for n, t in over], indent=1))
        print(f"overflow list -> {out / 'token_overflow.json'}")


def _build_processor():
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    # Official usage: global bf16 autocast (see examples notebook); required
    # because perflib.fused.addmm_act casts FFN outputs to bf16.
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    model = build_sam3_image_model(
        checkpoint_path=os.environ["SAM3_CKPT"], load_from_HF=False)
    return Sam3Processor(model), torch


def _contract_report(state, torch):
    for key in ("masks", "boxes", "scores", "masks_logits"):
        t = state.get(key)
        if t is None:
            print(f"  {key}: ABSENT")
            continue
        if t.numel() == 0:
            print(f"  {key}: EMPTY (no instance above threshold)")
            continue
        print(f"  {key}: shape={tuple(t.shape)} dtype={t.dtype} "
              f"range=[{t.min().item():.4f}, {t.max().item():.4f}]")
    if state.get("scores") is not None and state["scores"].numel():
        print(f"  scores device: {state['scores'].device}, "
              f"mask bool: {state['masks'].dtype == torch.bool}")


def _default_image():
    repo = Path(os.environ["SAM3_REPO"])
    imgs = sorted((repo / "assets" / "images").glob("*"))
    if not imgs:
        raise FileNotFoundError("no demo image under $SAM3_REPO/assets/images")
    return str(imgs[0])


def cmd_contract(args):
    import numpy as np
    from PIL import Image

    processor, torch = _build_processor()
    image_path = args.image or _default_image()
    state = processor.set_image(Image.open(image_path).convert("RGB"))
    state = processor.set_text_prompt(state=state, prompt=args.prompt)
    print(f"image: {image_path}")
    print(f"prompt: {args.prompt!r}  (conf thr "
          f"{processor.confidence_threshold})")
    _contract_report(state, torch)


def cmd_exemplar(args):
    import numpy as np
    from PIL import Image

    processor, torch = _build_processor()
    image_path = args.image or _default_image()
    im = Image.open(image_path).convert("RGB")
    w, h = im.size
    state = processor.set_image(im)
    # centre ROI as a positive box exemplar, corner as negative
    pos = [0.5, 0.5, 0.3, 0.3]
    neg = [0.1, 0.1, 0.1, 0.1]
    state = processor.add_geometric_prompt(state=state, box=pos, label=True)
    state = processor.add_geometric_prompt(state=state, box=neg, label=False)
    print(f"image: {image_path} ({w}x{h})")
    print(f"pos box (cxcywh norm): {pos}  neg box: {neg}")
    _contract_report(state, torch)


def cmd_multi(args):
    import numpy as np
    from PIL import Image

    processor, torch = _build_processor()
    image_path = args.image or _default_image()
    state = processor.set_image(Image.open(image_path).convert("RGB"))
    print(f"image: {image_path}")
    for phrase in args.phrases:
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(state=state, prompt=phrase)
        n = int(state["scores"].shape[0]) if state.get("scores") is not None else 0
        top = float(state["scores"].max().item()) if n else float("nan")
        print(f"  phrase {phrase!r}: {n} instances, top score {top:.4f}")
    # joint multi-phrase string behaviour (single call, several concepts)
    joint = "; ".join(args.phrases)
    processor.reset_all_prompts(state)
    state = processor.set_text_prompt(state=state, prompt=joint)
    n = int(state["scores"].shape[0]) if state.get("scores") is not None else 0
    print(f"  joint {joint!r}: {n} instances")
    _contract_report(state, torch)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tokens")
    t.add_argument("--dataset", default=None,
                   help="registry key, e.g. SIOR / NWPU / FAST")
    t.add_argument("--classes-file", default=None,
                   help="one class name per line (overrides --dataset)")
    t.set_defaults(fn=cmd_tokens)

    for name, fn in [("contract", cmd_contract), ("exemplar", cmd_exemplar)]:
        s = sub.add_parser(name)
        s.add_argument("--image", default=None)
        s.add_argument("--prompt", default="airplane")
        s.set_defaults(fn=fn)

    m = sub.add_parser("multi")
    m.add_argument("--image", default=None)
    m.add_argument("--phrases", nargs="+",
                   default=["airplane", "ship", "vehicle"])
    m.set_defaults(fn=cmd_multi)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
