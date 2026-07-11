"""Modal App for the from-scratch 125M SLM build (Phases 0 to 4)."""

from __future__ import annotations

import modal

import config

app = modal.App(config.PROJECT)

# CPU base. All pip/apt build steps MUST come before add_local_* (Modal rule).
_cpu_base = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wamerican")  # /usr/share/dict/words for the OCR gate
    .pip_install(
        "datasets==3.6.0",
        "huggingface_hub==0.34.4",
        "langdetect==1.0.9",
        "pyarrow==17.0.0",
        "datasketch==1.6.5",
    )
)
cpu_image = _cpu_base.add_local_python_source("config", "cleaning", "dedup")

volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}


def _stream_source(source: "config.Source", n: int):
    from datasets import load_dataset

    ds = load_dataset(source.hf_id, source.config_name, split=source.split, streaming=True)
    for i, record in enumerate(ds):
        if i >= n:
            break
        yield record


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 15)
def smoke_test(n_per_source: int = 10) -> dict:
    from cleaning import clean_document

    summary: dict[str, dict] = {}
    for source in config.DATA_MIX:
        print("\n" + "=" * 78)
        print(f"SOURCE: {source.name}  ({source.hf_id}, split={source.split}, "
              f"field='{source.text_field}')")
        print("=" * 78)
        kept = 0
        reasons: dict[str, int] = {}
        for i, record in enumerate(_stream_source(source, n_per_source)):
            text = record.get(source.text_field) or ""
            if not isinstance(text, str):
                text = str(text)
            result = clean_document(text)
            reasons[result.reason] = reasons.get(result.reason, 0) + 1
            kept += int(result.kept)
            excerpt = (result.text[:240] if result.kept else text[:160]).replace("\n", " / ")
            print(f"\n[{source.name} #{i}] raw={result.raw_chars:>7} clean={result.clean_chars:>7} "
                  f"-> {result.reason.upper()}")
            print(f"    {excerpt}")
        summary[source.name] = {"streamed": n_per_source, "kept": kept, "reasons": reasons}
    print("\nSMOKE TEST SUMMARY")
    for name, s in summary.items():
        print(f"  {name:<12} kept {s['kept']}/{s['streamed']}  reasons={s['reasons']}")
    return summary


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 20)
def measure_sources(n_per_source: int = 2000) -> dict:
    from cleaning import clean_document

    TOTAL_ROWS = {"case-law": 282_390, "sec": 48_543, "fineweb-edu": 9_670_000}
    out: dict[str, dict] = {}
    for source in config.DATA_MIX:
        clean_chars = kept = 0
        for record in _stream_source(source, n_per_source):
            text = record.get(source.text_field) or ""
            if not isinstance(text, str):
                text = str(text)
            r = clean_document(text)
            if r.kept:
                kept += 1
                clean_chars += r.clean_chars
        avg_clean = clean_chars / n_per_source if n_per_source else 0
        total = TOTAL_ROWS[source.name]
        est = total * avg_clean / config.CHARS_PER_TOKEN
        out[source.name] = {"est_clean_tokens": int(est), "keep_rate": round(kept / n_per_source, 3)}
        print(f"{source.name:<12} keep={kept/n_per_source:.0%}  avg_clean={avg_clean:>7.0f} ch/doc  "
              f"rows={total:>9,}  est_clean_tokens={est/1e9:.2f}B")
    print(f"TOTAL est clean tokens: {sum(v['est_clean_tokens'] for v in out.values())/1e9:.2f}B")
    return out


# ---- Phase 1: stream + clean, one worker per parquet shard ----
_SOURCE_BY_NAME = {s.name: s for s in config.DATA_MIX}


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 60)
def clean_shard(source_name: str, url: str, shard_index: int, token_cap: int) -> dict:
    import os

    from datasets import load_dataset

    from cleaning import clean_document

    source = _SOURCE_BY_NAME[source_name]
    out_dir = f"{config.CLEAN_DIR}/{source_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/shard-{shard_index:03d}.txt"
    ds = load_dataset("parquet", data_files=url, split="train", streaming=True)
    streamed = kept = clean_chars = 0
    reasons: dict[str, int] = {}
    with open(out_path, "w", encoding="utf-8") as fh:
        for record in ds:
            streamed += 1
            text = record.get(source.text_field) or ""
            if not isinstance(text, str):
                text = str(text)
            r = clean_document(text, strict_ocr=source.strict_ocr)
            reasons[r.reason] = reasons.get(r.reason, 0) + 1
            if r.kept:
                fh.write(r.text.replace("\n", " ").strip() + "\n")
                kept += 1
                clean_chars += r.clean_chars
                if clean_chars / config.CHARS_PER_TOKEN >= token_cap:
                    break
    volume.commit()
    est_tokens = int(clean_chars / config.CHARS_PER_TOKEN)
    print(f"[{source_name} shard {shard_index:03d}] streamed={streamed} kept={kept} "
          f"est_tokens={est_tokens/1e6:.1f}M reasons={reasons}")
    return {"source": source_name, "shard": shard_index, "streamed": streamed,
            "kept": kept, "est_tokens": est_tokens, "reasons": reasons}


def _parquet_urls(hf_id: str, config_name: str, split: str) -> list[str]:
    import json
    import urllib.request

    api = f"https://datasets-server.huggingface.co/parquet?dataset={hf_id}"
    req = urllib.request.Request(api, headers={"User-Agent": "slm-125m"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    return [f["url"] for f in data.get("parquet_files", [])
            if f.get("config") == config_name and f.get("split") == split]


@app.local_entrypoint()
def clean(fineweb_shards: int = 1, only: str = ""):
    def cfg(s):
        return s.config_name or "default"

    sources = [s for s in config.DATA_MIX if not only or s.name == only]
    work = []
    for s in sources:
        urls = _parquet_urls(s.hf_id, cfg(s), s.split)
        if s.name == "fineweb-edu":
            urls = urls[:fineweb_shards]
        per_shard_cap = s.token_budget // max(1, len(urls))
        for i, url in enumerate(urls):
            work.append((s.name, url, i, per_shard_cap))
        print(f"{s.name:<12} {len(urls)} shard(s), per-shard cap ~{per_shard_cap/1e6:.0f}M tokens")
    print(f"Launching {len(work)} clean workers...")
    results = list(clean_shard.starmap(work))
    report: dict[str, dict] = {}
    for r in results:
        agg = report.setdefault(r["source"], {"streamed": 0, "kept": 0, "est_tokens": 0, "reasons": {}})
        agg["streamed"] += r["streamed"]
        agg["kept"] += r["kept"]
        agg["est_tokens"] += r["est_tokens"]
        for k, v in r["reasons"].items():
            agg["reasons"][k] = agg["reasons"].get(k, 0) + v
    print("PHASE 1 DROP REPORT")
    total = 0
    for name, a in report.items():
        total += a["est_tokens"]
        print(f"  {name:<12} streamed={a['streamed']:>8} kept={a['kept']:>8} "
              f"est_tokens={a['est_tokens']/1e9:.2f}B drops={a['reasons']}")
    print(f"  TOTAL est_clean_tokens={total/1e9:.2f}B")
    save_report.remote(report)


ocr_image = cpu_image

# ---- Phase 2: dedup + contamination strip ----
SHINGLE_K = 5
MINHASH_PERM = 32
MINHASH_THRESHOLD = 0.8
DECONTAM_NGRAM = 13
SIG_DIR = f"{config.DATA_ROOT}/tmp/minhash_sigs"
NEAR_DUPS_PATH = f"{config.DATA_ROOT}/tmp/near_dups.json"
DECONTAM_SOURCES = {"case-law", "sec"}
CLEAN_SHARDS = {"case-law": 10, "sec": 5, "fineweb-edu": 5}


def _build_contamination_ngrams() -> set:
    from datasets import load_dataset

    from dedup import word_ngrams, words

    grams: set = set()
    for hf_id, cfg_name in [("casehold/casehold", None), ("coastalcph/lex_glue", "case_hold")]:
        try:
            urls = _parquet_urls(hf_id, cfg_name or "default", "test")
            if not urls:
                urls = _parquet_urls(hf_id, cfg_name or "default", "train")
            ds = load_dataset("parquet", data_files=urls, split="train", streaming=True)
            for rec in ds:
                text = " ".join(str(v) for v in rec.values() if isinstance(v, str))
                grams |= word_ngrams(words(text), DECONTAM_NGRAM)
        except Exception as e:
            print(f"  [decontam] could not load {hf_id}: {e}")
    print(f"  [decontam] {len(grams):,} eval 13-grams loaded")
    return grams


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 20, cpu=4.0, memory=4_096)
def minhash_shard(shard_basename: str) -> dict:
    import os

    import numpy as np
    from datasketch import MinHash

    from dedup import shingles, words

    path = f"{config.CLEAN_DIR}/case-law/{shard_basename}"
    sigs, idxs = [], []
    with open(path, encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.rstrip("\n")
            if not line:
                continue
            m = MinHash(num_perm=MINHASH_PERM)
            sh = list(shingles(words(line), SHINGLE_K))
            if sh:
                m.update_batch(sh)
            sigs.append(m.hashvalues.astype(np.uint64))
            idxs.append(idx)
    os.makedirs(SIG_DIR, exist_ok=True)
    np.savez(f"{SIG_DIR}/{shard_basename}.npz",
             sigs=np.vstack(sigs), idxs=np.asarray(idxs, dtype=np.int64))
    volume.commit()
    print(f"[minhash {shard_basename}] {len(idxs):,} docs")
    return {"shard": shard_basename, "n": len(idxs)}


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 20, memory=8_192)
def build_near_dups() -> int:
    import glob
    import json
    import os

    import numpy as np
    from datasketch import MinHash, MinHashLSH

    near: dict[str, list[int]] = {}
    lsh = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=MINHASH_PERM)
    for npz_path in sorted(glob.glob(f"{SIG_DIR}/*.npz")):
        shard = os.path.basename(npz_path)[: -len(".npz")]
        data = np.load(npz_path)
        for row, idx in zip(data["sigs"], data["idxs"]):
            m = MinHash(num_perm=MINHASH_PERM, hashvalues=row)
            if lsh.query(m):
                near.setdefault(shard, []).append(int(idx))
            else:
                lsh.insert(f"{shard}:{int(idx)}", m)
    os.makedirs(os.path.dirname(NEAR_DUPS_PATH), exist_ok=True)
    with open(NEAR_DUPS_PATH, "w", encoding="utf-8") as fh:
        json.dump(near, fh)
    volume.commit()
    total = sum(len(v) for v in near.values())
    print(f"[near-dups] {total:,} case-law near-duplicates")
    return total


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 30, cpu=4.0, memory=8_192)
def write_corpus_shard(source_name: str, shard_basename: str) -> dict:
    import json
    import os

    from dedup import exact_hash, word_ngrams, words

    near: set[int] = set()
    if source_name == "case-law":
        with open(NEAR_DUPS_PATH, encoding="utf-8") as fh:
            near = set(json.load(fh).get(shard_basename, []))
    contam = _build_contamination_ngrams() if source_name in DECONTAM_SOURCES else None
    in_path = f"{config.CLEAN_DIR}/{source_name}/{shard_basename}"
    out_dir = f"{config.CORPUS_DIR}/{source_name}"
    os.makedirs(out_dir, exist_ok=True)
    seen: set[str] = set()
    kept = clean_chars = 0
    reasons = {"near_dup": 0, "exact_dup": 0, "contaminated": 0, "kept": 0}
    with open(in_path, encoding="utf-8") as fin, \
            open(f"{out_dir}/{shard_basename}", "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            text = line.rstrip("\n")
            if not text:
                continue
            if idx in near:
                reasons["near_dup"] += 1
                continue
            h = exact_hash(text)
            if h in seen:
                reasons["exact_dup"] += 1
                continue
            if contam and (word_ngrams(words(text), DECONTAM_NGRAM) & contam):
                reasons["contaminated"] += 1
                continue
            seen.add(h)
            fout.write(text + "\n")
            kept += 1
            clean_chars += len(text)
            reasons["kept"] += 1
    volume.commit()
    print(f"[corpus {source_name}/{shard_basename}] kept={kept} drops={reasons}")
    return {"source": source_name, "shard": shard_basename, "kept": kept,
            "est_tokens": int(clean_chars / config.CHARS_PER_TOKEN), "reasons": reasons}


@app.function(image=cpu_image, volumes=VOLUMES)
def write_phase2_report(results: list) -> dict:
    import json

    report: dict[str, dict] = {}
    for r in results:
        if not r:
            continue
        agg = report.setdefault(r["source"], {"kept": 0, "est_tokens": 0,
              "reasons": {"near_dup": 0, "exact_dup": 0, "contaminated": 0, "kept": 0}})
        agg["kept"] += r["kept"]
        agg["est_tokens"] += r["est_tokens"]
        for k, v in r["reasons"].items():
            agg["reasons"][k] = agg["reasons"].get(k, 0) + v
    total = sum(v["est_tokens"] for v in report.values())
    print("PHASE 2 REPORT")
    for name, a in report.items():
        print(f"  {name:<12} kept={a['kept']:>8} est_tokens={a['est_tokens']/1e9:.2f}B drops={a['reasons']}")
    print(f"  TOTAL corpus est tokens: {total/1e9:.2f}B")
    with open(f"{config.CORPUS_DIR}/phase2_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    volume.commit()
    return report


@app.local_entrypoint()
def dedup(compute_sigs: bool = True):
    if compute_sigs:
        names = [f"shard-{i:03d}.txt" for i in range(CLEAN_SHARDS["case-law"])]
        print(f"1/3 MinHash signatures for {len(names)} case-law shards...")
        list(minhash_shard.map(names))
    print("2/3 building near-dup set (LSH)...")
    build_near_dups.remote()
    work = [(src, f"shard-{i:03d}.txt") for src, n in CLEAN_SHARDS.items() for i in range(n)]
    print(f"3/3 writing final corpus ({len(work)} shards, parallel)...")
    results = list(write_corpus_shard.starmap(work))
    write_phase2_report.remote(results)


# ---- Phase 3: train the 16K byte-level BPE tokenizer ----
ml_image = _cpu_base.pip_install("transformers==4.46.3").add_local_python_source(
    "config", "cleaning", "dedup")


def _corpus_line_iter():
    import glob

    for path in sorted(glob.glob(f"{config.CORPUS_DIR}/*/*.txt")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line:
                    yield line


@app.function(image=ml_image, volumes=VOLUMES, timeout=60 * 40, cpu=8.0, memory=16_384)
def train_tokenizer() -> dict:
    import os

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    specials = list(config.SPECIAL_TOKENS.values()) + list(config.EXTRA_CHAT_TOKENS)
    tok = Tokenizer(models.BPE(unk_token=config.SPECIAL_TOKENS["unk_token"]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=config.MODEL.vocab_size, special_tokens=specials,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)
    print("training BPE...")
    tok.train_from_iterator(_corpus_line_iter(), trainer=trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=config.SPECIAL_TOKENS["bos_token"],
        eos_token=config.SPECIAL_TOKENS["eos_token"],
        pad_token=config.SPECIAL_TOKENS["pad_token"],
        unk_token=config.SPECIAL_TOKENS["unk_token"],
        additional_special_tokens=list(config.EXTRA_CHAT_TOKENS))
    os.makedirs(config.TOKENIZER_DIR, exist_ok=True)
    fast.save_pretrained(config.TOKENIZER_DIR)
    volume.commit()
    for s in ["The plaintiff shall bear the burden of proof by a preponderance of the evidence.",
              "The Company's net revenues increased 12% year over year pursuant to the agreement."]:
        ids = fast.encode(s)
        print(f"  '{s[:40]}...' -> {len(ids)} tokens | roundtrip={fast.decode(ids).strip() == s}")
    print(f"vocab_size={fast.vocab_size}")
    return {"vocab_size": fast.vocab_size}


@app.local_entrypoint()
def tokenizer():
    train_tokenizer.remote()


# ---- Phase 4: tokenize + pack into uint16 1024-token windows, split 99/1 ----
TOKENIZE_SHARDS = {"case-law": 4, "sec": 6, "fineweb-edu": 4}
ENCODE_BATCH = 1_000


@app.function(image=ml_image, volumes=VOLUMES, timeout=60 * 40, cpu=8.0, memory=16_384)
def tokenize_shard(source_name: str, shard_index: int, num_shards: int) -> dict:
    import glob
    import os

    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    eos_id = tok.convert_tokens_to_ids(config.SPECIAL_TOKENS["eos_token"])
    seq_len = config.SEQ_LEN
    os.makedirs(config.TRAIN_TOKENS_DIR, exist_ok=True)
    os.makedirs(config.VAL_TOKENS_DIR, exist_ok=True)
    train_path = f"{config.TRAIN_TOKENS_DIR}/{source_name}-{shard_index:03d}.bin"
    val_path = f"{config.VAL_TOKENS_DIR}/{source_name}-{shard_index:03d}.bin"
    buf: list[int] = []
    win_count = n_train = n_val = 0
    corpus_files = sorted(glob.glob(f"{config.CORPUS_DIR}/{source_name}/*.txt"))

    def _doc_iter():
        for path in corpus_files:
            with open(path, encoding="utf-8") as fh:
                for idx, line in enumerate(fh):
                    if idx % num_shards == shard_index:
                        line = line.rstrip("\n")
                        if line:
                            yield line

    with open(train_path, "wb") as ftr, open(val_path, "wb") as fva:
        batch: list[str] = []

        def _flush():
            nonlocal win_count, n_train, n_val
            if not batch:
                return
            for ids in tok(batch, add_special_tokens=False)["input_ids"]:
                buf.extend(ids)
                buf.append(eos_id)
            while len(buf) >= seq_len:
                window = np.asarray(buf[:seq_len], dtype=np.uint16)
                del buf[:seq_len]
                if win_count % config.VAL_EVERY_N_WINDOWS == 0:
                    window.tofile(fva)
                    n_val += 1
                else:
                    window.tofile(ftr)
                    n_train += 1
                win_count += 1

        for doc in _doc_iter():
            batch.append(doc)
            if len(batch) >= ENCODE_BATCH:
                _flush()
                batch = []
        _flush()
    volume.commit()
    print(f"[{source_name} {shard_index:03d}] train_win={n_train} val_win={n_val} "
          f"train_tok={n_train*seq_len/1e6:.1f}M")
    return {"source": source_name, "shard": shard_index, "train_windows": n_train,
            "val_windows": n_val, "train_tokens": n_train * seq_len, "val_tokens": n_val * seq_len}


@app.function(image=ml_image, volumes=VOLUMES)
def write_token_index(results: list) -> dict:
    import json

    shards = [r for r in results if r]
    total = {"seq_len": config.SEQ_LEN, "dtype": config.TOKENS_DTYPE,
             "train_windows": sum(r["train_windows"] for r in shards),
             "val_windows": sum(r["val_windows"] for r in shards),
             "train_tokens": sum(r["train_tokens"] for r in shards),
             "val_tokens": sum(r["val_tokens"] for r in shards), "shards": shards}
    with open(f"{config.TOKENS_DIR}/index.json", "w", encoding="utf-8") as fh:
        json.dump(total, fh, indent=2)
    volume.commit()
    print(f"index: train={total['train_tokens']/1e9:.2f}B tok ({total['train_windows']} win), "
          f"val={total['val_tokens']/1e6:.1f}M tok ({total['val_windows']} win)")
    return total


@app.local_entrypoint()
def tokenize():
    work = [(name, i, n) for name, n in TOKENIZE_SHARDS.items() for i in range(n)]
    print(f"Launching {len(work)} tokenize workers...")
    results = list(tokenize_shard.starmap(work))
    write_token_index.remote(results)


# ---- OCR-threshold analysis (optional; informs config.CLEAN.nonword_ratio_max) ----
@app.function(image=ocr_image, timeout=60 * 15)
def ocr_sample(n_docs: int = 3000) -> dict:
    import re

    from cleaning import clean_document

    with open("/usr/share/dict/words", encoding="utf-8", errors="ignore") as fh:
        vocab = {w.strip().lower() for w in fh if w.strip().isalpha()}
    tokre = re.compile(r"[A-Za-z]{3,}")
    source = _SOURCE_BY_NAME["case-law"]
    ratios: list[float] = []
    for record in _stream_source(source, n_docs):
        text = record.get(source.text_field) or ""
        if not isinstance(text, str):
            text = str(text)
        r = clean_document(text)
        if not r.kept:
            continue
        toks = [t.lower() for t in tokre.findall(r.text)]
        if len(toks) < 50:
            continue
        ratios.append(sum(1 for t in toks if t not in vocab) / len(toks))
    ratios.sort()
    n = len(ratios)
    for t in [0.10, 0.15, 0.20, 0.25, 0.30]:
        d = sum(1 for x in ratios if x > t)
        print(f"  drop if non-word ratio >{int(t*100)}%: {d} docs ({d/n:.1%})")
    return {"scored": n}


@app.local_entrypoint()
def ocr(n_docs: int = 3000):
    ocr_sample.remote(n_docs)


@app.function(image=cpu_image, volumes=VOLUMES)
def save_report(report: dict) -> None:
    import json

    with open(f"{config.CLEAN_DIR}/phase1_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    volume.commit()


@app.local_entrypoint()
def main(n_per_source: int = 10):
    smoke_test.remote(n_per_source)


@app.local_entrypoint()
def measure(n_per_source: int = 2000):
    measure_sources.remote(n_per_source)


# ---- Phase 5: pretrain the 125M model (8x H100, DDP, ~1 epoch) ----
# GPU image adds PyTorch on top of the ML image. Build steps before add_local_*.
gpu_image = (
    _cpu_base
    .pip_install("torch==2.5.1", "transformers==4.46.3")
    .add_local_python_source("config", "cleaning", "dedup")
)

PRETRAIN_WORLD_SIZE = 8
H100_BF16_PEAK_FLOPS = 989.5e12  # dense bf16 per GPU, for MFU reporting


def _train_worker(rank: int, world_size: int, max_epochs: float, max_steps: int,
                  use_compile: bool) -> None:
    import contextlib
    import glob
    import json
    import math
    import os
    import time

    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import LlamaConfig, LlamaForCausalLM

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    is_main = rank == 0
    torch.manual_seed(config.TRAIN.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    T = config.TRAIN
    seq_len = config.SEQ_LEN
    global_batch_windows = T.global_batch_tokens // seq_len          # 512
    per_gpu_windows = global_batch_windows // world_size             # 64
    micro = min(T.micro_batch_size, per_gpu_windows)                 # 32
    grad_accum = max(1, per_gpu_windows // micro)                    # 2
    per_gpu_windows = micro * grad_accum
    eff_global_tokens = world_size * per_gpu_windows * seq_len       # 524288

    # Build a flat index of (file_idx, local_window_idx) over all train shards.
    files = sorted(glob.glob(f"{config.TRAIN_TOKENS_DIR}/*.bin"))
    mmaps, index = [], []
    for fi, path in enumerate(files):
        arr = np.memmap(path, dtype=np.uint16, mode="r")
        n = arr.shape[0] // seq_len
        mmaps.append(arr)
        index.extend((fi, li) for li in range(n))
    total_windows = len(index)
    steps_per_epoch = total_windows // global_batch_windows
    total_steps = int(steps_per_epoch * max_epochs)
    if max_steps > 0:
        total_steps = min(total_steps, max_steps)
    total_tokens = int(total_windows * seq_len * max_epochs)

    # Deterministic global shuffle, then shard windows round-robin by rank.
    perm = np.random.default_rng(T.seed).permutation(total_windows)
    shard = perm[rank::world_size]

    def get_batch(gis) -> "torch.Tensor":
        rows = np.empty((len(gis), seq_len), dtype=np.int64)
        for j, gi in enumerate(gis):
            fidx, lidx = index[gi]
            rows[j] = mmaps[fidx][lidx * seq_len:(lidx + 1) * seq_len]
        return torch.from_numpy(rows)

    if is_main:
        print(f"train windows={total_windows:,} | steps/epoch={steps_per_epoch:,} | "
              f"total_steps={total_steps:,} | micro={micro} grad_accum={grad_accum} | "
              f"global_batch={eff_global_tokens:,} tok", flush=True)

    kwargs = config.MODEL.to_llama_kwargs()
    llama_cfg = LlamaConfig(**kwargs)
    llama_cfg._attn_implementation = "sdpa"
    raw_model = LlamaForCausalLM(llama_cfg).to(device)
    n_params = sum(p.numel() for p in raw_model.parameters())
    if is_main:
        print(f"model params: {n_params:,} (~{n_params/1e6:.1f}M)", flush=True)

    decay = [p for p in raw_model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in raw_model.parameters() if p.requires_grad and p.dim() < 2]
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": T.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=T.lr, betas=(T.beta1, T.beta2), fused=True)

    start_step, tokens_seen = 0, 0
    if os.path.exists(config.RESUME_CKPT_PATH):
        ck = torch.load(config.RESUME_CKPT_PATH, map_location="cpu")
        raw_model.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optim"])
        for state in optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        start_step = int(ck["step"])
        tokens_seen = int(ck["tokens_seen"])
        if is_main:
            print(f"resumed from step {start_step} ({tokens_seen/1e9:.3f}B tokens)", flush=True)

    def build_ddp(module):
        return DDP(module, device_ids=[rank], gradient_as_bucket_view=True)

    train_module = torch.compile(raw_model) if use_compile else raw_model
    ddp = build_ddp(train_module)

    # Trigger compilation early so a compile failure fails fast (and cheap).
    try:
        warm = get_batch(shard[:micro]).to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            ddp(input_ids=warm, labels=warm).loss
        optim.zero_grad(set_to_none=True)
    except Exception as exc:  # fall back to eager, keep the run alive
        if use_compile:
            if is_main:
                print(f"torch.compile failed ({exc}); falling back to eager", flush=True)
            del ddp, train_module
            train_module = raw_model
            ddp = build_ddp(train_module)
        else:
            raise

    def get_lr(seen: int) -> float:
        if seen < T.warmup_tokens:
            return T.lr * (seen + 1) / T.warmup_tokens
        if seen >= total_tokens:
            return T.min_lr
        ratio = (seen - T.warmup_tokens) / max(1, total_tokens - T.warmup_tokens)
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return T.min_lr + coeff * (T.lr - T.min_lr)

    @torch.no_grad()
    def evaluate(n_windows: int = 64) -> float:
        ddp.eval()
        vfiles = sorted(glob.glob(f"{config.VAL_TOKENS_DIR}/*.bin"))
        vmmaps, vindex = [], []
        for fi, path in enumerate(vfiles):
            arr = np.memmap(path, dtype=np.uint16, mode="r")
            n = arr.shape[0] // seq_len
            vmmaps.append(arr)
            vindex.extend((fi, li) for li in range(n))
        vshard = np.arange(len(vindex))[rank::world_size][:n_windows]
        losses = []
        for s in range(0, len(vshard), micro):
            chunk = vshard[s:s + micro]
            rows = np.empty((len(chunk), seq_len), dtype=np.int64)
            for j, gi in enumerate(chunk):
                fidx, lidx = vindex[gi]
                rows[j] = vmmaps[fidx][lidx * seq_len:(lidx + 1) * seq_len]
            xb = torch.from_numpy(rows).to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                losses.append(ddp(input_ids=xb, labels=xb).loss.float())
        lt = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)
        dist.all_reduce(lt, op=dist.ReduceOp.AVG)
        ddp.train()
        return lt.item()

    def save_ckpt(step: int) -> None:
        if not is_main:
            return
        os.makedirs(config.BASE_CKPT_DIR, exist_ok=True)
        raw_model.save_pretrained(config.BASE_CKPT_DIR)
        torch.save({"model": raw_model.state_dict(), "optim": optim.state_dict(),
                    "step": step, "tokens_seen": tokens_seen}, config.RESUME_CKPT_PATH)
        try:  # mid-run commit is best-effort; parent guarantees a final commit
            volume.commit()
            print(f"  [ckpt] saved + committed at step {step} -> {config.BASE_CKPT_DIR}", flush=True)
        except Exception as exc:
            print(f"  [ckpt] saved at step {step}; commit deferred ({exc})", flush=True)

    ddp.train()
    ptr = start_step * per_gpu_windows
    t0 = time.time()
    for step in range(start_step, total_steps):
        lr = get_lr(tokens_seen)
        for pg in optim.param_groups:
            pg["lr"] = lr
        optim.zero_grad(set_to_none=True)
        loss_accum = torch.zeros((), device=device)
        for micro_i in range(grad_accum):
            gis = shard[ptr:ptr + micro]
            ptr += micro
            xb = get_batch(gis).to(device, non_blocking=True)
            sync = micro_i == grad_accum - 1
            ctx = contextlib.nullcontext() if sync else ddp.no_sync()
            with ctx:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = ddp(input_ids=xb, labels=xb).loss / grad_accum
                loss.backward()
            loss_accum += loss.detach()
        torch.nn.utils.clip_grad_norm_(ddp.parameters(), T.grad_clip)
        optim.step()
        tokens_seen += eff_global_tokens

        if step % T.log_every_steps == 0 or step == total_steps - 1:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
            if is_main:
                dt = time.time() - t0
                done = step - start_step + 1
                tps = done * eff_global_tokens / dt
                mfu = 6 * n_params * tps / (world_size * H100_BF16_PEAK_FLOPS)
                print(f"step {step:>5}/{total_steps} | loss {loss_accum.item():.4f} | "
                      f"lr {lr:.2e} | {tokens_seen/1e9:.3f}B tok | "
                      f"{tps/1e3:.0f}K tok/s | mfu {mfu*100:.1f}% | {dt/60:.1f}m",
                      flush=True)

        if step > start_step and step % T.eval_every_steps == 0:
            vloss = evaluate()
            if is_main:
                print(f"  [eval] step {step} val_loss {vloss:.4f}", flush=True)

        if step > start_step and step % T.ckpt_every_steps == 0:
            save_ckpt(step)

    vloss = evaluate()
    if is_main:
        print(f"FINAL val_loss {vloss:.4f} after {tokens_seen/1e9:.3f}B tokens", flush=True)
    save_ckpt(total_steps)
    dist.barrier()
    dist.destroy_process_group()


@app.function(image=gpu_image, volumes=VOLUMES, gpu=f"H100:{PRETRAIN_WORLD_SIZE}",
              timeout=60 * 60 * 4)
def pretrain(max_epochs: float = 1.0, max_steps: int = 0, use_compile: bool = True) -> dict:
    import os

    import torch.multiprocessing as mp

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    world_size = PRETRAIN_WORLD_SIZE
    # Use fork so workers inherit the Modal runtime (needed for volume.commit()).
    # Safe here because the parent has not initialized CUDA before forking.
    ctx = mp.get_context("fork")
    procs = []
    for r in range(world_size):
        p = ctx.Process(target=_train_worker,
                        args=(r, world_size, max_epochs, max_steps, use_compile))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    codes = [p.exitcode for p in procs]
    volume.commit()
    if any(c != 0 for c in codes):
        raise RuntimeError(f"pretrain workers exited with codes {codes}")
    return {"world_size": world_size, "max_epochs": max_epochs, "exit_codes": codes}


@app.local_entrypoint()
def pretrain_run(max_epochs: float = 1.0, max_steps: int = 0, use_compile: bool = True):
    pretrain.remote(max_epochs=max_epochs, max_steps=max_steps, use_compile=use_compile)


# ---- Exact perplexity over the FULL validation set (single GPU) ----
@app.function(image=gpu_image, volumes=VOLUMES, gpu="H100", timeout=60 * 20)
def eval_val(batch_size: int = 32) -> dict:
    import glob
    import math
    import os

    import numpy as np
    import torch
    from transformers import LlamaForCausalLM

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    model = LlamaForCausalLM.from_pretrained(config.BASE_CKPT_DIR).to(device).eval()
    seq_len = config.SEQ_LEN

    per_source: dict[str, dict] = {}
    loss_sum = 0.0
    windows = 0
    with torch.no_grad():
        for path in sorted(glob.glob(f"{config.VAL_TOKENS_DIR}/*.bin")):
            name = os.path.basename(path).rsplit("-", 1)[0]
            arr = np.memmap(path, dtype=np.uint16, mode="r")
            n = arr.shape[0] // seq_len
            if n == 0:
                continue
            grid = np.asarray(arr[:n * seq_len]).reshape(n, seq_len)
            s = per_source.setdefault(name, {"loss_sum": 0.0, "windows": 0})
            for i in range(0, n, batch_size):
                batch = grid[i:i + batch_size].astype(np.int64)
                xb = torch.from_numpy(batch).to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(input_ids=xb, labels=xb).loss.float().item()
                bs = batch.shape[0]
                s["loss_sum"] += loss * bs
                s["windows"] += bs
                loss_sum += loss * bs
                windows += bs

    results: dict[str, dict] = {}
    print("PER-SOURCE VALIDATION")
    for name, s in sorted(per_source.items()):
        avg = s["loss_sum"] / s["windows"]
        results[name] = {"val_loss": avg, "perplexity": math.exp(avg), "windows": s["windows"]}
        print(f"  {name:<14} loss={avg:.4f}  ppl={math.exp(avg):.2f}  ({s['windows']:,} win)")
    overall = loss_sum / windows
    print(f"OVERALL val_loss={overall:.4f}  perplexity={math.exp(overall):.2f}  "
          f"over {windows:,} windows ({windows * seq_len / 1e6:.1f}M tokens)")
    return {"overall_val_loss": overall, "overall_perplexity": math.exp(overall),
            "windows": windows, "per_source": results}


@app.local_entrypoint()
def eval_perplexity(batch_size: int = 32):
    eval_val.remote(batch_size=batch_size)
