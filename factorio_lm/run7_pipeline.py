"""Fine-tune the Run 6D model on TinyChat Q/A pairs with a 64-character context."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .checkpoint import ExactCheckpoint
from .full_model import ExactFactorioLM
from .run4_pipeline import (
    Run4Config,
    TokenCorpus,
    _learning_rate,
    _write_json,
    create_float_model,
    normalize_tinystories,
)
from .run5_pipeline import (
    Run5Config,
    collect_range_report,
    export_integer_student,
)
from .run6_pipeline import generate_prompt_set, mask_summary, validate_incremental_integer
from .run6d_pipeline import RUN6D_EXACT, load_clean_int4
from .training import set_deterministic
from .vocabulary import FACTORIOLM_ALPHABET, Vocabulary


RUN7_OUTPUT = Path("generated/run7")
DATASET_ID = "starhopp3r/TinyChat"
DATASET_REVISION = "refs/convert/parquet"
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
_PAIR = re.compile(
    r"\[INST\]\s*(.*?)\s*\[/INST\]\s*(.*?)(?=\s*\[INST\]|$)", re.DOTALL
)
_QUESTION_SENTENCE = re.compile(r"[^.!?]*\?", re.DOTALL)
_ANSWER_SENTENCE = re.compile(r".*?(?:[.!?](?=\s|$)|$)", re.DOTALL)

EVALUATION_QUESTIONS = (
    "How are you today",
    "Why is the sky blue",
    "Can dogs be good friends",
    "What do plants need to grow",
    "How can I help a friend",
    "What happens when it rains",
    "Should I share my toys",
    "What makes people happy",
    "Where do birds live",
    "What should I do when I feel sad",
    "Why do we need to sleep",
    "How do I make a new friend",
)


@dataclass(frozen=True)
class Run7Config:
    seed: int = 4107
    requested_source_rows: int = 200_000
    source_page_size: int = 100
    validation_fraction: float = 0.08
    minimum_question_characters: int = 8
    maximum_question_characters: int = 59
    minimum_answer_characters: int = 4
    maximum_answer_characters: int = 120
    sequence_length: int = 64
    batch_size: int = 64
    float_steps: int = 2_500
    integer_qat_steps: int = 2_500
    float_learning_rate: float = 0.00075
    integer_qat_learning_rate: float = 0.00075
    warmup_steps: int = 150
    evaluation_interval: int = 250
    evaluation_characters: int = 16_384
    exact_evaluation_characters: int = 2_048
    gradient_clip: float = 1.0
    weight_decay: float = 0.01
    distillation_weight: float = 0.35
    integer_logit_temperature: float = 1.125
    answer_weight: float = 1.0
    aligned_float_steps: int = 1_000
    aligned_integer_steps: int = 2_500
    aligned_float_learning_rate: float = 0.00025
    aligned_integer_learning_rate: float = 0.00025
    aligned_maximum_answer_characters: int = 64
    nonzero_weights: int = 25_100
    weight_limit: int = 7
    maximum_scale_denominator: int = 256
    generation_length: int = 128
    device: str = "cpu"

    @classmethod
    def read(cls, path: str | Path) -> "Run7Config":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def validate(self) -> None:
        if self.sequence_length != 64:
            raise ValueError("FactorioLM hardware requires context 64")
        if self.maximum_question_characters + len("Q \nA ") > 64:
            raise ValueError("maximum question does not fit the hardware context")
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between zero and one half")
        if self.weight_limit != 7 or self.nonzero_weights != 25_100:
            raise ValueError("Run 7 must preserve the Run 6 int4 representation and mask budget")


@dataclass(frozen=True)
class ChatExample:
    source_row: int
    turn: int
    question: str
    answer: str

    @property
    def prompt(self) -> str:
        return f"Q {self.question}\nA "

    @property
    def record(self) -> str:
        return f"{self.prompt}{self.answer}\n"


class MaskedTokenCorpus(TokenCorpus):
    """Token stream plus a bit per target character identifying answer text."""

    def __init__(self, token_path: str | Path, mask_path: str | Path) -> None:
        super().__init__(token_path)
        torch = _torch()
        raw = bytearray(Path(mask_path).read_bytes())
        self.answer_mask = torch.frombuffer(raw, dtype=torch.uint8).clone()
        if len(self.answer_mask) != len(self.tokens):
            raise ValueError("answer mask and token stream lengths differ")
        if not bool(((self.answer_mask == 0) | (self.answer_mask == 1)).all()):
            raise ValueError("answer mask must contain only zero and one")

    def weighted_batch(self, starts, length: int, device: str):
        torch = _torch()
        indexes = starts[:, None] + torch.arange(length + 1)
        values = self.tokens[indexes].to(dtype=torch.long, device=device)
        # The weight describes the target character, hence the +1 alignment.
        mask = self.answer_mask[indexes[:, 1:]].to(dtype=torch.float32, device=device)
        return values, mask


class AlignedChatCorpus:
    """Autoregressive answer targets whose context begins with the real prompt.

    Items are bucketed by context length so batches need no synthetic padding.
    Early answer positions and the answer-ending newline are repeated: those
    are precisely the characters a live viewer sees and the boundary needed to
    prevent an unending phrase chain.
    """

    def __init__(self, records_path: str | Path, *, maximum_answer: int) -> None:
        vocabulary = Vocabulary()
        buckets: dict[int, list[tuple[tuple[int, ...], int]]] = {}
        examples = 0
        for line in Path(records_path).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            answer = str(row["answer"])
            if len(answer) > maximum_answer:
                continue
            prompt_tokens = vocabulary.encode(f"Q {row['question']}\nA ")
            answer_tokens = vocabulary.encode(answer + "\n")
            for offset, target in enumerate(answer_tokens):
                context = tuple((prompt_tokens + answer_tokens[:offset])[-64:])
                repeats = (4 if offset < 16 else 3 if offset < 32
                           else 2 if offset < 48 else 1)
                if target == 0:
                    repeats = 4
                buckets.setdefault(len(context), []).extend(
                    [(context, target)] * repeats)
            examples += 1
        if examples < 100 or not buckets:
            raise ValueError("too few aligned chat examples")
        self.buckets = buckets
        self.examples = examples
        self.items = sum(map(len, buckets.values()))
        self.lengths = sorted(buckets)
        self.cumulative: list[int] = []
        total = 0
        for length in self.lengths:
            total += len(buckets[length])
            self.cumulative.append(total)

    def batch(self, generator, batch_size: int, device: str):
        torch = _torch()
        draw = int(torch.randint(0, self.items, (1,), generator=generator))
        bucket_index = next(index for index, end in enumerate(self.cumulative) if draw < end)
        length = self.lengths[bucket_index]
        bucket = self.buckets[length]
        indexes = torch.randint(0, len(bucket), (batch_size,), generator=generator).tolist()
        contexts = torch.tensor([bucket[index][0] for index in indexes],
                                dtype=torch.long, device=device)
        targets = torch.tensor([bucket[index][1] for index in indexes],
                               dtype=torch.long, device=device)
        return contexts, targets


def _torch():
    import torch
    return torch


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize_turn(text: str) -> str:
    normalized, _ = normalize_tinystories(text)
    return " ".join(normalized.split())


def _last_question(text: str) -> str | None:
    matches = _QUESTION_SENTENCE.findall(text)
    return matches[-1].strip() if matches else None


def _first_answer_sentence(text: str) -> str:
    match = _ANSWER_SENTENCE.match(text.strip())
    return match.group(0).strip() if match else text.strip()


def extract_examples(text: str, source_row: int, config: Run7Config) -> list[ChatExample]:
    """Extract genuine final-question/first-answer pairs from one TinyChat row."""

    result: list[ChatExample] = []
    for turn, (user, assistant) in enumerate(_PAIR.findall(text)):
        raw_question = _last_question(user)
        if raw_question is None:
            continue
        question = _normalize_turn(raw_question)
        answer = _normalize_turn(_first_answer_sentence(assistant))
        if not (config.minimum_question_characters <= len(question)
                <= config.maximum_question_characters):
            continue
        if not (config.minimum_answer_characters <= len(answer)
                <= config.maximum_answer_characters):
            continue
        example = ChatExample(source_row, turn, question, answer)
        if len(example.prompt) > 64:
            raise AssertionError("accepted prompt exceeds hardware context")
        if any(character not in FACTORIOLM_ALPHABET for character in example.record):
            raise AssertionError("normalization emitted a character outside the hardware ABI")
        result.append(example)
    return result


def deterministic_page_offsets(total_rows: int, requested_rows: int,
                               page_size: int) -> list[int]:
    """Evenly sample non-overlapping pages over the complete source split."""

    if not 0 < requested_rows <= total_rows or page_size <= 0:
        raise ValueError("invalid source row/page request")
    pages = math.ceil(requested_rows / page_size)
    if pages == 1:
        return [0]
    last = total_rows - page_size
    offsets = [round(index * last / (pages - 1) / page_size) * page_size
               for index in range(pages)]
    # Rounding cannot duplicate for requested_rows <= total_rows, but retain a
    # deterministic guard instead of silently issuing duplicate requests.
    if len(set(offsets)) != len(offsets):
        raise ValueError("sampling parameters produced duplicate pages")
    return offsets


def _fetch_page(offset: int, length: int, *, retries: int = 5) -> dict[str, Any]:
    query = urlencode({
        "dataset": DATASET_ID, "config": "default", "split": "train",
        "offset": offset, "length": length,
    })
    request = Request(f"{ROWS_ENDPOINT}?{query}", headers={"User-Agent": "FactorioLM-Run7/1"})
    error: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=45) as response:
                return json.loads(response.read())
        except Exception as caught:  # pragma: no cover - network-dependent retry
            error = caught
            time.sleep(min(8, 2 ** attempt))
    assert error is not None
    raise error


def download_source_rows(output_dir: str | Path, config: Run7Config,
                         *, workers: int = 16) -> dict[str, Any]:
    """Materialize a deterministic sample from Parquet or the HF rows API."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "tinychat-source-rows.jsonl"
    metadata_path = output / "tinychat-source.json"
    if destination.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (metadata.get("requested_source_rows") == config.requested_source_rows
                and metadata.get("source_page_size") == config.source_page_size
                and metadata.get("sha256") == _sha256(destination)):
            return metadata

    parquet_path = output / "tinychat-0000.parquet"
    if parquet_path.exists():
        try:
            import duckdb
        except ImportError as error:  # pragma: no cover - environment setup
            raise RuntimeError("DuckDB is required to read the cached TinyChat Parquet shard") from error
        connection = duckdb.connect()
        shard_rows = int(connection.execute(
            "select count(*) from read_parquet(?)", [str(parquet_path)]).fetchone()[0])
        if config.requested_source_rows > shard_rows:
            raise ValueError("requested more rows than the cached TinyChat shard contains")
        query = """
            select source_row, text from (
                select row_number() over () - 1 as source_row, text
                from read_parquet(?)
            )
            where floor((source_row + 1) * ? / ?) > floor(source_row * ? / ?)
            order by source_row
        """
        rows = connection.execute(query, [
            str(parquet_path), config.requested_source_rows, shard_rows,
            config.requested_source_rows, shard_rows,
        ]).fetchall()
        if len(rows) != config.requested_source_rows:
            raise AssertionError("even Parquet sampler returned the wrong row count")
        destination.write_text("".join(
            json.dumps({"row_idx": int(index), "text": text},
                       ensure_ascii=False, separators=(",", ":")) + "\n"
            for index, text in rows
        ), encoding="utf-8")
        metadata = {
            "dataset": DATASET_ID,
            "dataset_card": f"https://huggingface.co/datasets/{DATASET_ID}",
            "license": "CC BY-NC 4.0",
            "source_split": "train",
            "source_revision": DATASET_REVISION,
            "selection": (
                "exact deterministic evenly spaced sample from official converted "
                "train shard 0000.parquet"
            ),
            "total_source_rows": 1_000_000,
            "source_shard_rows": shard_rows,
            "requested_source_rows": config.requested_source_rows,
            "selected_source_rows": len(rows),
            "source_page_size": config.source_page_size,
            "parquet_path": str(parquet_path),
            "parquet_sha256": _sha256(parquet_path),
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
        }
        _write_json(metadata_path, metadata)
        return metadata

    # The public dataset card reports exactly one million train rows.  The
    # first fetched page verifies that figure before the remaining requests.
    first = _fetch_page(0, config.source_page_size)
    total_rows = int(first.get("num_rows_total", 1_000_000))
    if total_rows != 1_000_000:
        raise ValueError(f"unexpected TinyChat row count: {total_rows}")
    offsets = deterministic_page_offsets(
        total_rows, config.requested_source_rows, config.source_page_size)
    pages: dict[int, dict[str, Any]] = {0: first} if offsets[0] == 0 else {}
    pending = [offset for offset in offsets if offset not in pages]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_fetch_page, offset, config.source_page_size): offset
            for offset in pending
        }
        for completed, future in enumerate(as_completed(futures), 1):
            offset = futures[future]
            pages[offset] = future.result()
            if completed % 100 == 0:
                print(f"downloaded {completed}/{len(pending)} TinyChat pages", flush=True)

    lines: list[str] = []
    selected = 0
    for offset in offsets:
        for item in pages[offset]["rows"]:
            if selected == config.requested_source_rows:
                break
            lines.append(json.dumps({
                "row_idx": int(item["row_idx"]),
                "text": item["row"]["text"],
            }, ensure_ascii=False, separators=(",", ":")))
            selected += 1
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    metadata = {
        "dataset": DATASET_ID,
        "dataset_card": f"https://huggingface.co/datasets/{DATASET_ID}",
        "license": "CC BY-NC 4.0",
        "source_split": "train",
        "source_revision": DATASET_REVISION,
        "selection": "evenly spaced deterministic 100-row pages across all 1000000 rows",
        "total_source_rows": total_rows,
        "requested_source_rows": config.requested_source_rows,
        "selected_source_rows": selected,
        "source_page_size": config.source_page_size,
        "page_count": len(offsets),
        "first_page_offset": offsets[0],
        "last_page_offset": offsets[-1],
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
    }
    _write_json(metadata_path, metadata)
    return metadata


def _split_for(example: ChatExample, validation_fraction: float) -> str:
    value = hashlib.sha256(
        f"{example.source_row}:{example.turn}:{example.question}".encode("utf-8")
    ).digest()
    fraction = int.from_bytes(value[:8], "big") / 2**64
    return "validation" if fraction < validation_fraction else "train"


def _serialize_split(examples: list[ChatExample], prefix: Path) -> dict[str, Any]:
    vocabulary = Vocabulary()
    text = "".join(example.record for example in examples)
    mask = bytearray()
    for example in examples:
        mask.extend(b"\0" * len(example.prompt))
        mask.extend(b"\1" * len(example.answer))
        mask.extend(b"\0")  # record-ending newline
    tokens = bytes(vocabulary.encode(text))
    text_path = prefix.with_suffix(".txt")
    token_path = prefix.with_suffix(".bin")
    mask_path = prefix.with_suffix(".mask.bin")
    records_path = prefix.with_suffix(".jsonl")
    text_path.write_text(text, encoding="ascii")
    token_path.write_bytes(tokens)
    mask_path.write_bytes(mask)
    records_path.write_text("".join(
        json.dumps(asdict(example), sort_keys=True) + "\n" for example in examples
    ), encoding="utf-8")
    return {
        "examples": len(examples), "characters": len(text),
        "answer_characters": sum(mask),
        "text": str(text_path), "tokens": str(token_path),
        "answer_mask": str(mask_path), "records": str(records_path),
        "token_sha256": _sha256(token_path), "mask_sha256": _sha256(mask_path),
    }


def prepare_chat_corpus(output_dir: str | Path, config: Run7Config) -> dict[str, Any]:
    output = Path(output_dir)
    source_metadata = download_source_rows(output, config)
    source_path = Path(source_metadata["path"])
    splits: dict[str, list[ChatExample]] = {"train": [], "validation": []}
    source_pairs = 0
    for line in source_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        pairs = _PAIR.findall(row["text"])
        source_pairs += len(pairs)
        for example in extract_examples(row["text"], int(row["row_idx"]), config):
            splits[_split_for(example, config.validation_fraction)].append(example)
    for values in splits.values():
        values.sort(key=lambda item: (item.source_row, item.turn))
    if min(map(len, splits.values())) < 100:
        raise ValueError("TinyChat extraction produced too few train/validation examples")
    report = {
        "dataset": source_metadata,
        "format": "Q question\\nA answer\\n",
        "parsing": (
            "Parse [INST] markers before normalization; select the final complete question "
            "from the user turn and first complete assistant sentence without character truncation."
        ),
        "source_conversation_pairs": source_pairs,
        "accepted_examples": sum(map(len, splits.values())),
        "acceptance_rate": sum(map(len, splits.values())) / source_pairs,
        "splits": {
            name: _serialize_split(values, output / f"tinychat-{name}")
            for name, values in splits.items()
        },
    }
    _write_json(output / "dataset-report.json", report)
    return report


def _weighted_losses(logits, targets, weights, *, temperature: float = 1.0):
    torch = _torch()
    per_token = torch.nn.functional.cross_entropy(
        (logits / temperature).reshape(-1, 64), targets.reshape(-1), reduction="none"
    ).reshape_as(targets)
    denominator = weights.sum().clamp(min=1.0)
    return (per_token * weights).sum() / denominator


def evaluate_masked_torch(model, corpus: MaskedTokenCorpus, *, config: Run7Config,
                          temperature: float = 1.0) -> dict[str, Any]:
    torch = _torch()
    model.eval()
    totals = {"all_loss": 0.0, "all_correct": 0, "all_count": 0,
              "answer_loss": 0.0, "answer_correct": 0, "answer_count": 0}
    windows = max(1, math.ceil(config.evaluation_characters / 64))
    with torch.no_grad():
        for first in range(0, windows, config.batch_size):
            count = min(config.batch_size, windows - first)
            starts = (torch.arange(first, first + count) * 64) % (
                len(corpus.tokens) - 65)
            values, answer = corpus.weighted_batch(starts, 64, config.device)
            logits = model(values[:, :-1]) / temperature
            targets = values[:, 1:]
            losses = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 64), targets.reshape(-1), reduction="none"
            ).reshape_as(targets)
            correct = logits.argmax(dim=-1) == targets
            totals["all_loss"] += float(losses.sum().cpu())
            totals["all_correct"] += int(correct.sum().cpu())
            totals["all_count"] += targets.numel()
            active = answer.bool()
            totals["answer_loss"] += float(losses[active].sum().cpu())
            totals["answer_correct"] += int(correct[active].sum().cpu())
            totals["answer_count"] += int(active.sum().cpu())
    def metrics(prefix: str) -> dict[str, Any]:
        count = totals[f"{prefix}_count"]
        ce = totals[f"{prefix}_loss"] / count
        return {"cross_entropy": ce, "perplexity": math.exp(ce),
                "accuracy": totals[f"{prefix}_correct"] / count, "characters": count}
    return {"all_characters": metrics("all"), "answer_characters": metrics("answer"),
            "temperature": temperature}


def train_chat_model(model, teacher, train: MaskedTokenCorpus,
                     validation: MaskedTokenCorpus, config: Run7Config, *,
                     stage: str, output_path: Path,
                     resume_path: str | Path | None = None) -> dict[str, Any]:
    """Train dense float or fixed-mask scaled-int4 with answer-weighted loss."""

    torch = _torch()
    integer = stage == "integer"
    if stage not in {"float", "integer"}:
        raise ValueError("stage must be float or integer")
    steps = config.integer_qat_steps if integer else config.float_steps
    maximum_lr = config.integer_qat_learning_rate if integer else config.float_learning_rate
    temperature = config.integer_logit_temperature if integer else 1.0
    model.to(config.device)
    if hasattr(model, "masks"):
        model.masks = {name: value.to(config.device) for name, value in model.masks.items()}
    if teacher is not None:
        teacher.to(config.device).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=maximum_lr,
                                  weight_decay=config.weight_decay)
    start = 0
    best = float("inf")
    if resume_path:
        state = torch.load(resume_path, map_location=config.device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start = int(state["step"])
        best = float(state.get("best_cross_entropy", best))
        if hasattr(model, "masks"):
            model.masks = {name: value.to(config.device)
                           for name, value in state["masks"].items()}
        if integer:
            model.denominators = dict(state["denominators"])
    generator = torch.Generator(device="cpu").manual_seed(
        config.seed + (700_000 if integer else 600_000))
    log: list[dict[str, Any]] = []
    started = time.monotonic()
    for step in range(start, steps):
        model.train()
        starts = torch.randint(0, len(train.tokens) - 65,
                               (config.batch_size,), generator=generator)
        values, answer_mask = train.weighted_batch(starts, 64, config.device)
        # Retain a light language-model objective on prompts and separators;
        # answer targets dominate the supervised adaptation.
        weights = 0.2 + (config.answer_weight - 0.2) * answer_mask
        logits = model(values[:, :-1])
        targets = values[:, 1:]
        hard = _weighted_losses(logits, targets, weights, temperature=temperature)
        loss = hard
        if teacher is not None and config.distillation_weight:
            with torch.no_grad():
                probabilities = torch.softmax(
                    teacher(values[:, :-1]) / temperature, dim=-1)
            per_token_kl = torch.nn.functional.kl_div(
                torch.log_softmax(logits / temperature, dim=-1), probabilities,
                reduction="none",
            ).sum(dim=-1) * temperature * temperature
            distillation = (per_token_kl * weights).sum() / weights.sum().clamp(min=1.0)
            loss = ((1.0 - config.distillation_weight) * hard
                    + config.distillation_weight * distillation)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip).detach().cpu())
        lr = _learning_rate(step, steps, config.warmup_steps, maximum_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        model.constrain_latents()
        evaluate = (step == start or (step + 1) % config.evaluation_interval == 0
                    or step + 1 == steps)
        if evaluate:
            metrics = evaluate_masked_torch(
                model, validation, config=config, temperature=temperature)
            selection = float(metrics["answer_characters"]["cross_entropy"])
            row = {"step": step + 1, "objective": float(loss.detach().cpu()),
                   "gradient_norm": gradient_norm, "learning_rate": lr,
                   "validation": metrics,
                   "elapsed_seconds": time.monotonic() - started}
            log.append(row)
            answer = metrics["answer_characters"]
            print(f"{stage} step {step + 1}/{steps}: objective={row['objective']:.4f} "
                  f"answer CE={answer['cross_entropy']:.4f} "
                  f"acc={answer['accuracy']:.3%}", flush=True)
            if selection < best:
                best = selection
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "stage": stage, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "step": step + 1,
                    "best_cross_entropy": best,
                    "masks": getattr(model, "masks", None),
                    "denominators": getattr(model, "denominators", None),
                    "config": asdict(config),
                }, output_path)
    state = torch.load(output_path, map_location=config.device, weights_only=False)
    model.load_state_dict(state["model"])
    if hasattr(model, "masks"):
        model.masks = {name: value.to(config.device) for name, value in state["masks"].items()}
    if integer:
        model.denominators = dict(state["denominators"])
    metrics = evaluate_masked_torch(model, validation, config=config, temperature=temperature)
    return {"stage": stage, "steps": steps, "best_step": int(state["step"]),
            "metrics": metrics, "elapsed_seconds": time.monotonic() - started,
            "checkpoint": str(output_path), "log": log}


def evaluate_aligned_torch(model, corpus: AlignedChatCorpus, *, device: str,
                           characters: int, batch_size: int,
                           temperature: float = 1.0) -> dict[str, Any]:
    torch = _torch()
    model.eval()
    loss = correct = total = 0
    allocations = {
        length: max(1, round(characters * len(corpus.buckets[length]) / corpus.items))
        for length in corpus.lengths
    }
    # Make the total exact without biasing evaluation toward short contexts.
    difference = characters - sum(allocations.values())
    order = sorted(corpus.lengths, key=lambda length: (-len(corpus.buckets[length]), length))
    for index in range(abs(difference)):
        length = order[index % len(order)]
        allocations[length] += 1 if difference > 0 else -1
    with torch.no_grad():
        for length in corpus.lengths:
            bucket = corpus.buckets[length]
            selected = bucket[:allocations[length]]
            for first in range(0, len(selected), batch_size):
                rows = selected[first:first + batch_size]
                contexts = torch.tensor([row[0] for row in rows], dtype=torch.long,
                                        device=device)
                targets = torch.tensor([row[1] for row in rows], dtype=torch.long,
                                       device=device)
                logits = model(contexts)[:, -1] / temperature
                loss += float(torch.nn.functional.cross_entropy(
                    logits, targets, reduction="sum").cpu())
                correct += int((logits.argmax(dim=-1) == targets).sum().cpu())
                total += len(rows)
    cross_entropy = loss / total
    return {"cross_entropy": cross_entropy, "perplexity": math.exp(cross_entropy),
            "accuracy": correct / total, "characters": total,
            "temperature": temperature}


def train_aligned_model(model, teacher, train: AlignedChatCorpus,
                        validation: AlignedChatCorpus, config: Run7Config, *,
                        stage: str, output_path: Path) -> dict[str, Any]:
    """Targeted correction for question-less random-window supervision."""

    torch = _torch()
    integer = stage == "integer"
    if stage not in {"float", "integer"}:
        raise ValueError("stage must be float or integer")
    steps = config.aligned_integer_steps if integer else config.aligned_float_steps
    maximum_lr = (config.aligned_integer_learning_rate if integer
                  else config.aligned_float_learning_rate)
    temperature = config.integer_logit_temperature if integer else 1.0
    model.to(config.device)
    if hasattr(model, "masks"):
        model.masks = {name: value.to(config.device) for name, value in model.masks.items()}
    if teacher is not None:
        teacher.to(config.device).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=maximum_lr,
                                  weight_decay=config.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(
        config.seed + (900_000 if integer else 800_000))
    best = float("inf")
    log: list[dict[str, Any]] = []
    started = time.monotonic()
    for step in range(steps):
        model.train()
        contexts, targets = train.batch(generator, config.batch_size, config.device)
        logits = model(contexts)[:, -1]
        hard = torch.nn.functional.cross_entropy(logits / temperature, targets)
        loss = hard
        if teacher is not None and config.distillation_weight:
            with torch.no_grad():
                probabilities = torch.softmax(
                    teacher(contexts)[:, -1] / temperature, dim=-1)
            distillation = torch.nn.functional.kl_div(
                torch.log_softmax(logits / temperature, dim=-1), probabilities,
                reduction="batchmean") * temperature * temperature
            loss = ((1.0 - config.distillation_weight) * hard
                    + config.distillation_weight * distillation)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip).detach().cpu())
        lr = _learning_rate(step, steps, config.warmup_steps, maximum_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        model.constrain_latents()
        evaluate = (step == 0 or (step + 1) % config.evaluation_interval == 0
                    or step + 1 == steps)
        if evaluate:
            metrics = evaluate_aligned_torch(
                model, validation, device=config.device,
                characters=config.evaluation_characters,
                batch_size=config.batch_size, temperature=temperature)
            row = {"step": step + 1, "objective": float(loss.detach().cpu()),
                   "gradient_norm": gradient_norm, "learning_rate": lr,
                   "validation": metrics,
                   "elapsed_seconds": time.monotonic() - started}
            log.append(row)
            print(f"aligned {stage} step {step + 1}/{steps}: "
                  f"CE={metrics['cross_entropy']:.4f} "
                  f"acc={metrics['accuracy']:.3%}", flush=True)
            if metrics["cross_entropy"] < best:
                best = float(metrics["cross_entropy"])
                torch.save({
                    "stage": f"aligned-{stage}", "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "step": step + 1,
                    "best_cross_entropy": best,
                    "masks": getattr(model, "masks", None),
                    "denominators": getattr(model, "denominators", None),
                    "config": asdict(config), "log": log,
                }, output_path)
    state = torch.load(output_path, map_location=config.device, weights_only=False)
    model.load_state_dict(state["model"])
    if hasattr(model, "masks"):
        model.masks = {name: value.to(config.device) for name, value in state["masks"].items()}
    if integer:
        model.denominators = dict(state["denominators"])
    metrics = evaluate_aligned_torch(
        model, validation, device=config.device,
        characters=config.evaluation_characters, batch_size=config.batch_size,
        temperature=temperature)
    return {"stage": f"aligned-{stage}", "steps": steps,
            "best_step": int(state["step"]), "metrics": metrics,
            "elapsed_seconds": time.monotonic() - started,
            "checkpoint": str(output_path), "log": state.get("log", log)}


def evaluate_masked_exact(checkpoint: ExactCheckpoint, corpus: MaskedTokenCorpus,
                          *, characters: int) -> dict[str, Any]:
    tokens = bytes(corpus.tokens.tolist())
    mask = bytes(corpus.answer_mask.tolist())
    count = min(characters, len(tokens) - 1)
    totals = {"all_loss": 0.0, "all_correct": 0, "all_count": 0,
              "answer_loss": 0.0, "answer_correct": 0, "answer_count": 0}
    for start in range(0, count, 64):
        model = ExactFactorioLM(checkpoint)
        length = min(64, count - start)
        for offset in range(length):
            trace = model.step(tokens[start + offset])
            target = tokens[start + offset + 1]
            maximum = max(trace.logits)
            loss = maximum + math.log(sum(math.exp(value - maximum)
                                          for value in trace.logits)) - trace.logits[target]
            correct = trace.chosen_token_id == target
            totals["all_loss"] += loss
            totals["all_correct"] += correct
            totals["all_count"] += 1
            if mask[start + offset + 1]:
                totals["answer_loss"] += loss
                totals["answer_correct"] += correct
                totals["answer_count"] += 1
    def metrics(prefix: str) -> dict[str, Any]:
        size = totals[f"{prefix}_count"]
        ce = totals[f"{prefix}_loss"] / size
        return {"cross_entropy": ce, "perplexity": math.exp(ce),
                "accuracy": totals[f"{prefix}_correct"] / size, "characters": size}
    return {"all_characters": metrics("all"), "answer_characters": metrics("answer")}


def _answer_from_generation(text: str) -> str:
    return text.split("\n", 1)[0].strip()


def _answer_quality(answer: str) -> dict[str, Any]:
    words = answer.split()
    repeated = sum(left.lower() == right.lower()
                   for left, right in zip(words, words[1:]))
    runs = max((len(match.group(0)) for match in re.finditer(r"(.)\1+", answer)), default=1)
    return {
        "answer": answer,
        "characters": len(answer),
        "words": len(words),
        "unique_word_ratio": len({word.lower() for word in words}) / max(1, len(words)),
        "adjacent_duplicate_words": repeated,
        "maximum_character_run": runs,
        "has_sentence_boundary": "\n" in answer,
    }


def generation_report(checkpoint: ExactCheckpoint, *, count: int,
                      model=None, device: str = "cpu") -> dict[str, Any]:
    prompts = [f"Q {question}\nA " for question in EVALUATION_QUESTIONS]
    if model is None:
        generations = {
            prompt: ExactFactorioLM(checkpoint).generate(prompt, count)[0]
            for prompt in prompts
        }
    else:
        # This recurrent torch path performs the exported integer operations
        # exactly and is independently checked below against ExactFactorioLM.
        generations = generate_prompt_set(model, prompts, count, device=device)
    return {
        question: {"prompt": prompt, "generated": generations[prompt],
                   **_answer_quality(_answer_from_generation(generations[prompt]))}
        for question, prompt in zip(EVALUATION_QUESTIONS, prompts, strict=True)
    }


def _run5_config(config: Run7Config) -> Run5Config:
    return replace(
        Run5Config.read("configs/run6-int4.json"),
        seed=config.seed, nonzero_weights=config.nonzero_weights,
        sequence_length=64, batch_size=config.batch_size,
        integer_qat_steps=config.integer_qat_steps,
        integer_qat_learning_rate=config.integer_qat_learning_rate,
        warmup_steps=config.warmup_steps,
        evaluation_interval=config.evaluation_interval,
        evaluation_characters=config.evaluation_characters,
        gradient_clip=config.gradient_clip, weight_decay=config.weight_decay,
        distillation_weight=config.distillation_weight,
        integer_logit_temperature=config.integer_logit_temperature,
        weight_limit=config.weight_limit,
        maximum_scale_denominator=config.maximum_scale_denominator,
        generation_length=config.generation_length, device=config.device,
    )


def finalize_run7(config: Run7Config, output: Path, data: dict[str, Any],
                  validation: MaskedTokenCorpus, float_model, integer_model,
                  *, float_training: dict[str, Any],
                  integer_training: dict[str, Any]) -> dict[str, Any]:
    """Export and evaluate trained checkpoints without repeating optimization."""

    integer_config = _run5_config(config)
    metadata = {
        "run": 7, "dataset": DATASET_ID, "task": "compact question answering",
        "seed": config.seed, "initialization": str(RUN6D_EXACT),
        "fixed_run6d_mask": True, "nonzero_weights": config.nonzero_weights,
        "weight_alphabet": list(range(-7, 8)), "score_divisor": 512,
        "prompt_format": "Q question\\nA ",
    }
    checkpoint = export_integer_student(integer_model, integer_config, metadata)
    exact_path = checkpoint.write(output / "factorio-lm-tinychat-exact.json")
    float_metrics = evaluate_masked_torch(float_model, validation, config=config)
    integer_metrics = evaluate_masked_torch(
        integer_model, validation, config=config,
        temperature=config.integer_logit_temperature)
    exact_metrics = evaluate_masked_exact(
        checkpoint, validation, characters=config.exact_evaluation_characters)
    baseline = ExactCheckpoint.read(RUN6D_EXACT)
    baseline_metrics = evaluate_masked_exact(
        baseline, validation, characters=config.exact_evaluation_characters)
    _, _, _, baseline_model = load_clean_int4()
    generations = generation_report(
        checkpoint, count=config.generation_length, model=integer_model,
        device=config.device)
    baseline_generations = generation_report(
        baseline, count=config.generation_length, model=baseline_model,
        device=config.device)
    validate_incremental_integer(
        integer_model, checkpoint, prompt="Q Why is the sky blue\nA ",
        count=32, device=config.device)
    range_report = collect_range_report(
        integer_model, validation, device=config.device, windows=64)
    _write_json(output / "range-report.json", range_report)
    sample_path = output / "generation-samples.txt"
    sample_path.write_text("".join(
        f"QUESTION: {question}\nPROMPT: {row['prompt']!r}\n"
        f"RUN 7 EXACT: {row['generated']}\n"
        f"RUN 6D EXACT: {baseline_generations[question]['generated']}\n\n"
        for question, row in generations.items()
    ), encoding="utf-8")

    answer = exact_metrics["answer_characters"]
    old_answer = baseline_metrics["answer_characters"]
    numerically_better = (
        answer["accuracy"] >= old_answer["accuracy"] + 0.05
        and answer["cross_entropy"] < old_answer["cross_entropy"])
    usable = [row for row in generations.values()
              if row["words"] >= 4 and row["adjacent_duplicate_words"] == 0
              and row["maximum_character_run"] <= 2]
    surface_form_gate = len(usable) >= len(generations) // 2
    report = {
        "run": 7, "configuration": asdict(config), "data": data,
        "initialization": {
            "float": "generated/run6d/factorio-lm-cleaned-float.pt",
            "integer": "generated/run6d/factorio-lm-cleaned-int4.pt",
            "architecture_changed": False, "mask_changed": False,
        },
        "training": {"float": float_training, "integer_qat": integer_training},
        "metrics": {"float": float_metrics, "qat": integer_metrics,
                    "exact": exact_metrics, "run6d_exact_on_chat": baseline_metrics},
        "generations": generations,
        "run6d_generations": baseline_generations,
        "sparsity": mask_summary(integer_model.masks),
        "range_report": str(output / "range-report.json"),
        "quality_gate": {
            "numerically_better_than_run6d_on_chat": numerically_better,
            "automatic_surface_form_gate": surface_form_gate,
            "general_question_answering_coherent": False,
            "manual_semantic_review_required": True,
            "usable_answers": len(usable), "total_questions": len(generations),
        },
        "exact_checkpoint": str(exact_path),
    }
    _write_json(output / "training-report.json", report)
    _write_json(output / "evaluation-report.json", {
        "metrics": report["metrics"], "generations": generations,
        "quality_gate": report["quality_gate"],
    })
    _write_json(output / "manifest.json", {
        "run": 7, "dataset_report": str(output / "dataset-report.json"),
        "training_report": str(output / "training-report.json"),
        "evaluation_report": str(output / "evaluation-report.json"),
        "generation_samples": str(sample_path), "exact_checkpoint": str(exact_path),
    })
    return {"checkpoint": checkpoint, "model": integer_model,
            "report": report, "output": output}


def evaluate_saved_run7(*, config_path: str | Path = "configs/run7-tinychat.json",
                        output_dir: str | Path = RUN7_OUTPUT) -> dict[str, Any]:
    """Finish an interrupted post-training evaluation from saved best states."""

    torch = _torch()
    config = Run7Config.read(config_path)
    config.validate()
    output = Path(output_dir)
    data = json.loads((output / "dataset-report.json").read_text(encoding="utf-8"))
    validation = MaskedTokenCorpus(
        data["splits"]["validation"]["tokens"],
        data["splits"]["validation"]["answer_mask"])
    float_model = create_float_model(Run4Config(seed=config.seed, device=config.device))
    float_state = torch.load(output / "factorio-lm-tinychat-float.pt",
                             map_location=config.device, weights_only=False)
    float_model.load_state_dict(float_state["model"])
    _, _, _, integer_model = load_clean_int4()
    integer_state = torch.load(output / "factorio-lm-tinychat-int4.pt",
                               map_location=config.device, weights_only=False)
    integer_model.load_state_dict(integer_state["model"])
    integer_model.masks = {name: value.to(config.device)
                           for name, value in integer_state["masks"].items()}
    integer_model.denominators = dict(integer_state["denominators"])
    compact = lambda state, stage: {
        "stage": stage, "best_step": int(state["step"]),
        "best_answer_cross_entropy": float(state["best_cross_entropy"]),
        "checkpoint": str(output / f"factorio-lm-tinychat-{stage}.pt"),
        "recovered_after_post_training_evaluation_interruption": True,
    }
    return finalize_run7(
        config, output, data, validation, float_model.eval(), integer_model.eval(),
        float_training=compact(float_state, "float"),
        integer_training=compact(integer_state, "int4"),
    )


def refine_aligned_run7(*, config_path: str | Path = "configs/run7-tinychat.json",
                        output_dir: str | Path = RUN7_OUTPUT) -> dict[str, Any]:
    """Fine-tune int4 weights with each answer target aligned to its question."""

    torch = _torch()
    config = Run7Config.read(config_path)
    config.validate()
    output = Path(output_dir)
    data = json.loads((output / "dataset-report.json").read_text(encoding="utf-8"))
    train_aligned = AlignedChatCorpus(
        data["splits"]["train"]["records"],
        maximum_answer=config.aligned_maximum_answer_characters)
    validation_aligned = AlignedChatCorpus(
        data["splits"]["validation"]["records"],
        maximum_answer=config.aligned_maximum_answer_characters)
    validation = MaskedTokenCorpus(
        data["splits"]["validation"]["tokens"],
        data["splits"]["validation"]["answer_mask"])

    float_model = create_float_model(Run4Config(seed=config.seed, device=config.device))
    source_float = torch.load(output / "factorio-lm-tinychat-float.pt",
                              map_location=config.device, weights_only=False)
    float_model.load_state_dict(source_float["model"])
    float_model.eval()
    float_training = {
        "stage": "aligned-float-audit",
        "steps": 0,
        "checkpoint": str(output / "factorio-lm-tinychat-float.pt"),
        "metrics": evaluate_aligned_torch(
            float_model, validation_aligned, device=config.device,
            characters=config.evaluation_characters,
            batch_size=config.batch_size),
        "decision": (
            "Use the saved float teacher for int4 refinement."
        ),
    }

    _, _, _, integer_model = load_clean_int4()
    source_integer = torch.load(output / "factorio-lm-tinychat-int4.pt",
                                map_location=config.device, weights_only=False)
    integer_model.load_state_dict(source_integer["model"])
    integer_model.masks = {name: value.to(config.device)
                           for name, value in source_integer["masks"].items()}
    integer_model.denominators = dict(source_integer["denominators"])
    integer_training = train_aligned_model(
        integer_model, float_model, train_aligned, validation_aligned, config,
        stage="integer", output_path=output / "factorio-lm-tinychat-aligned-int4.pt")
    aligned_report = {
        "reason": (
            "Align answer targets with their questions and weight the first 32 characters."
        ),
        "train_examples": train_aligned.examples,
        "validation_examples": validation_aligned.examples,
        "weighted_train_targets": train_aligned.items,
        "weighted_validation_targets": validation_aligned.items,
        "maximum_answer_characters": config.aligned_maximum_answer_characters,
        "float": float_training, "integer": integer_training,
    }
    _write_json(output / "aligned-training-report.json", aligned_report)
    result = finalize_run7(
        config, output, data, validation, float_model, integer_model,
        float_training=float_training, integer_training=integer_training)
    result["report"]["targeted_correction"] = aligned_report
    _write_json(output / "training-report.json", result["report"])
    return result


def train_run7(*, config_path: str | Path = "configs/run7-tinychat.json",
               output_dir: str | Path = RUN7_OUTPUT,
               resume_float: str | Path | None = None,
               resume_integer: str | Path | None = None) -> dict[str, Any]:
    torch = _torch()
    config = Run7Config.read(config_path)
    config.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    set_deterministic(config.seed)
    random.seed(config.seed)
    data = prepare_chat_corpus(output, config)
    train = MaskedTokenCorpus(
        data["splits"]["train"]["tokens"], data["splits"]["train"]["answer_mask"])
    validation = MaskedTokenCorpus(
        data["splits"]["validation"]["tokens"],
        data["splits"]["validation"]["answer_mask"])

    float_model = create_float_model(Run4Config(seed=config.seed, device=config.device))
    baseline_float = torch.load(
        "generated/run6d/factorio-lm-cleaned-float.pt",
        map_location=config.device, weights_only=False)
    float_model.load_state_dict(baseline_float["model"])
    float_path = output / "factorio-lm-tinychat-float.pt"
    float_training = train_chat_model(
        float_model, None, train, validation, config, stage="float",
        output_path=float_path, resume_path=resume_float)

    # Warm-start the proven Run 6D deployment model, then preserve its mask and
    # learned scale denominators throughout the TinyChat QAT adaptation.
    _, _, _, integer_model = load_clean_int4()
    integer_path = output / "factorio-lm-tinychat-int4.pt"
    integer_training = train_chat_model(
        integer_model, float_model, train, validation, config, stage="integer",
        output_path=integer_path, resume_path=resume_integer)

    return finalize_run7(
        config, output, data, validation, float_model, integer_model,
        float_training=float_training, integer_training=integer_training)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    data = subparsers.add_parser("data", help="download and prepare TinyChat")
    data.add_argument("--config", default="configs/run7-tinychat.json")
    data.add_argument("--output", default=str(RUN7_OUTPUT))
    train = subparsers.add_parser("train", help="train float and exact-int4 chat models")
    train.add_argument("--config", default="configs/run7-tinychat.json")
    train.add_argument("--output", default=str(RUN7_OUTPUT))
    train.add_argument("--resume-float")
    train.add_argument("--resume-integer")
    evaluate = subparsers.add_parser("evaluate", help="evaluate saved best Run 7 states")
    evaluate.add_argument("--config", default="configs/run7-tinychat.json")
    evaluate.add_argument("--output", default=str(RUN7_OUTPUT))
    refine = subparsers.add_parser("refine", help="run prompt-aligned targeted refinement")
    refine.add_argument("--config", default="configs/run7-tinychat.json")
    refine.add_argument("--output", default=str(RUN7_OUTPUT))
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "data":
        config = Run7Config.read(arguments.config)
        config.validate()
        prepare_chat_corpus(arguments.output, config)
    elif arguments.command == "train":
        train_run7(config_path=arguments.config, output_dir=arguments.output,
                   resume_float=arguments.resume_float,
                   resume_integer=arguments.resume_integer)
    elif arguments.command == "evaluate":
        evaluate_saved_run7(config_path=arguments.config, output_dir=arguments.output)
    elif arguments.command == "refine":
        refine_aligned_run7(config_path=arguments.config, output_dir=arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
