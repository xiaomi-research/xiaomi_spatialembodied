"""
Type-grouped batch sampler for ms-swift.

Ensures every batch (across all distributed ranks) contains samples of the same
modality type (text-only / image / video).  When combined with ZeRO-3 this
avoids gathering parameters for modules that are not used in a given step
(e.g. the vision encoder for a text-only batch), reducing peak memory and
improving effective sequence length.

Activation:
    export SWIFT_GROUP_BY_DATA_TYPE=1

The sampler is activated by monkey-patching ``DataLoaderMixin.get_train_dataloader``
(see :func:`patch_dataloader_mixin`).  Call it from the model's register file
(``--custom_register_path``) so the patch is applied before the Trainer is
constructed.

Compatible with: DDP, DeepSpeed ZeRO-2/3, ``group_by_length``, ``skip_batches``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional

import torch

from swift.dataloader.shard import BatchSamplerShard, DataLoaderShard
from swift.utils import seed_worker

logger = logging.getLogger(__name__)

# ── data-type constants ──────────────────────────────────────────────────────
DATA_TYPE_TEXT = 0
DATA_TYPE_IMAGE = 1
DATA_TYPE_VIDEO = 2

_TYPE_NAMES = {DATA_TYPE_TEXT: "text", DATA_TYPE_IMAGE: "image", DATA_TYPE_VIDEO: "video"}


# ── Phase 1: infer per-sample data types from the raw HfDataset ─────────────
def infer_data_types(dataset) -> List[int]:
    """Return a list of ``DATA_TYPE_*`` ints, one per sample.

    Detection strategy (fast → slow):
    1. If *images* / *videos* columns exist, use them directly (O(1) per sample).
    2. Otherwise scan the *messages* column for media content markers.
    """
    n = len(dataset)
    data_types = [DATA_TYPE_TEXT] * n
    columns = set(dataset.column_names) if hasattr(dataset, "column_names") else set()

    if "images" in columns or "videos" in columns:
        if "images" in columns:
            for i, val in enumerate(dataset["images"]):
                if val:
                    data_types[i] = DATA_TYPE_IMAGE
        if "videos" in columns:
            for i, val in enumerate(dataset["videos"]):
                if val:
                    data_types[i] = DATA_TYPE_VIDEO
        return data_types

    if "messages" not in columns:
        return data_types

    for i, messages in enumerate(dataset["messages"]):
        has_image = has_video = False
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        t = item.get("type", "")
                        has_image = has_image or t == "image"
                        has_video = has_video or t == "video"
            elif isinstance(content, str):
                has_image = has_image or "<image>" in content
                has_video = has_video or "<video>" in content
            if has_image and has_video:
                break
        if has_video:
            data_types[i] = DATA_TYPE_VIDEO
        elif has_image:
            data_types[i] = DATA_TYPE_IMAGE
    return data_types


# ── Phase 2: TypeGroupedBatchSampler ─────────────────────────────────────────
class TypeGroupedBatchSampler(BatchSamplerShard):
    """``BatchSamplerShard`` that groups samples by modality type.

    The global index space is partitioned by type, shuffled *within* each type,
    then cut into **full** chunks of ``batch_size × world_size``.  Chunks are
    shuffled across types so that different modalities are interleaved epoch to
    epoch.  Each rank takes its strided slice from a chunk, guaranteeing all
    ranks see the same type at the same training step.

    Incomplete trailing chunks are handled according to ``drop_last``:

    * ``drop_last=True`` — trailing chunks are dropped, so every rank yields the
      same number of samples at every step.  Up to one full chunk per type is
      discarded per epoch.
    * ``drop_last=False`` — a trailing chunk is kept as long as it holds at least
      ``world_size`` samples; a smaller one is dropped, because otherwise some rank
      would get an empty slice and the ranks would de-synchronise (they have to
      yield the same number of batches).

      Note that a kept trailing chunk has fewer than ``batch_size × world_size``
      entries, so **the per-rank batch sizes in that step can differ by one** (e.g.
      ``[2, 2, 2, 1]`` for ``world_size=4``).  DDP all-reduces gradients with equal
      weight per rank, so such a step is mildly mis-weighted.  This is a training
      quality nuance, not a crash, but the earlier version of this docstring claimed
      the opposite ("dropped to keep all ranks synchronised").
    """

    def __init__(
        self,
        total_samples: int,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        data_seed: Optional[int],
        data_types: List[int],
        tp_size: int = 1,
        group_by_length: bool = False,
        lengths=None,
    ):
        super().__init__(
            total_samples,
            batch_size,
            shuffle,
            drop_last,
            data_seed,
            tp_size=tp_size,
            group_by_length=group_by_length,
            lengths=lengths,
        )
        self.data_types = data_types

    # ── iterator ─────────────────────────────────────────────────────────
    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.curr_seed)

        global_n = self.total_samples * self.world_size
        chunk_size = self.batch_size * self.world_size

        # 1. Group global indices by type.
        type_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx in range(global_n):
            t = self.data_types[idx] if idx < len(self.data_types) else DATA_TYPE_TEXT
            type_to_indices[t].append(idx)

        # 2. Per-type: shuffle (or length-group), then cut full chunks.
        all_chunks: List[List[int]] = []
        for type_id in sorted(type_to_indices):
            indices = type_to_indices[type_id]

            if self.group_by_length and self.lengths is not None:
                type_lengths = [self.lengths[i] if i < len(self.lengths) else 0 for i in indices]
                from transformers.trainer_pt_utils import get_length_grouped_indices
                ordered = get_length_grouped_indices(type_lengths, chunk_size, generator=generator)
                indices = [indices[i] for i in ordered]
            elif self.shuffle:
                perm = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[i] for i in perm]

            for start in range(0, len(indices), chunk_size):
                chunk = indices[start : start + chunk_size]
                if len(chunk) == chunk_size:
                    all_chunks.append(chunk)
                elif not self.drop_last and len(chunk) >= self.world_size:
                    all_chunks.append(chunk)

        # 3. Shuffle chunk order (inter-type mixing across the epoch).
        if self.shuffle:
            perm = torch.randperm(len(all_chunks), generator=generator).tolist()
            all_chunks = [all_chunks[i] for i in perm]

        # 4. Yield per-rank batches.
        for chunk in all_chunks:
            my_batch = chunk[self.rank :: self.world_size]
            if my_batch:
                yield list(my_batch)

    # ── length ───────────────────────────────────────────────────────────
    def __len__(self) -> int:
        global_n = self.total_samples * self.world_size
        chunk_size = self.batch_size * self.world_size

        type_counts: Dict[int, int] = defaultdict(int)
        for idx in range(global_n):
            t = self.data_types[idx] if idx < len(self.data_types) else DATA_TYPE_TEXT
            type_counts[t] += 1

        total_batches = 0
        for count in type_counts.values():
            total_batches += count // chunk_size
            remainder = count % chunk_size
            if not self.drop_last and remainder >= self.world_size:
                total_batches += 1
        return total_batches


# ── Phase 3: monkey-patch DataLoaderMixin ────────────────────────────────────
_original_get_train_dataloader = None


def _patched_get_train_dataloader(self, skip_batches=0):
    """Drop-in replacement for ``DataLoaderMixin.get_train_dataloader``."""
    # Sequence-parallel takes a completely different path; leave it alone.
    if self.template.sequence_parallel_size > 1:
        dl = self.get_sp_dataloader(self.train_dataset, self._train_batch_size, skip_batches=skip_batches)
        if dl is not None:
            return dl

    if self.train_dataset is None:
        raise ValueError("Trainer: training requires a train_dataset.")
    if not hasattr(self.train_dataset, "__len__"):
        logger.warning("TypeGroupedBatchSampler: IterableDataset detected, falling back to default dataloader.")
        return _original_get_train_dataloader(self, skip_batches)

    args = self.args
    train_dataset = self.train_dataset

    # ── infer data types ──
    data_types = infer_data_types(train_dataset)
    type_dist = defaultdict(int)
    for t in data_types:
        type_dist[t] += 1
    logger.info(
        "TypeGroupedBatchSampler: data type distribution: %s",
        {_TYPE_NAMES.get(k, k): v for k, v in sorted(type_dist.items())},
    )

    # ── build sampler ──
    batch_sampler_kwargs = dict(
        drop_last=args.dataloader_drop_last,
        shuffle=args.train_dataloader_shuffle,
        data_seed=args.data_seed,
        data_types=data_types,
        tp_size=(
            args.deepspeed["tensor_parallel"]["autotp_size"]
            if args.deepspeed and "tensor_parallel" in args.deepspeed
            else 1
        ),
    )
    if args.group_by_length:
        batch_sampler_kwargs["group_by_length"] = True
        batch_sampler_kwargs["lengths"] = train_dataset["lengths"]

    batch_sampler = TypeGroupedBatchSampler(
        len(train_dataset), batch_size=self._train_batch_size, **batch_sampler_kwargs
    )

    if skip_batches > 0:
        from accelerate.data_loader import SkipBatchSampler
        batch_sampler = SkipBatchSampler(batch_sampler, skip_batches=skip_batches)

    dataloader = DataLoaderShard(
        train_dataset,
        device=self.accelerator.device,
        batch_sampler=batch_sampler,
        collate_fn=self.data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.dataloader_pin_memory,
        persistent_workers=args.dataloader_persistent_workers,
        prefetch_factor=args.dataloader_prefetch_factor,
        worker_init_fn=partial(seed_worker, num_workers=args.dataloader_num_workers, rank=args.process_index),
    )
    return dataloader


def patch_dataloader_mixin():
    """Monkey-patch ``DataLoaderMixin.get_train_dataloader`` (idempotent)."""
    global _original_get_train_dataloader
    if _original_get_train_dataloader is not None:
        return
    from swift.trainers.mixin import DataLoaderMixin

    _original_get_train_dataloader = DataLoaderMixin.get_train_dataloader
    DataLoaderMixin.get_train_dataloader = _patched_get_train_dataloader
    logger.info("TypeGroupedBatchSampler: patched DataLoaderMixin.get_train_dataloader")
