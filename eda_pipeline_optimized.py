from collections import defaultdict
from datetime import datetime
import gc
import json
from pathlib import Path
import random
from typing import TYPE_CHECKING, Any, Dict, Generator, Iterator, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from llm_parallelization.new_processor import NewProcessor

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from prompts import (
    AUGMENTATION_REGISTRY,
    TextAttackAugmentation,
    get_augmentation,
    get_default_backend,
    is_llm_augmentation,
    is_textattack_augmentation,
)
from prompts.fast_augmentations import get_fast_augmentation, is_fast_augmentation_available


def _process_single_textattack_task(
    task: Dict[str, Any], op_type: str, params: Dict[str, Any], use_fast: bool = True
) -> List[Dict[str, Any]]:
    """
    Process a single TextAttack task.
    Module-level function for joblib pickling compatibility.
    """
    from prompts import get_augmentation
    from prompts.fast_augmentations import get_fast_augmentation, is_fast_augmentation_available

    text = task["text"]
    count = task["_count"]

    # Use fast augmentation if requested and available
    if use_fast and is_fast_augmentation_available(op_type):
        augmenter = get_fast_augmentation(op_type, params)
        augmented_texts = augmenter.augment(text, n=count)
    else:
        # Fall back to TextAttack
        augmentation = get_augmentation(op_type, backend="textattack")
        augmenter = augmentation.get_augmenter(params)
        augmented_texts = []
        for _ in range(count):
            try:
                result = augmenter.augment(text)
                augmented_texts.append(result[0] if result else text)
            except Exception:
                augmented_texts.append(text)

    task_results = []
    for aug_text in augmented_texts:
        result = {k: v for k, v in task.items() if not k.startswith("_") and k != "text"}
        result["text"] = aug_text
        task_results.append(result)

    return task_results


class AugmentedText(BaseModel):
    rewritten: str = Field(description="The augmented text")


class CheckpointManager:
    """Manages checkpointing for pipeline resumption with granular batch-level saves."""

    def __init__(self, checkpoint_dir: Path, save_frequency: int = 1):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.checkpoint_dir / "pipeline_state.json"
        self.output_file = self.checkpoint_dir / "checkpoint_output.csv"
        self.live_file = self.checkpoint_dir / "live_results.csv"
        self.save_frequency = save_frequency
        self._batch_counter = 0
        self._pending_rows: List[pd.DataFrame] = []
        self._live_file_initialized = False
        self._column_order: Optional[List[str]] = None

    def save_state(
        self,
        completed_chunks: List[int],
        current_chunk: int,
        current_round: str,
        global_text_id: int,
        total_output_count: int,
        metadata: Dict[str, Any] = None,
        batch_in_round: int = 0,
        completed_rounds_in_chunk: List[str] = None,
    ):
        state = {
            "completed_chunks": completed_chunks,
            "current_chunk": current_chunk,
            "current_round": current_round,
            "global_text_id": global_text_id,
            "total_output_count": total_output_count,
            "batch_in_round": batch_in_round,
            "completed_rounds_in_chunk": completed_rounds_in_chunk or [],
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self) -> Optional[Dict[str, Any]]:
        if not self.state_file.exists():
            return None
        with open(self.state_file, "r") as f:
            return json.load(f)

    def append_output(self, df: pd.DataFrame, first_write: bool = False):
        df.to_csv(
            self.output_file,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
        )

    def append_batch_to_live(
        self,
        results: List[Dict[str, Any]],
        force_flush: bool = False,
        column_order: Optional[List[str]] = None,
    ) -> int:
        if not results:
            return 0

        if column_order and not self._column_order:
            self._column_order = column_order

        batch_df = pd.DataFrame(results)

        if self._column_order:
            for col in self._column_order:
                if col not in batch_df.columns:
                    batch_df[col] = None

        self._pending_rows.append(batch_df)
        self._batch_counter += 1

        rows_written = 0

        if force_flush or self._batch_counter >= self.save_frequency:
            rows_written = self._flush_to_live()

        return rows_written

    def _flush_to_live(self) -> int:
        if not self._pending_rows:
            return 0

        if self._column_order:
            normalized_dfs = []
            for df in self._pending_rows:
                for col in self._column_order:
                    if col not in df.columns:
                        df[col] = None
                available_cols = [c for c in self._column_order if c in df.columns]
                extra_cols = sorted([c for c in df.columns if c not in self._column_order])
                df = df[available_cols + extra_cols]
                normalized_dfs.append(df)
            combined_df = pd.concat(normalized_dfs, ignore_index=True)
        else:
            combined_df = pd.concat(self._pending_rows, ignore_index=True)

        if self._column_order:
            available_cols = [c for c in self._column_order if c in combined_df.columns]
            extra_cols = sorted([c for c in combined_df.columns if c not in self._column_order])
            combined_df = combined_df[available_cols + extra_cols]

        is_first_write = not self._live_file_initialized
        combined_df.to_csv(
            self.live_file,
            mode="w" if is_first_write else "a",
            header=is_first_write,
            index=False,
        )

        self._live_file_initialized = True
        rows_written = len(combined_df)

        self._pending_rows = []
        self._batch_counter = 0

        return rows_written

    def flush_pending(self) -> int:
        return self._flush_to_live()

    def load_output(self) -> Optional[pd.DataFrame]:
        if not self.output_file.exists():
            return None
        return pd.read_csv(self.output_file)

    def load_live_results(self) -> Optional[pd.DataFrame]:
        if not self.live_file.exists():
            return None
        return pd.read_csv(self.live_file)

    def get_live_stats(self) -> Dict[str, Any]:
        if not self.live_file.exists():
            return {"exists": False, "rows": 0}

        with open(self.live_file, "r") as f:
            row_count = sum(1 for _ in f) - 1

        return {
            "exists": True,
            "rows": row_count,
            "path": str(self.live_file),
            "last_modified": datetime.fromtimestamp(self.live_file.stat().st_mtime).isoformat(),
        }

    def clear(self):
        if self.state_file.exists():
            self.state_file.unlink()
        if self.output_file.exists():
            self.output_file.unlink()
        if self.live_file.exists():
            self.live_file.unlink()
        self._reset_live_state()

    def _reset_live_state(self):
        self._batch_counter = 0
        self._pending_rows = []
        self._live_file_initialized = False
        self._column_order = None

    def exists(self) -> bool:
        return self.state_file.exists() and self.output_file.exists()


class EDAPipelineOptimized:
    """Optimized EDA Pipeline with hybrid LLM/TextAttack support."""

    PIPELINE_COLUMNS = {
        "text_id",
        "source_id",
        "text",
        "round",
        "prompt",
        "augmentation_chain",
        "depth",
        "parent_text_id",
        "backend",
    }

    def __init__(
        self,
        processor: Optional["NewProcessor"] = None,
        reference_df: Optional[pd.DataFrame] = None,
        reference_text_col: str = "text",
        min_semantic_similarity: float = 0.75,
        enable_validation: bool = False,
        similarity_model_name: str = "all-MiniLM-L6-v2",
        checkpoint_dir: Optional[str] = None,
        checkpoint_save_frequency: int = 1,
    ):
        self.processor = processor  # Can be None for TextAttack-only pipelines
        self.reference_df = reference_df
        self.reference_text_col = reference_text_col
        self.min_semantic_similarity = min_semantic_similarity
        self.enable_validation = enable_validation
        self.checkpoint_save_frequency = checkpoint_save_frequency

        if checkpoint_dir:
            self.checkpoint_mgr = CheckpointManager(
                Path(checkpoint_dir), save_frequency=checkpoint_save_frequency
            )
        else:
            self.checkpoint_mgr = None

        self.metadata_columns: List[str] = []
        self._column_order: Optional[List[str]] = None

        if reference_df is not None and not reference_df.empty:
            self._precompute_reference_buckets()

        if self.enable_validation:
            print(f"📊 Loading semantic similarity model: {similarity_model_name}...")
            self.similarity_model = SentenceTransformer(similarity_model_name)
            print("✅ Similarity model loaded")
        else:
            self.similarity_model = None

    def _precompute_reference_buckets(self):
        self.reference_buckets = defaultdict(list)
        lengths = self.reference_df[self.reference_text_col].str.len()

        for idx, length in enumerate(lengths):
            bucket = length // 50 * 50
            self.reference_buckets[bucket].append(idx)

        self.reference_bucket_keys = sorted(self.reference_buckets.keys())
        print(f"📦 Pre-computed {len(self.reference_bucket_keys)} reference buckets")

    def _sample_reference_fast(self, text_len: int) -> str:
        target_bucket = text_len // 50 * 50
        min_bucket = max(0, target_bucket - 100)
        max_bucket = target_bucket + 100

        eligible_buckets = [
            k for k in self.reference_bucket_keys if min_bucket <= k <= max_bucket
        ]

        if not eligible_buckets:
            eligible_buckets = self.reference_bucket_keys

        chosen_bucket = random.choice(eligible_buckets)
        chosen_idx = random.choice(self.reference_buckets[chosen_bucket])

        return self.reference_df.iloc[chosen_idx][self.reference_text_col]

    def _validate_semantic_similarity_batch(
        self,
        original_texts: List[str],
        augmented_texts: List[str],
    ) -> List[bool]:
        if not self.enable_validation or self.similarity_model is None:
            return [True] * len(original_texts)

        all_texts = original_texts + augmented_texts
        embeddings = self.similarity_model.encode(
            all_texts, batch_size=64, show_progress_bar=False
        )

        n = len(original_texts)
        orig_embeddings = embeddings[:n]
        aug_embeddings = embeddings[n:]

        orig_norms = np.linalg.norm(orig_embeddings, axis=1, keepdims=True)
        aug_norms = np.linalg.norm(aug_embeddings, axis=1, keepdims=True)

        similarities = np.sum(orig_embeddings * aug_embeddings, axis=1) / (
            orig_norms.flatten() * aug_norms.flatten()
        )

        return (similarities >= self.min_semantic_similarity).tolist()

    def build_prompt(self, aug_type: str, text: str, params: Dict[str, Any]) -> str:
        if pd.isna(text) or not isinstance(text, str) or not text.strip():
            raise ValueError(f"Invalid text value for augmentation: {text!r}")

        augmentation = get_augmentation(aug_type, backend="llm")

        if augmentation.requires_reference:
            if "reference" not in params:
                params["reference"] = self._sample_reference_fast(len(text))

        return augmentation.build_prompt(text, params)

    def _extract_metadata(self, row: pd.Series, text_column: str) -> Dict[str, Any]:
        return {k: v for k, v in row.items() if k != text_column}

    def _preserve_metadata(
        self, source: Dict[str, Any], exclude_pipeline_cols: bool = True
    ) -> Dict[str, Any]:
        if exclude_pipeline_cols:
            return {k: v for k, v in source.items() if k not in self.PIPELINE_COLUMNS}
        return {k: v for k, v in source.items() if k not in {"text", "prompt"}}

    def _get_column_order(self) -> List[str]:
        if self._column_order:
            return self._column_order

        pipeline_cols = [
            "text_id",
            "source_id",
            "text",
            "round",
            "augmentation_chain",
            "depth",
            "parent_text_id",
            "backend",
        ]
        metadata_cols = sorted(self.metadata_columns)
        self._column_order = pipeline_cols + metadata_cols
        return self._column_order

    def _dataframe_chunks(
        self, df: pd.DataFrame, chunk_size: int
    ) -> Generator[Tuple[int, pd.DataFrame], None, None]:
        for start_idx in range(0, len(df), chunk_size):
            yield start_idx, df.iloc[start_idx : start_idx + chunk_size]

    def _get_operation_backend(self, operation: Dict[str, Any]) -> str:
        """Determine which backend to use for an operation."""
        if "backend" in operation:
            return operation["backend"]
        return get_default_backend(operation["type"])

    def _partition_operations_by_backend(
        self, operations: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Partition operations into LLM and TextAttack groups."""
        llm_ops = []
        textattack_ops = []

        for op in operations:
            backend = self._get_operation_backend(op)
            if backend == "llm":
                llm_ops.append(op)
            else:
                textattack_ops.append(op)

        return llm_ops, textattack_ops

    def _build_prompts_vectorized(
        self,
        source_texts: List[Dict[str, Any]],
        operation: Dict[str, Any],
        round_name: str,
    ) -> List[Dict[str, Any]]:
        op_type = operation["type"]
        count = operation.get("count", 1)
        params = operation.get("params", {})

        prompts = []

        for source in source_texts:
            text = source.get("text")
            if pd.isna(text) or not isinstance(text, str) or not text.strip():
                continue

            metadata = self._preserve_metadata(source)
            base_chain = source.get("augmentation_chain", [])
            base_depth = source.get("depth", 0)

            for _ in range(count):
                op_params = params.copy()

                if op_type == "style_rewrite" and "reference_pool" in params:
                    ref_pool = params.get("_reference_pool_texts", [])
                    if ref_pool:
                        op_params["reference"] = random.choice(ref_pool)

                prompt_text = self.build_prompt(op_type, source["text"], op_params)

                prompts.append(
                    {
                        **metadata,
                        "prompt": prompt_text,
                        "source_id": source["source_id"],
                        "parent_text_id": source.get("text_id"),
                        "round": round_name,
                        "augmentation_chain": base_chain + [op_type],
                        "depth": base_depth + 1,
                        "backend": "llm",
                    }
                )

        return prompts

    def _build_textattack_tasks(
        self,
        source_texts: List[Dict[str, Any]],
        operation: Dict[str, Any],
        round_name: str,
    ) -> List[Dict[str, Any]]:
        """Build tasks for TextAttack processing."""
        op_type = operation["type"]
        count = operation.get("count", 1)
        params = operation.get("params", {})
        # Allow explicit backend override: "fast" or "textattack"
        preferred_backend = operation.get("backend", "auto")

        tasks = []

        for source in source_texts:
            text = source.get("text")
            if pd.isna(text) or not isinstance(text, str) or not text.strip():
                continue

            metadata = self._preserve_metadata(source)
            base_chain = source.get("augmentation_chain", [])
            base_depth = source.get("depth", 0)

            tasks.append(
                {
                    **metadata,
                    "text": source["text"],
                    "source_id": source["source_id"],
                    "parent_text_id": source.get("text_id"),
                    "round": round_name,
                    "augmentation_chain": base_chain + [op_type],
                    "depth": base_depth + 1,
                    "backend": "textattack",
                    "_op_type": op_type,
                    "_count": count,
                    "_params": params,
                    "_preferred_backend": preferred_backend,
                }
            )

        return tasks

    def _process_textattack_batch(
        self,
        tasks: List[Dict[str, Any]],
        all_texts_lookup: Dict[str, List[Dict[str, Any]]],
        show_progress: bool = True,
        workers: int = 1,
    ) -> List[Dict[str, Any]]:
        """Process a batch of TextAttack tasks with optional parallelization."""
        if not tasks:
            return []

        results = []

        # Group tasks by operation type for efficiency
        tasks_by_op: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for task in tasks:
            tasks_by_op[task["_op_type"]].append(task)

        for op_type, op_tasks in tasks_by_op.items():
            # Get params and backend preference from first task
            params = op_tasks[0]["_params"] if op_tasks else {}
            preferred_backend = (
                op_tasks[0].get("_preferred_backend", "auto") if op_tasks else "auto"
            )

            # Determine whether to use fast augmentation
            use_fast = False
            if preferred_backend == "fast":
                use_fast = is_fast_augmentation_available(op_type)
                if not use_fast:
                    print(
                        f"      ⚠️ Fast backend requested but not available for {op_type}, using TextAttack"
                    )
            elif preferred_backend == "auto":
                use_fast = is_fast_augmentation_available(op_type)
            # If preferred_backend == "textattack", use_fast stays False

            if workers > 1:
                # Parallel processing with joblib
                mode = "[FAST]" if use_fast else "[TextAttack]"
                if show_progress:
                    print(
                        f"      {mode} {op_type}: {len(op_tasks)} tasks with {workers} workers..."
                    )

                # Use joblib - each worker will check fast availability
                parallel_results = Parallel(n_jobs=workers, backend="loky", verbose=0)(
                    delayed(_process_single_textattack_task)(task, op_type, params, use_fast)
                    for task in op_tasks
                )

                for task_results in parallel_results:
                    results.extend(task_results)

                if show_progress:
                    total_augmented = sum(len(r) for r in parallel_results)
                    print(f"      ✓ {op_type}: generated {total_augmented} texts")
            else:
                # Sequential processing
                fast_aug = None
                textattack_augmenter = None

                if use_fast:
                    fast_aug = get_fast_augmentation(op_type, params)
                    print(f"      [FAST] Processing {len(op_tasks)} tasks for {op_type}")
                else:
                    augmentation = get_augmentation(op_type, backend="textattack")
                    textattack_augmenter = augmentation.get_augmenter(params)
                    print(
                        f"      [TextAttack BATCH] Processing {len(op_tasks)} tasks for {op_type}"
                    )

                if fast_aug:
                    # Fast augmentation - process one by one (already fast)
                    task_iter = op_tasks
                    if show_progress:
                        task_iter = tqdm(
                            op_tasks,
                            desc=f"      ⚡ {op_type}",
                            leave=False,
                            colour="#FFA500",
                        )

                    for task in task_iter:
                        text = task["text"]
                        count = task["_count"]
                        augmented_texts = fast_aug.augment(text, n=count)
                        for aug_text in augmented_texts:
                            result = {
                                k: v
                                for k, v in task.items()
                                if not k.startswith("_") and k != "text"
                            }
                            result["text"] = aug_text
                            results.append(result)
                else:
                    # TextAttack - use batch processing with augment_many
                    # Group by count to batch efficiently
                    tasks_by_count: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
                    for task in op_tasks:
                        tasks_by_count[task["_count"]].append(task)

                    for count, count_tasks in tasks_by_count.items():
                        texts = [t["text"] for t in count_tasks]

                        # Process each count iteration
                        for _ in range(count):
                            if show_progress:
                                print(f"      ⚡ {op_type}: batch of {len(texts)} texts...")

                            try:
                                # Use augment_many for batch processing
                                augmented_batch = textattack_augmenter.augment_many(text)
                            except Exception as e:
                                print(
                                    f"      ⚠️ TextAttack batch error: {e}, falling back to sequential"
                                )
                                augmented_batch = [[t] for t in texts]

                            # augment_many returns list of lists
                            for task, aug_results in zip(count_tasks, augmented_batch):
                                aug_text = aug_results[0] if aug_results else task["text"]
                                result = {
                                    k: v
                                    for k, v in task.items()
                                    if not k.startswith("_") and k != "text"
                                }
                                result["text"] = aug_text
                                results.append(result)

        # Validate if enabled
        if self.enable_validation and results:
            original_texts = []
            augmented_texts = []

            for result in results:
                source_id = result["source_id"]
                original_text = None
                if "original" in all_texts_lookup:
                    for t in all_texts_lookup["original"]:
                        if t["source_id"] == source_id:
                            original_text = t["text"]
                            break

                if original_text:
                    original_texts.append(original_text)
                    augmented_texts.append(result["text"])
                else:
                    original_texts.append(result["text"])
                    augmented_texts.append(result["text"])

            validity_flags = self._validate_semantic_similarity_batch(
                original_texts, augmented_texts
            )
            results = [r for r, v in zip(results, validity_flags) if v]

        return results

    def _build_compound_prompts_vectorized(
        self,
        source_texts: List[Dict[str, Any]],
        operation: Dict[str, Any],
        round_name: str,
    ) -> List[Dict[str, Any]]:
        sequences = operation.get("sequences", [])
        count = operation.get("count", 1)

        prompts = []

        for source in source_texts:
            text = source.get("text")
            if pd.isna(text) or not isinstance(text, str) or not text.strip():
                continue

            metadata = self._preserve_metadata(source)
            base_chain = source.get("augmentation_chain", [])
            base_depth = source.get("depth", 0)

            for sequence in sequences:
                compound_prompt = self._build_compound_prompt_text(source["text"], sequence)

                for _ in range(count):
                    prompts.append(
                        {
                            **metadata,
                            "prompt": compound_prompt,
                            "source_id": source["source_id"],
                            "parent_text_id": source.get("text_id"),
                            "round": round_name,
                            "augmentation_chain": base_chain + sequence,
                            "depth": base_depth + len(sequence),
                            "backend": "llm",
                        }
                    )

        return prompts

    def _build_compound_prompt_text(self, text: str, sequence: List[str]) -> str:
        operations_desc = ", then ".join(sequence)

        return f"""Apply the following transformations in sequence to the text:
{operations_desc}

CRITICAL: Preserve exact semantic content and severity throughout all transformations.

Input: {text}

Respond in JSON: {{"rewritten": "..."}}"""

    def apply_round_streaming(
        self,
        round_config: Dict[str, Any],
        round_name: str,
        source_texts_iter: Iterator[List[Dict[str, Any]]],
        all_texts_lookup: Dict[str, List[Dict[str, Any]]],
        batch_size: int = 25,
        prompt_batch_size: int = 1000,
        on_batch_complete: Optional[callable] = None,
        show_progress: bool = True,
        textattack_workers: int = 1,
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """Apply a round with streaming, routing to appropriate backend."""

        print(f"\n🔄 Starting {round_name} (streaming mode)...")

        operations = round_config.get("operations", [])

        # Partition operations by backend
        llm_ops, textattack_ops = self._partition_operations_by_backend(operations)

        if llm_ops:
            print(f"   📝 LLM operations: {[op['type'] for op in llm_ops]}")
        if textattack_ops:
            worker_info = f", {textattack_workers} workers" if textattack_workers > 1 else ""
            print(
                f"   ⚡ TextAttack operations: {[op['type'] for op in textattack_ops]}{worker_info}"
            )

        # Pre-fetch reference pools if needed
        for op in llm_ops:
            if op["type"] == "style_rewrite" and "reference_pool" in op.get("params", {}):
                pool_name = op["params"]["reference_pool"]
                if pool_name in all_texts_lookup:
                    op["params"]["_reference_pool_texts"] = [
                        t["text"] for t in all_texts_lookup[pool_name]
                    ]

        llm_prompt_buffer = []
        textattack_task_buffer = []
        total_generated = 0
        batch_idx = 0
        source_count = 0

        # Wrap source iterator with progress
        for source_batch in source_texts_iter:
            source_count += len(source_batch)

            # Build LLM prompts
            for op in llm_ops:
                op_type = op["type"]

                if op_type == "compound":
                    prompts = self._build_compound_prompts_vectorized(
                        source_batch, op, round_name
                    )
                else:
                    prompts = self._build_prompts_vectorized(source_batch, op, round_name)

                llm_prompt_buffer.extend(prompts)

            # Build TextAttack tasks
            for op in textattack_ops:
                tasks = self._build_textattack_tasks(source_batch, op, round_name)
                textattack_task_buffer.extend(tasks)

            # Process LLM prompts when buffer is full
            while len(llm_prompt_buffer) >= prompt_batch_size:
                batch_to_process = llm_prompt_buffer[:prompt_batch_size]
                llm_prompt_buffer = llm_prompt_buffer[prompt_batch_size:]

                results = self._process_llm_prompt_batch(
                    batch_to_process, all_texts_lookup, batch_size
                )
                total_generated += len(results)
                batch_idx += 1

                if results:
                    if on_batch_complete:
                        on_batch_complete(results, batch_idx, round_name)
                    yield results

            # Process TextAttack tasks when buffer is full
            while len(textattack_task_buffer) >= prompt_batch_size:
                batch_to_process = textattack_task_buffer[:prompt_batch_size]
                textattack_task_buffer = textattack_task_buffer[prompt_batch_size:]

                print(f"   📦 Processing TextAttack batch ({len(batch_to_process)} tasks)...")
                results = self._process_textattack_batch(
                    batch_to_process,
                    all_texts_lookup,
                    show_progress=show_progress,
                    workers=textattack_workers,
                )
                total_generated += len(results)
                batch_idx += 1

                if results:
                    if on_batch_complete:
                        on_batch_complete(results, batch_idx, round_name)
                    yield results

        # Process remaining LLM prompts
        if llm_prompt_buffer:
            results = self._process_llm_prompt_batch(
                llm_prompt_buffer, all_texts_lookup, batch_size
            )
            total_generated += len(results)
            batch_idx += 1

            if results:
                if on_batch_complete:
                    on_batch_complete(results, batch_idx, round_name)
                yield results

        # Process remaining TextAttack tasks
        if textattack_task_buffer:
            print(
                f"   📦 Processing final TextAttack batch ({len(textattack_task_buffer)} tasks)..."
            )
            results = self._process_textattack_batch(
                textattack_task_buffer,
                all_texts_lookup,
                show_progress=show_progress,
                workers=textattack_workers,
            )
            total_generated += len(results)
            batch_idx += 1

            if results:
                if on_batch_complete:
                    on_batch_complete(results, batch_idx, round_name)
                yield results

        print(f"✅ {round_name}: {source_count} sources → {total_generated} augmented texts")

    def _process_llm_prompt_batch(
        self,
        prompt_batch: List[Dict[str, Any]],
        all_texts_lookup: Dict[str, List[Dict[str, Any]]],
        batch_size: int,
    ) -> List[Dict[str, Any]]:
        """Process a batch of LLM prompts."""
        if not prompt_batch:
            return []

        if self.processor is None:
            raise RuntimeError(
                "LLM processor is not initialized but LLM operations were requested. "
                "This should not happen - check your config for backend specifications."
            )

        prompt_texts = [p["prompt"] for p in prompt_batch]

        self.processor.process_with_schema(
            prompts=prompt_texts, schema=AugmentedText, batch_size=batch_size
        )
        results: List[AugmentedText] = self.processor.parse_results_with_schema(
            schema=AugmentedText
        )

        valid_indices = []
        original_texts = []
        augmented_texts = []

        for idx, (prompt_data, result) in enumerate(zip(prompt_batch, results)):
            if result is None or not result.rewritten:
                continue

            augmented_text = result.rewritten.strip()
            source_id = prompt_data["source_id"]

            original_text = None
            if "original" in all_texts_lookup:
                for t in all_texts_lookup["original"]:
                    if t["source_id"] == source_id:
                        original_text = t["text"]
                        break

            if original_text is None:
                continue

            valid_indices.append(idx)
            original_texts.append(original_text)
            augmented_texts.append(augmented_text)

        if self.enable_validation and original_texts:
            validity_flags = self._validate_semantic_similarity_batch(
                original_texts, augmented_texts
            )
        else:
            validity_flags = [True] * len(original_texts)

        output_texts = []
        for i, (idx, is_valid) in enumerate(zip(valid_indices, validity_flags)):
            if not is_valid:
                continue

            prompt_data = prompt_batch[idx].copy()
            prompt_data["text"] = augmented_texts[i]
            prompt_data.pop("prompt", None)
            output_texts.append(prompt_data)

        return output_texts

    def run_pipeline_chunked(
        self,
        input_df: pd.DataFrame,
        config: Dict[str, Any],
        text_column: str = "text",
        chunk_size: int = 10000,
        output_path: Optional[Path] = None,
        resume: bool = True,
    ) -> pd.DataFrame:
        """Run the pipeline in chunks with granular batch-level checkpointing."""
        print(f"🚀 Starting Optimized EDA Pipeline (chunk_size={chunk_size})...")
        print(f"📊 Total input rows: {len(input_df)}")

        self.metadata_columns = [c for c in input_df.columns if c != text_column]
        print(f"📋 Preserving metadata columns: {self.metadata_columns}")

        batch_size = config.get("pipeline", {}).get("batch_size", 25)
        prompt_batch_size = config.get("pipeline", {}).get("prompt_batch_size", 2000)
        textattack_workers = config.get("pipeline", {}).get("textattack_workers", 1)
        rounds = config.get("rounds", [])

        start_chunk_idx = 0
        global_text_id = 0
        total_output_count = 0
        first_write = True
        completed_chunks: List[int] = []
        completed_rounds_in_chunk: List[str] = []

        if resume and self.checkpoint_mgr and self.checkpoint_mgr.exists():
            state = self.checkpoint_mgr.load_state()
            if state:
                completed_chunks = state.get("completed_chunks", [])
                global_text_id = state.get("global_text_id", 0)
                total_output_count = state.get("total_output_count", 0)
                completed_rounds_in_chunk = state.get("completed_rounds_in_chunk", [])
                first_write = False
                print("🔄 Resuming from checkpoint:")
                print(f"   - Completed chunks: {len(completed_chunks)}")
                print(f"   - Global text ID: {global_text_id}")
                print(f"   - Total output so far: {total_output_count}")

                live_stats = self.checkpoint_mgr.get_live_stats()
                if live_stats["exists"]:
                    print(f"   - Live results file: {live_stats['rows']} rows")
                    print(f"   - Live file path: {live_stats['path']}")

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        all_texts_lookup: Dict[str, List[Dict[str, Any]]] = {}

        column_order = self._get_column_order()

        def on_batch_complete(results: List[Dict[str, Any]], batch_idx: int, round_name: str):
            nonlocal total_output_count
            if self.checkpoint_mgr and results:
                rows_written = self.checkpoint_mgr.append_batch_to_live(
                    results, column_order=column_order
                )
                if rows_written > 0:
                    print(f"   💾 Saved {rows_written} rows to live file (batch {batch_idx})")

        chunk_list = list(self._dataframe_chunks(input_df, chunk_size))

        for chunk_idx, (chunk_start, chunk_df) in enumerate(
            tqdm(chunk_list, desc="Processing chunks", colour="#4CAF50")
        ):
            if chunk_idx in completed_chunks:
                print(f"⏭️  Skipping completed chunk {chunk_idx}")
                continue

            chunk_texts_lookup: Dict[str, List[Dict[str, Any]]] = {}

            original_texts = []
            nan_count = 0
            for idx, row in chunk_df.iterrows():
                text_value = row[text_column]
                if pd.isna(text_value) or (
                    isinstance(text_value, str) and not text_value.strip()
                ):
                    nan_count += 1
                    continue

                original_texts.append(
                    {
                        "text_id": global_text_id,
                        "source_id": idx,
                        "text": str(text_value),
                        "round": "original",
                        "augmentation_chain": [],
                        "depth": 0,
                        "parent_text_id": None,
                        "backend": "original",
                        **self._extract_metadata(row, text_column),
                    }
                )
                global_text_id += 1

            if nan_count > 0:
                print(f"   ⚠️  Skipped {nan_count} rows with NaN/empty text")

            chunk_texts_lookup["original"] = original_texts

            if self.checkpoint_mgr:
                self.checkpoint_mgr.append_batch_to_live(
                    original_texts, force_flush=True, column_order=column_order
                )

            for round_config in rounds:
                round_name = round_config["name"]
                apply_to = round_config.get("apply_to", ["original"])

                if round_name in completed_rounds_in_chunk:
                    print(f"⏭️  Skipping completed round {round_name}")
                    continue

                source_texts = []
                for source_round in apply_to:
                    if source_round in chunk_texts_lookup:
                        source_texts.extend(chunk_texts_lookup[source_round])
                    elif source_round in all_texts_lookup:
                        source_texts.extend(all_texts_lookup[source_round])

                if not source_texts:
                    continue

                round_outputs = []

                def source_iter():
                    batch = []
                    for t in source_texts:
                        batch.append(t)
                        if len(batch) >= 500:
                            yield batch
                            batch = []
                    if batch:
                        yield batch

                batch_in_round = 0
                for batch_results in self.apply_round_streaming(
                    round_config=round_config,
                    round_name=round_name,
                    source_texts_iter=source_iter(),
                    all_texts_lookup={**all_texts_lookup, **chunk_texts_lookup},
                    batch_size=batch_size,
                    prompt_batch_size=prompt_batch_size,
                    on_batch_complete=on_batch_complete,
                    textattack_workers=textattack_workers,
                ):
                    for result in batch_results:
                        result["text_id"] = global_text_id
                        global_text_id += 1

                    round_outputs.extend(batch_results)
                    batch_in_round += 1

                    if self.checkpoint_mgr:
                        self.checkpoint_mgr.save_state(
                            completed_chunks=completed_chunks,
                            current_chunk=chunk_idx,
                            current_round=round_name,
                            global_text_id=global_text_id,
                            total_output_count=total_output_count + len(round_outputs),
                            metadata={"text_column": text_column, "chunk_size": chunk_size},
                            batch_in_round=batch_in_round,
                            completed_rounds_in_chunk=completed_rounds_in_chunk,
                        )

                chunk_texts_lookup[round_name] = round_outputs
                completed_rounds_in_chunk.append(round_name)

                if self.checkpoint_mgr:
                    self.checkpoint_mgr.flush_pending()
                    self.checkpoint_mgr.save_state(
                        completed_chunks=completed_chunks,
                        current_chunk=chunk_idx,
                        current_round=round_name,
                        global_text_id=global_text_id,
                        total_output_count=total_output_count,
                        metadata={"text_column": text_column, "chunk_size": chunk_size},
                        completed_rounds_in_chunk=completed_rounds_in_chunk,
                    )

            chunk_output = []
            for round_name, texts in chunk_texts_lookup.items():
                chunk_output.extend(texts)

            if chunk_output:
                chunk_df_out = pd.DataFrame(chunk_output)

                pipeline_cols = [
                    "text_id",
                    "source_id",
                    "text",
                    "round",
                    "augmentation_chain",
                    "depth",
                    "parent_text_id",
                    "backend",
                ]
                other_cols = [c for c in chunk_df_out.columns if c not in pipeline_cols]
                ordered_cols = pipeline_cols + sorted(other_cols)
                ordered_cols = [c for c in ordered_cols if c in chunk_df_out.columns]
                chunk_df_out = chunk_df_out[ordered_cols]

                if output_path:
                    chunk_df_out.to_csv(
                        output_path,
                        mode="a" if not first_write else "w",
                        header=first_write,
                        index=False,
                    )
                    first_write = False

                if self.checkpoint_mgr:
                    self.checkpoint_mgr.append_output(
                        chunk_df_out,
                        first_write=(chunk_idx == 0 and chunk_idx not in completed_chunks),
                    )

                total_output_count += len(chunk_output)

            completed_chunks.append(chunk_idx)
            completed_rounds_in_chunk = []

            if self.checkpoint_mgr:
                self.checkpoint_mgr.save_state(
                    completed_chunks=completed_chunks,
                    current_chunk=chunk_idx,
                    current_round="completed",
                    global_text_id=global_text_id,
                    total_output_count=total_output_count,
                    metadata={"text_column": text_column, "chunk_size": chunk_size},
                    completed_rounds_in_chunk=[],
                )

            for round_config in rounds:
                round_name = round_config["name"]
                if round_name in chunk_texts_lookup:
                    if round_name not in all_texts_lookup:
                        all_texts_lookup[round_name] = []
                    sample_size = min(1000, len(chunk_texts_lookup[round_name]))
                    if sample_size > 0:
                        all_texts_lookup[round_name].extend(
                            random.sample(chunk_texts_lookup[round_name], sample_size)
                        )

            del chunk_texts_lookup
            gc.collect()

        print(f"\n✅ Pipeline complete! Generated {total_output_count} total texts.")
        print(f"📋 Preserved columns: {self.metadata_columns}")

        if self.checkpoint_mgr:
            print("🧹 Clearing checkpoint files (pipeline completed successfully)")
            self.checkpoint_mgr.clear()

        if output_path and output_path.exists():
            return pd.read_csv(output_path)

        return pd.DataFrame()

    def run_pipeline(
        self,
        input_df: pd.DataFrame,
        config: Dict[str, Any],
        text_column: str = "text",
        resume: bool = True,
    ) -> pd.DataFrame:
        chunk_size = config.get("pipeline", {}).get("chunk_size", 50000)

        if len(input_df) > chunk_size:
            print(
                f"📊 Large dataset detected ({len(input_df)} rows), using chunked processing..."
            )
            output_folder = config.get("output", {}).get("output_folder", "./output")
            output_path = Path(output_folder) / "final_augmented_dataset.csv"
            return self.run_pipeline_chunked(
                input_df=input_df,
                config=config,
                text_column=text_column,
                chunk_size=chunk_size,
                output_path=output_path,
                resume=resume,
            )

        output_folder = config.get("output", {}).get("output_folder", "./output")
        output_path = Path(output_folder) / "final_augmented_dataset.csv"
        return self.run_pipeline_chunked(
            input_df=input_df,
            config=config,
            text_column=text_column,
            chunk_size=len(input_df) + 1,
            output_path=output_path,
            resume=resume,
        )

    def get_live_results(self) -> Optional[pd.DataFrame]:
        if self.checkpoint_mgr:
            return self.checkpoint_mgr.load_live_results()
        return None

    def get_pipeline_status(self) -> Dict[str, Any]:
        status = {
            "checkpoint_exists": False,
            "live_results": {"exists": False, "rows": 0},
            "state": None,
        }

        if self.checkpoint_mgr:
            status["checkpoint_exists"] = self.checkpoint_mgr.exists()
            status["live_results"] = self.checkpoint_mgr.get_live_stats()
            if status["checkpoint_exists"]:
                status["state"] = self.checkpoint_mgr.load_state()

        return status


def load_dataframe(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    elif ext == ".jsonl":
        return pd.read_json(file_path, lines=True)
    elif ext in [".pkl", ".pickle"]:
        return pd.read_pickle(file_path)
    elif ext == ".parquet":
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def load_dataframe_lazy(
    file_path: str, chunk_size: int = 10000
) -> Generator[pd.DataFrame, None, None]:
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".csv":
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            yield chunk
    elif ext == ".parquet":
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(file_path)
        for batch in parquet_file.iter_batches(batch_size=chunk_size):
            yield batch.to_pandas()
    else:
        yield load_dataframe(file_path)


def inspect_live_results(checkpoint_dir: str) -> Optional[pd.DataFrame]:
    live_file = Path(checkpoint_dir) / "live_results.csv"
    if live_file.exists():
        return pd.read_csv(live_file)
    return None


def get_checkpoint_status(checkpoint_dir: str) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_dir)
    state_file = checkpoint_path / "pipeline_state.json"
    live_file = checkpoint_path / "live_results.csv"

    status = {
        "checkpoint_exists": state_file.exists(),
        "live_file_exists": live_file.exists(),
        "state": None,
        "live_rows": 0,
        "last_modified": None,
    }

    if state_file.exists():
        with open(state_file, "r") as f:
            status["state"] = json.load(f)

    if live_file.exists():
        with open(live_file, "r") as f:
            status["live_rows"] = sum(1 for _ in f) - 1
        status["last_modified"] = datetime.fromtimestamp(live_file.stat().st_mtime).isoformat()

    return status
