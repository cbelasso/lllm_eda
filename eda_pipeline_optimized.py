from collections import defaultdict
from datetime import datetime
import gc
import json
from pathlib import Path
import random
from typing import Any, Dict, Generator, Iterator, List, Optional, Set, Tuple

from llm_parallelization.new_processor import NewProcessor
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from prompts import AUGMENTATION_REGISTRY, get_augmentation


class AugmentedText(BaseModel):
    rewritten: str = Field(description="The augmented text")


class CheckpointManager:
    """Manages checkpointing for pipeline resumption with granular batch-level saves."""

    def __init__(self, checkpoint_dir: Path, save_frequency: int = 1):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoint files
            save_frequency: Save to live file every N batches (1 = every batch)
        """
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
        """Save pipeline state for resumption with batch-level granularity."""
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
        """Load previous pipeline state if exists."""
        if not self.state_file.exists():
            return None
        with open(self.state_file, "r") as f:
            return json.load(f)

    def append_output(self, df: pd.DataFrame, first_write: bool = False):
        """Append results to checkpoint output file."""
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
        """
        Append batch results to live file for real-time inspection.

        Args:
            results: List of result dictionaries to append
            force_flush: Force write even if batch threshold not met
            column_order: Column ordering for consistent output

        Returns:
            Number of rows written (0 if buffered, >0 if flushed)
        """
        if not results:
            return 0

        # Store column order for consistent output
        if column_order and not self._column_order:
            self._column_order = column_order

        batch_df = pd.DataFrame(results)

        # Ensure all expected columns exist (fill missing with None)
        if self._column_order:
            for col in self._column_order:
                if col not in batch_df.columns:
                    batch_df[col] = None

        self._pending_rows.append(batch_df)
        self._batch_counter += 1

        rows_written = 0

        # Flush if threshold met or forced
        if force_flush or self._batch_counter >= self.save_frequency:
            rows_written = self._flush_to_live()

        return rows_written

    def _flush_to_live(self) -> int:
        """Flush pending rows to live file."""
        if not self._pending_rows:
            return 0

        # Ensure all dataframes have the same columns before concatenation
        if self._column_order:
            normalized_dfs = []
            for df in self._pending_rows:
                # Add missing columns
                for col in self._column_order:
                    if col not in df.columns:
                        df[col] = None
                # Reorder columns (keep only expected + any extra)
                available_cols = [c for c in self._column_order if c in df.columns]
                extra_cols = sorted([c for c in df.columns if c not in self._column_order])
                df = df[available_cols + extra_cols]
                normalized_dfs.append(df)
            combined_df = pd.concat(normalized_dfs, ignore_index=True)
        else:
            combined_df = pd.concat(self._pending_rows, ignore_index=True)

        # Apply final column ordering
        if self._column_order:
            available_cols = [c for c in self._column_order if c in combined_df.columns]
            extra_cols = sorted([c for c in combined_df.columns if c not in self._column_order])
            combined_df = combined_df[available_cols + extra_cols]

        # Write to live file
        is_first_write = not self._live_file_initialized
        combined_df.to_csv(
            self.live_file,
            mode="w" if is_first_write else "a",
            header=is_first_write,
            index=False,
        )

        self._live_file_initialized = True
        rows_written = len(combined_df)

        # Clear buffer
        self._pending_rows = []
        self._batch_counter = 0

        return rows_written

    def flush_pending(self) -> int:
        """Force flush any pending rows to live file."""
        return self._flush_to_live()

    def load_output(self) -> Optional[pd.DataFrame]:
        """Load checkpoint output if exists."""
        if not self.output_file.exists():
            return None
        return pd.read_csv(self.output_file)

    def load_live_results(self) -> Optional[pd.DataFrame]:
        """Load current live results for inspection."""
        if not self.live_file.exists():
            return None
        return pd.read_csv(self.live_file)

    def get_live_stats(self) -> Dict[str, Any]:
        """Get statistics about live results without loading full dataframe."""
        if not self.live_file.exists():
            return {"exists": False, "rows": 0}

        # Count rows efficiently
        with open(self.live_file, "r") as f:
            row_count = sum(1 for _ in f) - 1  # Subtract header

        return {
            "exists": True,
            "rows": row_count,
            "path": str(self.live_file),
            "last_modified": datetime.fromtimestamp(self.live_file.stat().st_mtime).isoformat(),
        }

    def clear(self):
        """Clear checkpoint files."""
        if self.state_file.exists():
            self.state_file.unlink()
        if self.output_file.exists():
            self.output_file.unlink()
        if self.live_file.exists():
            self.live_file.unlink()
        self._reset_live_state()

    def _reset_live_state(self):
        """Reset live file tracking state."""
        self._batch_counter = 0
        self._pending_rows = []
        self._live_file_initialized = False
        self._column_order = None

    def exists(self) -> bool:
        """Check if a checkpoint exists."""
        return self.state_file.exists() and self.output_file.exists()


class EDAPipelineOptimized:
    """Optimized EDA Pipeline with metadata preservation and granular checkpointing."""

    # Columns added by the pipeline (not from original data)
    PIPELINE_COLUMNS = {
        "text_id",
        "source_id",
        "text",
        "round",
        "prompt",
        "augmentation_chain",
        "depth",
        "parent_text_id",
    }

    def __init__(
        self,
        processor: NewProcessor,
        reference_df: Optional[pd.DataFrame] = None,
        reference_text_col: str = "text",
        min_semantic_similarity: float = 0.75,
        enable_validation: bool = False,
        similarity_model_name: str = "all-MiniLM-L6-v2",
        checkpoint_dir: Optional[str] = None,
        checkpoint_save_frequency: int = 1,
    ):
        """
        Initialize the EDA Pipeline.

        Args:
            processor: LLM processor instance
            reference_df: Reference corpus dataframe for style transfer
            reference_text_col: Column name for reference texts
            min_semantic_similarity: Minimum similarity threshold for validation
            enable_validation: Enable semantic similarity validation
            similarity_model_name: Sentence transformer model for validation
            checkpoint_dir: Directory for checkpoints
            checkpoint_save_frequency: Save to live file every N batches
        """
        self.processor = processor
        self.reference_df = reference_df
        self.reference_text_col = reference_text_col
        self.min_semantic_similarity = min_semantic_similarity
        self.enable_validation = enable_validation
        self.checkpoint_save_frequency = checkpoint_save_frequency

        # Checkpoint manager
        if checkpoint_dir:
            self.checkpoint_mgr = CheckpointManager(
                Path(checkpoint_dir), save_frequency=checkpoint_save_frequency
            )
        else:
            self.checkpoint_mgr = None

        # Track metadata columns from input
        self.metadata_columns: List[str] = []

        # Column ordering for consistent output
        self._column_order: Optional[List[str]] = None

        # Pre-sample reference texts for faster access
        if reference_df is not None and not reference_df.empty:
            self._precompute_reference_buckets()

        if self.enable_validation:
            print(f"📊 Loading semantic similarity model: {similarity_model_name}...")
            self.similarity_model = SentenceTransformer(similarity_model_name)
            print("✅ Similarity model loaded")
        else:
            self.similarity_model = None

    def _precompute_reference_buckets(self):
        """Pre-bucket reference texts by length for O(1) sampling."""
        self.reference_buckets = defaultdict(list)
        lengths = self.reference_df[self.reference_text_col].str.len()

        for idx, length in enumerate(lengths):
            bucket = length // 50 * 50
            self.reference_buckets[bucket].append(idx)

        self.reference_bucket_keys = sorted(self.reference_buckets.keys())
        print(f"📦 Pre-computed {len(self.reference_bucket_keys)} reference buckets")

    def _sample_reference_fast(self, text_len: int) -> str:
        """Fast reference sampling using pre-computed buckets."""
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
        """Batch validation for efficiency."""
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
        """Build prompt using the augmentation registry."""
        # Defensive check for invalid text
        if pd.isna(text) or not isinstance(text, str) or not text.strip():
            raise ValueError(f"Invalid text value for augmentation: {text!r}")

        augmentation = get_augmentation(aug_type)

        if augmentation.requires_reference:
            if "reference" not in params:
                params["reference"] = self._sample_reference_fast(len(text))

        return augmentation.build_prompt(text, params)

    def _extract_metadata(self, row: pd.Series, text_column: str) -> Dict[str, Any]:
        """Extract all metadata columns from a row."""
        return {k: v for k, v in row.items() if k != text_column}

    def _preserve_metadata(
        self, source: Dict[str, Any], exclude_pipeline_cols: bool = True
    ) -> Dict[str, Any]:
        """Preserve metadata from source, optionally excluding pipeline columns."""
        if exclude_pipeline_cols:
            return {k: v for k, v in source.items() if k not in self.PIPELINE_COLUMNS}
        return {k: v for k, v in source.items() if k not in {"text", "prompt"}}

    def _get_column_order(self) -> List[str]:
        """Get consistent column ordering for output files."""
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
        ]
        metadata_cols = sorted(self.metadata_columns)
        self._column_order = pipeline_cols + metadata_cols
        return self._column_order

    def _dataframe_chunks(
        self, df: pd.DataFrame, chunk_size: int
    ) -> Generator[Tuple[int, pd.DataFrame], None, None]:
        """Yield dataframe chunks with their starting index."""
        for start_idx in range(0, len(df), chunk_size):
            yield start_idx, df.iloc[start_idx : start_idx + chunk_size]

    def _build_prompts_vectorized(
        self,
        source_texts: List[Dict[str, Any]],
        operation: Dict[str, Any],
        round_name: str,
    ) -> List[Dict[str, Any]]:
        """Build prompts with metadata preservation."""
        op_type = operation["type"]
        count = operation.get("count", 1)
        params = operation.get("params", {})

        prompts = []

        for source in source_texts:
            # Skip sources with invalid text
            text = source.get("text")
            if pd.isna(text) or not isinstance(text, str) or not text.strip():
                continue

            # Preserve all metadata from source
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
                        **metadata,  # All original metadata columns
                        "prompt": prompt_text,
                        "source_id": source["source_id"],
                        "parent_text_id": source.get("text_id"),
                        "round": round_name,
                        "augmentation_chain": base_chain + [op_type],
                        "depth": base_depth + 1,
                    }
                )

        return prompts

    def _build_compound_prompts_vectorized(
        self,
        source_texts: List[Dict[str, Any]],
        operation: Dict[str, Any],
        round_name: str,
    ) -> List[Dict[str, Any]]:
        """Build compound prompts with metadata preservation."""
        sequences = operation.get("sequences", [])
        count = operation.get("count", 1)

        prompts = []

        for source in source_texts:
            # Skip sources with invalid text
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
                        }
                    )

        return prompts

    def _build_compound_prompt_text(self, text: str, sequence: List[str]) -> str:
        """Build a compound prompt that chains multiple operations."""
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
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Apply a round with streaming to reduce memory footprint.

        Args:
            round_config: Configuration for this round
            round_name: Name of the round
            source_texts_iter: Iterator of source text batches
            all_texts_lookup: Lookup dict for all texts by round
            batch_size: LLM batch size
            prompt_batch_size: Number of prompts to accumulate before processing
            on_batch_complete: Optional callback after each batch is processed
        """

        print(f"\n🔄 Starting {round_name} (streaming mode)...")

        operations = round_config.get("operations", [])

        # Pre-fetch reference pools if needed
        for op in operations:
            if op["type"] == "style_rewrite" and "reference_pool" in op.get("params", {}):
                pool_name = op["params"]["reference_pool"]
                if pool_name in all_texts_lookup:
                    op["params"]["_reference_pool_texts"] = [
                        t["text"] for t in all_texts_lookup[pool_name]
                    ]

        prompt_buffer = []
        total_generated = 0
        batch_idx = 0

        for source_batch in source_texts_iter:
            for op in operations:
                op_type = op["type"]

                if op_type == "compound":
                    prompts = self._build_compound_prompts_vectorized(
                        source_batch, op, round_name
                    )
                else:
                    prompts = self._build_prompts_vectorized(source_batch, op, round_name)

                prompt_buffer.extend(prompts)

            while len(prompt_buffer) >= prompt_batch_size:
                batch_to_process = prompt_buffer[:prompt_batch_size]
                prompt_buffer = prompt_buffer[prompt_batch_size:]

                results = self._process_prompt_batch(
                    batch_to_process, all_texts_lookup, batch_size
                )
                total_generated += len(results)
                batch_idx += 1

                if results:
                    if on_batch_complete:
                        on_batch_complete(results, batch_idx, round_name)
                    yield results

        if prompt_buffer:
            results = self._process_prompt_batch(prompt_buffer, all_texts_lookup, batch_size)
            total_generated += len(results)
            batch_idx += 1

            if results:
                if on_batch_complete:
                    on_batch_complete(results, batch_idx, round_name)
                yield results

        print(f"✅ Generated {total_generated} texts in {round_name}")

    def _process_prompt_batch(
        self,
        prompt_batch: List[Dict[str, Any]],
        all_texts_lookup: Dict[str, List[Dict[str, Any]]],
        batch_size: int,
    ) -> List[Dict[str, Any]]:
        """Process a batch of prompts and validate results."""
        if not prompt_batch:
            return []

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

        # Identify and store metadata columns
        self.metadata_columns = [c for c in input_df.columns if c != text_column]
        print(f"📋 Preserving metadata columns: {self.metadata_columns}")

        batch_size = config.get("pipeline", {}).get("batch_size", 25)
        prompt_batch_size = config.get("pipeline", {}).get("prompt_batch_size", 2000)
        rounds = config.get("rounds", [])

        # Check for existing checkpoint
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

                # Show live file stats
                live_stats = self.checkpoint_mgr.get_live_stats()
                if live_stats["exists"]:
                    print(f"   - Live results file: {live_stats['rows']} rows")
                    print(f"   - Live file path: {live_stats['path']}")

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        all_texts_lookup: Dict[str, List[Dict[str, Any]]] = {}

        # Initialize column order
        column_order = self._get_column_order()

        # Define batch completion callback for live saving
        def on_batch_complete(results: List[Dict[str, Any]], batch_idx: int, round_name: str):
            nonlocal total_output_count
            if self.checkpoint_mgr and results:
                rows_written = self.checkpoint_mgr.append_batch_to_live(
                    results, column_order=column_order
                )
                if rows_written > 0:
                    print(f"   💾 Saved {rows_written} rows to live file (batch {batch_idx})")

        # Process input in chunks
        chunk_list = list(self._dataframe_chunks(input_df, chunk_size))

        for chunk_idx, (chunk_start, chunk_df) in enumerate(
            tqdm(chunk_list, desc="Processing chunks", colour="#4CAF50")
        ):
            # Skip already completed chunks
            if chunk_idx in completed_chunks:
                print(f"⏭️  Skipping completed chunk {chunk_idx}")
                continue

            chunk_texts_lookup: Dict[str, List[Dict[str, Any]]] = {}

            # Initialize original texts with ALL metadata (filter out NaN texts)
            original_texts = []
            nan_count = 0
            for idx, row in chunk_df.iterrows():
                text_value = row[text_column]
                # Skip rows with NaN or empty text
                if pd.isna(text_value) or (
                    isinstance(text_value, str) and not text_value.strip()
                ):
                    nan_count += 1
                    continue

                original_texts.append(
                    {
                        "text_id": global_text_id,
                        "source_id": idx,
                        "text": str(text_value),  # Ensure string type
                        "round": "original",
                        "augmentation_chain": [],
                        "depth": 0,
                        "parent_text_id": None,  # Ensure column exists for alignment
                        **self._extract_metadata(row, text_column),  # All metadata
                    }
                )
                global_text_id += 1

            if nan_count > 0:
                print(f"   ⚠️  Skipped {nan_count} rows with NaN/empty text")

            chunk_texts_lookup["original"] = original_texts

            # Save original texts to live file
            if self.checkpoint_mgr:
                self.checkpoint_mgr.append_batch_to_live(
                    original_texts, force_flush=True, column_order=column_order
                )

            # Process each round
            for round_config in rounds:
                round_name = round_config["name"]
                apply_to = round_config.get("apply_to", ["original"])

                # Skip already completed rounds in current chunk
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
                ):
                    for result in batch_results:
                        result["text_id"] = global_text_id
                        global_text_id += 1

                    round_outputs.extend(batch_results)
                    batch_in_round += 1

                    # Update checkpoint after each batch within a round
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

                # Flush any pending rows after each round
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

            # Combine all chunk outputs and write incrementally
            chunk_output = []
            for round_name, texts in chunk_texts_lookup.items():
                chunk_output.extend(texts)

            if chunk_output:
                chunk_df_out = pd.DataFrame(chunk_output)

                # Ensure consistent column ordering
                pipeline_cols = [
                    "text_id",
                    "source_id",
                    "text",
                    "round",
                    "augmentation_chain",
                    "depth",
                    "parent_text_id",
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

                # Also save to checkpoint
                if self.checkpoint_mgr:
                    self.checkpoint_mgr.append_output(
                        chunk_df_out,
                        first_write=(chunk_idx == 0 and chunk_idx not in completed_chunks),
                    )

                total_output_count += len(chunk_output)

            # Mark chunk as completed
            completed_chunks.append(chunk_idx)
            completed_rounds_in_chunk = []  # Reset for next chunk

            # Update checkpoint
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

            # Update global lookup for cross-chunk references
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

        # Clear checkpoint on successful completion
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
        """Standard pipeline entry point with automatic chunking for large datasets."""
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

        # For smaller datasets, still use chunked but with full dataset as one chunk
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
        """Get current live results for inspection while pipeline is running."""
        if self.checkpoint_mgr:
            return self.checkpoint_mgr.load_live_results()
        return None

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status including live results stats."""
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
    """Load dataframe from various formats."""
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
    """Lazy load dataframe in chunks for very large files."""
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


# Utility function for inspecting live results from outside the pipeline
def inspect_live_results(checkpoint_dir: str) -> Optional[pd.DataFrame]:
    """
    Load and inspect live results from a running or interrupted pipeline.

    Args:
        checkpoint_dir: Path to the checkpoint directory

    Returns:
        DataFrame with current live results, or None if no results exist
    """
    live_file = Path(checkpoint_dir) / "live_results.csv"
    if live_file.exists():
        return pd.read_csv(live_file)
    return None


def get_checkpoint_status(checkpoint_dir: str) -> Dict[str, Any]:
    """
    Get status of a checkpoint without loading full data.

    Args:
        checkpoint_dir: Path to the checkpoint directory

    Returns:
        Dictionary with checkpoint status information
    """
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
