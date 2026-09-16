import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_text_encoder.qwen_encoder_layer import QwenEncoderLayer
from mflux.models.qwen.model.qwen_text_encoder.qwen_rms_norm import QwenRMSNorm
from mflux.models.qwen.model.qwen_text_encoder.qwen_rope import QwenRotaryEmbedding


class QwenEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 152064,
        hidden_size: int = 3584,
        num_hidden_layers: int = 28,
        max_position_embeddings: int = 128000,
        rope_theta: float = 1000000.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.image_token_id = 151655

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [QwenEncoderLayer() for i in range(num_hidden_layers)]
        self.norm = QwenRMSNorm(hidden_size, eps=1e-6)
        self.rotary_emb = QwenRotaryEmbedding(
            dim=hidden_size // 28,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            rope_type="default",
        )

        self.visual = None

    def get_image_features(self, pixel_values: mx.array, image_grid_thw: mx.array) -> mx.array:
        if self.visual is None:
            raise RuntimeError("Vision transformer not initialized. Call load_visual_weights() first.")

        pixel_values = pixel_values.astype(mx.float32)
        image_embeds = self.visual(pixel_values, image_grid_thw)
        original_split_sizes = image_grid_thw.prod(axis=-1).astype(mx.int32)
        split_sizes = (original_split_sizes // 4).astype(mx.int32)
        split_sizes = [int(s) for s in split_sizes.tolist()]
        split_sizes = [s for s in split_sizes if s > 0]
        image_embeds_split = []
        start_idx = 0
        for split_size in split_sizes:
            end_idx = start_idx + split_size
            image_embeds_split.append(image_embeds[start_idx:end_idx])
            start_idx = end_idx
        return image_embeds_split

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
    ) -> mx.array:
        batch_size, seq_len = input_ids.shape
        inputs_embeds = self.embed_tokens(input_ids)

        if pixel_values is not None and image_grid_thw is not None:
            image_embeds_split = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = mx.concatenate(image_embeds_split, axis=0)

            image_positions = input_ids == self.image_token_id
            n_image_tokens = int(mx.sum(image_positions).item())

            if n_image_tokens != image_embeds.shape[0]:
                raise ValueError(
                    "Qwen image features and image tokens do not match: "
                    f"{image_embeds.shape[0]} features for {n_image_tokens} tokens."
                )
            if n_image_tokens > 0:
                image_positions_flat = image_positions.flatten()
                inputs_embeds_flat = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
                image_offsets = mx.cumsum(image_positions_flat.astype(mx.int32), axis=0) - 1
                safe_offsets = mx.maximum(image_offsets, mx.zeros_like(image_offsets))
                replacement_embeds = image_embeds[safe_offsets]
                inputs_embeds_flat = mx.where(
                    mx.expand_dims(image_positions_flat, axis=-1),
                    replacement_embeds,
                    inputs_embeds_flat,
                )
                inputs_embeds = inputs_embeds_flat.reshape(inputs_embeds.shape)
        position_ids = self._compute_position_ids(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            batch_size=batch_size,
            seq_len=seq_len,
        )
        padding_mask = mx.where(
            attention_mask == 1,
            mx.zeros_like(attention_mask).astype(mx.float32),
            mx.ones_like(attention_mask).astype(mx.float32) * (-float("inf")),
        )
        padding_mask = mx.expand_dims(mx.expand_dims(padding_mask, axis=1), axis=1)

        idx = mx.arange(seq_len, dtype=mx.int32)
        j = mx.expand_dims(idx, axis=0)
        i = mx.expand_dims(idx, axis=1)
        tri_bool = j > i
        zeros_2d = mx.zeros((seq_len, seq_len)).astype(mx.float32)
        neginf_2d = mx.ones((seq_len, seq_len)).astype(mx.float32) * (-float("inf"))
        causal_tri_mask = mx.where(tri_bool, neginf_2d, zeros_2d)
        causal_tri_mask = mx.expand_dims(mx.expand_dims(causal_tri_mask, axis=0), axis=0)
        causal_tri_mask = mx.broadcast_to(causal_tri_mask, (batch_size, 1, seq_len, seq_len))
        attention_mask_4d = causal_tri_mask + padding_mask
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, attention_mask_4d, position_embeddings)

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def _compute_position_ids(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        image_grid_thw: mx.array | None,
        batch_size: int,
        seq_len: int,
    ) -> mx.array:
        if image_grid_thw is None or int(mx.sum(input_ids == self.image_token_id).item()) == 0:
            cache_position = mx.arange(seq_len, dtype=mx.int32)
            position_ids = mx.expand_dims(mx.expand_dims(cache_position, axis=0), axis=0)
            return mx.broadcast_to(position_ids, (3, batch_size, seq_len))

        return self._compute_multimodal_position_ids(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            spatial_merge_size=self._spatial_merge_size(),
        )

    def _compute_multimodal_position_ids(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        image_grid_thw: mx.array,
        spatial_merge_size: int,
    ) -> mx.array:
        input_rows = input_ids.tolist()
        mask_rows = attention_mask.tolist()
        image_grids = image_grid_thw.tolist()
        batch_size = len(input_rows)
        seq_len = len(input_rows[0]) if input_rows else 0
        position_rows = [[[0 for _ in range(seq_len)] for _ in range(batch_size)] for _ in range(3)]
        grid_index = 0

        for batch_idx, input_row in enumerate(input_rows):
            active_indices = [idx for idx, keep in enumerate(mask_rows[batch_idx]) if int(keep) == 1]
            token_types = [1 if int(input_row[idx]) == self.image_token_id else 0 for idx in active_indices]
            current_pos = 0
            local_rows = [[], [], []]
            start_idx = 0

            while start_idx < len(token_types):
                modality_type = token_types[start_idx]
                end_idx = start_idx + 1
                while end_idx < len(token_types) and token_types[end_idx] == modality_type:
                    end_idx += 1

                if modality_type == 0:
                    text_len = end_idx - start_idx
                    for offset in range(text_len):
                        for axis in range(3):
                            local_rows[axis].append(current_pos + offset)
                    current_pos += text_len
                else:
                    if grid_index >= len(image_grids):
                        raise ValueError("Qwen image token groups exceed image_grid_thw entries.")
                    vision_rows = self._vision_position_ids(
                        start_position=current_pos,
                        grid_thw=image_grids[grid_index],
                        spatial_merge_size=spatial_merge_size,
                    )
                    expected_len = end_idx - start_idx
                    if len(vision_rows[0]) != expected_len:
                        raise ValueError(
                            "Qwen image token group length does not match image_grid_thw: "
                            f"{expected_len} tokens for grid-derived {len(vision_rows[0])} positions."
                        )
                    for axis in range(3):
                        local_rows[axis].extend(vision_rows[axis])
                    current_pos += max(int(image_grids[grid_index][1]), int(image_grids[grid_index][2])) // spatial_merge_size
                    grid_index += 1

                start_idx = end_idx

            if any(len(row) != len(active_indices) for row in local_rows):
                raise ValueError("Qwen multimodal position construction produced an invalid sequence length.")
            for local_idx, source_idx in enumerate(active_indices):
                for axis in range(3):
                    position_rows[axis][batch_idx][source_idx] = local_rows[axis][local_idx]

        if grid_index != len(image_grids):
            raise ValueError("Qwen image_grid_thw entries exceed image token groups.")

        return mx.array(position_rows, dtype=mx.int32)

    @staticmethod
    def _vision_position_ids(
        start_position: int,
        grid_thw: list[int],
        spatial_merge_size: int,
    ) -> list[list[int]]:
        grid_t = int(grid_thw[0])
        grid_h = int(grid_thw[1]) // spatial_merge_size
        grid_w = int(grid_thw[2]) // spatial_merge_size
        if grid_t <= 0 or grid_h <= 0 or grid_w <= 0:
            raise ValueError(f"Invalid Qwen image grid for multimodal RoPE: {grid_thw}.")

        rows = [[], [], []]
        for temporal in range(grid_t):
            for height in range(grid_h):
                for width in range(grid_w):
                    rows[0].append(start_position + temporal)
                    rows[1].append(start_position + height)
                    rows[2].append(start_position + width)
        return rows

    def _spatial_merge_size(self) -> int:
        if self.visual is None:
            return 2
        return int(getattr(self.visual, "spatial_merge_size", 2))
