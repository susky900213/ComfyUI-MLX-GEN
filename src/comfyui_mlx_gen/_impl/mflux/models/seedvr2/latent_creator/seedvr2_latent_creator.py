import mlx.core as mx


class SeedVR2LatentCreator:
    @staticmethod
    def create_noise_latents(
        seed: int,
        height: int,
        width: int,
        num_frames: int = 1,
        batch_size: int = 1,
        latent_channels: int = 16,
    ) -> mx.array:
        return mx.random.normal(
            shape=(batch_size, latent_channels, num_frames, height, width),
            key=mx.random.key(seed),
        )

    @staticmethod
    def create_condition(encoded_latent: mx.array) -> mx.array:
        # Ensure we have 5D (B, C, T, H, W)
        if encoded_latent.ndim == 4:
            encoded_latent = encoded_latent[:, :, None, :, :]

        batch_size = encoded_latent.shape[0]
        num_frames = encoded_latent.shape[2]
        height = encoded_latent.shape[3]
        width = encoded_latent.shape[4]
        mask = mx.ones((batch_size, 1, num_frames, height, width))
        condition_with_mask = mx.concatenate([encoded_latent, mask], axis=1)
        return condition_with_mask

    @staticmethod
    def unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        return latents
