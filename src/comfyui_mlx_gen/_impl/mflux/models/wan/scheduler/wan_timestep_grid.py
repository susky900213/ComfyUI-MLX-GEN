import numpy as np


class WanTimestepGrid:
    # Explicit denoising grids (0099): distilled Wan recipes (lightx2v
    # Wan22StepDistillScheduler) publish exact timestep lists such as
    # [1000, 750, 500, 250] instead of a step count. Grid entries are FINAL
    # (already-shifted) timesteps, so sigma follows the flow-matching identity
    # sigma = t / num_train_timesteps and flow_shift never applies - the
    # count path's shift already happened when the list was designed.

    @staticmethod
    def validate(denoising_step_list, num_train_timesteps: int) -> list[int]:
        if denoising_step_list is None or len(denoising_step_list) == 0:
            raise ValueError("denoising_step_list must contain at least one timestep.")
        validated: list[int] = []
        for entry in denoising_step_list:
            # bool is an int subclass; reject it explicitly.
            if isinstance(entry, bool) or not isinstance(entry, (int, np.integer)):
                raise ValueError(
                    f"denoising_step_list entries must be integers in [1, {num_train_timesteps}], got {entry!r}."
                )
            value = int(entry)
            if value < 1 or value > num_train_timesteps:
                raise ValueError(
                    f"denoising_step_list entries must be integers in [1, {num_train_timesteps}], got {value}."
                )
            validated.append(value)
        for previous, current in zip(validated, validated[1:]):
            if current >= previous:
                raise ValueError(
                    f"denoising_step_list must be strictly decreasing, got {validated} "
                    f"({current} does not decrease after {previous})."
                )
        return validated

    @staticmethod
    def sigmas(denoising_step_list: list[int], num_train_timesteps: int) -> np.ndarray:
        # float64 like the count path's linspace, before each scheduler casts.
        return np.asarray(denoising_step_list, dtype=np.float64) / float(num_train_timesteps)
