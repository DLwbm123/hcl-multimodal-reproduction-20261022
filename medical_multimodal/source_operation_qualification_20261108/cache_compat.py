"""Explicit compatibility adapter for the pinned CheXagent visual prefill.

Only an empty cache on a prefill containing image tokens is normalized to None.
Nonempty decode caches are preserved. This is shared correctness infrastructure,
not learned experience, and does not edit the original model assets.
"""
from contextlib import contextmanager


@contextmanager
def inspect_prefill(model, normalize_empty_cache=False):
    core = model.model
    original_forward, original_encode = core.forward, core.visual.encode
    record = {"first_forward": None, "visual_calls": 0, "visual_image_counts": [],
              "normalized_empty_caches": 0, "visual_shapes": [], "decode_steps": 0}

    def encode(paths, *args, **kwargs):
        record["visual_calls"] += 1
        record["visual_image_counts"].append(len(paths))
        result = original_encode(paths, *args, **kwargs)
        record["visual_shapes"].append(list(result.shape))
        return result

    def forward(*args, **kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        cache = kwargs.get("past_key_values")
        length = int(cache.get_seq_length()) if hasattr(cache, "get_seq_length") else None
        image_tokens = int((ids == core.tokenizer.img_start_id).sum().item()) if ids is not None else 0
        if record["first_forward"] is None:
            record["first_forward"] = {"cache_type": type(cache).__name__, "cache_length": length,
                                       "input_tokens": int(ids.shape[-1]) if ids is not None else None,
                                       "image_start_tokens": image_tokens}
        else:
            record["decode_steps"] += 1
        if normalize_empty_cache and cache is not None and length == 0 and image_tokens:
            kwargs["past_key_values"] = None
            record["normalized_empty_caches"] += 1
        return original_forward(*args, **kwargs)

    core.forward, core.visual.encode = forward, encode
    try:
        yield record
    finally:
        core.forward, core.visual.encode = original_forward, original_encode

