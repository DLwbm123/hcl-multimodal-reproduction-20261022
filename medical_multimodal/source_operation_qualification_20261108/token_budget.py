"""Conservative input admission for the pinned CheXagent interface.

This is shared tool infrastructure, not a learned policy. It does not truncate
images or silently convert a joint/temporal question into a single-image task.
"""


def check_budget(input_tokens, max_new_tokens, configured_context):
    if any(type(n) is not int or n <= 0 for n in (input_tokens, max_new_tokens, configured_context)):
        raise ValueError("Token counts and configured context must be positive integers")
    required = input_tokens + max_new_tokens
    return {
        "admissible": required <= configured_context,
        "input_tokens": input_tokens,
        "reserved_output_tokens": max_new_tokens,
        "required_tokens": required,
        "configured_context": configured_context,
        "reason": "within_declared_window" if required <= configured_context else "exceeds_declared_window",
    }


def encode_checked(tokenizer, image_paths, prompt, max_new_tokens, configured_context):
    if not image_paths:
        raise ValueError("At least one current-case image is required")
    query = tokenizer.from_list_format([*({"image": p} for p in image_paths), {"text": prompt}])
    tokens = tokenizer.apply_chat_template(
        [{"from": "system", "value": "You are a helpful assistant."},
         {"from": "human", "value": query}],
        add_generation_prompt=True, return_tensors="pt",
    )
    decision = check_budget(int(tokens.shape[1]), max_new_tokens, configured_context)
    return (tokens if decision["admissible"] else None), decision
