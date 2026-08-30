def build_window_spans(
    total_steps: int,
    window_size: int | None,
    window_stride: int | None,
) -> list[tuple[int, int]]:
    if total_steps < 1:
        return []
    if window_size is None:
        return [(0, total_steps)]
    if window_size < 1:
        raise ValueError("window_size must be positive")

    stride = window_size if window_stride is None else window_stride
    if stride < 1:
        raise ValueError("window_stride must be positive")
    if stride > window_size:
        raise ValueError("window_stride cannot exceed window_size")
    if total_steps <= window_size:
        return [(0, total_steps)]

    spans = [
        (start, start + window_size)
        for start in range(0, total_steps - window_size + 1, stride)
    ]
    final_span = (total_steps - window_size, total_steps)
    if spans[-1] != final_span:
        spans.append(final_span)
    return spans
