import math

def round_by_factor(number: float, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: float, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: float, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int, max_ratio: int = 200) -> tuple[int, int]:
    if max(height, width) / min(height, width) > max_ratio:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {max_ratio}, "
            f"got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def smart_resize_2(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    # Calculate the target number of pixels
    target_pixels = (min_pixels + max_pixels) // 2
    
    # Calculate initial dimensions
    original_aspect_ratio = width / height
    initial_width = math.sqrt(target_pixels * original_aspect_ratio)
    initial_height = target_pixels / initial_width
    
    # Round to nearest multiple of factor
    width = round(initial_width / factor) * factor
    height = round(initial_height / factor) * factor
    
    # Adjust dimensions to meet pixel range requirement
    while width * height < min_pixels:
        width += factor
        height = round(width / original_aspect_ratio / factor) * factor
    
    while width * height > max_pixels:
        width -= factor
        height = round(width / original_aspect_ratio / factor) * factor

    return (width, height)