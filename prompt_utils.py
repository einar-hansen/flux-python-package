import random

def generate_prompt_variant(base_prompt: str, config: dict) -> str:
    modifiers = [
        "in the style of {}",
        "with a {} color palette",
        "during {} time of day",
        "in a {} setting",
        "with a {} mood",
    ]

    modifier = random.choice(modifiers)
    if "style" in modifier:
        fill = random.choice(config['styles'])
    elif "color palette" in modifier:
        fill = random.choice(config['color_palettes'])
    elif "time of day" in modifier:
        fill = random.choice(config['times_of_day'])
    elif "setting" in modifier:
        fill = random.choice(config['settings'])
    elif "mood" in modifier:
        fill = random.choice(config['moods'])

    variant = f"{base_prompt}, {modifier.format(fill)}"

    return variant
