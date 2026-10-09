def build_encoder(name: str, device: str = "cuda"):
    name = name.lower()
    if name == "mert-v0":
        from .mert import MERTEncoder
        return MERTEncoder("m-a-p/MERT-v0", device=device)
    if name == "mert-v0-public":
        from .mert import MERTEncoder
        return MERTEncoder("m-a-p/MERT-v0-public", device=device)
    if name == "music2vec-v1":
        from .music2vec import Music2VecEncoder
        return Music2VecEncoder("m-a-p/music2vec-v1", device=device)
    if name == "dac-16k":
        from .dac import DACEncoder
        return DACEncoder("16khz", device=device)
    raise ValueError(f"unknown encoder: {name}")
