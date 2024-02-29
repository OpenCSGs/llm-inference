def _replace_prefix(model: str) -> str:
    return model.replace("--", "/")

def _reverse_prefix(model: str) -> str:
    return model.replace("/", "--")
