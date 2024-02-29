def has_ray():
    try:
        import ray  # noqa: F401

        return True
    except ImportError:
        return False


def has_backend():
    try:
        import llmserve.backend  # noqa: F401

        return True
    except ImportError:
        return True


def assert_has_ray():
    assert has_ray(), (
        "This command requires ray to be installed. "
        "Please install ray with `pip install 'ray[default]'`"
    )


def assert_has_backend():
    assert has_backend(), (
        "This command requires llmserve backend to be installed. "
        "Please install backend dependencies with `pip install llmserve[backend]`. "
    )
