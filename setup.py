from setuptools import find_packages, setup

with open("./requirements/requirements.txt") as f:
    required = f.read().splitlines()

with open("./requirements/requirements-backend.txt") as f:
    required_backend = f.read().splitlines()

with open("./requirements/requirements-vllm.txt") as f:
    required_vllm = f.read().splitlines()

with open("./requirements/requirements-frontend.txt") as f:
    required_frontend = f.read().splitlines()

with open("./requirements/requirements-dev.txt") as f:
    required_dev = f.read().splitlines()

with open("./requirements/requirements-docs.txt") as f:
    required_docs = f.read().splitlines()

setup(
    name="llm-serve",
    version="0.0.1",
    description="An LLM inference solution to quickly deploy productive LLM service",
    packages=find_packages(include="llmserve*"),
    include_package_data=True,
    package_data={"llmserve": ["models/*"]},
    entry_points={
        "console_scripts": [
            "llm-serve=llmserve.api.cli:app",
        ]
    },
    install_requires=required,
    extras_require={
        "backend": required_backend,
        "vllm": required_vllm,
        "frontend": required_frontend,
        "dev": required_dev,
        "test": required_dev,
        "docs": required_docs,
    },
    dependency_links=["https://download.pytorch.org/whl/cu118"],
    python_requires=">=3.10",
)
