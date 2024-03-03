from setuptools import find_packages, setup

setup(
    name="llm-serve",
    version="0.0.1",
    description="An LLM inference solution to quickly deploy productive LLM service",
    author="llm-serve authors",
    author_email="jasonhe258@163.com",
    packages=find_packages(include="llmserve*"),
    include_package_data=True,
    package_data={"llmserve": ["models/*"]},
    entry_points={
        "console_scripts": [
            "llm-serve=llmserve.api.cli:app",
        ]
    },
    install_requires=["typer>=0.9", "rich"],
    extras_require={
        "backend": [
            "async_timeout",
            "datasets",
            "ftfy",
            "tensorboard",
            "sentencepiece",
            "Jinja2",
            "numexpr>=2.7.3",
            "hf_transfer",
            "evaluate",
            "bitsandbytes",
            "numpy<1.24",
            "ninja",
            "protobuf<3.21.0",
            "optimum",
            "safetensors",
            "pydantic==1.10.7",
            "einops",
            "markdown-it-py[plugins]",
            "scipy==1.11.1",
            "jieba==0.42.1",
            "rouge_chinese==1.0.3",
            "nltk==3.8.1",
            "sqlalchemy==1.4.41",
            "typing-extensions==4.5.0",
            "linkify-it-py==2.0.2",
            "markdown-it-py==2.2.0",
            "gradio",
            "httpx[socks]==0.23.3",
            "torch==2.1.2",
            "torchaudio==2.1.2",
            "torchvision==0.16.2",
            "accelerate==0.25.0",
            "deepspeed==0.12.6",
            "torchmetrics==1.2.1",
            "llama_cpp_python==0.2.20",
            "transformers==4.33.3",
        ],
        "vllm": [
            "vllm==0.2.7",
        ],
        "frontend": [
            "gradio",
            "aiorwlock",
            "pymongo",
            "pandas",
            "boto3",
        ],
        "dev": [
            "pre-commit",
            "ruff==0.0.270",
            "black==23.3.0",
        ],
        "test": [
            "pytest",
        ],
        "docs": [
            "mkdocs-material",
        ],
    },
    dependency_links=["https://download.pytorch.org/whl/cu118"],
    python_requires=">=3.10",
)






