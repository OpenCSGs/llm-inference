# build from Dockerfile-base
FROM sean/llmray:base-0.0.2

RUN sudo apt-get update && sudo apt-get install -y vim && sudo apt-get clean

COPY "./dist" "/home/ray/dist"

RUN cd /home/ray/dist && pip install "$(ls *.whl | head -n1)[backend, frontend]"

# The build context should be the root of the repo
# So this gives the model definitions
COPY "./models" "/home/ray/models"

ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN echo "Testing llmserve install" && python -c "import llmserve.backend"

RUN pip cache purge && conda clean -a && rm -rf ~/.cache
