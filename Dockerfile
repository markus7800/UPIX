FROM python:3.13

RUN apt-get update;

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel=1.9
ENV PATH="/root/.juliaup/bin:${PATH}"
ENV JULIA_DEPOT_PATH="/root/.julia"
ENV OMP_NUM_THREADS=1

COPY pyproject.toml uv.lock ./
COPY upix ./upix
COPY evaluation/gmm/gen/Project.toml ./evaluation/gmm/gen/
COPY evaluation/gp/autogp/Project.toml ./evaluation/gp/autogp/

RUN /root/.local/bin/uv sync --frozen -p python3.13 --extra=cpu

# RUN /usr/local/bin/python3 -m pip install matplotlib

RUN julia --project=evaluation/gmm/gen -e "import Pkg; Pkg.instantiate(); Pkg.precompile()"

RUN PYTHON=./.venv/bin/python3 julia --project=evaluation/gp/autogp -e "import Pkg; Pkg.instantiate(); Pkg.precompile()"

COPY . .

# For running urn with BLOG (Milch)
# RUN apt-get install -y g++ cmake libopenblas-dev liblapack-dev libarmadillo-dev
# RUN make compile -C evaluation/urn/milch/swift/

ENTRYPOINT [ "bash" ]