FROM python:3.13

RUN apt-get update;

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel=1.9

COPY . .

RUN /root/.local/bin/uv sync --frozen -p python3.13 --extra=cpu

RUN /usr/local/bin/python3 -m pip install matplotlib

RUN /root/.juliaup/bin/julia --project=evaluation/gmm/gen -e "import Pkg; Pkg.instantiate()"

RUN /root/.juliaup/bin/julia --project=evaluation/gp/autogp -e "import Pkg; Pkg.instantiate()"

RUN apt-get install -y g++ cmake libopenblas-dev liblapack-dev libarmadillo-dev

RUN make compile -C evaluation/urn/milch/swift/

ENTRYPOINT [ "bash" ]