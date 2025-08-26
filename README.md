# UPPAX

uv run --extra gpu --python python3.11 --with-requirements evaluation/gp/requirements.txt evaluation/gp/run_scale_vi.py sequential pmap 10 10 1000 -host_device_count 10