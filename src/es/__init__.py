# es/ -- distributed OpenAI-ES harness (coordinator + stateless workers).
#
# Layering (strict, so a worker box needs nothing but numpy + stable-retro):
#   policy.py    pure-numpy MLP policy, no torch, no env imports
#   openes.py    the ES math: mirrored sampling from integer seeds, centered
#                ranks, Adam-on-the-mean, checkpointing
#   protocol.py  wire contract + the pure ChunkQueue leasing logic
#   coordinator.py / worker.py  the two runnable ends (stdlib HTTP only)
#
# Nothing in this package may import torch or stable_baselines3: workers run
# on machines that only have the requirements-es.txt deps installed.
