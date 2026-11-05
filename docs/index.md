# Laya

Multilingual, non-autoregressive System 1 decision engine: typed `choice`, `score` and `noul`
decisions over any state, in a single forward pass.

A `choice` question picks one of several options, a `score` question places the state on a
scale, and a `noul` question gives the probability that the answer is yes. Laya runs on your own
hardware, reads 100+ languages, and its `Router` picks the checkpoint for each request. A state can
carry an image as well as text, and a checkpoint trained with `--vision` answers from the pixels —
on a public benchmark it beats both a linear probe and its own tower's zero-shot head, and passes
the two gates that ask whether the image is doing the work and whether reordering the options
changes the answer. A state can also carry a **sound**: a checkpoint trained with `--audio` answers
from the recording — a dog bark, a raised voice, a failing engine — through the same block, layout
and gates the picture uses. Measured on public data: **75.5 %** on 10-way FSDD with the recording
removed dropping the answer to chance (gate 1 +0.655), and a frozen linear probe at 52.65 % on
50-way ESC-50 — see [the results](audio-benchmark-results.md).

## Start here

Install Laya and make a first decision with the README's
[Installation](https://github.com/thread3d/laya-words-sounds-vision#installation) and
[Quickstart](https://github.com/thread3d/laya-words-sounds-vision#quickstart). They stay in the README so there
is one copy to keep current.

2.0 adds the third modality and reorganises the layout the media share, without changing a text or
vision call site. If you are upgrading, [Upgrading to 2.0](upgrading.md) lists what moved; the
release history is in [CHANGELOG.md](https://github.com/thread3d/laya-words-sounds-vision/blob/main/CHANGELOG.md).

## Find a guide

| To | Read |
|---|---|
| get typed values back from a JSON schema or a pydantic model | [Schema-driven decisions](structured.md) |
| log, redact, cache or gate every decision without forking Laya | [Prediction hooks](hooks/index.md) |
| adopt Laya incrementally without granting execution permission | [Staged adoption](staged-adoption.md) |
| route, screen or triage inside a LangChain or LangGraph app | [LangChain & LangGraph](langchain.md) |
| run Laya from a terminal or connect its built-in MCP stdio server | [Command line and MCP server](cli-mcp.md) |
| hand each request to the checkpoint that fits its language and task | [Routing](routing.md) |
| call Laya over HTTP from a client written against the Jev wire protocol | [HTTP API](http-api.md) |
| run the SDK or the `laya-serve` HTTP API in a container, on CPU or an NVIDIA GPU | [Docker quickstart](docker.md) |
| build for ARM64 hosts or DGX Spark | [ARM64 and DGX Spark containers](docker-platforms.md) |
| specialise a checkpoint for your own decisions | [Browser-agent fine-tuning example](finetune_browser_agent.md) and the [fine-tuning notebook](https://github.com/thread3d/laya-words-sounds-vision/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb) |
| answer questions about a picture, or teach a checkpoint to look at one | [Vision fine-tuning](finetune.md#vision-teaching-a-checkpoint-to-look-at-a-picture) and [the measured results](vision-benchmark-results.md) |
| answer questions about a recording, or teach a checkpoint to listen to one | [Sounds](audio-decision-system.md) and [sound fine-tuning](finetune.md#sounds-teaching-a-checkpoint-to-listen) |
| judge benchmark results and deployment limits | [Benchmarks and known limits](benchmarks.md) |
| look up a class, function or parameter | [Python API reference](reference/index.md) |
| score a labelled dataset, compare to a baseline, or gate a build on it | [Evaluation harness](evals.md) |

Confidence gating is in the
[README](https://github.com/thread3d/laya-words-sounds-vision#automated-confidence-gating) for now.
