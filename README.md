# Open-Jev

**Open probability decisions with Qwen3.5-2B, Qwen3.5-9B and Qwen3.8-27B.**
Supply a context, questions and candidates; get typed probabilities directly,
without autoregressive answer generation or parsing generated JSON.

**Project website:** [zefan-cai.github.io/open-jev](https://zefan-cai.github.io/open-jev/) — demos, measured results and project background.

**Hugging Face:** [Collection](https://huggingface.co/collections/ZefanCai/open-jev) ·
[2B model](https://huggingface.co/ZefanCai/Open-Jev-2B) ·
[9B model](https://huggingface.co/ZefanCai/Open-Jev-9B) ·
[Dataset](https://huggingface.co/datasets/ZefanCai/Open-Jev)

**Try it online:** [Text & CSV workbench](https://zefan-cai.github.io/open-jev/workbench/) · [Open in Hugging Face](https://huggingface.co/spaces/ZefanCai/Open-Jev-Workbench)

**No-code workbench:** [User guide](docs/get-started.md) · [Workbench source](examples/workbench/) · [CPU deployment](deploy/huggingface-space/). Paste messages or import a CSV, define your categories, and download the original table with suggested labels and candidate probabilities. The hosted workbench uses the released Open-Jev-2B model on CPU. Start with a few rows: this trial is slower than the GPU deployment measured below. Its optional walkthrough is explicitly marked as an illustrative example.

**Benchmarks:** [Results and scope](docs/benchmarks.md) ·
[Website tables](https://zefan-cai.github.io/open-jev/benchmarks/) ·
[Source code](https://github.com/Zefan-Cai/Open-Jev)

**Public releases:** the dataset and completed 2B/9B checkpoints are available
at the links above. The 2B/9B artifacts are LoRA adapters plus a trained scalar decision
head and calibration temperature. They require the pinned upstream Qwen
weights and the Open-Jev loader; they are not merged base models or ordinary
text-generation checkpoints. The dedicated public JevBench evaluation is
complete. The new 27B training iteration is complete; its full internal
evaluation and independent audit are still in progress. New model results
will be published after that audit.

Open-Jev is an independent implementation inspired by TypeSafe's Jev. It does
not reproduce proprietary RLCD, private weights or training data, and does not
claim TypeSafe's advertised speedups or parity with every community demo.

Current results: [methods and quality](docs/provider-comparison.md) ·
[interactive comparison](https://zefan-cai.github.io/open-jev/#comparison) ·
[published X thread](https://x.com/Zefan_Cai/status/2101845509170417784) ·
[community cases and benchmark plan](docs/community-research-20260921.md).

**New external evaluation:** the [audited JevBench public-subset report](docs/jevbench-public.md)
compares the released 2B/9B baselines, Jev, GPT-5.6 Luna and GPT-6 Astra on all
231 available public tasks. It includes every decision, per-tier results,
probability validation and timing diagnostics, with candidate-order and
hardware/network limitations. See the [dedicated website table](https://zefan-cai.github.io/open-jev/#jevbench).

**Broader data iteration:** [new community cases](docs/community-research-broadening-20260921.md), [the frozen 148,639-row training mixture](docs/community-hard-training-v2.md), and the [Chinese project story](https://zefan-cai.github.io/open-jev/story/) document the sources, construction, controls and limitations. New model gains have not been measured.

**Consumer GPUs:** [running the 9B in bf16 on one 16 GB card](docs/consumer-gpu.md), with JevBench results on an RTX 4060 Ti, an RTX 5060 Ti and an H100, and the opt-in options that make the cached path faster.

**V3 data prepared:** [natural intent routing](docs/community-routing-v3.md),
[executable SQL](docs/sql-semantics-v3.md) and [approval/CMS controls](docs/community-workflow-v3.md)
add 129,288 decision rows, including 74,921 training rows. The frozen next-stage
mixture combines those training rows with 21,928 earlier replay rows for 96,849
training rows. The data are staged and tokenizer-checked; the next stage is
queued behind completion and evaluation of the current 27B run.
The [training recipe](docs/community-hard-training-v3.md) and
[internal evaluation guide](docs/internal-evaluation.md) explain held-out splits,
the fixed 1,280-row / 840-group comparison panel and equal-group reporting.
These are preparation and audit results; the released-model scores below are unchanged.

## Inference latency

The [latency report](docs/inference-latency.md) measures the released 2B
checkpoint with warmed in-process Predictor calls and real loopback HTTP
requests. It records P50/P95, context and candidate counts, hardware, cache
mode, all warmup/timed attempts, and cache-output parity. The
[website latency table and evidence video](https://zefan-cai.github.io/open-jev/#latency)
use those saved measurements; video playback duration is not inference time.

The same 11 saved workloads were measured against **Jev-1.13.0** from the same
client: 220 measured HTTPS requests and 33 warmups, with no request errors.
For customer service, median local Open-Jev HTTP latency is **85.03 ms** versus
**295.26 ms** for Jev HTTPS. At 1024 state tokens and 32 candidates, Open-Jev is
slower: **1015.90 ms** versus **301.37 ms**. Hardware and network paths differ;
this is observed deployment latency, not matched-hardware speedup. CUDA prefix
caching exceeded the probability tolerance on 9/11 workloads; all selected
decisions matched, and caching remains off by default. Full results, including unfavorable cases, are in the
[public evidence](reports/inference-latency/public).

The [provider comparison](docs/provider-comparison.md) also evaluates OpenAI
structured decisions on identical saved requests and maintains a separate
all-domain quality suite. Latency does not establish equal task quality.
For the same customer-service request, OpenAI Luna and Astra have P50 response
times of **918.13 ms** and **1938.39 ms**, respectively, with their recorded
reasoning settings and structured categorical outputs.

## Install and run

Use Python 3.10 or newer. Core task contracts, data generators and CPU tests
need only the source package. Model inference and training share the `train`
extra; a suitable GPU and the upstream model weights are required for the
published checkpoint workflow. The full GPU workflow targets Linux.

```bash
git clone https://github.com/Zefan-Cai/Open-Jev.git
cd Open-Jev
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v

# Add the pinned model runtime dependencies.
python -m pip install -e '.[train]'
hf download ZefanCai/Open-Jev-2B \
  --revision 0c7aa498b1627be8da4acf34c863ff0ee0a92785 --local-dir models/Open-Jev-2B
python -m jev.server --checkpoint models/Open-Jev-2B/package/checkpoint \
  --device cuda:0 --max-length 4096 --batch-size 1 --no-prefix-cache
```

The download is pinned to the published 2B revision. The 9B package uses the
same layout at revision `47e966881e489511c0c7f5633a9e1960a676a551`. The public
dataset revision is `c67699e13d0ae25e35b77165a4b6b079bedc8aba`; the original release remains reproducible at `341d9338462da1cf56ba57519fb0f3f5258b825f`.
Open **http://127.0.0.1:8791** for the text/CSV classification workbench,
**http://127.0.0.1:8791/examples/index.html** for the developer task lab, or
**http://127.0.0.1:8791/examples/painting/index.html** for probability painting.
Run from the checkout to serve the example UI. No live business action is
performed by the server.

To start directly from the base model, without an Open-Jev adapter:

```bash
python -m jev.server --model Qwen/Qwen3.5-2B \
  --revision 15852e8c16360a2fea060d615a32b45270f8a8fc --max-length 4096
```

This base-model initialization is distinct from the trained checkpoint above.
Optional extras are `.[phone]` for phone-number controls and `.[doom]` for
ViZDoom. Video rendering additionally needs Pillow, ffmpeg and ffprobe.

## Project videos

The [introduction video](release/social/open-jev-introduction.mp4) introduces
the project, and the [successful-demo video](release/social/open-jev-demos.mp4)
shows selected examples. Captions, transcripts and source evidence accompany
them in [release/social](release/social). The demo was published in [the launch quote](https://x.com/Zefan_Cai/status/2101782158658695388); the [introduction reply](https://x.com/Zefan_Cai/status/2101786019607740436) and [website reply](https://x.com/Zefan_Cai/status/2101789698947793231) accompany it. Their renderer is
[`scripts/render_launch_videos.py`](scripts/render_launch_videos.py).

## Typed decisions

With the server running, use the Python client from another terminal:

```python
from jev.client import Client

result = Client().ask(
    state="My order arrived damaged. Please refund it.",
    questions={
        "refund_requested": {
            "type": "noul", "instructions": "Is a refund explicitly requested?"
        },
        "route": {
            "type": "choice", "instructions": "Which team should handle this?",
            "criteria": {"billing": "Refunds and charges", "engineering": "Software defects"}
        },
        "frustration": {
            "type": "score", "instructions": "Rate expressed frustration.",
            "criteria": ["Calm", "Frustrated but civil", "Very angry"]
        }
    }
)
print(result["answers"])
```

Choice returns probabilities over supplied candidates and the highest-scoring
candidate. Noul returns a yes/no probability. Score returns probabilities over
ordered criteria and their expected index. A scalar head scores candidate
sequences; one calibrated probability distribution is assembled per question.
The loader rejects overlength inputs instead of silently truncating them.

[Prefix caching](docs/prefix-caching.md) is optional and off by default. It
reuses exact shared request/question token prefixes while preserving independent
attention, convolution and recurrent states for candidate branches. Tiny hybrid
CPU/LoRA checks pass. In the released 2B BF16/CUDA benchmark, probability
tolerance failed on 9/11 workloads while all 440 paired selected decisions
matched. Prefix caching remains experimental and off by default. See the full
latency evidence above; full 9B/27B comparisons remain pending.

## What is included

| Area | Runnable components | Scope |
| --- | --- | --- |
| API | Choice, Noul, Score, dynamic candidates, batching, calibration, Python client | Partial hosted-API compatibility |
| Workflows | Customer service, security incidents, agent traces, invoices | Original synthetic policies and offline action proposals |
| Games | Snake, T-Rex, tile platformer, Wiki, tic-tac-toe, optional ViZDoom | Local environments, replay, model/random/teacher evaluation |
| Painting | Palette, silhouette, binary RGB and HSL probability representations | Browser interface and generated geometry controls |
| Recipes | Search/ranking, RAG, guardrails, spans/dates, functions, skills, hierarchy, verification | Request builders and postprocessors; external execution where documented |
| Community adapters | Browser DOM, RuneScape, Pokémon, HEIST, drone, 4/4/255/28-field stress forms | State/action interfaces; external environments are not bundled |
| Extraction controls | Citation, entity alignment, amount, email and phone | Generated data and evaluation software; no trained quality claim on these newer corpora |

See [capability evidence](docs/public-capabilities.md),
[workflows](docs/workflows.md), [games](docs/games.md),
[painting](docs/painting.md), [recipes](docs/recipes.md) and
[community adapters](docs/community.md). The website selects reviewed successful
examples and explicit interface walkthroughs. It is an outcome-selected
showcase, not a representative success-rate estimate. Original failures remain
in the scientific reports.


The latest three original community corpora add **30,234 audited typed rows**
for [context retention](docs/context-retention-data.md),
[seven-way transcript categorization](docs/sponsor-segment-data.md) and
[silent API failure detection](docs/silent-failure-data.md). They are published
as three additional HF configs, with all earlier payloads preserved. These
corpora have not been used to retrain the released adapters. The broader
prepared inventory is **408,884 rows / 25 source identifiers**, including
268,493 training and 107,922 test/OOD rows; it includes the original Wiki
records omitted from the public redistribution.

[Graded retrieval controls](docs/ir-control-data.md) add 11,600 rows and six
executable reranking methods. [Multilingual mailroom controls](docs/mailroom-control-data.md)
add 114,800 rows with 11 question heads in English, Chinese and Turkish.
These are finite original controls. They have not been used to retrain the
released models, and do not establish TREC performance or production email
automation accuracy.

The [live provider comparison](https://zefan-cai.github.io/open-jev/#comparison)
keeps common cases and pending evaluations explicit. On the same 76 hard
coverage decisions, released 2B/9B have 65/72 correct, Jev has 66, GPT-5.6
Luna has 60 and GPT-6 Astra has 71. The released-model predictions are reused only after exact
row/checkpoint/hash validation. In the broader 140-hard-case coverage pass,
Jev scores 117, Luna 109 and Astra 135; on JF100's 300 option rotations, Jev and
Luna score 232 and 227 respectively, while Astra scores 300/300. All five API
quality suites are complete. Jev/Luna/Astra mailroom reference matches are
908/921, 900/921 and 913/921; all 1,616 OpenAI quality requests completed with
zero request errors. [Full results and IR ranking replay](docs/provider-comparison.md)
retain per-suite scope and label limitations. New Open-Jev GPU cases and the
full held-out inventory remain pending.

The separate [TREC-DL evaluation](reports/ir-control-v1/trec-holdout/README.md)
has completed Jev's 97 queries / 873 requests. DL19/DL20 nDCG@10 is
0.275836/0.190667 under strict validation and 0.728218/0.715734 in the
predeclared supplementary scalar analysis. Probability-mass failures affect
66 queries and remain zero in the strict metric. Luna also completed all 97
queries with strict nDCG@10 of 0.729911/0.702082, zero request errors and an
independently checked $0.799273 usage estimate. Astra also completed all 97
queries with strict nDCG@10 of 0.736610/0.714484, zero request errors and a
$39.62622 usage estimate. Open-Jev TREC results remain pending.

The frozen counts are reference matches. A subsequent game-label audit found
equivalent platformer actions and omitted ViZDoom policy constants; these
limitations are recorded in the provider-comparison method and must not be
interpreted as game-performance differences. Original scores remain preserved.
Applying the same six technical exclusions to the common slice gives 2B 60/70,
9B 67/70, Jev 64/70, Luna 57/70 and Astra 69/70. This is a post-hoc sensitivity
analysis, not a replacement benchmark or a population-level model ranking.

## Training data and measured results

Both completed models consumed **80,816 training rows** in one pass over the
original frozen `release-v2` training split: 20,204 optimizer steps, global
batch 4, rank-8 LoRA and a jointly trained decision head. Temperature was fitted
only on 512 calibration rows. Saved checkpoint reload checks passed.

The public `release-v2-redistributable` training projection contains **79,116
rows**. It excludes 1,700 original `wikispeedia-v1` training records because
redistribution permission for that archive has not been confirmed. It is not
byte-identical to the dataset used for the published adapters. Original
split hashes remain recorded; no scientific metric is relabeled as a result
on the reduced public projection.

The [independent complete evaluation](reports/full-data-eval-n1-v1/README.md)
covers 10,532 test and 15,920 OOD rows per model with zero missing, duplicate or
failed predictions. It uses the original dataset, including Wiki records.

| Model | Test hard correct / hard rows | OOD hard correct / hard rows |
| --- | ---: | ---: |
| 2B | 9,515 / 10,046 (94.71%) | 13,287 / 15,446 (86.02%) |
| 9B | 9,799 / 10,046 (97.54%) | 14,205 / 15,446 (91.97%) |

Hard accuracy excludes soft-target rows; the audit retains probability metrics
and every subgroup. No full-data baseline was run, so these figures do not
measure full-data training gain. Separate [2B](reports/fullpass-2b-n1/README.md)
and [9B](reports/fullpass-9b-n1/README.md) audits compare base/trained models on
512 test and 512 OOD selections. These synthetic decision results do not
establish real-world workflow completion, game wins or autonomous control.

The larger prepared inventory includes browser/drone and later extraction
controls. Completed 2B/9B training did not use the newer expansion or extraction
corpora. Final-model JF100 and closed-loop task results remain pending. Older
[pilot evaluations](reports/pilot-suite-n1-4k.md) retain failures and identify
their different checkpoints.

For data generation, training and evaluation, use the documented task-specific
commands in [training](docs/training.md), [data](docs/data.md),
[multidomain training](docs/multidomain-training.md),
[model evaluation](docs/model-evaluation.md) and
[checkpoint packaging](docs/checkpoint-package.md). Training records the Git
revision, so use a Git checkout. Historical cluster capture/supervisor scripts
are preserved for audit; their recorded host/process plans are not portable
launch configurations. Active private cluster policy and monitoring logs are
not part of this source release.

## License and provenance

Original source code is [MIT](LICENSE). The trained LoRA adapters and decision
heads are Apache-2.0 under their published model cards and pinned Qwen terms.
Original generated controls carry CC0-1.0 where marked; third-party data and
optional engines retain their own terms. See
[third-party notices](THIRD_PARTY_NOTICES.md) and
[model provenance](docs/model-provenance.md).

No base-model weights, credentials, private official Jev cases, commercial ROMs
or development Git history are included. This public repository is a reviewed
source snapshot; `PUBLIC_RELEASE.json` records its source commit, file
inventory and publication transformations.
