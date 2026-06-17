---
title: "Vision Extraction: The Local Gap Stays Small"
date: 2026-06-11
draft: false
tags: ["LLM", "SLM", "Multimodal", "Vision", "Ollama", "OCR", "Evaluation"]
categories: ["Technical"]
series: ["SLM Experiments"]
series_order: 2
summary: "I ran 30 thermal-print restaurant receipts through Gemini, GPT, and two local Gemma vision models. Cloud holds a three-point lead. The free 8B Gemma 4 E4B still hits 90.7% with 100% valid JSON — and the larger 26B Gemma is the one that quietly falls apart."
---

*Prerequisites: Familiarity with LLM prompting and JSON output. This post compares vision-capable models on a receipt-extraction task using the [CORD-v2 benchmark](https://huggingface.co/datasets/naver-clova-ix/cord-v2). Local models run through [Ollama](https://ollama.com/); cloud models go through the OpenAI and Vertex AI SDKs.*

A few weeks ago I ran a text-extraction shootout — nine models, 30 job postings, twelve fields — and found that the cloud premium had all but vanished for structured JSON extraction from prose. A free 8B local model landed within 1.2 percentage points of the best cloud option. ([Post here](/posts/slm-structured-extraction/) if you want the long version.)

That left me with an obvious follow-up. What happens if the input isn't a string of words but a *picture*? Receipts, invoices, scanned forms, screenshots of a UI someone needs you to parse — the schema-and-prompt shape is identical, but the model has to look at pixels first. Vision feels like the dimension where local SLMs ought to lose, and lose badly. Cloud providers have spent years and billions tuning their multimodal stacks; local open-weight vision models are comparatively new.

So I built the receipt version of the same benchmark: 30 real receipts, four models with native vision, the same five-field schema, same scoring loop. The headline:

**Cloud holds a ~3-point lead on vision, not a chasm.** Gemini 2.5 Flash at 93.3%, GPT-5.4-mini at 92.0%, the free local Gemma 4 E4B at 90.7%. The local 26B Gemma is the model that actually falls over here — it produces invalid JSON on one in six receipts.

The longer answer has some interesting wrinkles — counting items is hard for *everyone*, vision-token pricing changes the cost math from the text post, and the bigger-Gemma-is-worse pattern from the text run is now a louder signal — so I'll walk through it.

## The setup

For the dataset I used **CORD-v2** [[1]](#ref-1) — the Consolidated Receipt Dataset, a public benchmark of ~1,000 real restaurant receipts mostly photographed in Indonesia and Korea. CORD is the closest thing the field has to a standard receipt-OCR benchmark, and it's gnarly: small thermal-print receipts, often shot at an angle on a desk, with mixed Latin/Korean text and inconsistent use of `.` versus `,` as thousands separators. A representative example:

![CORD receipt RCP0001](receipt-example.png)
*Figure 1: One CORD receipt. The model has to identify three line items (J.STB PROMO, Y.B.BAT, Y.BASO PROM), read the first item's price (17,500), pull the total (91,000), and decide whether a tax line is present (no). All from the pixels.*

I took the first 30 receipts from the test split and hand-derived a small five-field ground truth from CORD's structured annotations:

```
num_items         (integer — how many menu line items)
first_item_name   (string — name of the first item, as printed)
first_item_price  (string — digits only)
total_price       (string — digits only)
has_tax           (boolean — separate tax/VAT line present?)
```

The prompt is one screen of plain text: *"Look at this receipt image and extract these fields as JSON..."*, the field list, and the image attached as base64. Temperature 0 everywhere. No vision-specific prompting tricks, no detail flags. The vanilla shape.

![Experiment setup: 30 receipts, 4 vision models, scored on 5 fields](setup.png)
*Figure 2: Same task as the text-extraction post, but the input is now a pixel buffer instead of a string. Only the model's vision encoder changes between rows.*

### The lineup

Picking vision-capable models is more constrained than picking text models — fewer of the open-weight families ship native multimodal weights, and Ollama's vision support specifically lags its text support. The four I tested:

| Model | Vision | Where it runs |
|---|---|---|
| `gemini-2.5-flash` | Native multimodal | Vertex AI |
| `gpt-5.4-mini` | Native multimodal | OpenAI API |
| `gemma4` (E4B) | Native multimodal [[2]](#ref-2) | Local · Ollama |
| `gemma4:26b` | Native multimodal | Local · Ollama |

Gemma 4 ships its vision encoder integrated into the same checkpoint as the text model; you don't load a separate vision adapter. That's worth flagging because it means the local memory footprint of "Gemma 4 with vision" is the same as "Gemma 4 with text" — about 5GB for E4B in 4-bit on Apple Silicon.

I considered adding Llama 3.2 Vision (11B), Qwen3-VL, and `gpt-oss:20b`, but each had a friction point — Llama 3.2 Vision wasn't installed cleanly on this hardware, Qwen-VL isn't yet in Ollama at the time of writing, and `gpt-oss` is text-only. The four-model lineup is the realistic apples-to-apples set on a current Apple Silicon laptop (36 GB unified memory).

## Headline result

Average accuracy across five fields × thirty receipts:

![Vision extraction accuracy by model on 30 CORD receipts](headline-accuracy.png)
*Figure 3: Cloud holds a ~3-point lead. The local Gemma E4B sits 2.6pp behind GPT-5.4-mini with 100% valid JSON. The larger 26B Gemma is the only model that struggles.*

The top three — Gemini 2.5 Flash (93.3%), GPT-5.4-mini (92.0%), Gemma 4 E4B (90.7%) — span 2.6 percentage points. Closer than I expected going in. The 26B Gemma drops to 79.3%, but most of that gap comes from the 5 out of 30 receipts where it returned invalid JSON; on the 25 where it parsed successfully, its mean score is in line with the others.

**The gap grew, but not by much.** In the [text-extraction post](/posts/slm-structured-extraction/) the cloud-to-best-local gap was 1.2 percentage points. On vision it's 2.6 — roughly double, still inside the noise floor on a 30-example single run. If you'd predicted "vision is where local SLMs fall off a cliff," the cliff isn't there yet. The cluster pattern looks similar to the text-extraction post; the cloud premium is real but small; the free local 8B is genuinely a working option for batch document extraction, in a way that wouldn't have been true two years ago.

A pilot version of this experiment used five hand-built synthetic invoices and produced a "three-way tie at 88%" result. That tie didn't survive the move to 30 real CORD receipts — Gemini opens a 2-3 point lead, and 26B-Gemma's instability becomes obvious. Same lesson as the text-extraction post: pilots are for debugging, not for deciding.

## Counting is harder than reading

Averaging across all four models, here's how the fields rank:

![Field difficulty averaged across all 4 vision models](field-accuracy.png)
*Figure 4: Reading labelled values is easy. Counting line items is meaningfully harder, because it requires segmenting the receipt visually rather than locating one label.*

The four "reading" fields — `has_tax`, `first_item_price`, `total_price`, `first_item_name` — sit between 87% and 94%. These all reduce to finding one labelled value or one keyword in the image. The vision encoder reads text well enough that the bottleneck is the model's ability to parse the *layout* and locate the right token.

`num_items` is the odd one out, at 81%. Counting line items requires segmenting the receipt visually: figure out where the menu section starts, where it ends, and how many distinct items live in between. That's a fundamentally different cognitive task from "find the number after 'TOTAL'." It involves spatial reasoning, not just text reading. Every model — including Gemini at 80% — was less reliable on this field than on any of the others.

This matches a pattern I've seen in production document-extraction work: extracting fields with explicit labels is the easy half. Counting, segmenting, or pulling list-of-objects (rather than a single value) is where vision models still drop accuracy. If your real schema has a `line_items: list[...]` field with N variable items per document, expect lower accuracy there than on flat scalar fields.

## The gemma4:26b problem is now a pattern

The most interesting finding from this run isn't about cloud-vs-local. It's about *which local Gemma is the right one to ship*.

`gemma4:26b` — the bigger Gemma, "should be better than the smaller one" — produced invalid JSON on **5 of 30 receipts**, all hard cases where the receipt was longer or more visually cluttered. The smaller `gemma4` E4B produced invalid JSON on **0 of 30**.

This is the same pattern that showed up in the text-extraction post: at n=10 the 26B Gemma lost 1 to invalid JSON; at n=30 it lost 1 more. On vision the failure rate is 17%, an order of magnitude worse than its smaller sibling. Each failed receipt scores 0 across all five fields, which is why the 26B's per-receipt score variance is **three times higher** than any other model in the run.

I have a working theory but not a clean confirmation. The 26B Gemma is a denser model than E4B (no MoE routing), so on Apple Silicon it sits much closer to the memory and compute ceiling. Generation pressure under that ceiling — especially for the longer, more complex receipts — pushes it into a state where it starts emitting markdown wrappers around its JSON, or truncates mid-array, or hallucinates extra keys. None of which break a cloud-served model the same way, because the cloud-served model isn't running on a laptop.

The practical implication is simple: **for vision extraction on a laptop, the smaller Gemma is the right Gemma.** I'd say that even if accuracy were identical, because reliability matters more than that last point of accuracy in a production pipeline. With accuracy *also* favouring the smaller model on this run, the case is unambiguous.

If you only take one operational tip from this post, take this one. Bigger isn't better when bigger doesn't fit.

## Latency: even more of a local penalty

Cloud responses on vision are slower than on text — these models are pushing pixel buffers and burning more tokens — but local responses are slower by a much bigger factor.

| Model | Source | Score | Avg latency / receipt |
|---|---|---|---|
| `gpt-5.4-mini` | cloud | 92.0% | **2.2s** |
| `gemini-2.5-flash` | cloud | 93.3% | 5.8s |
| `gemma4` (E4B) | local | 90.7% | 11.7s |
| `gemma4:26b` | local | 79.3% | 19.7s |

The 12-second per-receipt local latency on Gemma E4B is the laptop number, with all the caveats from the text-extraction post — single-stream, no batching, no GPU. The same model on vLLM behind a real GPU would be in the sub-second range. But for the realistic "I'm running this in a Jupyter notebook to see if it works" workflow, expect ~10 seconds per receipt on an M-series Mac, ~20 on the larger model.

There is a worthwhile detail in how the latency breaks down. Each receipt sends an image of roughly 600×800 pixels — which, depending on the model, encodes to somewhere between 250 and 1,000 input tokens [[3]](#ref-3)[[4]](#ref-4) before the prompt template even gets added. The cloud models tokenize the image server-side; the local models do it on-device. A meaningful chunk of the 12-second local latency is the image encoding step, not the JSON generation step. If you're tempted to optimise by tiling or downscaling images before sending them, the local case is where it'll show up first.

## The cost question, with the vision tax baked in

Vision inputs are billed differently from text inputs. Each receipt image, at the size CORD produces them, encodes to roughly:

- **OpenAI GPT-5.4-mini** (`detail: "auto"`): ~250-700 input tokens per image, depending on size [[3]](#ref-3)
- **Gemini 2.5 Flash**: ~258 input tokens per image at default tile size [[4]](#ref-4)
- **Local (Ollama)**: no per-token billing — just GPU time

That changes the cost math from the text post. The cheap-cloud option for vision is no longer a $60-a-month rounding error at a million requests; it's more like several hundred dollars when you add the per-image token cost on top of the prompt and output. For a pipeline at 1M images per month with this prompt shape (~500 image+prompt tokens in, ~100 tokens out):

| Option | Monthly cost @ 1M images | Accuracy |
|---|---|---|
| `gemini-2.5-flash` (cloud) | ~$400 | 93.3% |
| `gpt-5.4-mini` (cloud) | ~$700 | 92.0% |
| Gemma 4 E4B self-hosted on A100 ($3.67/hr) | ~$2,640 | 90.7% |

(These are rough; vision tokenisation rules change between providers and image sizes, so treat the cloud numbers as the right order of magnitude rather than to-the-dollar accurate.)

The cheap-cloud edge has shrunk but not disappeared. Gemini Flash on vision at $400/month for a million receipts is still hard to beat for low-to-mid volume. The self-hosting crossover for vision happens earlier than for text — somewhere around 4-7 million images per month against Gemini Flash on this prompt shape — because vision input pricing scales with image count, not paragraph length.

For the batch-document use case (overnight processing of yesterday's receipts, no real-time SLA), self-hosting at the high end of that range becomes attractive. For the interactive use case (a chatbot user attaches a receipt and wants an answer in two seconds), the latency picture tips it firmly back toward the cloud APIs.

## When this doesn't apply

Honest caveats, same shape as in the text-extraction post.

- **CORD is one kind of receipt.** Mostly Indonesian and Korean, mostly thermal prints, mostly photographed on the same desk under similar lighting. Other domains — pharmacy receipts in English, scanned PDFs from a corporate ERP, photographed handwritten invoices — will produce a different ranking. The relative ordering between models may hold; the absolute numbers won't.
- **CORD is a *public* benchmark.** The dataset has been on HuggingFace since 2019, which means every cloud model in this lineup almost certainly saw it during pre-training. The cloud scores at the top of the chart are partly "this model can read receipts" and partly "this model has seen these specific receipts before." The local Gemmas are newer and were trained on different corpora, but I can't rule out that they've seen CORD too — and the synthetic-data trick I used in the text-extraction post (where I knew none of the models had seen the test set) doesn't have an obvious analogue for vision. **The cloud's lead here is an upper bound, and the contamination-adjusted number is probably smaller than 2.6pp.** A capstone post later in this series digs into the contamination question in more rigorous terms.
- **Five fields is a small schema.** Real-world receipt parsing usually wants line items as a list of objects with name, quantity, unit price, and subtotal each — which is the `num_items`-style task done five times per receipt. The accuracy on that fuller schema would almost certainly be lower than 90% across the board.
- **No vision-specific tricks.** I didn't enable OpenAI's `detail: "high"` flag, didn't pre-rotate or crop the images, didn't use a layout-aware OCR pre-pass. Any of those would lift every model by some amount and probably compress the table further. I wanted the vanilla baseline.
- **Single run, no variance.** Same disclaimer as last time: one run per (model, receipt). The 2.6-point gap between Gemini and Gemma E4B is within plausible run-to-run noise.
- **No comparison to specialist OCR.** Donut [[5]](#ref-5), LayoutLM, and other document-understanding models are purpose-built for this kind of task and may outperform general-purpose VLMs at receipt extraction. They're also a heavier integration. The question I cared about was "can a general-purpose LLM with vision do this acceptably?" — for which the answer is yes.

## The methodology lesson

If the lesson from the text-extraction post was *pilot at n=10, decide at n≥30*, the lesson from this one is more specific:

> **Score per-image, then look at the distribution — not just the mean.**

The single number "gemma4:26b scored 79.3%" hides the actual failure mode. The model isn't 79% accurate; it's 100% accurate on the 25 receipts it parsed, and 0% accurate on the 5 it didn't. That's a completely different operational characteristic from "79% accurate uniformly." The first kind of failure is fixable with retries or a validator-and-retry loop; the second kind isn't.

Whenever you're evaluating a model for production use, plot the per-example score histogram, not just the average. A bimodal distribution where most examples are perfect and a handful are zero is a different bet from a unimodal distribution centred on the mean. The fix is different. The risk profile is different. The single average score is misleading on its own.

This is the kind of thing that's obvious in hindsight and easy to skip when you're in a rush.

## Where to go from here

Vision and text extraction are both tasks where the model gets one input and returns one structured output — the prompt-in, JSON-out shape. The local-cloud gap is small in both cases. What happens when you add a *retrieval* step in the middle — where the model has to first decide which pieces of a corpus are relevant, then synthesise an answer from them?

That's the RAG question, and it's where I expect the gap to widen the most. The model can be smart and the prompt can be sharp, but if the retriever hands it the wrong chunks, no amount of generation quality recovers. [The next post](/posts/slm-rag-shootout/) is a RAG shootout — same kind of head-to-head, but with the retriever as a first-class variable.

If you want to take this one further yourself:

- **Run CORD-v2 against your model of choice.** The dataset is on HuggingFace [[1]](#ref-1), the eval harness in this experiment is a couple of hundred lines, and the per-field scoring is forgiving (digit-stripping for prices, substring match for names). It's a useful baseline to keep around.
- **Add a JSON validator-and-retry loop.** Five of the 26B Gemma's 30 failures were just malformed JSON, not wrong content. A single retry with `format="json"` enforced [[6]](#ref-6) would recover most of those. That alone would move the 26B from 79% to ~89%, which would change every recommendation in this post.
- **Compare against a specialist.** If your only use case is receipts, run Donut [[5]](#ref-5) on the same 30 images. The interesting question isn't "which general-purpose VLM is best" but "is the specialist still worth it." I haven't tested that here, but it's the right next step for someone actually building a receipt pipeline.
- **Try larger images.** OpenAI's `detail: "high"` flag and Gemini's higher-resolution tile mode both increase per-image token cost meaningfully but should help on the hardest receipts. Worth measuring at what image size the accuracy lift stops paying for the extra tokens.

## References

<span id="ref-1">**[1]**</span> **Park et al. (2019)** — [CORD: A Consolidated Receipt Dataset for Post-OCR Parsing](https://openreview.net/forum?id=SJl3z659UH). The original CORD paper; the v2 dataset used here is hosted at [naver-clova-ix/cord-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) on HuggingFace.

<span id="ref-2">**[2]**</span> **Google DeepMind** — [Gemma model family](https://ai.google.dev/gemma). Gemma 4 ships native multimodal weights in the same checkpoint as the text model.

<span id="ref-3">**[3]**</span> **OpenAI** — [Vision input pricing](https://platform.openai.com/docs/guides/vision). Images are billed at 85 base tokens plus 170 tokens per 512×512 tile in high-detail mode.

<span id="ref-4">**[4]**</span> **Google Vertex AI** — [Gemini multimodal pricing and tokenisation](https://ai.google.dev/gemini-api/docs/pricing). Images under 384×384 are billed as 258 input tokens; larger images are tiled.

<span id="ref-5">**[5]**</span> **Kim et al. (2022)** — [Donut: Document Understanding Transformer without OCR](https://arxiv.org/abs/2111.15664). End-to-end transformer for document parsing without a separate OCR step; the closest specialist baseline for the CORD task.

<span id="ref-6">**[6]**</span> **Ollama** — [Structured outputs (JSON mode)](https://ollama.com/blog/structured-outputs). Passing `format="json"` forces strict JSON output and would have recovered most of the 26B Gemma's failures.
