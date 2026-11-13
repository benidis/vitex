This is an improved version of an existing tutorial (the --doc seed), with the
video as a second seed. It is a tutorial on making attention efficient for
large language models: why the KV cache dominates memory, then MQA, GQA, MLA
(with its RoPE handling), and sparse attention such as DeepSeek Sparse
Attention (DSA), each motivated by the limitation of the previous one.

1. Inference versus training. Make explicit, early and throughout, which part
   of the model's life each method improves — inference, training, or both —
   and why. Set up the two regimes before the methods: training processes whole
   sequences in parallel (compute-bound, no KV cache), while autoregressive
   decoding generates one token at a time and re-reads the cached keys and
   values for every new token (memory-bandwidth-bound, KV cache grows with
   batch x context length). Show with concrete numbers why the KV cache becomes
   the bottleneck at inference. For each method then state: what it saves (KV
   cache memory, memory bandwidth per decoded token, attention FLOPs), in which
   regime (prefill, decode, training), and what it costs (quality, extra
   compute, training changes such as uptraining GQA from a multi-head
   checkpoint or MLA's need to be trained from the start).
   Contrast this family with methods that target training only — data, tensor,
   pipeline, and sequence/context parallelism, activation checkpointing,
   mixed precision — briefly and only to place the tutorial's methods: they are
   out of scope beyond a short comparison (a paragraph or a table), and the
   reader should understand why they solve a different problem. Mention where
   methods help both (e.g. FlashAttention-style IO-aware kernels speed up
   training and prefill without changing the model; sparse attention can reduce
   long-context training cost too).

2. Sparse attention (Section 7 of the existing tutorial, DSA) was hard to
   follow. Rebuild it from the problem up: why dense attention is quadratic in
   the context length and where that hurts (prefill and long-context decoding),
   what "sparse" means (each query attends to a selected subset of keys), the
   simplest fixed patterns (local windows, strided) and their limitation, then
   learned/content-based selection. Explain DSA step by step: the lightning
   indexer (what it computes, its cost, why it is cheap), top-k selection, how
   attention is then computed over the selected tokens, how it interacts with
   MLA, how it is trained (warm-up and sparse stages), and a worked numerical
   example with a small sequence showing the indexer scores, the selected
   tokens, and the cost saved. Give the algorithm as it is run and the
   complexity before and after.

3. Keep the existing tutorial's good parts — its foundations of self-attention,
   the KV-cache explanation, MQA/GQA, MLA, and RoPE — fact-checked and
   improved, with cost analyses using concrete model sizes (e.g. bytes of KV
   cache per token for MHA vs GQA vs MLA for a real model configuration).
