A tutorial on optimizing and parallelizing the training of large models, built
from three seeds: a detailed set of notes (the --doc, which covers foundations,
communication primitives, DP, ZeRO, TP, sequence parallelism, their combination
in ms-swift, LoRA, and configuration decisions) and two videos (ZeRO/FSDP, and
an overview of five ways thousands of GPUs train one model).

1. Categorization is central. The tutorial must make unmistakably clear, for
   every method, where it operates and what it buys:
   - WHAT IT SPLITS or reduces: the batch, the model's layers, individual
     weight matrices, the sequence, the optimizer/gradient/parameter state, the
     experts, or nothing (a pure memory or precision trick).
   - WHICH BOTTLENECK it addresses: GPU memory (which part: parameters,
     gradients, optimizer states, activations, KV/attention memory),
     compute/time, or inter-device communication.
   - WHAT IT COSTS: extra communication (which collective, how much data, on
     which interconnect), idle time (pipeline bubbles), recomputation,
     implementation complexity, or constraints on the model or batch shape.
   - WHEN TO USE IT: the regime where it is the right answer (model fits or
     does not fit on one device, long sequences, limited interconnect
     bandwidth, many nodes, small or large batch), and how the methods compose
     in practice (e.g. DP x TP x PP x SP topologies, ZeRO stage with TP/SP,
     what to reach for first and what only when forced).
   Carry this through the whole tutorial: a comparison table of all methods
   along these axes, a short per-method summary of the same, and a decision
   guide at the end (given model size, device memory, sequence length, number
   of devices and interconnect, what configuration to choose and why). State
   explicitly which methods change training only, which affect inference too,
   and which change the model's mathematics versus only its execution.

2. The "5 ways thousands of GPUs train one AI" video was hard to follow.
   Rebuild everything it covers from first principles instead of following its
   presentation: for each parallelism (data, tensor, pipeline, sequence/context,
   expert) start from the concrete limitation it fixes, show what is placed on
   which device, follow one training step end to end (forward, backward,
   optimizer step) saying exactly which communication happens when and what is
   synchronized, and give a small worked example with real numbers (device
   count, shapes, bytes moved). Where a mechanism is subtle (pipeline schedules
   and bubbles, activation flow in sequence/context parallelism, ring attention,
   expert routing and load balancing, overlap of communication with compute),
   explain it step by step; a reader must be able to redraw the data movement
   from memory.

3. Also cover the memory and throughput optimizations that are not parallelism
   but belong in the same decision: mixed precision and bf16/fp8, activation
   checkpointing/recomputation, gradient accumulation, CPU/NVMe offload,
   fused and memory-efficient kernels (FlashAttention-style), and efficient
   fine-tuning (LoRA/QLoRA) and how it changes the memory picture. Put them in
   the same categorization so the reader sees them as alternatives to, or
   complements of, splitting the model.

4. Keep the practical content of the notes: the communication primitives, the
   per-method memory accounting with formulas and concrete numbers, the ZeRO
   stages and what each partitions, and the configuration decisions. The
   ms-swift/DeepSpeed specifics (the mpu question, gradient normalization,
   ZeRO-3 with frozen LoRA parameters) go in one compact practical section. Give the algorithms and
   configurations as they are actually run, fact-checked against the primary
   sources (ZeRO, Megatron-LM, GPipe/PipeDream, FSDP, ring attention, Mixture
   of Experts, FlashAttention).

5. Hardware foundation first. Before any method uses these terms, give a short,
   plain foundation on the hardware the methods depend on: GPU memory (HBM) vs
   host (CPU) memory, NVLink/NVSwitch within a node vs PCIe, InfiniBand/RoCE
   between nodes, and NVMe storage — what each is, its typical bandwidth and
   latency, and why the gaps between them decide which parallelism goes where
   (e.g. tensor parallelism inside a node, data parallelism across nodes).
   Refer back to it whenever a method's cost depends on the interconnect.

6. Readability over coverage. This should read like a textbook chapter, not
   like dense notes: fewer topics, each explained properly, rather than every
   detail of the sources compressed together. The main parallelisms (data
   parallelism with ZeRO/FSDP, tensor, pipeline, sequence/context, expert)
   and the single-GPU memory techniques deserve full treatment; schedules,
   variants, and framework specifics (e.g. zero-bubble schedules, Ulysses vs
   ring attention, ms-swift/DeepSpeed details) should be compact or mentions
   unless they are needed to understand a main method.
