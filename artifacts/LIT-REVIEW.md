# Literature Review: Knowledge Distillation for Large Language Models

## Distillation

### Distilling the Knowledge in a Neural Network
**Key:** `hintonDistillingKnowledgeNeural2015`

Introduces knowledge distillation, a technique for compressing the knowledge of an ensemble (or a large model) into a single smaller model by training it to match softened probability distributions of the teacher. Demonstrates surprising results on MNIST and significantly improves a commercial acoustic model. Also introduces specialist models that learn to distinguish fine-grained classes the full models confuse, forming a new type of ensemble.

---

### Towards Cross-Tokenizer Distillation: the Universal Logit Distillation Loss for LLMs
**Key:** `boizardCrossTokenizerDistillationUniversal2025`

Introduces Universal Logit Distillation (ULD), a loss function grounded in optimal transport that enables knowledge distillation between LLMs with different tokenizers and architectures. Standard logit-based distillation methods require shared tokenizers between teacher and student, limiting cross-family applicability. ULD overcomes this limitation, paving the way for more widespread use of distillation across diverse LLM families.

---

### Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching
**Key:** `minixhoferUniversalCrossTokenizerDistillation2025`

Develops a principled cross-tokenizer distillation method that is the first to enable effective distillation across fundamentally different tokenizers without requiring a next-token prediction loss. Demonstrates three use cases: tokenizer transfer (including subword to byte-level), distillation of math-specialized LLMs into general-purpose models with different tokenizers, and training-free tokenizer transfer via hypernetworks. Transferring models to the same tokenizer also enables ensembling to boost performance.

---

### Distribution-Aligned Sequence Distillation for Superior Long-CoT Reasoning
**Key:** `yanDistributionAlignedSequenceDistillation2026`

Introduces DASD-4B-Thinking, a lightweight reasoning model achieving SOTA performance among open-source models of comparable scale by critically reexamining SFT-based sequence-level distillation. Identifies three critical limitations in current practice: inadequate representation of the teacher's sequence-level distribution, misalignment between teacher output distribution and student learning capacity, and exposure bias. Proposes methodological innovations forming an enhanced sequence-level distillation pipeline, achieving competitive results with only 448K training samples.

---

### GKD: A General Knowledge Distillation Framework for Large-scale Pre-trained Language Model
**Key:** `tan2023GKDGeneralKnowledge`

Proposes GKD, a general knowledge distillation framework that supports distillation on larger-scale PLMs using various methods, enabling developers to build distillation models on memory-limited GPUs and easily switch between 25 mainstream distillation methods. Addresses the challenge of deploying complex distillation systems in real-world industrial-strength applications at scale. Demonstrates support for distillation of at least 100B-scale PLMs on 8 NVIDIA A100 (40GB) GPUs.

---

### SelecTKD: Selective Token-Weighted Knowledge Distillation for LLMs
**Key:** `huang2025SelecTKDSelectiveTokenWeighted`

Introduces SelecTKD, a plug-and-play Selective Token-Weighted distillation framework that shifts focus from "how to measure divergence" to "where to apply learning" through a propose-and-verify procedure where the student proposes tokens verified by the teacher. Accepted tokens receive full loss while rejected tokens are masked or down-weighted, inducing an implicit curriculum quantified by Token Acceptance Rate (TAR). Consistently improves strong baselines and achieves state-of-the-art results for small models across instruction following, math, code, and VLM settings.

---

### ERNIE 3.0 Titan: Exploring Larger-scale Knowledge Enhanced Pre-training for Language Understanding and Generation
**Key:** `wang2021ERNIE30Titan`

Trains ERNIE 3.0 Titan, a 260-billion-parameter Chinese dense pre-trained model, introducing self-supervised adversarial and controllable language modeling losses for credible, controllable text generation. Proposes an online distillation framework where the teacher model teaches students and trains itself simultaneously, reducing computation overhead and carbon emissions. Achieves state-of-the-art results on 68 NLP datasets.

---

### Embarrassingly Simple Self-Distillation Improves Code Generation
**Key:** `zhangEmbarrassinglySimpleSelfDistillation2026`

Demonstrates that simple self-distillation (SSD)—sampling solutions from a model and fine-tuning on those samples—improves code generation without requiring a verifier, teacher model, or reinforcement learning. Improves Qwen3-30B-Instruct from 42.4% to 55.3% pass@1 on LiveCodeBench v6, with gains concentrating on harder problems and generalizing across model families at 4B, 8B, and 30B scale. Traces gains to a precision-exploration conflict in decoding, showing SSD reshapes token distributions in a context-dependent way.

---

### Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs?
**Key:** `kimWhyDoesSelfDistillation2026`

Finds that self-distillation in mathematical reasoning can reduce response length while degrading performance, tracing this to suppression of epistemic verbalization—the model's expression of uncertainty during reasoning. Shows that conditioning the teacher on rich information suppresses uncertainty expression, enabling rapid in-domain optimization but harming OOD performance with drops of up to 40% across Qwen3-8B, DeepSeek-Distill-Qwen-7B, and OLMo3-7B-Instruct. Highlights that exposing appropriate levels of uncertainty is crucial for robust reasoning beyond merely reinforcing correct answer traces.

---

### Distilling 100B+ Models 40x Faster with TRL
**Key:** `patino2026_distilling_100b_models_40x_faster_with_trl`

Presents practical methods and infrastructure for performing knowledge distillation from 100B+ parameter teacher models at scale using the TRL library, achieving up to 40× speedups over naive approaches.

---

## On-Policy Distillation

### On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes
**Key:** `agarwalOnPolicyDistillationLanguage2024`

Introduces Generalized Knowledge Distillation (GKD) to address the distribution mismatch in auto-regressive sequence models, where training uses a fixed dataset but inference uses student-generated outputs. GKD trains the student on its own self-generated output sequences with the teacher providing on-policy feedback, and explores alternative divergences beyond forward KLD. Shows effectiveness in summarization, translation, arithmetic reasoning, and instruction-tuning, integrating naturally with RL fine-tuning.

---

### MiniLLM: On-Policy Distillation of Large Language Models
**Key:** `guMiniLLMOnPolicyDistillation2026`

Proposes MiniLLM, a knowledge distillation approach for white-box generative LLMs that replaces forward KLD with reverse KLD to prevent the student from overestimating low-probability regions of the teacher distribution. Derives an effective on-policy optimization approach where student models generate their own sequences during training. Shows that MiniLLM generates more precise responses with higher quality, lower exposure bias, better calibration, and improved long-text generation across models from 120M to 13B parameters.

---

### On-Policy Distillation
**Key:** `luOnPolicyDistillation2025`

Technical report from the Thinking Machines Lab providing analysis and discussion of on-policy distillation methods for large language models.

---

### Unmasking On-Policy Distillation: Where It Helps, Where It Hurts, and Why
**Key:** `armandpourUnmaskingOnPolicyDistillation2026`

Introduces a training-free diagnostic framework that operates at per-token, per-question, and per-teacher resolution to analyze when on-policy distillation helps vs. hurts. Derives an ideal per-node gradient and a scalable targeted-rollout algorithm to estimate it, defining a gradient alignment score that quantifies how well a distillation configuration approximates the ideal signal. Finds that distillation guidance is substantially more aligned on incorrect rollouts than correct ones, and that the optimal distillation context depends jointly on student capacity and target task.

---

### Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe
**Key:** `liRethinkingOnPolicyDistillation2026`

Provides a systematic investigation of OPD dynamics, identifying two conditions for success: teacher-student compatible thinking patterns, and the teacher offering genuinely new capabilities beyond the student's training data. Shows that successful OPD is characterized by progressive alignment on a small set of high-probability tokens (97–99% of mass) at student-visited states. Proposes practical strategies to recover failing OPD—off-policy cold start and teacher-aligned prompt selection—while raising questions about OPD's scalability to long-horizon distillation.

---

### Lightning OPD: Efficient Post-Training for Large Reasoning Models with Offline On-Policy Distillation
**Key:** `wuLightningOPDEfficient2026`

Investigates whether on-policy distillation can be performed offline, identifying "teacher consistency"—using the same teacher for both SFT and OPD—as a critical and previously overlooked condition. Proposes Lightning OPD, which precomputes teacher log-probabilities over SFT rollouts, eliminating the need for a live teacher server entirely while sharing the same optimum as standard OPD. Achieves 69.9% on AIME 2024 in just 30 GPU hours starting from Qwen3-8B-Base, a 4× speedup over standard OPD.

---

### TIP: Token Importance in On-Policy Distillation
**Key:** `xuTIPTokenImportance2026`

Proposes TIP, a two-axis taxonomy of token importance in OPD based on student entropy and teacher-student divergence, showing that not all token positions carry equal learning signal. Demonstrates that entropy-based sampling retaining 50% of tokens matches or exceeds full training while reducing peak memory by up to 47%, and that low-entropy high-divergence tokens (overconfident wrong predictions) carry dense corrective signal despite being less than 10% of tokens. Validates across Qwen3, Llama, and Qwen2.5 on MATH-500, AIME 2024/2025, and the DeepPlanning benchmark.

---

### Stable On-Policy Distillation through Adaptive Target Reformulation
**Key:** `jang2026StableOnPolicyDistillation`

Proposes Veto, an objective-level reformulation that constructs a geometric bridge in logit space to address training instabilities in on-policy KD caused by the wide distributional gap between novice student and expert teacher. Introduces a tunable parameter β that serves as both an Adaptive Gradient Veto (suppressing harmful gradients on low-confidence tokens) and a Decisiveness Knob (balancing reward-driven performance with output diversity). Consistently outperforms SFT and existing on-policy baselines across reasoning and generation tasks.

---

### Unlocking On-Policy Distillation for Any Model Family
**Key:** `patinoUnlockingOnPolicyDistillation2025`

Presents methods for enabling on-policy distillation across different model families, removing the constraint that teacher and student must share the same architecture or tokenizer.

---

### Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting
**Key:** `chenRetainingDoingRole2025`

Systematically compares forgetting patterns of SFT and RL post-training, finding RL leads to less forgetting than SFT while achieving comparable or higher task performance across Llama and Qwen model families. Identifies the mode-seeking nature of RL, which stems from its use of on-policy data, as the key factor enabling prior knowledge preservation during learning. Highlights the potential of approximately on-policy data for substantially more efficient forgetting mitigation.

---

### Speculative Knowledge Distillation: Bridging the Teacher-Student Gap Through Interleaved Sampling
**Key:** `xu2024SpeculativeKnowledgeDistillation`

Introduces Speculative Knowledge Distillation (SKD) that leverages cooperation between student and teacher to generate high-quality training data on-the-fly while aligning with the student's inference-time distribution. In SKD, the student proposes tokens and the teacher replaces poorly ranked ones based on its own distribution, transferring high-quality knowledge adaptively. Consistently outperforms existing KD methods across translation, summarization, math, and instruction following.

---

## Privileged Context

### Privileged Information Distillation for Language Models
**Key:** `penalozaPrivilegedInformationDistillation2026`

Studies the challenge of distilling frontier agents in multi-turn agentic environments where closed-source systems hide internal reasoning and expose only action trajectories. Introduces π-Distill, a joint teacher-student objective training a PI-conditioned teacher and unconditioned student simultaneously, and On-Policy Self-Distillation (OPSD) using RL with reverse KL-penalty between student and PI-conditioned teacher. Both methods effectively distill frontier agents using action-only privileged information, outperforming industry-standard SFT+RL pipelines across multiple agentic benchmarks.

---

### On-Policy Context Distillation for Language Models
**Key:** `yeOnPolicyContextDistillation2026`

Proposes On-Policy Context Distillation (OPCD), bridging on-policy distillation with context distillation by training a student on its own trajectories while minimizing reverse KL divergence against a context-conditioned teacher. Demonstrates effectiveness on experiential knowledge distillation (consolidating knowledge from historical solution traces) and system prompt distillation (internalizing beneficial behaviors from optimized prompts). Consistently outperforms baselines across math reasoning, text-based games, and domain-specific tasks while preserving out-of-distribution capabilities.

---

### Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models
**Key:** `zhaoSelfDistilledReasonerOnPolicy2026`

Introduces OPSD where a single model acts as both teacher and student by conditioning on different contexts—the teacher sees privileged verified reasoning traces while the student sees only the question. Trains by minimizing per-token divergence between these distributions over the student's own rollouts, eliminating the need for a separate larger teacher model. Achieves 4–8× token efficiency compared to GRPO and superior performance over off-policy distillation on mathematical reasoning benchmarks.

---

### Self-Distilled RLVR
**Keys:** `yangSelfDistilledRLVR2026`, `yangSelfDistilledRLVR2026a`

Demonstrates that learning signals derived solely from a privileged teacher (conditioned on reference answers) result in severe information leakage and unstable long-term training. Proposes RLSD, which leverages self-distillation to obtain token-level policy differences for fine-grained update magnitudes while using RLVR for reliable update directions from environmental feedback (e.g., response correctness). Combines strengths of both RLVR and on-policy self-distillation, achieving a higher convergence ceiling and superior training stability.

---

### Self-Distillation Enables Continual Learning
**Key:** `shenfeldSelfDistillationEnablesContinual2026`

Introduces Self-Distillation Fine-Tuning (SDFT), which enables on-policy learning from demonstrations by using a demonstration-conditioned model as its own teacher via in-context learning. Generates on-policy training signals that preserve prior capabilities while acquiring new skills, outperforming SFT with higher new-task accuracy and substantially reduced catastrophic forgetting. Enables sequential accumulation of multiple skills over time without performance regression.

---

### Reinforcement Learning via Self-Distillation
**Key:** `hubotterReinforcementLearningSelfDistillation2026`

Introduces Self-Distillation Policy Optimization (SDPO), which converts rich textual feedback (runtime errors, judge evaluations) into a dense learning signal without any external teacher, by treating the model conditioned on feedback as a self-teacher. Distills feedback-informed next-token predictions back into the policy, leveraging the model's ability to retrospectively identify its own mistakes in-context. Improves sample efficiency and final accuracy over strong RLVR baselines across scientific reasoning, tool use, and competitive programming.

---

### Self-Distillation Zero: Self-Revision Turns Binary Rewards into Dense Supervision
**Key:** `heSelfDistillationZeroSelfRevision2026`

Proposes SD-Zero, which trains a single model to play two roles: a Generator producing initial responses, and a Reviser conditioning on the response and its binary reward to produce improved responses. Performs on-policy self-distillation from reviser to generator, effectively transforming binary rewards into dense token-level self-supervision without an external teacher. Outperforms RFT, GRPO, and SDFT on math and code reasoning benchmarks, exhibiting token-level self-localization and iterative self-evolution.

---

## Distillation with Large Capacity Gap

### Improved Knowledge Distillation via Teacher Assistant
**Key:** `mirzadehImprovedKnowledgeDistillation2019`

Shows that student network performance degrades when the gap between student and teacher is large—a teacher can only effectively transfer knowledge to students up to a certain size. Introduces multi-step knowledge distillation using an intermediate-sized Teacher Assistant network to bridge the gap, extending the framework to multi-step distillation chains. Demonstrates effectiveness on CIFAR-10/100 and ImageNet with CNN and ResNet architectures.

---

### Small Models Struggle to Learn from Strong Reasoners
**Key:** `liSmallModelsStruggle2025`

Uncovers the Small Model Learnability Gap: models ≤3B parameters do not consistently benefit from long chain-of-thought reasoning or distillation from larger models, performing better with shorter, simpler reasoning chains aligned with their intrinsic capacity. Proposes Mix Distillation, which balances reasoning complexity by combining long and short CoT examples or reasoning from both larger and smaller models. Significantly improves small model reasoning performance compared to training on either data source alone.

---

### TAID: Temporally Adaptive Interpolated Distillation for Efficient Knowledge Transfer in Language Models
**Key:** `shingTAIDTemporallyAdaptive2025`

Introduces TAID, which dynamically interpolates student and teacher distributions through an adaptive intermediate distribution, gradually shifting from student's initial distribution toward the teacher's. Provides theoretical analysis demonstrating TAID's ability to prevent mode collapse while empirically addressing the capacity gap and balancing mode averaging. Develops two state-of-the-art compact foundation models: TAID-LLM-1.5B for language tasks and TAID-VLM-2B for vision-language tasks.

---

### Lifting the Curse of Capacity Gap in Distilling Language Models
**Key:** `zhangLiftingCurseCapacity2023`

Addresses the curse of capacity gap in knowledge distillation of pretrained language models, where larger teacher-student gaps lead to diminishing returns. Proposes MiniMoE, a Mixture of Minimal Experts approach that adds parameters to the student without significantly increasing inference compute, effectively bridging the capacity gap. Achieves state-of-the-art performance on GLUE and CoNLL benchmarks with high compression rates.

---

### Follow Your Path: A Progressive Method for Knowledge Distillation
**Key:** `shi2021FollowYourPath`

Proposes a progressive knowledge distillation method that guides the student model along an optimized path from its initial distribution toward the teacher's, rather than directly matching the teacher. This progressive approach helps bridge the capacity gap by providing intermediate learning targets that are more suitable for the student's current ability. Published in ECML PKDD 2021.

---

## Other / Supporting References

### American Invitational Mathematics Examination (AIME) 2025
**Key:** `aime25`

Mathematical competition benchmark commonly used to evaluate reasoning capabilities of LLMs on challenging high-school level math problems.

---

### The Smol Training Playbook: The Secrets to Building World-Class LLMs
**Key:** `allalSmolTrainingPlaybook2025`

Comprehensive guide covering the full pipeline for training world-class large language models, from data curation and pre-training to post-training and evaluation.

---

### Stream of Search (SoS): Learning to Search in Language
**Key:** `gandhiStreamSearchSoS2024`

Shows that language models can learn complex problem-solving strategies by pretraining on streams of search generated by heuristic solvers, representing the search process as a flattened string. Models can be further improved with policy-based methods like Advantage-Induced Policy Alignment, solving previously unsolvable problems.

---

### OLMo: Accelerating the Science of Language Models
**Key:** `groeneveldOLMoAcceleratingScience2024`

Presents OLMo, a competitive truly open language model released alongside open training data, training code, and evaluation code to enable scientific study of language models. Unlike most prior efforts that only release model weights and inference code, OLMo aims to empower open research and inspire innovation.

---

### DeepMath-103K: A Large-Scale, Challenging, Decontaminated, and Verifiable Mathematical Dataset for Advancing Reasoning
**Key:** `he2025DeepMath103KLargeScaleChallenging`

Introduces DeepMath-103K, a large-scale mathematical dataset designed with high difficulty (primarily levels 5–9), rigorous decontamination against numerous benchmarks, and verifiable answers for rule-based RL reward. Models trained on DeepMath-103K achieve state-of-the-art results on challenging mathematical benchmarks and demonstrate generalization beyond math to biology, physics, and chemistry.

---

### JustRL: Scaling a 1.5B LLM with a Simple RL Recipe
**Key:** `heJustRLScaling15B2025`

Demonstrates that a minimal RL approach using single-stage training with fixed hyperparameters achieves state-of-the-art performance on 1.5B reasoning models, outperforming sophisticated multi-stage pipelines with curriculum learning and dynamic schedules. Shows that complexity in RL for LLMs is not always necessary.

---

### Understanding Reasoning in LLMs through Strategic Information Allocation under Uncertainty
**Key:** `kimUnderstandingReasoningLLMs2026`

Introduces an information-theoretic framework that decomposes reasoning into procedural information and epistemic verbalization—the explicit externalization of uncertainty that supports downstream control actions. Shows that purely procedural reasoning can become informationally stagnant, whereas epistemic verbalization enables continued information acquisition and is critical for information sufficiency.

---

### Efficient Memory Management for Large Language Model Serving with PagedAttention
**Key:** `kwon2023efficient`

Introduces PagedAttention and the vLLM system for efficiently managing GPU memory in LLM serving, applying virtual memory and paging concepts to the KV cache. Achieves near-zero waste in KV cache memory and significantly improves serving throughput.

---

### Let's Verify Step by Step
**Key:** `lightmanLetsVerifyStep2023`

Finds that process supervision (feedback for each intermediate reasoning step) significantly outperforms outcome supervision (feedback for the final result only) for training models to solve challenging MATH dataset problems. Releases PRM800K, a dataset of 800,000 step-level human feedback labels, and shows active learning significantly improves the efficacy of process supervision.

---

### ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models
**Key:** `liuProRLProlongedReinforcement2025`

Demonstrates that prolonged RL (ProRL) training can uncover novel reasoning strategies inaccessible to base models even under extensive sampling, challenging the assumption that RL merely amplifies latent capabilities. Introduces KL divergence control, reference policy resetting, and diverse task suites to enable sustained RL training, showing reasoning boundary improvements correlate with task competence and training duration.

---

### DeepScaleR: Surpassing O1-Preview with a 1.5B Model by Scaling RL
**Key:** `luoDeepScaleRSurpassingO1Preview2025`

Demonstrates that scaling reinforcement learning with a 1.5B parameter model can achieve reasoning performance surpassing O1-Preview, showing the potential of compute-efficient RL scaling for small models.

---

### DataTrove: Large Scale Data Processing
**Key:** `penedo2024datatrove`

Open-source library for large-scale data processing pipelines, commonly used for curating pre-training and fine-tuning datasets for LLMs.

---

### GPQA: A Graduate-Level Google-Proof Q&A Benchmark
**Key:** `reinGPQAGraduateLevelGoogleProof2023`

Presents GPQA, a challenging dataset of 448 multiple-choice questions written by domain experts in biology, physics, and chemistry where experts reach 65% accuracy while non-experts reach only 34% despite unrestricted web access. Designed for scalable oversight experiments to develop methods for humans to reliably supervise AI systems.

---

### Inspect AI: Framework for Large Language Model Evaluations
**Key:** `UK_AI_Security_Institute_Inspect_AI_Framework_2024`

Evaluation framework developed by the UK AI Security Institute for conducting standardized evaluations of large language models.

---

### TRL: Transformers Reinforcement Learning
**Key:** `vonwerra2020trl`

Open-source library providing tools for training transformer language models with reinforcement learning, including implementations of PPO, DPO, SFT, and distillation methods.

---

### Qwen3 Technical Report
**Key:** `yangQwen3TechnicalReport2025`

Presents Qwen3, a series of LLMs from 0.6B to 235B parameters integrating thinking mode (for complex reasoning) and non-thinking mode (for rapid responses) into a unified framework. Introduces a thinking budget mechanism for adaptive computational resource allocation and expands multilingual support from 29 to 119 languages.

---

### τ-Bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains
**Key:** `yaoTbenchBenchmarkToolAgentUser2024`

Benchmark for evaluating agents' ability to interact with simulated users through dynamic conversations while using domain-specific API tools and following policy guidelines.

---

### SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild
**Key:** `zengSimpleRLZooInvestigatingTaming2025`

Investigates zero RL training (starting directly from base models with rule-based rewards) across 10 diverse base models spanning different families and sizes. Shares key design strategies including format reward adjustment and query difficulty control, observing the "aha moment" for the first time in small non-Qwen models.

---

### Good SFT Optimizes for SFT, Better SFT Prepares for Reinforcement Learning
**Key:** `zhangGoodSFTOptimizes2026`

Shows that after identical RL training, models initialized from stronger SFT checkpoints can significantly underperform those from weaker ones due to distribution mismatch between offline SFT data and online RL policy. Proposes PEAR, an SFT-stage method using importance sampling to reweight the SFT loss, consistently improving post-RL performance with pass@8 gains up to 14.6% on AIME 2025.

---

### Instruction-Following Evaluation for Large Language Models
**Key:** `zhou2023instructionfollowingevaluationlargelanguage`

Introduces IFEval, a straightforward and reproducible evaluation benchmark for measuring LLMs' ability to follow natural language instructions using verifiable criteria, avoiding the biases of human or LLM-based evaluation.

---

### Improving Language Understanding by Generative Pre-Training
**Key:** `zotero-item-425`

The original GPT paper demonstrating that generative pre-training of a language model on a diverse corpus of unlabeled text, followed by discriminative fine-tuning on specific tasks, achieves large gains on a range of NLP benchmarks.
