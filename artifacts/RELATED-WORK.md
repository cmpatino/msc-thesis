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

### Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs?
**Key:** `kimWhyDoesSelfDistillation2026`

Finds that self-distillation in mathematical reasoning can reduce response length while degrading performance, tracing this to suppression of epistemic verbalization—the model's expression of uncertainty during reasoning. Shows that conditioning the teacher on rich information suppresses uncertainty expression, enabling rapid in-domain optimization but harming OOD performance with drops of up to 40% across Qwen3-8B, DeepSeek-Distill-Qwen-7B, and OLMo3-7B-Instruct. Highlights that exposing appropriate levels of uncertainty is crucial for robust reasoning beyond merely reinforcing correct answer traces.

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

### Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting
**Key:** `chenRetainingDoingRole2025`

Systematically compares forgetting patterns of SFT and RL post-training, finding RL leads to less forgetting than SFT while achieving comparable or higher task performance across Llama and Qwen model families. Identifies the mode-seeking nature of RL, which stems from its use of on-policy data, as the key factor enabling prior knowledge preservation during learning. Highlights the potential of approximately on-policy data for substantially more efficient forgetting mitigation.

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

### Speculative Knowledge Distillation: Bridging the Teacher-Student Gap Through Interleaved Sampling
**Key:** `xu2024SpeculativeKnowledgeDistillation`

Introduces Speculative Knowledge Distillation (SKD) that leverages cooperation between student and teacher to generate high-quality training data on-the-fly while aligning with the student's inference-time distribution. In SKD, the student proposes tokens and the teacher replaces poorly ranked ones based on its own distribution, transferring high-quality knowledge adaptively. Consistently outperforms existing KD methods across translation, summarization, math, and instruction following.

---

### Stable On-Policy Distillation through Adaptive Target Reformulation
**Key:** `jang2026StableOnPolicyDistillation`

Proposes Veto, an objective-level reformulation that constructs a geometric bridge in logit space to address training instabilities in on-policy KD caused by the wide distributional gap between novice student and expert teacher. Introduces a tunable parameter β that serves as both an Adaptive Gradient Veto (suppressing harmful gradients on low-confidence tokens) and a Decisiveness Knob (balancing reward-driven performance with output diversity). Consistently outperforms SFT and existing on-policy baselines across reasoning and generation tasks.

---

## Context Distillation

### General definition of context distillation

**A General Language Assistant as a Laboratory for Alignment**
**Key:** `askell2021GeneralLanguageAssistant`

This is the origin paper for the term **context distillation** in the LLM alignment setting. The paper frames the problem as taking a behavior induced by a long prompt/context and distilling it into model parameters, so the model approximates the prompted distribution without carrying the prompt at inference. It is especially useful for your story because their context is a behavioral/alignment prompt, not merely factual knowledge.

**Learning by Distilling Context**
**Key:** `snell2022LearningDistillingContext`

This is the canonical generalization of context distillation. It explicitly says that context tokens such as instructions, explanations, scratchpads, and examples improve behavior, but those gains disappear when the context is removed; context distillation is then used to internalize those gains into weights. For your setup, this is the cleanest citation for the abstract template: teacher sees richer context, student learns to behave as if it had that context.

#### 2. Off-policy context distillation: teacher has the context, student imitates teacher rollouts

**Propagating Knowledge Updates to LMs Through Distillation**
**Key:** `padmanabhan2023PropagatingKnowledgeUpdates`

This paper gives a very clear off-policy knowledge-update version of context distillation. It generates a transfer set by prompting the LM with an entity definition, then updates the student so its distribution matches the definition-conditioned teacher. It is useful because it states the teacher/student split explicitly: the teacher is the LM conditioned on the definition, while the student is updated to behave that way without the definition.

**Generative Prompt Internalization**
**Key:** `shin2025GenerativePromptInternalization`

This paper is highly relevant for fixed-prompt internalization. It argues that simply imitating prompted-teacher outputs can be an indirect and weak way to learn the prompt, so it trains the model not only to match the prompted behavior but also to generate the prompt content and the reason the behavior should change. This is a useful “off-policy but more explicit” variant for your story, especially if you want to discuss why teacher-output imitation may underspecify the hidden context.

**Prompt Baking**
**Key:** `bhargava2024PromptBaking`

This paper formulates prompt internalization as directly converting a prompt into a weight update. The objective is to make an unprompted “baked” model match the distribution of the original prompted model, which is very close to your “compare against just giving c as a system prompt” framing. It is particularly useful for behavioral prompts because the paper evaluates baking instructions and personas, not only task knowledge.

## 3. Different off-policy techniques for context distillation

**Prompt Injection: Parameterization of Fixed Inputs** / **Fixed Input Parameterization for Efficient Prompting**
**Key:** `choi2022PromptInjectionParameterization`

This is an early prompt-to-parameters formulation. The paper proposes injecting a fixed prompt into model parameters so the prompt no longer needs to be concatenated at inference, especially for long fixed prompts. It is not always framed as teacher-student distillation, but it is important background for the broader “fixed context becomes weights” technique family.

**PromptIntern: Saving Inference Costs by Internalizing Recurrent Prompt during Large Language Model Fine-tuning**
**Key:** `zou2024PromptInternSavingInference`

This paper focuses on repeated prompt templates and few-shot examples that appear across many calls. Its technique combines instruction template compression, few-shot example absorption, and progressive internalization, reducing the need for long recurring prompts at inference. It is a strong engineering-style off-policy reference for recurrent system prompts or templates.

**Updating Parametric Knowledge with Context Distillation Retains Post-Training Capabilities**
**Key:** `padmanabhan2026UpdatingParametricKnowledge`

This paper introduces **DiSC**, or Distillation via Split Contexts, as a context-distillation method for continual knowledge adaptation. It derives teacher and student distributions from different segments of a training example and matches them via KL on shared tokens, avoiding explicit generation during training. It is useful for your story because it addresses a likely concern in your experiments: whether internalizing context harms unrelated post-training capabilities.

**SIEVE: Sample-Efficient Parametric Learning from Natural Language**
**Key:** `asawa2026SIEVESampleEfficientParametric`

SIEVE is about making context distillation sample-efficient when the context is natural language instructions, knowledge, or feedback. Its key idea is to decompose the context into smaller applicable units, generate synthetic rollouts against the relevant pieces, and then use context distillation to internalize them. This is a good paper for the “different techniques” section because it treats context distillation as a data-generation and curriculum-design problem.

**MEND: Meta dEmonstratioN Distillation for Efficient and Effective In-Context Learning**
**Key:** `li2024MENDMetaDEmonstratioN`

MEND distills long demonstrations into compact vectors that can be reused for in-context learning. It is not the same as fully removing context at inference, because the student still consumes distilled vectors, but it is relevant for the “distill demonstrations / examples” branch of the literature. I would cite it as adjacent rather than central to your hidden-system-prompt setting.

## 4. On-policy distillation: privileged-teacher context vs student-visible context

**On-Policy Context Distillation for Language Models**
**Key:** `yeOnPolicyContextDistillation2026`

This is the key paper for the first on-policy route you describe: the student generates rollouts without the hidden context, while the teacher is context-conditioned and supplies token-level supervision. The student is trained on its own trajectories under a reverse-KL objective against the context-conditioned teacher. This is the paper I would use as the main citation for “privileged context is attached to the teacher, and the student learns it through teacher logprobs.”
