Project context

This project studies privileged-context on-policy distillation for mathematical reasoning. The target domain is AIME-style math reasoning. The student models are small Qwen no-think variants, and the teachers are larger Qwen instruct models.

The main goal is:

Extract as much useful knowledge as possible from a much larger teacher, even when the teacher is around 100× larger than the student.

The core concern is the capacity gap: a very large teacher may know more, but its local predictions may be too different, too compressed, or not sufficiently visible to the smaller student under the chosen distillation objective.

Model pairs

Student models:
	•	HuggingFaceTB/Qwen3-4B-no-think
	•	HuggingFaceTB/Qwen3-0.6B-no-think

Teacher models:
	•	Qwen/Qwen3-4B-Instruct-2507
	•	Qwen/Qwen3-30B-A3B-Instruct-2507
	•	Qwen/Qwen3-235B-A22B-Instruct-2507

Main student-teacher combinations:
	•	4B-no-think student ← 30B teacher
	•	4B-no-think student ← 235B teacher
	•	0.6B-no-think student ← 4B teacher
	•	0.6B-no-think student ← 30B teacher
	•	0.6B-no-think student ← 235B teacher

The *-no-think students are Qwen models with modified chat templates that force no-think mode. Training targets should not include <think> tags or hidden reasoning.

Data and artifacts

Dataset:
	•	zwhe99/DeepMath-103K

Available artifacts:
	•	DeepMath prompts.
	•	Cached teacher completions for:
	•	Qwen3-4B-Instruct-2507
	•	Qwen3-30B-A3B-Instruct-2507
	•	Qwen3-235B-A22B-Instruct-2507
	•	Existing implementation for adding privileged context to teacher inputs.
	•	Existing successful top-1 OPD runs for all student-teacher combinations.
	•	vLLM infrastructure for teacher next-token logprobs.
	•	Ability to load the student locally and compute full student next-token distributions.

Core notation

For each problem:

x

is the original problem prompt.

z

is privileged context. It is visible only to the teacher.

\hat y

is a student on-policy rollout sampled from the student.

At token position t:

\hat y_{<t}

is the current student prefix.

The student distribution is:

\pi_S(\cdot \mid x,\hat y_{<t})

The teacher distribution is:

q_T(\cdot \mid x,z,\hat y_{<t})

The teacher sees privileged context z. The student never sees z.

Privileged-context OPD

The setup is inspired by privileged-context/self-distillation papers, but with different student and teacher models rather than self-distillation.

The teacher is not simply writing a full solution. Instead, the teacher defines a local next-token distribution at the student’s own prefix:

q_T(\cdot \mid x,z,\hat y_{<t})

This matters because distillation happens on student-induced states, not on teacher-written trajectories.

OPD

OPD = on-policy distillation.

The student generates rollouts. Then the teacher provides labels or distributions at the prefixes the student actually visits.

This is different from ordinary SFT on teacher-generated full solutions.

Top-1 OPD

In top-1 hard-label OPD, the teacher label is:

a_t = \arg\max_v q_T(v \mid x,z,\hat y_{<t})

The student trains on:

-\log \pi_S(a_t \mid x,\hat y_{<t})

This uses only the teacher’s single most likely token.

Key diagnostic finding

Top-1 OPD is a major information bottleneck.

The 30B and 235B teachers agree on top-1 around 85–90% of the time. This means that top-1 hard labels hide most of the 235B teacher’s additional information.

Teacher top-K

Teacher top-K means the set of K highest-probability next tokens under the teacher:

T_K = \text{TopK}(q_T)

The project focused on:

K \in \{10,50,100\}

For training, if using K=50, ask vLLM for exactly top_logprobs=50. Do not ask for 100 if only using 50, because teacher top-logprobs are expensive.

Forward KL

Forward KL is:

D_{KL}(q_T \parallel \pi_S)
=
\sum_v q_T(v)\log\frac{q_T(v)}{\pi_S(v)}

Equivalent training term:

-\sum_v q_T(v)\log \pi_S(v)

This is teacher-support-weighted. Teacher top-K diagnostics are naturally relevant to forward KL because the expectation is over the teacher distribution.

In this project, forward KL is not the main objective. It can be used as an ablation, but the literature and current setup favor reverse KL for on-policy distillation.

Reverse KL

Reverse KL is the main objective of interest:

D_{KL}(\pi_S \parallel q_T)
=
\sum_v \pi_S(v)\log\frac{\pi_S(v)}{q_T(v)}

It is student-support-weighted. This means the important question is:

Where does the student place probability mass, and how does the teacher score those tokens?

Reverse KL is mode-seeking and often preferred in on-policy distillation literature.

Bucketed top-K reverse KL

Because arbitrary teacher token scoring is not available, the project uses a teacher top-K plus OTHER bucket approximation.

For teacher top-K:

T_K = \{v_1,\ldots,v_K\}

Teacher mass on explicit tokens:

q_T(v), \quad v\in T_K

Student mass on explicit tokens:

\pi_S(v), \quad v\in T_K

OTHER bucket:

q_T(\text{OTHER}) = 1 - \sum_{v\in T_K} q_T(v)

\pi_S(\text{OTHER}) = 1 - \sum_{v\in T_K} \pi_S(v)

Bucketed reverse KL loss:

\mathcal L_K =
\sum_{v\in T_K}
\pi_S(v)
\left[
\log \pi_S(v)-\log q_T(v)
\right]
+
\pi_S(\text{OTHER})
\left[
\log \pi_S(\text{OTHER})-\log q_T(\text{OTHER})
\right]

Important implementation points:
	•	Do not renormalize teacher top-K to sum to 1.
	•	Always include the OTHER bucket.
	•	Teacher logprobs are detached.
	•	Student probabilities remain differentiable, including \pi_S(\text{OTHER}).
	•	top_logprobs requested from teacher should equal K during training.
	•	K=50 is the current recommended default.

OTHER bucket

The OTHER bucket represents all tokens not in teacher top-K.

It is necessary because reverse KL requires support for all tokens where the student has probability mass. Since teacher probabilities are only known for top-K tokens, all remaining teacher mass and student mass are collapsed into OTHER.

If K is large enough and teacher top-K covers most student mass, OTHER is small and the approximation is good.

Key RKL support diagnostic result

Teacher top-K covers almost all student probability mass:

K	Mean student mass covered
10	~0.948
50	~0.993
100	~0.997

Conclusion:
	•	K=10 is already strong.
	•	K=50 is the practical default.
	•	K=100 adds little beyond K=50.
	•	Arbitrary teacher token scoring is not needed for the next experiments.

Capacity gap

The capacity gap is the mismatch between teacher and student capabilities.

Initial concern:

The 235B teacher may be too large and too far from the 0.6B or 4B student.

Diagnostic conclusion:
	•	Under top-1 OPD, 235B advantage is mostly hidden.
	•	Under reverse KL support diagnostics, 235B is not less compatible with the student.
	•	235B assigns higher log-probability than 30B to student-supported tokens, especially for the 0.6B student.

So the capacity gap appears to be less about support mismatch and more about the loss/objective bottleneck.

Privileged context / PC

PC = privileged context.

The teacher receives z. The student does not.

Privileged context comes from cached teacher completions / DeepMath assistant messages, truncated to about 4000 characters in diagnostics.

Diagnostics showed:
	•	PC changes teacher top-1 in about 11–14% of prefixes.
	•	Most PC-induced top-1 changes land on reachable student tokens.
	•	Under reverse KL, PC has only a small effect on coverage because teacher top-K already covers most student mass.

Conclusion:
	•	Keep PC.
	•	PC is useful, especially for local top-1 decisions.
	•	But PC is not the main lever for top-K reverse KL; the main lever is using the teacher distribution instead of top-1.

Prompt levels

Three teacher prompt levels were tested.

L1

Distillation-only prompt.

Contains:
	•	student model ID;
	•	teacher model ID;
	•	top-1 OPD setup;
	•	no-think student;
	•	privileged context;
	•	current student prefix.

Does not contain domain or failure-mode information.

L2

L1 plus math-domain guidance.

Adds:
	•	mathematical reasoning domain;
	•	AIME-style downstream target;
	•	guidance about concise, checkable math reasoning.

L3

L2 plus student failure modes.

Adds broad failure-mode summaries, such as:
	•	misses constraints;
	•	arithmetic/algebra errors;
	•	shallow pattern matching;
	•	overcommits to wrong intermediate conclusions.

Prompt sensitivity

Prompt sensitivity measures whether L1/L2/L3 change the teacher’s local predictions.

Top-1 prompt sensitivity:
	•	Around 4–6%.
	•	Top-10 / top-K differences are also small.

Reverse-KL support sensitivity:
	•	Coverage differences across prompts are ≤ about 0.01.

Conclusion:
	•	GEPA/prompt optimization is low priority.
	•	The main lever is the objective: top-1 → bucketed top-K reverse KL.

GEPA

GEPA = Genetic-Pareto prompt optimizer / DSPy optimizer.

Originally considered to optimize teacher prompts.

Current conclusion:
	•	Not worth prioritizing now.
	•	Diagnostics show prompt wording has low leverage.
	•	Revisit only after top-K reverse KL training is implemented and evaluated.

Teacher-support diagnostics

These were the first diagnostic round.

They looked at:
	•	teacher top-1 agreement;
	•	teacher top-10 overlap;
	•	top-10 rescue tokens;
	•	PC-induced top-1 changes;
	•	prompt sensitivity;
	•	student reachability of teacher top-1.

They are most relevant to:
	•	top-1 hard OPD;
	•	forward-KL-like top-K thinking.

Key conclusion:

Top-1 OPD hides most of the 235B advantage. Top-K teacher distributions contain much more useful signal.

Reverse-KL support diagnostics

These were the second diagnostic round.

They looked at:
	•	student probability mass inside teacher top-K;
	•	student top-M overlap with teacher top-K;
	•	bucketed RKL approximation;
	•	30B vs 235B mass-weighted teacher advantage on student support;
	•	PC effect on student support;
	•	RKL-visible rescue mass.

Key conclusion:

Top-K reverse KL is well-supported. Teacher top-K covers almost all student mass, and 235B looks better than 30B on student-supported tokens.

Student mass in teacher top-K

Definition:

\sum_{v\in T_K}\pi_S(v)

This is the most important reverse-KL support metric.

It answers:

How much of the student distribution is visible inside the teacher’s returned top-K?

Observed:
	•	~94.8% at K=10
	•	~99.3% at K=50
	•	~99.7% at K=100

Top-1 bottleneck

Definition:

\Pr[
\arg\max q_{30B} = \arg\max q_{235B}
]

Observed:
	•	30B and 235B agree on top-1 around 85–90%.

Conclusion:

Top-1 hard-label OPD cannot express most of the 235B teacher’s advantage.

Top-K rescue

Teacher-support rescue means:

The 235B top-1 may match 30B or be unhelpful, but 235B top-K contains a reachable non-student-top-1 alternative.

This was high in the first diagnostic:
	•	about 40% for 4B student;
	•	about 62–64% for 0.6B student.

But RKL-visible rescue is mass-weighted by the student:

\sum_{v\in T_K,\ v\ne \arg\max\pi_S,\ \text{reachable}(v)}
\pi_S(v)

Observed:
	•	about 0.12 at K=10;
	•	about 0.16 at K=50/100.

Conclusion:
	•	The rescue signal is real.
	•	But the first rescue metric was partly forward-KL-flavored.
	•	Under RKL, rescue contributes a bounded but meaningful part of the gradient.

Reachability

A teacher token is considered reachable if the student NLL is in a band such as:

0.25 \le -\log \pi_S(v) \le 8.0

Too close:

-\log \pi_S(v) < 0.25

Too far:

-\log \pi_S(v) > 8.0

Reachability was used in diagnostics, but it is heuristic.

For bucketed reverse KL training, hard reachability filtering is not currently the main recommendation, because RKL already weights by student probability and top-K coverage is high.

Student top-M

Student top-M means:

S_M = \text{TopM}(\pi_S)

It is useful for diagnostics, especially reverse-KL support analysis.

But it should not define the main training loss unless the teacher can score arbitrary student tokens.

Since arbitrary teacher token scoring is not available, training should use:

T_K \cup \{\text{OTHER}\}

not student top-M.

Teacher top-logprobs

During training:
	•	For K=10, request top_logprobs=10.
	•	For K=50, request top_logprobs=50.
	•	For K=100, request top_logprobs=100.

Do not request more than K during training because extra teacher logprobs are expensive.

Earlier diagnostics requested top-logprobs=100 to evaluate K ∈ {10,50,100}; that was for diagnostics, not training.

Caching

Full teacher top-K caching is not required for heavy training.

Recommended modes:
	•	none: default for heavy training. Stream teacher top-K calls and discard token-level outputs after computing the loss.
	•	debug: cache a small sampled subset for inspection.
	•	full: cache everything only for diagnostics or small probe runs.

Reason:
	•	Heavy on-policy training could create enormous caches.
	•	Student prefixes change over time.
	•	Storing every teacher top-K result is expensive.

Recommended implementation default

For first serious training:

loss_type: bucketed_rkl
K: 50
teacher_top_logprobs: 50
with_privileged_context: true
teacher_topk_cache_mode: none

Recommended next experiments

Main experiment:

\text{235B privileged-context bucketed reverse KL, K=50}

Compare against:
	•	current top-1 OPD baseline;
	•	30B teacher bucketed RKL K=50;
	•	K=10 bucketed RKL;
	•	optionally K=100 bucketed RKL.

Minimal matrix:

Student	Teacher	Objective	K	PC
0.6B-no-think	235B	top-1 OPD	1	yes
0.6B-no-think	235B	bucketed RKL	10	yes
0.6B-no-think	235B	bucketed RKL	50	yes
0.6B-no-think	30B	bucketed RKL	50	yes
4B-no-think	235B	top-1 OPD	1	yes
4B-no-think	235B	bucketed RKL	10	yes
4B-no-think	235B	bucketed RKL	50	yes
4B-no-think	30B	bucketed RKL	50	yes

Most important conclusion

The diagnostics support this thesis direction:

The 235B teacher does contain more useful information than the 30B teacher, but top-1 hard-label OPD hides it. Bucketed top-K reverse KL exposes the 235B teacher’s distributional advantage while remaining student-compatible, because teacher top-K covers almost all of the student’s probability mass.

The next decisive question is empirical:

Does K=50 bucketed reverse KL with the 235B privileged teacher improve downstream math performance compared to top-1 OPD and 30B K=50 RKL?