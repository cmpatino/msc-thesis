You are working in an existing research repository for privileged-context on-policy distillation experiments.

Your task is to implement and run a diagnostic-first experiment. Do not implement GEPA prompt optimization yet. Do not run probe training. Do not launch full OPD training. The goal is to measure whether prompt optimization and/or top-k distillation are likely to be useful before we spend compute on them.

The user will inspect the diagnostic report and then decide how to proceed.

Context
=======

We are studying privileged-context on-policy distillation inspired by arXiv:2601.19897, but with different teacher and student models rather than self-distillation.

For each problem x:

- The student sees only x.
- The teacher sees x plus privileged context z.
- The student first samples an on-policy rollout:

    y_hat ~ pi_S(. | x)

- At token position t, the on-policy prefix is:

    prefix_t = y_hat_<t

- The student distribution is:

    pi_S(. | x, prefix_t)

- The privileged teacher distribution under teacher-conditioning prompt p is:

    q_T^p(. | x, z, prefix_t)

- In current top-1 hard-label OPD, the training label is:

    a_t = argmax_v q_T^p(v | x, z, prefix_t)

- The student is trained on:

    -log pi_S(a_t | x, prefix_t)

The privileged context z is never shown to the student. It must not be copied into the student training target. It is only used to condition the teacher distribution at student-induced prefixes. You can see an example of the implementation in @trl/experimental/distillation/distillation_trainer.py, line 1635.

The key scientific question for this diagnostic pass is:

Can the 235B teacher provide more useful and extractable privileged-context signal than the 30B teacher, or does top-1 hard-label OPD discard most of that advantage?

Model pairs
===========

Run diagnostics for these student-teacher combinations:

1. Student: HuggingFaceTB/Qwen3-4B-no-think
   Teacher: Qwen3-30B-A3B-Instruct-2507

2. Student: HuggingFaceTB/Qwen3-4B-no-think
   Teacher: Qwen3-235B-A22B-Instruct-2507

3. Student: HuggingFaceTB/Qwen3-0.6B-no-think
   Teacher: Qwen3-4B-Instruct-2507

4. Student: HuggingFaceTB/Qwen3-0.6B-no-think
   Teacher: Qwen3-30B-A3B-Instruct-2507

5. Student: HuggingFaceTB/Qwen3-0.6B-no-think
   Teacher: Qwen3-235B-A22B-Instruct-2507

If the repo already defines aliases or full Hugging Face model IDs, use the repo’s definitions. If missing, resolve teacher IDs under the Qwen namespace, for example:

- Qwen/Qwen3-4B-Instruct-2507
- Qwen/Qwen3-30B-A3B-Instruct-2507
- Qwen/Qwen3-235B-A22B-Instruct-2507

The `*-no-think` student models are Qwen models with a modified template that forces no-think mode. Never add `<think>` tags to student rollouts, teacher labels, or prompts.

Available infrastructure
========================

You have access to

1. Cached completions for all DeepMath prompts for:
   - Qwen3-235B-A22B-Instruct-2507: HuggingFaceTB/DeepMath-103K-Qwen3-235B-A22B-Instruct-2507
   - Qwen3-30B-A3B-Instruct-2507: HuggingFaceTB/DeepMath-103K-Qwen3-30B-A3B-Instruct-2507
   - Qwen3-4B-Instruct-2507: HuggingFaceTB/DeepMath-103K-Qwen3-4B-Instruct-2507
2. A working implementation for adding privileged context to the teacher. Check `use_privileged_context` and `teacher_prompt_template` from trl/experimental/distillation/distillation_config.py.
3. Successful OPD runs for all student-teacher combinations. Check dev/recipes/KD-Capacity-Gap/distillation/config_v01.00.yaml for OPD and dev/recipes/KD-Capacity-Gap/distillation/config_v07.00.yaml for OPD + PC.
4. Access to vLLM for generating student rollouts.
5. Access to a vLLM server for teacher next-token logprobs/top-k logprobs. You launch a vllm server and use `get_sequence_logprobs` in trl/generation/vllm_client.py.
6. Ability to load the student model into GPU memory and compute the full next-token distribution.

Use the existing privileged-context implementation as the source of truth. Do not invent a new privileged-context format unless the existing one cannot be reused.

You have access to 1 H100 GPU, but can launch jobs for more compute heavy tasks using SLURM. You can check an example of the resources for the teachers in dev/slurm/kd_capacity_gap_array.slurm.

Main task
=========

Implement and run diagnostics that answer these questions:

1. Does the teacher-conditioning prompt actually affect the teacher’s local top-1 labels?
   - Run L1, L2, and L3 prompts on the same 1,000 student prefixes.
   - Measure how often the teacher top-1 token changes across L1/L2/L3.
   - Measure how much the teacher top-10 distribution changes across L1/L2/L3.

2. How much does top-1 hard-label OPD limit the extractable advantage of the 235B teacher?
   - Compare 30B vs 235B on the same student, same problems, same privileged context, same student prefixes, and same prompt level.
   - Measure how often 30B and 235B have the same top-1 token.
   - Measure whether their top-10 distributions differ even when top-1 is the same.
   - Measure whether the 235B top-1 token is reachable by the student.
   - Measure whether useful 235B alternatives appear in top-10 even when top-1 is too far or identical to 30B.

3. Does privileged context affect local teacher decisions?
   - Compare teacher distributions with privileged context z versus without z on the same prefixes.
   - Measure top-1 change rate and top-10 distribution shift.
   - Compare whether 235B responds more usefully to privileged context than 30B.

4. Are the teacher labels learnable by the student?
   - For teacher top-1 and top-10 tokens, compute student logprob, NLL, rank, and reachability under the full student distribution.
   - Report whether 235B is more corrective but too far from the student.

Do not optimize prompts yet. Do not train models yet. Only implement and run diagnostics.

Prompt levels
=============

Implement three teacher-conditioning prompt levels. These prompts condition q_T^p(. | x, z, prefix_t). They are not prompts for generating complete standalone teacher solutions.

L1: Distillation-only teacher-conditioning prompt
------------------------------------------------

Allowed information:
- student model ID;
- teacher model ID;
- top-1 hard-label OPD setup;
- no-think student;
- problem x;
- privileged context z;
- current student prefix prefix_t.

Not allowed:
- explicit math/AIME domain description;
- explicit student failure-mode summary.

Seed prompt:

"""
You are the privileged-context teacher in a top-1 hard-label on-policy distillation step.

The student is {student_model_id}. The student sees only the original problem and its own current answer prefix. The student uses a no-think chat template, so labels must correspond to visible answer text only. Do not favor hidden-reasoning tags such as <think>.

The teacher is {teacher_model_id}. You see the same problem and prefix as the student, plus privileged context that the student will not see. Use the privileged context to form a better next-token distribution at the student's current prefix.

Important:
- Do not copy the privileged context verbatim.
- Do not mention that privileged context exists.
- Do not restart the solution unless the prefix has clearly failed and needs repair.
- Prefer the next token or continuation that a better, learnable version of the student should produce from this exact prefix.
- Stay close enough to the student's style that the student can imitate the correction.
- Use the privileged context to make local corrective decisions.

Continue from the student's prefix.
"""

L2: Distillation + math-domain teacher-conditioning prompt
---------------------------------------------------------

Allowed information:
- everything in L1;
- the domain is mathematical reasoning;
- target downstream behavior is AIME-style mathematical problem solving;
- mathematical-validity and answer-format guidance.

Not allowed:
- explicit student failure-mode summary.

Seed prompt:

"""
You are the privileged-context teacher in a top-1 hard-label on-policy distillation step.

The student is {student_model_id}. The student sees only the original problem and its own current answer prefix. The student uses a no-think chat template, so labels must correspond to visible answer text only. Do not favor hidden-reasoning tags such as <think>.

The teacher is {teacher_model_id}. You see the same problem and prefix as the student, plus privileged context that the student will not see. Use the privileged context to form a better next-token distribution at the student's current prefix.

The domain is mathematical reasoning, with the downstream target being AIME-style problem solving. The best teacher behavior is not to be maximally verbose. It is to make the local next-token decision that moves the student's current solution toward a short, correct, checkable mathematical argument.

At the current prefix:
- preserve correct partial reasoning when possible;
- repair incorrect reasoning when needed;
- favor decisive mathematical moves over boilerplate;
- make missing constraints explicit when they matter;
- avoid advanced jumps that the student is unlikely to imitate;
- avoid hidden-reasoning tags;
- favor final-answer formatting such as \\boxed{...} when the solution is ready to conclude.

Do not copy or mention the privileged context. Continue from the student's prefix.
"""

L3: Distillation + math-domain + student-failure-mode prompt
-----------------------------------------------------------

Allowed information:
- everything in L2;
- a concise failure-mode summary generated only from this student's training/probe rollouts.

Seed prompt:

"""
You are the privileged-context teacher in a top-1 hard-label on-policy distillation step.

The student is {student_model_id}. The student sees only the original problem and its own current answer prefix. The student uses a no-think chat template, so labels must correspond to visible answer text only. Do not favor hidden-reasoning tags such as <think>.

The teacher is {teacher_model_id}. You see the same problem and prefix as the student, plus privileged context that the student will not see. Use the privileged context to form a better next-token distribution at the student's current prefix.

The domain is mathematical reasoning, with the downstream target being AIME-style problem solving.

Known failure modes of this student, estimated only from training/probe rollouts:
{student_failure_modes}

At the current prefix:
- preserve correct partial reasoning when possible;
- repair the likely mistake when the prefix shows one of the known failure modes;
- include the missing constraint check when the student tends to skip it;
- favor the decisive modular, algebraic, geometric, or counting move when that is the bottleneck;
- avoid unnecessarily sophisticated continuations that are correct but not learnable by this student;
- avoid hidden-reasoning tags;
- favor final-answer formatting such as \\boxed{...} when the solution is ready to conclude.

Do not copy or mention the privileged context. Continue from the student's prefix.
"""

Failure-mode summaries for L3
=============================

If the repo already has student failure-mode summaries, reuse them.

Otherwise, create short provisional summaries for the two students using existing rollout/eval logs if available. Do not spend large compute here.

Suggested default L3 summaries:

For HuggingFaceTB/Qwen3-4B-no-think:

"""
- Sometimes follows a plausible solution path without checking hidden constraints such as integrality, positivity, range, or uniqueness.
- Can make arithmetic or algebraic simplification errors after identifying the right method.
- May skip a decisive modular, parity, or case-splitting observation when the problem requires it.
- Can produce an answer that is locally plausible but not verified against the original conditions.
"""

For HuggingFaceTB/Qwen3-0.6B-no-think:

"""
- Often starts with a plausible but shallow pattern instead of identifying the decisive mathematical structure.
- Frequently makes algebraic, arithmetic, or variable-tracking mistakes.
- Often misses modular, parity, counting, or casework constraints.
- May overcommit to an incorrect intermediate conclusion and continue confidently.
- Needs local corrections that are simple, explicit, and close to its current prefix.
"""

Save final summaries used in the run.

Data and prefix sampling
========================

Use DeepMath-103K or the repo’s existing math prompt split.

Create a fixed diagnostic prefix set.

Default target:

- 1,000 student prefixes per student model.

For each student model:

1. Select problems from the training/probe split only.
2. Generate or load student on-policy rollouts without privileged context.
3. Select informative prefixes from those rollouts.

Prefix selection should avoid scoring every token. Prefer prefixes where prompt and privileged context can plausibly matter:

- line boundaries;
- sentence boundaries;
- after equations;
- after words like "therefore", "so", "hence", "we need", "the key", "case", "mod", "congruent";
- before final-answer formatting;
- positions with high student entropy;
- positions where the student is likely wrong if verifier/eval logs are available;
- a small random sample for coverage.

If student entropy is needed for prefix selection, first compute student full next-token distribution at candidate prefixes and cache it.

Keep the same 1,000 prefixes fixed across all prompts and teachers for a given student.

If 1,000 prefixes is too expensive for smoke mode, use 8-32 prefixes. The full diagnostic mode should default to 1,000 prefixes.

Teacher logprob requirements
============================

Use the vLLM teacher server to obtain next-token top-k logprobs at each teacher-conditioned input.

For each query, request:

- max_tokens = 1
- temperature = 0
- top_logprobs = 10 or equivalent
- return the generated top-1 token and the top-10 logprobs

If the exact vLLM client/API differs, inspect the repo and use the existing teacher-logprob client.

For every teacher-prefix-prompt-context query, store:

- teacher_top1_token_id
- teacher_top1_token_text
- teacher_top1_logprob if available
- teacher_top10_token_ids
- teacher_top10_token_texts
- teacher_top10_logprobs
- whether top10 logprobs are normalized or raw logprobs
- backend metadata

If the teacher server can score arbitrary token IDs in addition to top-k, implement optional scoring of:
- the other teacher’s top-1 token;
- the student’s top-1 token.

If arbitrary-token scoring is unavailable, continue with top-10 membership metrics and mark arbitrary scoring unavailable.

Student distribution requirements
=================================

Load the student model into GPU memory and compute the full next-token distribution for each student prefix.

For every prefix, store:

- student_top1_token_id
- student_top1_token_text
- student_top10_token_ids
- student_top10_token_texts
- student_top10_logprobs
- student_entropy_full normalized by log(vocab_size)
- student_logprob for every teacher top-1 token
- student_NLL for every teacher top-1 token
- student_rank for every teacher top-1 token, if feasible
- student_logprob for every token in each teacher top-10
- student_NLL for every token in each teacher top-10
- student_rank for every token in each teacher top-10, if feasible

If exact rank over the full vocabulary is expensive, compute rank by sorting logits once per prefix and cache the sorted order or rank map. If still too expensive, compute:
- whether token is in student top-10;
- whether token is in student top-100;
- whether token is in student top-1000;
- student NLL.

Teacher input construction
==========================

Implement a single source-of-truth function:

    build_teacher_conditioned_input(
        problem: str,
        privileged_context: Optional[str],
        student_prefix: str,
        teacher_prompt: str,
        student_model_id: str,
        teacher_model_id: str,
        prompt_level: str,
    ) -> TeacherInput

The teacher input must clearly separate:

- the original problem x;
- the privileged context z, when present;
- the current student prefix prefix_t;
- the instruction prompt p.

For privileged-context ablations, call the same function with privileged_context=None or an explicit empty/no-context placeholder.

The teacher must be asked to continue from the exact student prefix. It must not be asked to restart the solution.

Do not include privileged context in any student input.

Diagnostics to implement
========================

Implement the following diagnostic groups.

Diagnostic A: Prompt sensitivity
--------------------------------

For each student-teacher pair, compare L1, L2, and L3 on the exact same 1,000 prefixes.

Compute pairwise and all-way metrics:

- top1_change_rate_L1_L2
- top1_change_rate_L1_L3
- top1_change_rate_L2_L3
- all_same_top1_rate
- any_prompt_changes_top1_rate
- top10_jaccard_L1_L2
- top10_jaccard_L1_L3
- top10_jaccard_L2_L3
- top10_overlap_count mean
- top10_conditional_JS_L1_L2
- top10_conditional_JS_L1_L3
- top10_conditional_JS_L2_L3
- change rate specifically on high-student-entropy prefixes
- change rate specifically on failed/likely-wrong prefixes if available

Definitions:

top10_jaccard(A, B) =
    |top10_A ∩ top10_B| / |top10_A ∪ top10_B|

top10_conditional_JS:
    Normalize each teacher’s top-10 logprobs over its top-10 set.
    Build the union of both top-10 sets.
    Assign epsilon probability to missing tokens.
    Renormalize over the union.
    Compute Jensen-Shannon divergence.
    Mark this as an approximate top-10-conditional JS, not full-distribution JS.

Interpretation flags:

- If any_prompt_changes_top1_rate < 0.03, prompt optimization may have low leverage for top-1 OPD.
- If top1 change is low but top10_conditional_JS is high, top-1 OPD may be hiding prompt effects.
- If L3 changes top1 much more than L1/L2, student-failure-mode prompting may be useful.
- If all prompt levels are nearly identical, GEPA may be unlikely to help unless it discovers much stronger prompt changes.

Diagnostic B: 30B vs 235B teacher comparison
--------------------------------------------

For student HuggingFaceTB/Qwen3-4B-no-think:

- compare Qwen3-30B-A3B-Instruct-2507 vs Qwen3-235B-A22B-Instruct-2507.

For student HuggingFaceTB/Qwen3-0.6B-no-think:

- compare Qwen3-30B-A3B-Instruct-2507 vs Qwen3-235B-A22B-Instruct-2507.
- also compare Qwen3-4B-Instruct-2507 vs Qwen3-30B-A3B-Instruct-2507.
- also compare Qwen3-4B-Instruct-2507 vs Qwen3-235B-A22B-Instruct-2507.

For each comparison, same prompt level, same x, same z, same student prefix:

Top-1 metrics:

- same_top1_rate
- different_top1_rate
- hard_label_identical_rate = same_top1_rate
- 235B_top1_student_NLL_mean
- 30B_top1_student_NLL_mean
- 235B_top1_student_rank_median
- 30B_top1_student_rank_median
- 235B_top1_reachable_rate
- 30B_top1_reachable_rate
- 235B_top1_too_close_rate
- 30B_top1_too_close_rate
- 235B_top1_too_far_rate
- 30B_top1_too_far_rate

Top-10 metrics:

- top10_jaccard
- top10_overlap_count
- top10_conditional_JS
- 235B_top1_in_30B_top10_rate
- 30B_top1_in_235B_top10_rate
- student_top1_in_235B_top10_rate
- student_top1_in_30B_top10_rate
- 235B_top10_reachable_mass
- 30B_top10_reachable_mass
- 235B_top10_expected_student_NLL
- 30B_top10_expected_student_NLL
- 235B_top10_min_student_NLL
- 30B_top10_min_student_NLL
- 235B_top10_contains_reachable_alternative_rate
- 30B_top10_contains_reachable_alternative_rate

Definitions:

teacher_top10_reachable_mass:
    Normalize teacher top-10 logprobs over the top-10 set.
    Sum the normalized probability of teacher top-10 tokens whose student NLL is in the reachable band.

teacher_top10_expected_student_NLL:
    Normalize teacher top-10 logprobs over the top-10 set.
    Compute sum_j q10_j * [-log pi_S(token_j | x, prefix_t)].

teacher_top10_min_student_NLL:
    Minimum student NLL among the teacher top-10 tokens.

teacher_top10_contains_reachable_alternative:
    True if teacher top-1 is too far but at least one token in teacher top-10 is reachable.

Reachability defaults:

- too_close if student_NLL < 0.25
- reachable if 0.25 <= student_NLL <= 8.0
- too_far if student_NLL > 8.0

Make these thresholds configurable.

Interpretation flags:

- If same_top1_rate is high, top-1 hard OPD has a hard ceiling on how much extra 235B signal can be extracted.
- If same_top1_rate is high but top10_conditional_JS is also high, top-1 labels are discarding distributional differences.
- If 235B_top1_reachable_rate is lower than 30B_top1_reachable_rate, 235B may be better but less distillable.
- If 235B_top10_reachable_mass is high while 235B_top1_reachable_rate is low, top-k or softened distillation may extract more than top-1 OPD.
- If 235B_top1 differs often and remains reachable, then prompt optimization or better prefix selection may unlock 235B advantage.

Diagnostic C: Privileged-context influence
------------------------------------------

For each student-teacher-prompt-level combination, compare teacher distributions with privileged context z versus without privileged context.

Same x, same student prefix, same teacher prompt p.

Compute:

- context_top1_change_rate
- context_top10_jaccard
- context_top10_conditional_JS
- context_changes_to_reachable_token_rate
- context_changes_to_too_far_token_rate
- context_top1_student_NLL_before
- context_top1_student_NLL_after
- context_top10_reachable_mass_before
- context_top10_reachable_mass_after

Interpretation flags:

- If privileged context rarely changes top-1 or top-10, the current teacher-conditioning format may not use z effectively.
- If privileged context changes top-1 but mostly to too-far tokens, the teacher may use z in an unlearnable way.
- If 235B shows stronger useful context influence than 30B, there is potential to extract more from 235B.
- If 235B uses context more but becomes less reachable, top-k or better prompting may be needed.

Diagnostic D: Student-conditioned OPD utility
---------------------------------------------

For each teacher top-1 label a_t:

Compute:

- student_NLL_on_teacher_top1
- student_rank_on_teacher_top1
- student_entropy_full
- d_hat = robust normalized NLL
- h_hat = robust normalized entropy
- soft_or = h_hat + d_hat - h_hat * d_hat
- reachable
- too_close
- too_far

Calibration:

For each student model, build normalization stats on a fixed calibration subset:

- q05_nll
- q95_nll
- q05_entropy
- q95_entropy

Use these to compute d_hat and h_hat.

Q-bucket proxy:

- Q1: high h_hat, high d_hat
- Q2: high h_hat, low d_hat
- Q3: low h_hat, high d_hat
- Q4: low h_hat, low d_hat

Default thresholds:
- high >= 0.65
- low < 0.35

Report per teacher/prompt:

- mean_soft_or
- mean_reachable_soft_or = mean(soft_or * reachable)
- q1_rate
- q2_rate
- q3_rate
- q4_rate
- q3_reachable_rate
- too_close_rate
- too_far_rate

Interpretation flags:

- Q3-like reachable tokens are especially interesting: the student is relatively confident, but the privileged teacher changes the action.
- High soft_or with high too_far_rate is not necessarily useful.
- High Q4 means little training signal.
- High Q2 may indicate uncertainty but not teacher correction.
- High Q3 with reachability suggests useful corrective OPD labels.

Diagnostic E: Top-1 bottleneck score
------------------------------------

Create explicit summary metrics for the top-1 limitation.

For each pair of teachers, especially 30B vs 235B:

1. hard_label_same_rate:
    Fraction of prefixes where both teachers have the same top-1 token.

2. top10_diff_hidden_by_top1_rate:
    Fraction of prefixes where:
      - top-1 token is the same;
      - top10_conditional_JS >= configurable threshold, default 0.10.

   This estimates how often top-1 hard labels hide distributional differences between teachers.

3. larger_teacher_unreachable_top1_rate:
    Fraction of prefixes where:
      - 235B top-1 differs from 30B top-1;
      - 235B top-1 is too far under the student distribution.

4. larger_teacher_reachable_difference_rate:
    Fraction of prefixes where:
      - 235B top-1 differs from 30B top-1;
      - 235B top-1 is reachable.

5. top10_rescue_rate:
    Fraction of prefixes where:
      - 235B top-1 is too far or same as 30B top-1;
      - at least one 235B top-10 token is reachable and differs from the student top-1.

6. top1_extractable_advantage_proxy:
    larger_teacher_reachable_difference_rate

7. top10_extractable_advantage_proxy:
    larger_teacher_reachable_difference_rate + top10_rescue_rate

These are proxies, not proof of downstream improvement. Label them clearly as diagnostics.

Report these metrics prominently.

Implementation structure
========================

Everything should go inside the pc_experiments directory.

Adapt names to the repository conventions, but add equivalents of:

configs/privileged_opd_checks.yaml
    model pairs;
    prompt levels;
    dataset paths;
    vLLM server settings;
    student model loading settings;
    prefix sampling settings;
    cache paths;
    thresholds;
    run modes.

src/privileged_opd_checks/prompts.py
    L1/L2/L3 prompts;
    rendering;
    prompt-level access control;
    prompt hashing.

src/privileged_opd_checks/prefix_sampling.py
    load/generate student rollouts;
    select informative prefixes;
    save fixed prefix set.

src/privileged_opd_checks/student_distribution.py
    load student model;
    compute full next-token distribution;
    compute entropy;
    compute token logprob/NLL/rank for arbitrary token IDs;
    cache per-prefix student distributions.

src/privileged_opd_checks/teacher_distribution.py
    query vLLM teacher server;
    compute teacher top-1/top-10 logprobs;
    handle with-context and no-context calls;
    cache teacher outputs.

src/privileged_opd_checks/input_building.py
    build_student_input(problem, prefix);
    build_teacher_conditioned_input(problem, privileged_context, prefix, teacher_prompt, ...).

src/privileged_opd_checks/metrics.py
    top1 metrics;
    top10 metrics;
    student extractability metrics;
    OPD utility metrics;
    Q-bucket metrics;
    top-1 bottleneck metrics.

src/privileged_opd_checks/analysis.py
    aggregate metrics;
    produce interpretation flags;
    compare prompts;
    compare teachers;
    compare with-context vs no-context.

scripts/run_privileged_opd_checks.py
    orchestrator with modes:
      smoke
      build_prefixes
      compute_student_distributions
      compute_teacher_distributions
      analyze
      full_check

scripts/summarize_privileged_opd_checks.py
    regenerate report from cached metric files.

tests/test_privileged_opd_metrics.py
tests/test_prompt_access_control.py
tests/test_top10_metrics.py
tests/test_cache_keys.py

Caching requirements
====================

Cache aggressively. The diagnostics must be restartable.

Cache keys must include:

- student_model_id
- teacher_model_id
- prompt_level
- prompt_hash
- problem_id
- privileged_context_hash
- student_rollout_id
- prefix_text_hash
- prefix_token_index if available
- with_privileged_context boolean
- vLLM backend/server identifier
- decoding/logprob parameters
- tokenizer name/hash if available
- repo commit hash if easy to include

Cache:

- selected prefix set;
- student rollouts;
- student full logits or compressed logprobs/ranks;
- student entropy;
- teacher top-1/top-10 outputs;
- no-context teacher top-1/top-10 outputs;
- per-prefix metrics;
- aggregate summaries.

Store raw data in JSONL or parquet. Prefer parquet for large per-prefix metrics and JSON/CSV for summaries.

Do not recompute cached distributions unless `--overwrite_cache` is set.

Run modes
=========

Implement these modes.

1. smoke

Small sanity check.

Defaults:
- 1 student model;
- 1 teacher model;
- 1 prompt level;
- 2 problems;
- 4-8 prefixes total.

Must verify:
- student rollout generated without privileged context;
- teacher receives privileged context;
- no-context teacher ablation works;
- teacher top-10 logprobs are returned;
- student full distribution is computed;
- student NLL on teacher top-1 is computed;
- top10 metrics compute without error;
- report is written.

2. build_prefixes

Build the fixed prefix set for each student.

Defaults:
- 1,000 prefixes per student;
- fixed seed;
- save prefix_set.jsonl.

3. compute_student_distributions

Load each student into GPU memory and compute full next-token distribution for all prefixes.

Save:
- entropy;
- student top-10;
- enough information to compute NLL/rank for teacher tokens later.

If storing full logits for every prefix is too large, store:
- entropy;
- sorted token ranks or a token->rank map if feasible;
- top-k student tokens;
- and recompute arbitrary token logprobs later if needed.

4. compute_teacher_distributions

For each teacher, prompt level, prefix, and context condition:

- with privileged context;
- without privileged context;

query vLLM server for max_tokens=1 and top_logprobs=10.

Save teacher outputs.

5. analyze

Read cached student and teacher outputs.

Compute:
- Prompt sensitivity diagnostics.
- Teacher comparison diagnostics.
- Privileged-context influence diagnostics.
- Student-conditioned OPD utility diagnostics.
- Top-1 bottleneck diagnostics.

Write report.

6. full_check

Run all steps in order:

- build_prefixes;
- compute_student_distributions;
- compute_teacher_distributions;
- analyze.

Do not run training.

Outputs
=======

Write all outputs under:

experiments/privileged_opd_checks/{run_name}/

Required files:

1. report.md

Include:

- run name;
- repo commit hash;
- exact config;
- model pair table;
- number of problems, rollouts, and prefixes;
- prompt texts used;
- failure-mode summaries used for L3;
- cache hit/miss summary;
- smoke/full status;
- limitations.

Main sections:

A. Prompt sensitivity

For each student-teacher pair:
- any_prompt_changes_top1_rate;
- pairwise L1/L2/L3 top1 change;
- pairwise top10 Jaccard;
- pairwise top10 conditional JS;
- interpretation flag.

B. 30B vs 235B teacher comparison

For each shared student and prompt level:
- same_top1_rate;
- top10 conditional JS;
- 235B top1 reachable rate;
- 30B top1 reachable rate;
- 235B top10 reachable mass;
- 30B top10 reachable mass;
- top10 rescue rate;
- interpretation flag.

C. Privileged-context influence

For each teacher/prompt:
- context_top1_change_rate;
- context_top10 conditional JS;
- context changes to reachable token rate;
- context changes to too-far token rate;
- interpretation flag.

D. Student-conditioned OPD utility

For each student-teacher-prompt:
- mean soft_or;
- mean reachable soft_or;
- Q1/Q2/Q3/Q4 rates;
- Q3 reachable rate;
- too close/far rates.

E. Top-1 bottleneck

For each teacher comparison:
- hard_label_same_rate;
- top10_diff_hidden_by_top1_rate;
- larger_teacher_unreachable_top1_rate;
- larger_teacher_reachable_difference_rate;
- top10_rescue_rate;
- top1_extractable_advantage_proxy;
- top10_extractable_advantage_proxy.

F. Recommendations

Make recommendation flags, but do not decide future experiments automatically.

Use these guidelines:

- If prompt top1 sensitivity is >= 5%, prompt optimization may be worth trying for top-1 OPD.
- If prompt top1 sensitivity is < 3% but top10 JS is high, prompt optimization may matter only for top-k or soft distillation.
- If 235B and 30B have same top1 on > 80% of prefixes, top-1 OPD likely limits extractable 235B advantage.
- If 235B top1 reachable rate is much lower than 30B, 235B may be too far for the student under hard top-1 labels.
- If 235B top10 reachable mass is high but 235B top1 reachable rate is low, top-k distillation may be promising.
- If privileged context rarely changes top1 or top10, the privileged-context formatting/prompt may be ineffective.
- If privileged context changes to reachable corrective tokens, the setup has promising signal.

2. aggregate_metrics.csv

One row per student-teacher-prompt-level-context-condition with aggregate metrics.

3. prompt_sensitivity.csv

One row per student-teacher-prompt-pair comparison.

4. teacher_comparison.csv

One row per student-teacherA-teacherB-prompt-level comparison.

5. context_influence.csv

One row per student-teacher-prompt-level with-context vs no-context comparison.

6. top1_bottleneck.csv

One row per teacher comparison and prompt level.

7. per_prefix_metrics.parquet

Per-prefix diagnostic metrics.

8. prefix_set.jsonl

The fixed 1,000-prefix sample.

9. config_resolved.yaml

The full resolved config.

10. commands_rerun.sh

Exact commands to reproduce:
- smoke;
- build_prefixes;
- compute_student_distributions;
- compute_teacher_distributions;
- analyze;
- full_check.

Metrics details
===============

Implement helper functions carefully.

Top-10 conditional distribution
-------------------------------

Given teacher top-10 logprobs:

1. Convert logprobs to probabilities.
2. Normalize over the returned top-10 tokens only.
3. This is a conditional distribution over the returned top-10 set, not the full teacher distribution.
4. Name metrics clearly as `top10_conditional_*`.

Jensen-Shannon over top-10 union
--------------------------------

For two top-10 conditional distributions A and B:

1. Build token union U.
2. For tokens missing from a distribution, assign epsilon probability, default 1e-12.
3. Renormalize A and B over U.
4. Compute JS divergence.

Top-10 Jaccard
--------------

top10_jaccard(A, B) =
    |set(A_top10) ∩ set(B_top10)| / |set(A_top10) ∪ set(B_top10)|

Reachability
------------

For a token v under student distribution:

    nll_S(v) = -log pi_S(v | x, prefix_t)

Default thresholds:

- too_close: nll_S(v) < 0.25
- reachable: 0.25 <= nll_S(v) <= 8.0
- too_far: nll_S(v) > 8.0

These are configurable.

Student entropy
---------------

Compute:

    H(pi_S) = -sum_v pi_S(v) log pi_S(v)

Normalized:

    H_norm = H(pi_S) / log(vocab_size)

Use stable log-softmax.

OPD utility
-----------

For a teacher top-1 token a_t:

    d_t = -log pi_S(a_t | x, prefix_t)

Normalize by student-specific calibration quantiles:

    d_hat = clip((d_t - q05_nll) / (q95_nll - q05_nll + eps), 0, 1)

    h_hat = clip((H_norm - q05_entropy) / (q95_entropy - q05_entropy + eps), 0, 1)

Soft-OR:

    soft_or = h_hat + d_hat - h_hat * d_hat

Reachable soft-OR:

    reachable_soft_or = soft_or if reachable else 0

Q-buckets:

- Q1 if h_hat >= 0.65 and d_hat >= 0.65
- Q2 if h_hat >= 0.65 and d_hat < 0.35
- Q3 if h_hat < 0.35 and d_hat >= 0.65
- Q4 if h_hat < 0.35 and d_hat < 0.35
- otherwise mixed

Top-k extractability
--------------------

For teacher top-10 conditional distribution q10:

    expected_student_NLL = sum_v q10(v) * [-log pi_S(v)]

    reachable_mass = sum_v q10(v) * 1[reachable(v)]

    too_far_mass = sum_v q10(v) * 1[too_far(v)]

    too_close_mass = sum_v q10(v) * 1[too_close(v)]

Also compute:

    min_student_NLL_over_top10
    min_reachable_student_NLL_over_top10, if any reachable token exists
    contains_reachable_alternative =
        teacher_top1 is too_far AND any top10 token is reachable

Top-1 bottleneck metrics
------------------------

For larger teacher L = 235B and smaller teacher M = 30B:

    hard_label_same_rate =
        mean( top1_L == top1_M )

    top10_diff_hidden_by_top1_rate =
        mean( top1_L == top1_M AND top10_conditional_JS(L, M) >= 0.10 )

    larger_teacher_unreachable_top1_rate =
        mean( top1_L != top1_M AND top1_L is too_far under student )

    larger_teacher_reachable_difference_rate =
        mean( top1_L != top1_M AND top1_L is reachable under student )

    top10_rescue_rate =
        mean(
            (top1_L is too_far OR top1_L == top1_M)
            AND exists token v in L_top10 such that:
                v != student_top1
                AND v is reachable under student
        )

    top1_extractable_advantage_proxy =
        larger_teacher_reachable_difference_rate

    top10_extractable_advantage_proxy =
        larger_teacher_reachable_difference_rate + top10_rescue_rate

These are diagnostic proxies, not claims about final training performance.

Acceptance criteria
===================

Before finishing:

1. Implement the diagnostic code.
2. Add unit tests for:
   - prompt access control;
   - top-10 Jaccard;
   - top-10 conditional JS;
   - reachability buckets;
   - OPD utility formula;
   - top-1 bottleneck metrics;
   - cache-key construction.
3. Run the tests.
4. Run smoke mode.
5. If resources are available, run full_check on the default 1,000-prefix diagnostic set.
6. If full_check is too expensive or blocked, run as much as possible and clearly mark the blocker in report.md.
7. Always produce report.md, even for partial execution.

Do not silently skip failures. If the teacher vLLM server, student GPU load, tokenizer, privileged-context code, or dataset path is unavailable, write the exact blocker and the exact command/config needed to continue.

Final response
=========================

When done, report:

1. Files added/changed.
2. Commands executed.
3. Tests passed/failed.
4. Smoke mode status.
5. Full diagnostic status.
6. Path to report.md.
7. Most important diagnostic findings:
   - prompt top-1 sensitivity;
   - 30B vs 235B same-top1 rate;
   - top10 differences hidden by top1;
   - 235B reachability vs 30B reachability;
   - privileged-context influence;
   - top10 rescue rate.
8. Blockers, if any.
9. Exact next commands to rerun or extend the checks.

Important constraints
=====================

- Do not run GEPA.
- Do not run probe training.
- Do not launch full OPD training.
- Do not optimize prompts yet.
- Do not use final AIME evaluation data.
- Student rollouts must not receive privileged context.
- Teacher queries with privileged context must use the existing repo implementation.
- Compare all prompts and teachers on the exact same fixed prefixes.
- Cache every expensive distribution query.
- Report top-1 and top-10 metrics separately.
- Clearly distinguish full-distribution student metrics from top-10-conditional teacher metrics.
- Clearly label all top-k metrics as diagnostic proxies, not final proof of downstream learning.