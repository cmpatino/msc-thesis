You are continuing work in the existing privileged-context OPD diagnostic branch.

Important: this is a follow-up diagnostic pass, not a training run.

The previous diagnostic pass focused on:
- teacher top-1 agreement;
- teacher top-10 rescue tokens;
- privileged-context influence;
- prompt sensitivity.

Those diagnostics were useful for top-1 OPD and teacher-support/top-k analysis, but they are not sufficient for reverse KL.

Current correction
==================

The training objective of interest is reverse KL:

    D_KL(pi_S(. | x, y_hat_<t) || q_T(. | x, z, y_hat_<t))

or approximations of it.

Reverse KL is an expectation under the student distribution. Therefore, diagnostics must focus on:

    Where does the student put probability mass, and how much of that mass is visible inside the teacher's returned top-k distribution?

Do not assume teacher top-k rescue tokens are useful for reverse KL just because they are reachable by an NLL threshold. A teacher top-k token with small student probability contributes little to reverse KL.

Goal of this follow-up
======================

Implement and run reverse-KL support diagnostics restricted to:

    K in {10, 50, 100}

The main question is:

    Can a top-{10,50,100} teacher distribution expose enough of the student-supported probability mass to make filtered/top-k reverse-KL distillation meaningful?

Also answer:

1. Does the 235B teacher cover more or less of the student's probability mass than the 30B teacher?
2. Does privileged context increase or decrease teacher support on student-likely tokens?
3. Are the previous top-k rescue findings actually visible under the student distribution?
4. Is top-100 enough, or do we need larger returned teacher top_logprobs / arbitrary token scoring in the future?

Important infrastructure constraint
===================================

We do NOT have a way to score arbitrary next-token candidates under the teacher. Check this is actually the case in trl/generation/vllm_client.py.

Therefore:
- do not require teacher logprob for arbitrary student top-M tokens;
- do not implement arbitrary candidate scoring;
- instead, request a larger teacher top_logprobs set from vLLM.

Use:

    requested_teacher_top_logprobs = 500

if vLLM supports it.

If 500 fails because of server/API/backend limits, fallback to:

    requested_teacher_top_logprobs = 100

and clearly report the fallback.

The formal diagnostic tables should be restricted to:

    K = 10, 50, 100

If top_logprobs=500 is available, additionally report a "returned_topN_coverage" diagnostic as an appendix / sanity check, but do not treat K=500 as a main experimental condition.

Use previous code where possible
================================

Reuse the existing code from the previous diagnostic implementation, including:
- prompt levels L1/L2/L3;
- student rollout/prefix sampling;
- privileged-context input construction;
- teacher vLLM server querying;
- student full next-token distribution computation;
- existing caches where valid;
- existing report structure and model-pair configs.

Do not reimplement the whole pipeline from scratch.

Add new modules/functions only for reverse-KL support analysis.

Do not run:
- GEPA;
- prompt optimization;
- probe training;
- full OPD training.

Use the same model pairs
========================

Run support diagnostics for the same model pairs:

1. Student: HuggingFaceTB/Qwen3-4B-no-think
   Teacher: Qwen/Qwen3-30B-A3B-Instruct-2507

2. Student: HuggingFaceTB/Qwen3-4B-no-think
   Teacher: Qwen/Qwen3-235B-A22B-Instruct-2507

3. Student: HuggingFaceTB/Qwen3-0.6B-no-think
   Teacher: Qwen/Qwen3-4B-Instruct-2507

4. Student: HuggingFaceTB/Qwen3-0.6B-no-think
   Teacher: Qwen/Qwen3-30B-A3B-Instruct-2507

5. Student: HuggingFaceTB/Qwen3-0.6B-no-think
   Teacher: Qwen/Qwen3-235B-A22B-Instruct-2507

Use the same L1/L2/L3 teacher-conditioning prompts from the previous diagnostic unless the repo has updated versions.

Use the same fixed prefix set
=============================

Prefer reusing the exact fixed prefixes from the previous full diagnostic run:

    pc_opd_checks_v01

If those prefixes are available, do not resample.

If they are unavailable, rebuild a comparable fixed prefix set:
- same students;
- same data source;
- same seeds;
- same prefix-selection heuristics;
- around 600 prefixes per student by default, or 1000 if the repo config already supports it.

The key requirement is that all teachers, prompt levels, and with/without privileged-context variants are evaluated on the exact same prefixes for a given student.

Teacher top-logprob collection
==============================

The previous run used top-10 teacher logprobs. That is not enough for this follow-up.

Add a mode or config option to query teacher top-logprobs with:

    max_tokens = 1
    temperature = 0
    top_logprobs = requested_teacher_top_logprobs

where requested_teacher_top_logprobs defaults to 500 and falls back to 100 if necessary.

For every teacher query, store:
- teacher_topN_token_ids
- teacher_topN_token_texts
- teacher_topN_logprobs
- requested_teacher_top_logprobs
- actual_returned_top_logprobs
- whether the backend returned fewer than requested
- whether logprobs appear to be full-vocabulary logprobs or only top-N conditional logprobs
- sum_exp_teacher_topN_logprobs
- backend metadata

Important:
vLLM/OpenAI-style top_logprobs are expected to be full-distribution logprobs for the returned tokens, not logprobs renormalized over the top-N. Verify this assumption where possible:
- convert returned logprobs to probabilities;
- check that sum over returned top-N is <= 1 plus small tolerance;
- if sum > 1.001, mark logprobs as not full-distribution-compatible and skip bucketed reverse-KL metrics that require q_other.

Student distribution requirements
=================================

Use the existing student full next-token distribution cache if valid.

For every prefix, we need:
- full student logprobs or logits;
- student top-10, top-50, top-100 token IDs;
- student top-10, top-50, top-100 probabilities/logprobs;
- student entropy;
- ability to compute pi_S(v) for every token returned by teacher top-N.

If the previous cache only stored student scores for teacher top-10 tokens, extend it or recompute full student distributions.

Do not give privileged context to the student.

Main K values
=============

All headline diagnostics must be reported for:

    K in {10, 50, 100}

For student top-M overlap, use:

    M in {10, 50, 100}

If teacher top_logprobs=500 is available, also compute:

    returned_topN_student_mass_covered

where N is actual_returned_top_logprobs. This is only a truncation/coverage sanity check, not a main K condition.

Diagnostics to implement
========================

Diagnostic A: Student mass covered by teacher top-K
---------------------------------------------------

For each prefix, teacher, prompt level, and context condition, define:

    T_K = teacher top-K token set
    pi = student next-token distribution

Compute:

    student_mass_in_teacher_topK@K =
        sum_{v in T_K} pi(v)

for K = 10, 50, 100.

Also compute:

    student_mass_outside_teacher_topK@K =
        1 - student_mass_in_teacher_topK@K

Report aggregate:
- mean;
- median;
- p10;
- p25;
- p75;
- p90;
- fraction >= 0.25;
- fraction >= 0.50;
- fraction >= 0.75;
- fraction >= 0.90.

Interpretation:
- If student_mass_in_teacher_top100 is low, then teacher top-100 does not expose enough of the student distribution for reverse KL to be accurately approximated.
- If 235B has lower coverage than 30B, that is direct evidence of capacity-gap mismatch under reverse KL.
- If top-100 coverage is much higher than top-10 coverage, top-10 is probably too narrow for reverse-KL distillation.
- If returned_topN coverage is still low even for N=500, arbitrary teacher scoring or a different objective may be needed.

Diagnostic B: Student top-M overlap with teacher top-K
------------------------------------------------------

For M,K in {10,50,100}, compute:

    overlap_count@M,K =
        |TopM(pi_S) ∩ TopK(q_T)|

    overlap_jaccard@M,K =
        |TopM(pi_S) ∩ TopK(q_T)| / |TopM(pi_S) ∪ TopK(q_T)|

    student_topM_mass_in_teacher_topK@M,K =
        sum_{v in TopM(pi_S) ∩ TopK(q_T)} pi_S(v)

Also compute:
- student_top1_in_teacher_topK@K;
- teacher_top1_in_student_topM@M;
- teacher_topK_mass_on_student_topM@M,K =
    sum_{v in TopM(pi_S) ∩ TopK(q_T)} q_T(v), using unnormalized/full returned teacher probabilities.

Report all metrics per:
- student;
- teacher;
- prompt level;
- with_privileged_context.

Interpretation:
- Reverse KL cares much more about student_topM_mass_in_teacher_topK than raw overlap_count.
- If student_top1 is often outside teacher top100, reverse KL pressure against the student's current mode is strong but cannot be resolved in detail using only teacher top100.
- If teacher top1 is often outside student top100, hard top-1 OPD may be off-support even if teacher top-K has some student-supported tokens.

Diagnostic C: Bucketed reverse-KL approximation
-----------------------------------------------

This diagnostic is only valid if teacher top-K logprobs are full-distribution logprobs, not top-K-renormalized logprobs.

For each K in {10,50,100}, define a partition:

    {each token v in T_K individually} plus {OTHER}

Teacher distribution on the partition:

    q_K(v) = q_T(v), for v in T_K

    q_K(OTHER) = max(eps, 1 - sum_{v in T_K} q_T(v))

Student distribution on the same partition:

    pi_K(v) = pi_S(v), for v in T_K

    pi_K(OTHER) = max(eps, 1 - sum_{v in T_K} pi_S(v))

Then compute:

    bucketed_RKL@K =
        sum_{v in T_K} pi_K(v) * [log pi_K(v) - log q_K(v)]
        + pi_K(OTHER) * [log pi_K(OTHER) - log q_K(OTHER)]

This is a coarse-grained reverse KL over the teacher top-K partition.

Important labeling:
- This is NOT exact full-vocabulary reverse KL.
- It is a coarse partition diagnostic.
- Because each teacher has a different T_K partition, comparisons between teachers are heuristic unless using a common partition.
- It is still useful for seeing whether teacher top-K captures enough student support.

Also compute:
- visible_token_rkl_contribution@K =
    sum_{v in T_K} pi(v) * [log pi(v) - log q_T(v)]
- other_bucket_rkl_contribution@K =
    pi_OTHER * [log pi_OTHER - log q_OTHER]
- other_bucket_fraction_of_bucketed_RKL@K.

Interpretation:
- If the OTHER bucket dominates, top-K is too small for reverse-KL diagnostics/training.
- If bucketed_RKL stabilizes from K=50 to K=100, top-100 may be enough.
- If bucketed_RKL changes a lot from K=50 to K=100, larger K may matter.

Diagnostic D: Common-partition teacher comparison where possible
----------------------------------------------------------------

Comparing bucketed_RKL across teachers is tricky because each teacher has a different T_K.

For teacher comparisons, especially 30B vs 235B, implement common-partition diagnostics as far as possible without arbitrary candidate scoring.

For each K in {10,50,100}, define:

    U_K = TopK(teacher_A) ∪ TopK(teacher_B)

We have q_A(v) for v in U_K only if v appears in teacher_A's returned top-N.
We have q_B(v) for v in U_K only if v appears in teacher_B's returned top-N.

Because requested top-N is 500 or 100, many union tokens may be scoreable if they appear in both returned top-N sets.

For each comparison, compute:
- union_size@K;
- union_tokens_scored_by_both_rate@K;
- student_mass_on_union@K;
- student_mass_on_tokens_scored_by_both@K;
- student_mass_on_A_only_tokens@K;
- student_mass_on_B_only_tokens@K;
- student_mass_on_neither_topN@K.

On the subset of union tokens scored by both teachers, compute:

    student_mass_weighted_logprob_advantage_B_over_A@K =
        sum_{v in both_scored_subset} pi(v) * [log q_B(v) - log q_A(v)]

where B is the larger teacher, for example 235B, and A is 30B.

Also compute the normalized version:

    normalized_advantage_B_over_A@K =
        sum_{v in both_scored_subset} pi(v) * [log q_B(v) - log q_A(v)]
        / max(eps, sum_{v in both_scored_subset} pi(v))

For 30B vs 235B, positive means:
    235B assigns higher logprob than 30B on the student mass that is visible and scored by both teachers.

Important:
- Do not pretend this is exact reverse KL.
- Clearly report the student mass over which the comparison is valid.
- If both-scored student mass is small, mark the comparison as low-confidence.

Diagnostic E: Privileged-context effect on student support
----------------------------------------------------------

For each student-teacher-prompt-level, compare with PC vs without PC.

For K in {10,50,100}, compute:
- student_mass_in_teacher_topK_with_PC;
- student_mass_in_teacher_topK_without_PC;
- delta_student_mass_covered = with_PC - without_PC;
- student_top1_in_teacher_topK_with_PC;
- student_top1_in_teacher_topK_without_PC;
- bucketed_RKL_with_PC if valid;
- bucketed_RKL_without_PC if valid;
- delta_bucketed_RKL = without_PC - with_PC, where positive means PC lowers reverse-KL mismatch;
- overlap between teacher topK with PC and without PC;
- student mass on tokens added by PC;
- student mass on tokens removed by PC.

Definitions:

    added_by_PC_K = T_K_with_PC \ T_K_without_PC
    removed_by_PC_K = T_K_without_PC \ T_K_with_PC

    student_mass_added_by_PC@K =
        sum_{v in added_by_PC_K} pi(v)

    student_mass_removed_by_PC@K =
        sum_{v in removed_by_PC_K} pi(v)

Interpretation:
- If PC changes teacher top-k but added tokens have tiny student mass, reverse KL may not see the PC signal.
- If PC increases student_mass_in_teacher_topK and lowers bucketed_RKL, PC is reverse-KL-compatible.
- If PC decreases support but improves top-1 correctness/rescue, PC may be useful for forward/hard labels but less compatible with reverse KL.

Diagnostic F: Reverse-KL-visible rescue
---------------------------------------

The previous diagnostic used teacher top-10 rescue rate. For reverse KL, replace it with mass-weighted rescue.

For K in {10,50,100}, define:

    student_top1 = argmax pi_S

    rescue_set_K =
        {v in T_K :
            v != student_top1
            and reachable(v)
        }

where:

    reachable(v) iff 0.25 <= -log pi_S(v) <= 8.0

Compute:

    rkl_visible_rescue_mass@K =
        sum_{v in rescue_set_K} pi_S(v)

    teacher_mass_on_rescue_tokens@K =
        sum_{v in rescue_set_K} q_T(v)

    rescue_token_count@K =
        |rescue_set_K|

    has_rkl_visible_rescue@K =
        rkl_visible_rescue_mass@K >= threshold

Use thresholds:
- 0.001
- 0.005
- 0.01
- 0.05

Report fraction of prefixes exceeding each threshold.

Interpretation:
- If old teacher-topK rescue was high but rkl_visible_rescue_mass is low, the previous conclusion was forward-KL-biased.
- If rkl_visible_rescue_mass is substantial, filtered top-k reverse KL may exploit the 235B advantage.

Diagnostic G: Prompt sensitivity under reverse-KL support
---------------------------------------------------------

For each student-teacher pair, compare L1/L2/L3 using support metrics rather than just top-1 changes.

For K in {10,50,100}, compute pairwise L1/L2/L3 differences:
- absolute delta in student_mass_in_teacher_topK;
- absolute delta in bucketed_RKL, if valid;
- student mass on tokens that enter/leave topK due to prompt change;
- topK set Jaccard;
- prompt_changes_student_support_rate:
    fraction of prefixes where |delta student_mass_in_teacher_topK| >= 0.01.

Interpretation:
- If prompt levels barely change student support, GEPA remains low-priority for reverse KL too.
- If top-1 prompt sensitivity is low but support sensitivity is high, prompt optimization may matter for reverse-KL/top-k even if not for hard top-1.

Diagnostic H: Top-K size sufficiency
------------------------------------

For each condition, compare K=10,50,100.

Compute:
- coverage_gain_10_to_50 =
    student_mass_in_teacher_top50 - student_mass_in_teacher_top10

- coverage_gain_50_to_100 =
    student_mass_in_teacher_top100 - student_mass_in_teacher_top50

- bucketed_RKL_change_10_to_50 if valid;
- bucketed_RKL_change_50_to_100 if valid;
- rkl_visible_rescue_gain_10_to_50;
- rkl_visible_rescue_gain_50_to_100.

If returned_topN=500 is available, compute:

    coverage_gap_100_to_returned_topN =
        student_mass_in_teacher_returned_topN - student_mass_in_teacher_top100

This is only a diagnostic for whether K>100 or arbitrary teacher scoring might be needed later.

Interpretation:
- If 50->100 adds little coverage, K=50 may be enough.
- If 50->100 adds substantial coverage, K=100 should be considered.
- If top100 coverage remains far below returned_topN coverage, K>100 may matter, but the user wants current analysis restricted to top-{10,50,100}.
- If returned_topN coverage is low, arbitrary scoring would be needed to approximate reverse KL well.

New report
==========

Write outputs under:

    experiments/reverse_kl_support_checks/{run_name}/

Required files:

1. report.md

Include:
- run name;
- repo commit hash;
- whether prefixes were reused from previous diagnostic;
- number of prefixes per student;
- requested_teacher_top_logprobs;
- actual_returned_top_logprobs by teacher/server;
- whether teacher logprobs were full-distribution-compatible;
- prompt levels used;
- with/without PC status;
- model pairs.

Main sections:

A. Why this follow-up was needed

Explain:
- previous teacher top-k diagnostics were informative but teacher-support-oriented;
- reverse KL is student-support-weighted;
- this run measures student mass covered by teacher top-{10,50,100}.

B. Student mass covered by teacher top-K

Tables for K=10,50,100:
- mean/median/p10/p90 coverage;
- fraction coverage >= 0.25/0.50/0.75/0.90;
- compare 30B vs 235B;
- compare with PC vs no PC.

C. Student top-M overlap with teacher top-K

Tables for M,K in {10,50,100}:
- overlap_count;
- overlap_jaccard;
- student_topM_mass_in_teacher_topK;
- student_top1_in_teacher_topK;
- teacher_top1_in_student_topM.

D. Bucketed reverse-KL approximation

For K=10,50,100:
- bucketed_RKL;
- visible token contribution;
- OTHER contribution;
- OTHER fraction;
- validity flags.

If invalid because teacher logprobs are not full-distribution-compatible, say so and skip this section cleanly.

E. 30B vs 235B on reverse-KL-visible support

For each shared student and prompt level:
- coverage difference;
- union tokens scored by both rate;
- student mass on both-scored subset;
- mass-weighted 235B-over-30B logprob advantage on both-scored subset;
- confidence flag based on both-scored student mass.

F. Privileged-context effect on student support

For each teacher/prompt:
- delta coverage with PC;
- delta bucketed_RKL if valid;
- student mass added/removed by PC;
- whether PC helps or hurts reverse-KL support.

G. Reverse-KL-visible rescue

For K=10,50,100:
- rkl_visible_rescue_mass;
- teacher mass on rescue tokens;
- rescue count;
- fraction of prefixes above thresholds 0.001/0.005/0.01/0.05.

Explicitly compare against the previous top-k rescue conclusion:
- If visible rescue mass is low, say the old top-k rescue was probably forward-KL-flavored.
- If visible rescue mass is substantial, say filtered top-k reverse KL remains promising.

H. Prompt sensitivity under reverse-KL support

Report whether L1/L2/L3 differ meaningfully in coverage or bucketed_RKL.

I. Top-K size sufficiency

Compare K=10 vs 50 vs 100:
- coverage gains;
- rescue gains;
- bucketed_RKL changes;
- if returned_topN=500, top100-to-returned-topN coverage gap.

J. Recommendations

Do not automatically choose training settings, but include interpretation flags:

- `top100_sufficient_for_rkl`: true if coverage@100 is high and 50->100 gain is small.
- `top50_sufficient_for_rkl`: true if coverage@50 is high and 50->100 gain is tiny.
- `topk_rkl_not_visible`: true if coverage@100 and rkl_visible_rescue_mass are low.
- `larger_teacher_less_student_compatible`: true if 235B coverage is consistently below 30B coverage.
- `larger_teacher_rkl_promising`: true if 235B coverage is comparable/higher and rkl-visible rescue mass is substantial.
- `privileged_context_rkl_compatible`: true if PC increases coverage or lowers bucketed_RKL without large support loss.
- `need_arbitrary_teacher_scoring`: true if coverage@100 is low and returned_topN coverage is much higher or still low.

2. support_coverage.csv

One row per:
- student;
- teacher;
- prompt level;
- with_privileged_context;
- K.

Columns:
- coverage mean/median/p10/p90;
- coverage thresholds;
- outside mass;
- returned_topN coverage if available.

3. topm_topk_overlap.csv

One row per:
- student;
- teacher;
- prompt;
- context;
- M;
- K.

4. bucketed_rkl.csv

One row per:
- student;
- teacher;
- prompt;
- context;
- K.

Include validity flags.

5. teacher_comparison_rkl_support.csv

One row per:
- student;
- teacher_A;
- teacher_B;
- prompt;
- context;
- K.

6. pc_effect_rkl_support.csv

One row per:
- student;
- teacher;
- prompt;
- K.

7. rkl_visible_rescue.csv

One row per:
- student;
- teacher;
- prompt;
- context;
- K.

8. prompt_support_sensitivity.csv

One row per:
- student;
- teacher;
- context;
- prompt comparison;
- K.

9. per_prefix_rkl_support_metrics.parquet

Per-prefix raw metrics.

10. config_resolved.yaml

Full resolved config.

11. commands_rerun.sh

Exact commands to reproduce:
- smoke;
- teacher top-N collection;
- analysis;
- full support check.

Implementation files
====================

Adapt to repo conventions. Add or modify equivalents of:

src/privileged_opd_checks/reverse_kl_support.py
    student mass covered by teacher top-K;
    topM/topK overlap;
    bucketed reverse KL;
    rkl-visible rescue;
    PC support deltas;
    prompt support sensitivity;
    teacher comparison support metrics.

src/privileged_opd_checks/teacher_distribution.py
    add requested_teacher_top_logprobs support;
    store actual_returned_top_logprobs;
    store topN logprob semantics validation;
    support topN cache keys.

src/privileged_opd_checks/analysis.py
    add reverse-KL support report generation.

scripts/run_reverse_kl_support_checks.py
    modes:
      smoke
      collect_teacher_topN
      analyze
      full_support_check

Or extend the existing run_privileged_opd_checks.py with:
    --analysis reverse_kl_support
    --requested_teacher_top_logprobs 500

Prefer the least invasive implementation that reuses existing infrastructure.

Caching
=======

Do not overwrite previous top-10 caches unless required.

Add top-N cache keys that include:
- requested_teacher_top_logprobs;
- actual_returned_top_logprobs;
- teacher model ID;
- student model ID;
- prompt hash;
- prompt level;
- with_privileged_context;
- privileged context hash;
- prefix hash;
- teacher server URL/backend identifier.

Important:
A previous top-10 cache is insufficient for K=50 or K=100. Re-query teachers unless a valid top-N cache already exists.

Unit tests
==========

Add tests for:

1. student_mass_in_teacher_topK
2. student_topM_mass_in_teacher_topK
3. topM/topK overlap count and Jaccard
4. bucketed reverse KL with OTHER bucket
5. bucketed reverse KL validity when teacher probabilities sum > 1
6. PC added/removed student mass
7. rkl_visible_rescue_mass
8. common-partition teacher comparison with missing tokens
9. cache key includes requested_teacher_top_logprobs

Run tests before full analysis.

Smoke mode
==========

Smoke mode should use:
- one student;
- one teacher;
- one prompt level;
- with PC and without PC;
- 4-8 prefixes;
- requested_teacher_top_logprobs = min(100, configured default) unless a 500-token smoke is cheap.

Smoke must verify:
- teacher returns more than 10 logprobs;
- student full distribution is available;
- coverage@10/50/100 computes, with K clipped if fewer than 100 returned in smoke;
- bucketed_RKL computes if valid;
- report writes.

Execution expectation
=====================

After implementation:

1. Run unit tests.
2. Run smoke.
3. Run full_support_check if teacher endpoints and GPU resources are available.
4. If teacher top_logprobs=500 is rejected by vLLM, automatically fallback to 100 and continue.
5. If 30B or 235B endpoints are unavailable, run available subsets and clearly mark missing pairings.

Do not silently skip teacher comparisons.

Final response from Codex
=========================

When finished, report:

1. Files changed.
2. Commands executed.
3. Tests passed/failed.
4. Smoke status.
5. Full support-check status.
6. requested_teacher_top_logprobs and actual returned values.
7. Path to report.md.
8. Headline findings:
   - student mass in teacher top-10/50/100;
   - whether 235B covers more or less student mass than 30B;
   - whether PC increases or decreases reverse-KL-visible support;
   - whether previous top-k rescue remains visible under student mass;
   - whether K=10/50/100 seems sufficient.
9. Blockers and exact rerun commands.

Critical constraints
====================

- Do not run GEPA.
- Do not run training.
- Do not run full OPD.
- Do not require arbitrary teacher scoring.
- Restrict main K diagnostics to top-{10,50,100}.
- Request large teacher top_logprobs, preferably 500, fallback 100.
- Clearly distinguish:
    teacher-support top-k diagnostics
    from reverse-KL student-support diagnostics.
- Clearly label bucketed_RKL as a coarse approximation, not exact full-vocabulary reverse KL.
- Do not claim that top-k reverse KL will improve training until probe training is run later.