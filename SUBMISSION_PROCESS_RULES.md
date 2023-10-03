# MLCommons™ AlgoPerf: Submission Process Rules

**Version:** 0.0.2 *(Last updated 03 Oktober 2023)*

## Table of Contents <!-- omit from toc -->

- [Basics](#basics)
- [Schedule](#schedule)
  - [Dates](#dates)
  - [Version freeze](#version-freeze)
  - [Submission deadline](#submission-deadline)
- [Submission](#submission)
  - [Register a submission](#register-a-submission)
  - [How to submit](#how-to-submit)
    - [Submission repository](#submission-repository)
    - [Licensing](#licensing)
  - [Multiple Submission](#multiple-submission)
- [Scoring](#scoring)
  - [Self-reporting scores](#self-reporting-scores)
    - [Verifying scores](#verifying-scores)
  - [Sampling held-out workloads and hyperparameters](#sampling-held-out-workloads-and-hyperparameters)
  - [Leaderboard](#leaderboard)
- [Sprit jury \& challenging submissions](#sprit-jury--challenging-submissions)
- [Awards and prize money](#awards-and-prize-money)
  - [Awards committee](#awards-committee)
- [Ineligibility and conflict of interest](#ineligibility-and-conflict-of-interest)

## Basics

This is the submission process rules for the AlgoPerf: Training Algorithms Benchmark. It describes the process of submitting a new training algorithm and details how it will be scored. This process applies to both the external tuning ruleset and the self-tuning ruleset although, for all intents and purposes, they are two separate competitions, with separate leaderboards.

Three additional documents complement this document:

- [**Benchmark rules**](RULES.md): While the submission process rules detail the *logistical* aspects of submitting to the AlgoPerf: Training Algorithms Benchmark, the [rules document](RULES.md) describes the *scientific* rules of the competition. This includes, for example, how tuning is performed in each ruleset, what types of submissions are allowed, or how the benchmark score is computed.
- [**AlgoPerf paper**](https://arxiv.org/abs/2306.07179): The paper titled ["Benchmarking Neural Network Training Algorithms"](https://arxiv.org/abs/2306.07179) motivates the need for the benchmark, explains the rules, and justifies the specific design choices of the AlgoPerf: Training Algorithms Benchmark. Additionally, it evaluates baseline submissions, constructed using various optimizers like Adam, Shampoo, or SAM, on the benchmark, demonstrating the feasibility but also the difficulty of the benchmark.
- [**Benchmark codebase**](https://github.com/mlcommons/algorithmic-efficiency): The codebase implements the rules, provides exact specifications of the workloads, and it will ultimately be used to score submissions.

## Schedule

### Dates

- **Publication of the call for submission: 17. Oktober 2023 (08:00 AM UTC)**
- Registration deadline for submissions: 15. December 2023 (08:00 AM UTC)
- Version freeze for the benchmark codebase: 17. January 2024 (08:00 AM UTC)
- **Submission deadline: 15. February 2024 (08:00 AM UTC)**
- Sampling the held-out workloads and hyperparameters: 16. February 2024 (08:00 AM UTC)
- Deadline for specifying the submission batch sizes for held-out workloads: 28. February 2024 (08:00 AM UTC)
- Deadline for self-reporting results: 10. April 2024 (08:00 AM UTC)
- **[extra tentative] Announcement of all results: 22. May 2024 (08:00 AM UTC)**

The presented dates are subject to change and adjustments may be made by the [MLCommmons Algorithms Working Group](https://mlcommons.org/en/groups/research-algorithms/).

### Version freeze

The benchmark code base is subject to change after the call for submissions is published. For example, while interacting with the codebase, if submitters encounter bugs or API limitations, they have the option to issue a bug report. This might lead to modifications of the benchmark codebase even after the publication of the call for submissions.

To ensure that all submitters can develop their submissions based on the same code that will be utilized for scoring, we will freeze the package versions of the codebase dependencies before the submission deadline. By doing so, we level the playing field for everyone involved, ensuring fairness and consistency in the assessment of submissions. We will also try to minimize changes to the benchmark codebase as best as possible.

### Submission deadline

With the submission deadline, all submissions need to be available as a *public* repository with the appropriate license (see the [Licensing section](#licensing)). No changes to the submission code are allowed after the submission deadline (with the notable exception of specifying the batch size for the - at that point unknown - held-out workloads). Once the submission deadline has passed, the working group will publish a list of all submitted algorithms, along with their associated repositories. Anyone has the right to challenge a submission, i.e. request a review by the spirit jury to determine whether a submission violates the rules of the competition, see the [Spirit jury section](#sprit-jury--challenging-submissions).

Directly after the submission deadline, all randomized aspects of the competition are fixed. This includes sampling the held-out workloads from the set of randomized workloads, as well as, sampling the hyperparameters for each submission in the external tuning ruleset (for more details see the [Sampling held-out workloads and hyperparameters section](#sampling-held-out-workloads-and-hyperparameters)). After that, submitters can now ascertain the appropriate batch size of their submission on each held-out workload and self-report scores on either the qualification set or the full benchmarking set of workloads including both fixed and held-out workloads (see the [Self-reporting scores section](#self-reporting-scores)).

## Submission

For a guide on the technical steps and details on how to write a submission, please refer to the [**Getting started document**](GETTING_STARTED.md). Additionally, the folders [/reference_algorithms](/reference_algorithms/) and [/baselines](/baselines/) provide example submissions that can serve as a template for creating new submissions.

In the following, we describe the logistical steps required to submit a training algorithm to the AlgoPerf: Training Algorithms Benchmark.

### Register a submission

All submitters need to register an intent to submit before the submission registration deadline. This registration is mandatory, i.e. required for all submissions, but not binding, i.e. you don't have to submit a registered submission. This registration is necessary, to estimate the number of submissions and provide support for potential submitters.

To register a submission, please write an email to <algorithms-chairs@mlcommons.org> with the subject "[Registration] *submission_name*" and the following information:

- Name of the submission (e.g. name of the algorithm, or any other arbitrary identifier).
- Ruleset under which the submission will be scored.
- Name of all submitters associated with this submission.
- Email of all submitters associated with this submission.
- Affiliations of all submitters associated with this submission.

In return, the submission will be issued a unique **submission ID** that will be used throughout the submission process.

### How to submit

Submitters have the flexibility to submit their training algorithm anytime between the registration of the submission and the submission deadline. To submit a submission, please write an email to <algorithms-chairs@mlcommons.org> with the subject "[Submission] *submission_ID*" and the following information:

- Submission ID.
- URL of the associated *public* GitHub repository.
- If applicable, a list of all changes to the names, emails, or affiliations compared to the registration of the submission.
- A digital version of all relevant licensing documents (see the [Licensing section](#licensing)).

#### Submission repository

The *public* GitHub repository needs to be a clone of the frozen `main` branch of the [benchmark codebase](https://github.com/mlcommons/algorithmic-efficiency). All elements of the original codebase,  except for the `/submission` directory need to be unaltered from the original benchmark code. In particular, the repository must use the same [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0) as the benchmark codebase. Once the submission deadline has passed, modifications of the submission repository's code are generally prohibited. The sole exception to this rule is the definition of the batch sizes for the held-out workloads.

Any software dependencies required for the submission need to be defined in a `requirements.txt` file within the `/submission` directory. This file needs to be `pip` readable, i.e. installable via `pip install -r requirements.txt`. In order to comply with the rules, submissions are not allowed to modify the used package version of the software dependencies of the benchmarking codebase, e.g. by using a different version of PyTorch or JAX (see [](RULES.md#disallowed-submissions)).

#### Licensing

Submitting to the AlgoPerf: Training Algorithms Benchmark requires the following legal considerations:

- A signed [Contributor License Agreement (CLA) "Corporate CLA"](https://mlcommons.org/en/policies/) of MLCommons.
- *Either* a membership in MLCommons *or* a signed [non-member test agreement](https://mlcommons.org/en/policies/).
- A signed trademark license agreement, either the member or the non-member version, as appropriate. These license agreements are available upon request to [support@mlcommons.org](mailto:support@mlcommons.org).

We furthermore require all submissions to be made available open source on the submission deadline under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).

### Multiple Submission

Our benchmark allows multiple submissions by the same submitter(s). However, we would like to prevent submitters from circumventing the purpose of the benchmark by, for example, submitting dozens of copies of the same submission with slightly different hyperparameters. Such a bulk submission would result in an unfair advantage on the randomized workloads and is not in the spirit of the benchmark.

Submitters may submit algorithms marked as *baselines*. These might include existing algorithms with different search spaces or learning rate schedules. These baseline algorithms are not eligible for winning the competition or prize money but they are also not required to be "substantially different" from other submissions by the same submitters.

## Scoring

### Self-reporting scores

Submitters are expected to self-report scores on the full benchmark set before the deadline for self-reporting results. Reporting the scores involves providing all unmodified logs that the benchmarking codebase automatically generates in a separate `/results` directory within the `/submission` folder. For submissions competing in the external tuning ruleset, this includes all the logs of the tuning trials using the [hyperparameter samples provided by the working group](#sampling-held-out-workloads-and-hyperparameters). Note, that while the tuning runs can be performed on non-competition hardware, they still need to show that the "winning hyperparameter configuration" in each study was selected according to the [tuning rules](/RULES.md#external-tuning-ruleset), i.e. the fastest hyperparameter to reach the validation target. Additionally, the logs of the "winning hyperparameter configuration" (or each trial, in the self-tuning ruleset) in each of the five studies need to be computed on the competition hardware, to allow wall-clock runtime comparisons.

Submitters unable to self-fund scoring costs can instead self-report only on the [qualification set of workloads](/RULES.md#qualification-set) that excludes some of the most expensive workloads. Based on this performance on the qualification set, the working group will provide - as funding allows - compute to evaluate and score the most promising submissions. Additionally, we encourage researchers to reach out to the [working group](mailto:algorithms@mlcommons.org) to find potential collaborators with the resources to run larger, more comprehensive experiments for both developing and scoring submissions.

#### Verifying scores

The working group will independently verify the scores of the highest-scoring submissions in each ruleset. Results that have been verified by the working group will be clearly marked on the leaderboard.

### Sampling held-out workloads and hyperparameters

After the submission deadline has passed and all submission code is frozen, the working group will sample a specific instance of held-out workloads from the set of randomized workloads. Additionally, every submission in the external tuning ruleset will receive its specific set of 5x20 hyperparameter values grouped by study. This set of hyperparameter values is sampled from the search space provided by the submitters.

The sampling code for the held-out workloads and the hyperparameters is publicly available (**TODO link to both functions!**). Both sampling functions take as input a random seed, which will be provided by a trusted third party after the submission deadline.

### Leaderboard

The announcement of the results will contain two separate leaderboards, one for the self-tuning and one for the external tuning ruleset. All valid submissions will be ranked by the benchmark score, taking into account all workloads, including the held-out ones. The leaderboard will clearly mark scores that were verified by the working group.

## Sprit jury & challenging submissions

The spirit jury, consisting of selected active members of the working group, will be responsible for deciding whether a submission violates the "spirit of the rules". Submitters with specific concerns about a particular submission can request a review by the spirit jury to determine whether a submission violates the rules of the competition. To challenge a submission, please write an email to <algorithms-chairs@mlcommons.org> with the subject "[Challenge] *submission_name*". The email needs to link to the challenged submission and include a detailed description of why the submission should be reviewed. This request must be made reasonably in advance of the results announcement deadline to allow the Spirit Jury sufficient time to conduct a thorough review.

The spirit jury may then hear the justifications of the submitters, inspect the code, and also ask the submitters to explain how the submission was produced, for example, by disclosing their intermediate experiments. Example cases that might be reviewed by the spirit jury are cases of multiple similar submissions by the same submitter or extensive workload-specific tuning.

## Awards and prize money

An awards committee will award a prize for the "*Best Performance*" in each ruleset as well as a "*Jury Award*". The prize for the best-performing submission will take into account the [benchmark score](RULES.md#benchmark-score-using-performance-profiles) on the full benchmark. The "*Jury Award*" will favor more out-of-the-box ideas that show great potential, even though the method may not be of practical value with the current landscape of models, software, etc.

The prize money for "*Best Performance*" in a ruleset is $20,000 each. The winner of the "*Jury Award*" will be awarded $10,000. We reserve the right to split the prize money and distribute it among multiple submissions.

If a submission is ineligible to win prize money it can still win an award. The prize money will then go to the highest-ranking eligible submission.

### Awards committee

The awards committee will be responsible for awarding prize money to submissions. The committee will try to reach a consensus on how to award prize money and settle disagreements by majority vote, if necessary.

**TODO Who is on the Awards committee?**

## Ineligibility and conflict of interest

To ensure a fair process and avoid conflicts of interest, some individuals and institutions are ineligible to win prize money. This includes:

- The chairs of the MLCommons Algorithms Working Group (presently *George Dahl* and *Frank Schneider*) and their institutions (currently *Google Inc.* and the *University of Tübingen*)
- All individuals serving on the awards committee and their institutions.

A submission with at least one ineligible submitter may still win an award, but the prize money will then be awarded to the top-ranked submission that is eligible for prize money.

Additionally, we require members of the spirit jury to abstain from being involved in a review if:

- They are part of the reviewed submission.
- The reviewed submission contains individuals from their institution.

The spirit jury can still take a decision if at least one member of the jury is without a conflict of interest.
