# Contributing

The best way to contribute to the MLCommons is to get involved with one of our many project communities. You find more information about getting involved with MLCommons [here](https://mlcommons.org/en/get-involved/#getting-started).

Generally we encourage people to become a MLCommons member if they wish to contribute to MLCommons projects, but outside pull requests are very welcome too.

To get started contributing code, you or your organization needs to sign the MLCommons CLA found at the [MLC policies page](https://mlcommons.org/en/policies/). Once you or your organization has signed the corporate CLA, please fill out this [CLA sign up form](https://forms.gle/Ew1KkBVpyeJDuRw67) form to get your specific GitHub handle authorized so that you can start contributing code under the proper license.

MLCommons project work is tracked with issue trackers and pull requests. Modify the project in your own fork and issue a pull request once you want other developers to take a look at what you have done and discuss the proposed changes. Ensure that cla-bot and other checks pass for your Pull requests.

## Table of Contents
- Set up your workspace
- Developing with Docker
- Presubmit testing
- Merging your Change

### Submitting PRs 
New PRs will be merged on the dev branch by default, given that they pass the presubmits.

### Testing
We run tests with GitHub Actions, configured in the [.github/workflows](https://github.com/mlcommons/algorithmic-efficiency/tree/main/.github/workflows) folder.

#### Style Testing
We run yapf and linting tests on PRs. You can view and fix offending errors with these instructions.

To run the below commands, use the versions installed via `pip install -e '.[dev]'`.

To automatically fix formatting errors, run the following (*WARNING:* this will edit your code, so it is suggested to make a git commit first!):
```bash
yapf -i -r -vv -p algorithmic_efficiency baselines datasets reference_algorithms tests *.py
```

To sort all import orderings, run the following:
```bash
isort .
```

To just print out all offending import orderings, run the following:
```bash
isort . --check --diff
```

To print out all offending pylint issues, run the following:
```bash
pylint algorithmic_efficiency
pylint baselines
pylint datasets
pylint reference_algorithms
pylint submission_runner.py
pylint tests
```

#### Unit and integration tests
We run unit tests and integration tests as part of the of github actions as well. 
You can also use `python tests/reference_algorithm_tests.py` to run a single model update and two model evals for each workload using the reference algorithm in `reference_algorithms/development_algorithms/`.

#### Regression tests
We also have regression tests available in [.github/workflows/regression_tests.yml](https://github.com/mlcommons/algorithmic-efficiency/tree/main/.github/workflows/regression_tests.yml) that can be run semi-automatically.
The regression tests are shorter end-to-end submissions run in a containerized environment across all 8 workloads, in both the jax and pytorch frameworks. 
The regression tests run on self-hosted runners and are triggered for pull requests that target the main branch.
To trigger a regression test:
1. The self-hosted runner has to be on.
2. The self-hosted runner application is active for the runner to accept jobs.
3. Open a pull request to trigger the workflow.