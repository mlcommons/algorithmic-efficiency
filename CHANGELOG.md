# Change log

# TODO: algorithmic-efficiency 0.1.0
First release of AlgoPerf benchmarking code.

**Disclaimer**: The Conformer Pytorch workload has memory fragmentation issue after upgrading to 
Pytorch 2.0.1. To circumvent this issues we have tuned the pytorch memory allocation configuration,
which slows down the workload by a factor of 2x. For submitters, this means that the Conformer Pytorch 
submission times will be about 2x slower. 

Tracking issue: [issue/497](https://github.com/mlcommons/algorithmic-efficiency/issues/497). 