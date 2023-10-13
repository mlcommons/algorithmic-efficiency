# Change log

# TODO: algorithmic-efficiency 0.1.0
First release of AlgoPerf benchmarking code.

**Disclaimer**: The Conformer Pytorch workload has memory fragmentation issue after upgrading to 
Pytorch 2.0.1, which led to out of memory errors. To circumvent this issue we have tuned the pytorch 
memory allocation configuration, which slows down the workload by a factor of roughly 2x. For submitters, this 
means that the Conformer Pytorch submission times will be roughly 2x slower. 
Tracking in issue/497(https://github.com/mlcommons/algorithmic-efficiency/issues/497).  

Tracking issue: [issue/497](https://github.com/mlcommons/algorithmic-efficiency/issues/497). 