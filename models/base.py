from abc import ABC, abstractmethod
from typing import Callable, Tuple

from specification.draft_api import *


class ReferenceModel(ABC):

    """
    Fixed functions
    """
    @staticmethod
    @abstractmethod
    def preprocess_for_train(selected_raw_input_batch: Tensor,
                             selected_label_batch: Tensor,
                             train_mean: Tensor,
                             train_stddev: Tensor,
                             seed: Seed) -> Tuple[Tensor, Tensor]:

        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def preprocess_for_eval(raw_input_batch: Tensor,
                            train_mean: Tensor,
                            train_stddev: Tensor) -> Tensor:

        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def init_model_fn(param_shapes: ParameterShapeTree,
                      seed: Seed) -> ParameterTree:

        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def model_fn(params: ParameterTree,
                 preprocessed_input_batch: Tensor,
                 mode: ForwardPassMode,
                 seed: Seed,
                 update_batch_norm: bool) -> Tuple[OutputTree, Tensor]:

        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def loss_fn(label_batch: Tensor,
                logits_batch: Tensor,
                loss_type: LossType) -> float:

        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def output_activation_fn(logits_batch: Tensor,
                             loss_type: LossType) -> Tensor:

        raise NotImplementedError

    """
    Submission functions
    """
    @staticmethod
    def init_optimizer_state(params_shapes: ParameterShapeTree,
                             hyperparameters: Hyperparameters,
                             seed: Seed) -> OptimizerState:

        raise NotImplementedError


    @staticmethod
    def update_params(current_params: ParameterTree,
                      current_params_types: ParameterTypeTree,
                      hyperparameters: Hyperparameters,
                      augmented_and_preprocessed_input_batch: Tensor,
                      label_batch: Tensor,
                      loss_type: LossType,
                      model_fn: Callable,
                      optimizer_state: OptimizerState,
                      global_step: int,
                      seed: Seed) -> Tuple[OptimizerState, ParameterTree]:

        raise NotImplementedError

    @staticmethod
    def data_selection(input_queue: InputQueue,
                       optimizer_state: OptimizerState,
                       current_params: ParameterTree,
                       loss_type: LossType,
                       hyperparameters: Hyperparameters,
                       global_step: int,
                       seed: Seed) -> Tuple[Tensor, Tensor]:

        raise NotImplementedError
