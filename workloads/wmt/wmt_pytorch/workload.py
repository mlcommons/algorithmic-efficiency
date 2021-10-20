import contextlib
import itertools
from typing import Tuple
from collections import OrderedDict

import spec
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from transformer.Models import Transformer
from torchtext.datasets import TranslationDataset
from torchtext.data import Field, Dataset, BucketIterator
import transformer.Constants as Constants



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class WMTWorkload(spec.Workload):

    def __init__(self):
        self.batch_size = 256
        self.embs_share_weight = True
        self.proj_share_weight = True
        self.data_pkl = 'm30k_deen_shr.pkl'
        self.train_path = './bpe_deen/deen-train'
        self.val_path = './bpe_deen/deen-val'
        self.max_token_seq_len = None
        self.src_pad_idx = None
        self.trg_pad_idx = None
        self.src_vocab_size = self.trg_vocab_size = None
        self.trg_emb_prj_weight_sharing = self.proj_share_weight
        self.emb_src_trg_weight_sharing = self.embs_share_weight
        self.d_k = 64
        self.d_v = 64
        self.d_model = 512
        self.d_word_vec = self.d_model
        self.cuda = True
        self.d_inner_hid = 2048
        self.d_inner = self.d_inner_hid
        self.n_layers = 6
        self.n_head = 8
        self.dropout = 0.1
        self.scale_emb_or_prj = 'prj'

    def _build_dataset(self,
                       data_rng: spec.RandomState,
                       split: str,
                       data_dir: str,
                       batch_size: int
                       ):
        return self.prepare_dataloaders_from_bpe_files(
            split,
            data_dir,
            batch_size
        )

    def build_input_queue(
      self,
      data_rng: spec.RandomState,
      split: str,
      data_dir: str,
      batch_size: int):
        return iter(self._build_dataset(data_rng, split, data_dir, batch_size))


    def init_model_fn(
      self, rng: spec.RandomState) -> Tuple[spec.ParameterContainer, spec.ModelAuxiliaryState]:
        torch.random.manual_seed(rng[0])
        model = Transformer(
            self.src_vocab_size,
            self.trg_vocab_size,
            self.src_pad_idx,
            self.trg_pad_idx,
            self.trg_emb_prj_weight_sharing,
            self.emb_src_trg_weight_sharing,
            self.d_k,
            self.d_v,
            self.d_model,
            self.d_word_vec,
            self.d_inner,
            self.n_layers,
            self.n_head,
            self.dropout,
            self.scale_emb_or_prj)

        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model.to(DEVICE)
        return model, None

    def prepare_dataloaders_from_bpe_files(self, split, data_dir, batch_size, ):
        batch_size = batch_size
        MIN_FREQ = 2
        if not self.embs_share_weight:
            raise

        data = pickle.load(open(data_dir, 'rb'))
        MAX_LEN = data['settings'].max_len
        field = data['vocab']
        fields = (field, field)

        def filter_examples_with_length(x):
            return len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN

        train = TranslationDataset(
            fields=fields,
            path=self.train_path,
            exts=('.src', '.trg'),
            filter_pred=filter_examples_with_length)
        val = TranslationDataset(
            fields=fields,
            path=self.val_path,
            exts=('.src', '.trg'),
            filter_pred=filter_examples_with_length)

        self.max_token_seq_len = MAX_LEN + 2
        self.src_pad_idx = self.trg_pad_idx = field.vocab.stoi[Constants.PAD_WORD]
        self.src_vocab_size = self.trg_vocab_size = len(field.vocab)

        if split == 'train':
            train_iterator = BucketIterator(train, batch_size=batch_size, device=DEVICE, train=True, shuffle=True)
            val_iterator = BucketIterator(val, batch_size=batch_size, device=DEVICE, shuffle=True)
        else:
            train_iterator = BucketIterator(train, batch_size=batch_size, device=DEVICE, train=False, shuffle=False)
            val_iterator = BucketIterator(val, batch_size=batch_size, device=DEVICE, shuffle=False)

        return train_iterator, val_iterator

    def prepare_dataloaders(self, device):
        batch_size = self.batch_size
        data = pickle.load(open(self.data_pkl, 'rb'))

        self.max_token_seq_len = data['settings'].max_len
        self.src_pad_idx = data['vocab']['src'].vocab.stoi[Constants.PAD_WORD]
        self.trg_pad_idx = data['vocab']['trg'].vocab.stoi[Constants.PAD_WORD]

        self.src_vocab_size = len(data['vocab']['src'].vocab)
        self.trg_vocab_size = len(data['vocab']['trg'].vocab)

        # ========= Preparing Model =========#
        if self.embs_share_weight:
            assert data['vocab']['src'].vocab.stoi == data['vocab']['trg'].vocab.stoi, \
                'To sharing word embedding the src/trg word2idx table shall be the same.'

        fields = {'src': data['vocab']['src'], 'trg': data['vocab']['trg']}

        train = Dataset(examples=data['train'], fields=fields)
        val = Dataset(examples=data['valid'], fields=fields)

        train_iterator = BucketIterator(train, batch_size=batch_size, device=device, train=True)
        val_iterator = BucketIterator(val, batch_size=batch_size, device=device)

        return train_iterator, val_iterator


    def model_fn(
      self,
      params: spec.ParameterContainer,
      input_batch: spec.Tensor,
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
        del model_state
        del rng
        del update_batch_norm

        model = params
        if mode == spec.ForwardPassMode.EVAL:
            model.eval()

        contexts = {
            spec.ForwardPassMode.EVAL: torch.no_grad,
            spec.ForwardPassMode.TRAIN: contextlib.nullcontext
        }

        with contexts[mode]():
            logits_batch = model(input_batch)

        return logits_batch, None

    def loss_fn(
      self,
      label_batch: spec.Tensor,  # Dense (not one-hot) labels.
      logits_batch: spec.Tensor) -> spec.Tensor:
        F.cross_entropy(label_batch, logits_batch)