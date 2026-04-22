from pathlib import Path
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader

import pytest
import torch

from remora import RemoraError
from remora.util import parse_device


pytestmark = pytest.mark.unit


def load_model_module(model_path):
    loader = SourceFileLoader("tmp_model_module", str(model_path))
    module = module_from_spec(spec_from_loader(loader.name, loader))
    loader.exec_module(module)
    return module


def test_transformer_model_forward_shape():
    model_path = (
        Path(__file__).absolute().parent.parent
        / "models"
        / "Transformer_w_ref.py"
    )
    module = load_model_module(model_path)
    model = module.network(size=16, kmer_len=9, num_out=3)

    sigs = torch.randn(4, 1, 100)
    seqs = torch.randn(4, 36, 100)
    out = model(sigs, seqs)

    assert out.shape == (4, 3)


def test_parse_device_accepts_metal_alias():
    if not torch.backends.mps.is_available():
        with pytest.raises(RemoraError):
            parse_device("metal")
    else:
        assert parse_device("metal").type == "mps"


def test_parse_device_integer_still_maps_to_cuda():
    if not torch.cuda.is_available():
        with pytest.raises(RemoraError):
            parse_device("0")
    else:
        assert parse_device("0").type == "cuda"
