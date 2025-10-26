
import matplotlib
import copy
import pickle as pkl
import sys
import os
import numpy as np
from tqdm import tqdm
import tensorflow_datasets as tfds
import math

import numba
import awkward
import vector
import fastjet
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep

import boost_histogram as bh
import mplhep

import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from torch import Tensor

import onnxscript
import onnx
import onnxruntime as rt
from onnxconverter_common import float16
from onnxscript.function_libs.torch_lib.tensor_typing import TFloat



import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

# contrib op: https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftmultiheadattention
# CMSSW ONNXRuntime version: https://github.com/cms-sw/cmsdist/blob/REL/CMSSW_14_1_0_pre3/el9_amd64_gcc12/onnxruntime.spec
# ONNXRuntime compatiblity table: https://onnxruntime.ai/docs/reference/compatibility.html

# with pytorch 2.5.0, we should use at least opset 20 (previous opsets did not work)
# from onnxscript import opset20 as op

opset_version = 17
# custom_opset = onnxscript.values.Opset(domain="onnx-script", version=1)
# msft_op = onnxscript.values.Opset("com.microsoft", 1)

mplhep.style.use("CMS")

import onnxruntime as ort

DATA_DIR = "/local/laurits/ml-tau-en-reg/ntuples/20240924_lowered_recoPtCut"
DATASETS = ["p8_ee_qq_ecm380", "p8_ee_ZH_Htautau_ecm380", "p8_ee_Z_Ztautau_ecm380"]
MODEL_DIR = "/home/laurits/mltau-ONNX"
TASKS = ["dm_multiclass", "jet_regression", "binary_classification"]



def print_software_environment():
    print("--------------------------------------")
    print("For CUDA, we must use onnxruntime-gpu (not onnxruntime):")
    print(f"ONNXRuntime version: {rt.__path__, rt.__version__}")
    print(f"Pytorch version: {torch.__version__}")
    print("--------------------------------------")


def validate_conversion(inputs: dict, onnx_model_path: str, pytorch_model):
    # Evaluate ONNX model
    ort_session = ort.InferenceSession(onnx_model_path)
    outputs = ort_session.run(None, {key: value.numpy() for key, value in inputs.items()})
    # outputs is a list (one element per output tensor)
    print("ONNX output shape:", outputs[0].shape)

    # Get Pytorch model predictions
    with torch.no_grad():
        torch_output = pytorch_model(**inputs)
    # Convert PyTorch output to NumPy
    torch_output_np = torch_output.numpy()

    # Check if the outputs match for the two models
    np.testing.assert_allclose(torch_output_np, outputs[0], rtol=1e-3, atol=1e-5)
    print("ONNX output matches PyTorch output!")


def load_model(model_dir: str, task: str, cfg: DictConfig, device: str = "cpu"):
    # Load model state
    model_path = os.path.join(model_dir, f"{task}.pt")
    model_state = torch.load(model_path, map_location=torch.device(device), weights_only=True)
    model = instantiate(cfg.models.ParticleTransformer).to(device=device)
    model.eval()
    model.load_state_dict(model_state, strict=False)
    model = model.to(device=device)
    return model


@hydra.main(config_path="../config", config_name="model_training", version_base=None)
def model_to_onnx(cfg: DictConfig) -> None:
    task = cfg.training_type
    print_software_environment()
    print(f"Processing {task}")
    model = load_model(model_dir=MODEL_DIR, task=task, cfg=cfg)

    # Create dummy input as is used by the ParticleTransformer model
    dummy_inputs = {
        "cand_features": torch.randn(1, 10, 16),
        "cand_kinematics_pxpypze":  torch.randn(1, 4, 16),
        "cand_mask": torch.ones(1, 1, 16)#, dtype=torch.bool)
    }

    # Convert the model to ONNX format
    onnx_model_path = os.path.join(MODEL_DIR, f"{task}.onnx")
    torch.onnx.export(
        model,
        tuple(dummy_inputs.values()),
        onnx_model_path,
        opset_version=13,
        verbose=False,
        input_names=list(dummy_inputs.keys()),
        output_names=["prediction"],
        # dynamo=True
    )
    validate_conversion(inputs=dummy_inputs, onnx_model_path=onnx_model_path, pytorch_model=model)


# ONNX graph will explicitly contain the standard attention computations (matmuls + softmax), not a fused "Attention" node. This can improve compatibility but sacrifices performance.


if __name__ == '__main__':
    model_to_onnx()