import torch
import onnx
from pancake.logger import setup_logger

l = setup_logger(__name__)

def export_onnx(model, onnx_path, input_tensor):
    """
    :param model: PyTorch model
    :param onnx_path: target location for new onnx model
    :param input: tensor with input data size
    """
    try:
        torch.onnx.export(
            model, input_tensor, onnx_path, 
            input_names=["input"], output_names=["output"], 
            opset_version=12,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            training=torch.onnx.TrainingMode.EVAL,
            do_constant_folding=False,
            verbose=True
        )

        # Checks
        model_onnx = onnx.load(onnx_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

    except Exception as e:
        l.critical(f'{prefix} export failure: {e}')