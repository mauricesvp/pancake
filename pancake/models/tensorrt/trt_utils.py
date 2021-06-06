import torch
import onnx
from pancake.logger import setup_logger
from pancake.utils.general import check_requirements, file_size

l = setup_logger(__name__)

def export_onnx(
    model, 
    onnx_path, 
    input_tensor, 
    dynamic_axes: bool=True, 
    simplify: bool=True
    ):
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
            dynamic_axes={'input': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
                          'output': {0: 'batch', 2: 'y', 3: 'x'}} if dynamic_axes else None,
            verbose=False
        )

        # Checks
        model_onnx = onnx.load(onnx_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify
        if simplify:
            try:
                check_requirements(['onnx-simplifier'])
                import onnxsim

                l.info(f'Simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(
                    model_onnx,
                    dynamic_input_shape=dynamic_axes,
                    input_shapes={'input': list(input_tensor.shape)} if dynamic_axes else None)
                assert check, 'assert check failed'
                onnx.save(model_onnx, onnx_path)
            except Exception as e:
                l.info(f'Simplifier failure: {e}')
                return

        l.info(f'Export success, saved as {onnx_path} ({file_size(onnx_path):.1f} MB)')
    except Exception as e:
        l.critical(f'Export failure: {e}')