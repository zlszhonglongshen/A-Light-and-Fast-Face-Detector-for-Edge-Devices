import logging
import numpy
import sys
sys.path.append('/home/heyonghao/libs/incubator-mxnet/python')
import mxnet
from mxnet.contrib import onnx as onnx_mxnet
from onnx import checker
import onnx


def generate_onnx_file():
    logging.basicConfig(level=logging.INFO)

    symbol_path = '../symbol_farm/symbol_20_320_16L_4scales_v3_deploy.json'
    param_path = '../saved_model/configuration_20_320_16L_4scales_v3_2019-07-18-01-27-03/train_20_320_16L_4scales_v3_iter_1000000.params'
    onnx_path = './onnx_files/v3.onnx'

    net_symbol = mxnet.symbol.load(symbol_path)
    net_params_raw = mxnet.nd.load(param_path)
    net_params = dict()
    for k, v in net_params_raw.items():
        tp, name = k.split(':', 1)
        net_params.update({name: v})

    input_shape = (1, 3, 480, 640)

    converted_model_path = onnx_mxnet.export_model(net_symbol, net_params, [input_shape], numpy.float32, onnx_path, verbose=True)


def check_onnx_file():
    onnx_file_path = './onnx_files/v3.onnx'
    # Load onnx model
    model_proto = onnx.load_model(onnx_file_path)

    # Check if converted ONNX protobuf is valid
    checker.check_graph(model_proto.graph)


if __name__ == '__main__':
    generate_onnx_file()
    check_onnx_file()
