import tvm
from tvm import rpc, relay
from tvm.contrib import utils, xcode, coreml_runtime, ndk
from tvm.contrib.debugger import debug_runtime
from tvm.contrib.download import download_testdata
from tvm import relay, auto_scheduler

import os
import onnx
import numpy as np
from tvm.relay.transform import recast
import argparse
from tvm.relay.expr_functor import ExprMutator, Call
from tvm.relay import transform

from tvm.target import Target


USE_VM = True
RUN_ON_HOST = True

parser = argparse.ArgumentParser(description=
    "tunes network on ios")
required = parser.add_argument_group('required arguments')
required.add_argument('-m', '--input_model', required=True, type=str, help="path to compiled .so file")
required.add_argument('-p', '--precision', required=False, type=str, help="precision to tune")
args = parser.parse_args()

target_c = "opencl -device=adreno"
#target_c = "opencl"
#target_c = "llvm -mcpu=tigerlake"
if RUN_ON_HOST:
    target_h = "llvm -mcpu=tigerlake"
else:
    target_h = "llvm -mtriple=arm64-linux-android"

#target = "opencl -device=intel_graphics"
#target = "opencl -device=IrisXe"
#target = "opencl"


def get_model():
    dtype = "float32"
    input_shape = (1, 512, 28, 28)
    filter_shape = (512, 512, 3, 3)
    #bias_shape = (1, 512, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    #bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    D = relay.nn.conv2d(
        A,
        B,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[1, 1, 1, 1],
        channels=512,
        kernel_size=[3, 3],
        out_dtype=dtype,
    )
    #D = relay.op.add(D, bias)
    #D = relay.op.nn.relu(D)

    #mod = relay.Function([A, B, bias], D)
    mod = relay.Function([A, B], D)
    np.random.seed(0)
    filter_data = np.zeros(filter_shape).astype(dtype)
    #bias_data = np.zeros(bias_shape).astype(dtype)
    params1 = {
        "weight": tvm.nd.array(filter_data),
        #"bias": tvm.nd.array(bias_data),
    }
    module = tvm.IRModule({})
    module["main"] = mod
    return module, params1


#def create_lib():
#    name = args.input_model
#    onnx_model = onnx.load(name)
#    shape_dict = {}
#    #shape_dict["input"] = [relay.Any(), 3, 224, 224]
#    shape_dict["data"] = [1, 3, 224, 224]
#    model, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)
#    target = Target(target_c, host=target_h)
#    #model, params = relay.frontend.from_onnx(onnx_model, freeze_params=True)
#
#    vmc = relay.backend.vm.compile(model, target=target, params=params) 
#    vmc.mod.export_library(f"{name}.vm.so")
#    text_file = open(f"{name}.vm.json", "w")
#    text_file.write(vmc.bytecode)
#    text_file.close()
#
#    code, lib = vmc.save()
#    lib.export_library(f"{name}.vm.kernels.so")
#    vmc.move_late_bound_consts(f"{name}.vm.const", byte_limit=256)
#    with open(f"{name}.vm.ro", "wb") as fo:
#        fo.write(code)

def create_lib_vm():
    name = "my_conv2d"
    model, params = get_model()
    target = Target(target_c, host=target_h)
    #model, params = relay.frontend.from_onnx(onnx_model, freeze_params=True)

    with tvm.transform.PassContext(opt_level=3):
        vmc = relay.backend.vm.compile(model, target=target, params=params)
        if RUN_ON_HOST:
            vmc.mod.export_library(f"{name}.vm.so")
        else:
            vmc.mod.export_library(f"{name}.vm.so", ndk.create_shared)
        text_file = open(f"{name}.vm.json", "w")
        text_file.write(vmc.bytecode)
        text_file.close()

    code, lib = vmc.save()
    if RUN_ON_HOST:
        lib.export_library(f"{name}.vm.kernels.so")
    else:
        lib.export_library(f"{name}.vm.kernels.so", ndk.create_shared)
    vmc.move_late_bound_consts(f"{name}.vm.const", byte_limit=256)
    with open(f"{name}.vm.ro", "wb") as fo:
        fo.write(code)

def create_lib_graph():
    name = "my_conv2d"
    model, params = get_model()
    target = Target(target_c, host=target_h)
    #model, params = relay.frontend.from_onnx(onnx_model, freeze_params=True)

    with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = relay.build(model, target=target, params=params)
        if RUN_ON_HOST:
            lib.export_library(f"{name}.graph.so")
        else:
            lib.export_library(f"{name}.graph.so", ndk.create_shared)


if __name__ == '__main__':
    if USE_VM:
        create_lib_vm()
    else:
        create_lib_graph()


