import tvm
from tvm import relay
from tvm.contrib import ndk
from tvm.runtime.vm import VirtualMachine
from tvm.contrib import graph_executor

import os
import numpy as np
import onnx

from tvm.target import Target


USE_VM = True
RUN_ON_HOST = True

target_c = "opencl -device=adreno"
if RUN_ON_HOST:
    target_h = "llvm -mcpu=tigerlake"
    target_h = "llvm"
else:
    target_h = "llvm -mtriple=arm64-linux-android"


def get_model():
    dtype = "float32"
    input_name = "data"
    input_shape = (relay.Any(), 3, 224, 224)
    filter_shape = (64, 3, 7, 7)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)

    D = relay.nn.conv2d(
        A,
        B,
        padding=[3, 3, 3, 3],
        strides=[2, 2],
        channels=64,
        kernel_size=[7, 7],
        out_dtype=dtype,
    )

    D = relay.nn.max_pool2d(D, pool_size=[3, 3], strides=[2, 2], padding=[1, 1, 1, 1])

    mod = relay.Function([A, B], D)
    np.random.seed(0)
    filter_data = np.zeros(filter_shape).astype(dtype)
    params1 = {
        "weight": tvm.nd.array(filter_data),
    }
    module = tvm.IRModule({})
    module["main"] = mod
    input_shape = (1, 3, 224, 224)
    return module, params1, input_name, input_shape


def get_ssd_model():
    dtype = "float32"
    input_name = "data"
    input_shape = (1, 3, 1200, 1200)
    filter_shape = (64, 3, 7, 7)
    filter_shape2 = (64, 64, 3, 3)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    E = relay.var("weight2", shape=filter_shape2, dtype=dtype)

    D = relay.nn.conv2d(
        A,
        B,
        padding=[3, 3, 3, 3],
        strides=[2, 2],
        channels=64,
        kernel_size=[7, 7],
        out_dtype=dtype,
    )
    D = relay.nn.relu(D)

    MP = relay.nn.max_pool2d(D, pool_size=[3, 3], strides=[2, 2], padding=[1, 1, 1, 1])

    D = relay.nn.conv2d(
        MP,
        E,
        padding=[1, 1, 1, 1],
        channels=64,
        kernel_size=[3, 3],
        out_dtype=dtype,
    )
    #D = relay.nn.relu(D)

    #D = relay.nn.conv2d(
    #    D,
    #    E,
    #    padding=[1, 1, 1, 1],
    #    channels=64,
    #    kernel_size=[3, 3],
    #    out_dtype=dtype,
    #)
    #D = relay.add(D, MP)
    #D = relay.nn.relu(D)

    mod = relay.Function([A, B, E], D)
    np.random.seed(0)
    filter_data = np.zeros(filter_shape).astype(dtype)
    filter_data2 = np.zeros(filter_shape2).astype(dtype)
    params1 = {
        "weight": tvm.nd.array(filter_data),
        "weight2": tvm.nd.array(filter_data2),
    }
    module = tvm.IRModule({})
    module["main"] = mod
    return module, params1, input_name, input_shape


def download_resnet_model():
    print("Download model...")
    model_file = "resnet50-v2-7.onnx"
    model_url = "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx"
    if not os.path.exists(model_file):
        import urllib.request
        urllib.request.urlretrieve(model_url, model_file)
    return model_file


def get_resnet_model():
    model_file = download_resnet_model()
    print("Import model...")
    onnx_model = onnx.load(model_file)
    input_name = "data"
    input_shape = (1, 3, 224, 224)
    #shape_dict = {
    #    input_name: [*input_shape]
    #}
    #model, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)
    model, params = relay.frontend.from_onnx(onnx_model, freeze_params=True)
    print("=" * 10)
    print(model)
    print("=" * 10)
    return model, params, input_name, input_shape


def download_resnet_ssd_model():
    print("Download model...")
    model_file = "resnet-ssd.onnx"
    model_url = "https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/ssd/model/ssd-12.onnx"
    if not os.path.exists(model_file):
        import urllib.request
        urllib.request.urlretrieve(model_url, model_file)
    return model_file


def get_resnet_ssd_model():
    model_file = download_resnet_ssd_model()
    print("Import model...")
    onnx_model = onnx.load(model_file)
    input_name = "image"
    input_shape = (1, 3, 1200, 1200)
    #shape_dict = {
    #    input_name: [*input_shape]
    #}
    #model, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)
    model, params = relay.frontend.from_onnx(onnx_model, freeze_params=True)
    print("=" * 10)
    print(model)
    print("=" * 10)
    return model, params, input_name, input_shape


def compile_model_for_vm(name, model, params, target):
    with tvm.transform.PassContext(opt_level=3):
        vmc = relay.vm.compile(model, target=target, params=params)
        if RUN_ON_HOST:
            vmc.mod.export_library(f"{name}.vm.so")
        else:
            vmc.mod.export_library(f"{name}.vm.so", ndk.create_shared)
        text_file = open(f"{name}.vm.json", "w")
        text_file.write(vmc.bytecode)
        text_file.close()

    return vmc


def compile_model_for_ge(name, model, params, target):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(model, target=target, params=params)
        if RUN_ON_HOST:
            lib.export_library(f"{name}.graph.so")
        else:
            lib.export_library(f"{name}.graph.so", ndk.create_shared)
    return lib


if __name__ == '__main__':
    target = Target(target_c, host=target_h)
    #name = "resnet_vm_model"
    #model, params, input_name, input_shape = get_resnet_model()
    #name = "my_conv2d"
    #model, params, input_name, input_shape = get_model()
    name = "resnet_ssd_vm_model"
    model, params, input_name, input_shape = get_resnet_ssd_model()
    #name = "my_ssd_vm_model"
    #model, params, input_name, input_shape = get_ssd_model()
    img = np.random.rand(*input_shape).astype("float32")
    if USE_VM:
        vm_exec = compile_model_for_vm(name, model, params, target)
        dev = tvm.cl()
        vm = VirtualMachine(vm_exec, dev, "naive")
        vm.set_input("main", **{input_name: img})
        tvm_res = vm.run()
    else:
        lib = compile_model_for_ge(name, model, params, target)
        dev = tvm.cl()
        module = graph_executor.GraphModule(lib["default"](dev))
        module.set_input(input_name, img)
        module.run()
