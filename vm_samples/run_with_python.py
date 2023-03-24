import tvm
from tvm import relay
from tvm.contrib import ndk
from tvm.runtime.vm import VirtualMachine

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
    input_shape = (1, 512, 28, 28)
    filter_shape = (512, 512, 3, 3)
    bias_shape = (1, 512, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

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
    D = relay.op.add(D, bias)
    D = relay.op.nn.relu(D)

    mod = relay.Function([A, B, bias], D)
    np.random.seed(0)
    filter_data = np.zeros(filter_shape).astype(dtype)
    bias_data = np.zeros(bias_shape).astype(dtype)
    params1 = {
        "weight": tvm.nd.array(filter_data),
        "bias": tvm.nd.array(bias_data),
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
    shape_dict = {
        input_name: [*input_shape]
    }
    model, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)
    return model, params, input_name, input_shape

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

def create_conv_lib_vm():
    name = "my_conv2d"
    model, params, input_name, input_shape  = get_model()
    target = Target(target_c, host=target_h)
    #model, params = relay.frontend.from_onnx(onnx_model, freeze_params=True)

    with tvm.transform.PassContext(opt_level=3):
        vmc = relay.vm.compile(model, target=target, params=params)
        if RUN_ON_HOST:
            vmc.mod.export_library(f"{name}.vm.so")
        else:
            vmc.mod.export_library(f"{name}.vm.so", ndk.create_shared)
        text_file = open(f"{name}.vm.json", "w")
        text_file.write(vmc.bytecode)
        text_file.close()

    #code, lib = vmc.save()
    #if RUN_ON_HOST:
    #    lib.export_library(f"{name}.vm.kernels.so")
    #else:
    #    lib.export_library(f"{name}.vm.kernels.so", ndk.create_shared)
    #vmc.move_late_bound_consts(f"{name}.vm.const", byte_limit=256)
    #with open(f"{name}.vm.ro", "wb") as fo:
    #    fo.write(code)
    return vmc, input_name, input_shape

def create_resnet_lib_vm():
    name = "resnet_vm_model"
    model, params, input_name, input_shape  = get_resnet_model()
    target = Target(target_c, host=target_h)

    with tvm.transform.PassContext(opt_level=3):
        vmc = relay.vm.compile(model, target=target, params=params)
        if RUN_ON_HOST:
            vmc.mod.export_library(f"{name}.vm.so")
        else:
            vmc.mod.export_library(f"{name}.vm.so", ndk.create_shared)
        text_file = open(f"{name}.vm.json", "w")
        text_file.write(vmc.bytecode)
        text_file.close()

    #code, lib = vmc.save()
    #if RUN_ON_HOST:
    #    lib.export_library(f"{name}.vm.kernels.so")
    #else:
    #    lib.export_library(f"{name}.vm.kernels.so", ndk.create_shared)
    #vmc.move_late_bound_consts(f"{name}.vm.const", byte_limit=256)
    #with open(f"{name}.vm.ro", "wb") as fo:
    #    fo.write(code)
    return vmc, input_name, input_shape

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
        #vm_exec, input_name, input_shape = create_conv_lib_vm()
        vm_exec, input_name, input_shape = create_resnet_lib_vm()
        dev = tvm.cl()
        vm = VirtualMachine(vm_exec, dev, "naive")
        img = np.random.rand(*input_shape).astype("float32")
        vm.set_input("main", **{input_name: img})
        tvm_res = vm.run()
    else:
        create_lib_graph()



