import onnx
from onnx import helper, AttributeProto, TensorProto, GraphProto

A = helper.make_tensor_value_info("A", TensorProto.UINT32, [None, 2])
B = helper.make_tensor_value_info("B", TensorProto.UINT32, [None, 2])
C = helper.make_tensor_value_info("C", TensorProto.UINT32, [None, 2])

node_def = [
    helper.make_node("Add", ["A", "B"], ["C"])
]

graph_def = helper.make_graph(node_def, "test_model", [A, B], [C])

model_def = helper.make_model(graph_def, producer_name="onnx-example")

onnx.checker.check_model(model_def)

onnx.save(model_def, "batched_add.onnx")