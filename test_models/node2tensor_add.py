import onnx
from onnx import helper, AttributeProto, TensorProto, GraphProto

A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [3, 2])
B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [3, 2])
A_plus_B = helper.make_tensor_value_info("A+B", TensorProto.FLOAT, [3, 2])
C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [3, 2])

D = helper.make_tensor_value_info("(A+B)+C", TensorProto.FLOAT, [3, 2])

node_def = [
    helper.make_node("Add", ["A", "B"], ["A+B"]),
    helper.make_node("Add", ["A+B", "C"], ["(A+B)+C"]),
]

graph_def = helper.make_graph(node_def, "test_model", [A, B, C], [D])

model_def = helper.make_model(graph_def, producer_name="onnx-example")

onnx.checker.check_model(model_def)

onnx.save(model_def, "node2tensor_add.onnx")