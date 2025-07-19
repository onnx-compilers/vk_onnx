import onnx
from onnx import helper, AttributeProto, TensorProto, GraphProto

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 2])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 2])
Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [3, 2])

node_def = [
    helper.make_node("Add", ["X", "Y"], ["Z"])
]

graph_def = helper.make_graph(node_def, "test_model", [X, Y], [Z])

model_def = helper.make_model(graph_def, producer_name="onnx-example")

onnx.save(model_def, "simple_add.onnx")