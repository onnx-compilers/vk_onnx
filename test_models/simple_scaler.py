import onnx
from onnx import helper, AttributeProto, TensorProto, GraphProto

# shape = [3, 2, 500, 20]
shape = [3, 2]

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, shape)

node_def = [
    helper.make_node("Scaler", ["X"], ["Y"], offset=[1.0], scale=[2.0])
]

graph_def = helper.make_graph(node_def, "test_model", [X], [Y])

model_def = helper.make_model(graph_def, producer_name="onnx-example")

onnx.save(model_def, "simple_scaler.onnx")