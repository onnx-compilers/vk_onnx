use onnx_ir::Node;
use onnx_ir::ir::{self as onnx, ArgType, Argument};
// use rspirv::binary::{Assemble, Disassemble};
use rspirv::dr::Builder;
use rspirv::spirv;
// use vulkano::VulkanLibrary;
// use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};

pub fn compile_add(b: &mut Builder, node: &Node) -> spirv::Word {
    assert_eq!(node.inputs.len(), 2);
    assert_eq!(node.outputs.len(), 1);
    let lhs = &node.inputs[0];
    let rhs = &node.inputs[1];
    let out = &node.outputs[0];
    let (_shape, ty) = get_shape_and_type(&lhs, &rhs, &out).unwrap();
    let _spirv_ty = match ty {
        onnx::ElementType::Float32 => b.type_float(32),
        onnx::ElementType::Int32 => b.type_int(32, 1),
        _ => unimplemented!(),
    };
    b.memory_model(
        spirv::AddressingModel::Logical,
        spirv::MemoryModel::GLSL450,
    );
    let void = b.type_void();
    let voidf = b.type_function(void, vec![]);
    let fun = b
        .begin_function(
            void,
            None,
            spirv::FunctionControl::DONT_INLINE | spirv::FunctionControl::CONST,
            voidf,
        )
        .unwrap();
    b.begin_block(None).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    fun
}

fn get_shape_and_type<'a>(
    lhs: &'a Argument,
    rhs: &'a Argument,
    out: &'a Argument,
) -> Result<(&'a [usize], onnx::ElementType), ()> {
    match (&lhs.ty, &rhs.ty, &out.ty) {
        (
            ArgType::Tensor(onnx::TensorType {
                elem_type: lhs_type,
                rank: lhs_rank,
                static_shape: Some(lhs_shape),
            }),
            ArgType::Tensor(onnx::TensorType {
                elem_type: rhs_type,
                rank: rhs_rank,
                static_shape: Some(rhs_shape),
            }),
            ArgType::Tensor(onnx::TensorType {
                elem_type: out_type,
                rank: out_rank,
                ..
            }),
        ) if lhs_type == rhs_type
            && lhs_type == out_type
            && lhs_rank == rhs_rank
            && lhs_rank == out_rank
            && lhs_shape == rhs_shape =>
        {
            Ok((&lhs_shape[..], lhs_type.clone()))
        }
        _ => Err(()),
    }
}

#[cfg(test)]
mod tests {
    use onnx_ir::parse_onnx;
    use rspirv::binary::Disassemble;

    use super::*;
    use std::path::Path;

    #[test]
    fn test_compile_add() {
        let mut b = rspirv::dr::Builder::new();
        b.set_version(1, 0);
        b.capability(spirv::Capability::Shader);
        b.memory_model(
            spirv::AddressingModel::Logical,
            spirv::MemoryModel::GLSL450,
        );
        let graph_path = Path::new(file!())
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("test_models")
            .join("simple_add.onnx");
        let graph = parse_onnx(&graph_path);
        assert_eq!(graph.nodes.len(), 1);
        let node = &graph.nodes[0];
        let fun = compile_add(&mut b, node);
        b.entry_point(
            spirv::ExecutionModel::GLCompute,
            fun,
            "simple_add",
            vec![],
        );
        let module = b.module();
        let dis = module.disassemble();
        println!("{}", dis);
    }
}
