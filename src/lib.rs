// use rspirv::binary::{Assemble, Disassemble};
// use vulkano::VulkanLibrary;
// use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};

pub mod protos;
pub mod l_base;
pub mod l0;

#[cfg(test)]
mod tests {
    // use super::*;
    use std::path::Path;

    use protobuf::{Enum, Message};

    use crate::protos::onnx::{self, ModelProto};

    #[test]
    fn test_compile_add() {
        let models_dir = Path::new(file!())
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("test_models");

        let graph_path = models_dir.join("simple_tripple_add.onnx");
        let bytes = std::fs::read(graph_path).unwrap();
        let model = ModelProto::parse_from_bytes(&bytes[..]).unwrap();
        eprintln!("{:#?}", model);
    }
}