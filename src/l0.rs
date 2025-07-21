use std::collections::HashMap;

use protobuf::Enum;

use crate::l_base::TranslateFrom;
use crate::protos::onnx::tensor_proto::DataType;
use crate::protos::onnx::tensor_shape_proto::{Dimension, dimension};
use crate::protos::onnx::{self, TensorShapeProto, ValueInfoProto, type_proto};

#[derive(Clone, Debug)]
pub struct IR {
    pub instructions: Vec<Instruction>,
    pub inputs: Vec<Input>,
    pub outputs: Vec<Output>,
}

#[derive(Clone, Debug)]
pub enum Instruction {
    BinOp(Ty, BinOp<Value>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Value {
    Parameter(Parameter),
    Temporary(Temporary),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Parameter(pub usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Temporary(pub usize);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Input {
    pub name: String,
    pub ty: Ty,
    pub shape: Vec<usize>,
    pub batch: Option<Batch>,
}

#[derive(Clone, Debug)]
pub struct Output {
    pub name: String,
    pub value: Value,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Batch {
    First,
    Last,
}

#[derive(Clone, Copy, Debug)]
pub enum BinOpKind {
    Add,
}

#[derive(Clone, Debug)]
pub struct BinOp<R> {
    pub kind: BinOpKind,
    pub lhs: R,
    pub rhs: R,
    pub shapes: BinOpShapes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ty {
    F32,
    S32,
    U32,
}

#[derive(Clone, Debug)]
pub enum BinOpShapes {
    SS,                                  // scalar-scalar
    TS(Vec<usize>),                      // tensor-scalar
    TTExact(Vec<usize>),                 // tensor-tensor with same shape
    TTBroadcast(Vec<usize>, Vec<usize>), // tensor-tensor with broadcast
}

#[derive(Default)]
struct IRBuilder<'a> {
    model: &'a onnx::ModelProto,
    instructions: Vec<Instruction>,
    inputs: Vec<Input>,
    outputs: Vec<Output>,
    input_map: HashMap<&'a str, usize>,
    tmp_count: usize,
}

#[derive(Debug)]
pub enum IRBuildError {
    InputNotTensor(usize),
    UntypedInput(usize),
    InvalidElementType(usize),
    ShapelessTensor(usize),
    MissingName(usize),
    MissingDimension { index: usize, dimension: usize },
}

type Result<V> = core::result::Result<V, IRBuildError>;

impl<'a> IRBuilder<'a> {
    fn new(model: &'a onnx::ModelProto) -> Self {
        Self {
            model,
            inputs: Vec::with_capacity(model.graph.input.len()),
            outputs: Vec::with_capacity(model.graph.output.len()),
            input_map: HashMap::with_capacity(model.graph.input.len()),
            tmp_count: 0,
            ..Default::default()
        }
    }

    fn build(mut self) -> Result<IR> {
        self.build_inputs()?;
        self.build_instructions()?;
        Ok(IR {
            instructions: self.instructions,
            inputs: self.inputs.to_vec(),
            outputs: self.outputs,
        })
    }

    fn build_inputs(&mut self) -> Result<()> {
        for (i, input) in self.model.graph.input.iter().enumerate() {
            let ValueInfoProto {
                name,
                type_: raw_ty,
                ..
            } = input;
            let name = name.as_ref().ok_or(IRBuildError::MissingName(i))?;

            let tensor_ty = raw_ty
                .as_ref()
                .and_then(|raw_ty| raw_ty.value.as_ref())
                .ok_or(IRBuildError::UntypedInput(i))
                .and_then(|ty_val| match ty_val {
                    type_proto::Value::TensorType(t) => Ok(t),
                    _ => Err(IRBuildError::InputNotTensor(i)),
                })?;

            let elem_ty = DataType::from_i32(tensor_ty.elem_type())
                .map(parse_ty)
                .ok_or(IRBuildError::InvalidElementType(i))?;

            let (dims, batch_config) = tensor_ty
                .shape
                .as_ref()
                .ok_or(IRBuildError::ShapelessTensor(i))
                .and_then(|shape_proto| parse_shape(i, shape_proto))?;

            self.input_map.insert(name.as_str(), i);

            self.inputs.push(Input {
                name: name.clone(),
                ty: elem_ty,
                shape: dims,
                batch: batch_config,
            });
        }

        Ok(())
    }

    fn build_instructions(&mut self) -> Result<()> {
        Ok(())
    }
}

impl TranslateFrom<onnx::ModelProto> for IR {
    type Error = IRBuildError;
    fn translate_from(model: &onnx::ModelProto) -> Result<IR> {
        let builder = IRBuilder::new(model);
        builder.build()
    }
}

fn parse_ty(ty: DataType) -> Ty {
    match ty {
        DataType::FLOAT => Ty::F32,
        DataType::INT32 => Ty::S32,
        DataType::UINT32 => Ty::U32,
        _ => todo!(),
    }
}

fn parse_shape(
    idx: usize,
    shape: &TensorShapeProto,
) -> Result<(Vec<usize>, Option<Batch>)> {
    let TensorShapeProto { dim: raw_dims, .. } = shape;
    let batch = if raw_dims.len() > 1 {
        let Dimension { value: first, .. } = &raw_dims[0];
        let Dimension { value: last, .. } = &raw_dims[raw_dims.len() - 1];
        first
            .is_none()
            .then_some(Batch::First)
            .or_else(|| last.is_none().then_some(Batch::Last))
    } else {
        None
    };

    let raw_dims = batch.as_ref().map_or_else(
        || &raw_dims[..],
        |batch| match batch {
            Batch::First => &raw_dims[1..],
            Batch::Last => &raw_dims[..raw_dims.len() - 1],
        },
    );

    let mut dims = Vec::with_capacity(raw_dims.len());
    for (j, raw_dim) in raw_dims.iter().enumerate() {
        let Dimension { value, .. } = raw_dim;
        let value = value
            .as_ref()
            .ok_or(IRBuildError::MissingDimension {
                index: idx,
                dimension: j,
            })
            .and_then(|v| match v {
                &dimension::Value::DimValue(v) => Ok(v),
                dimension::Value::DimParam(_) => {
                    todo!("Named parameter dimensions")
                }
            })?;
        dims.push(value as usize);
    }

    Ok((dims, batch))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    use protobuf::Message;

    use crate::protos::onnx::ModelProto;

    #[test]
    fn test_translate_simple_add() {
        let models_dir = Path::new(file!())
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("test_models");

        let graph_path = models_dir.join("simple_add.onnx");
        let bytes = std::fs::read(graph_path).unwrap();
        let model = ModelProto::parse_from_bytes(&bytes[..]).unwrap();
        let ir = IR::translate_from(&model).unwrap();
        assert_eq!(ir.inputs.len(), 2);
        assert_eq!(ir.inputs[0], Input {
            name: "X".to_string(),
            ty: Ty::F32,
            shape: vec![3, 2],
            batch: None
        });
        assert_eq!(ir.inputs[1], Input {
            name: "Y".to_string(),
            ty: Ty::F32,
            shape: vec![3, 2],
            batch: None
        });
        eprintln!("{:#?}", ir);
    }
}
