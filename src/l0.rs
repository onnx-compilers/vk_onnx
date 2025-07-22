use std::collections::HashMap;
use std::fmt::Debug;

use protobuf::Enum;

use crate::l_base::TranslateFrom;
use crate::protos::onnx::tensor_proto::DataType;
use crate::protos::onnx::tensor_shape_proto::{Dimension, dimension};
use crate::protos::onnx::{self, TensorShapeProto, ValueInfoProto, type_proto};

#[derive(Clone, Debug)]
pub struct IR {
    pub operations: Vec<Operation>,
    pub input_names: Vec<String>,
    pub inputs: Vec<Input>,
    pub output_names: Vec<String>,
    pub outputs: Vec<Output>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Operation {
    BinOp(BinOp<Value>),
}

#[derive(Clone, Copy, PartialEq, Eq)]
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
    pub ty: Ty,
    pub shape: Vec<Option<usize>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Output {
    pub value: Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOpKind {
    Add,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BinOp<R: PartialEq + Eq> {
    pub kind: BinOpKind,
    pub lhs: R,
    pub rhs: R,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ty {
    F32,
    S32,
    U32,
}

#[derive(Default)]
struct IRBuilder<'a> {
    model: &'a onnx::ModelProto,
    operations: Vec<Operation>,
    inputs: Vec<Input>,
    outputs: Vec<Output>,
    input_map: HashMap<&'a str, usize>,
    output_map: HashMap<&'a str, Temporary>,
    tmp_count: usize,
}

#[derive(Debug)]
pub enum IRBuildError {
    InputNotTensor(usize),
    UntypedInput(usize),
    InvalidElementType(usize),
    ShapelessTensor(usize),
    MissingName(usize),
    MissingOpName(usize),
    WrongInputCount { index: usize, expected: usize },
    WrongOutputCount { index: usize, expected: usize },
    UnknownInput { index: usize, nth: usize },
    UnknownOutput(usize),
}

type Result<V> = core::result::Result<V, IRBuildError>;

impl From<usize> for Parameter {
    fn from(idx: usize) -> Self {
        Self(idx)
    }
}

impl From<usize> for Temporary {
    fn from(idx: usize) -> Self {
        Self(idx)
    }
}

impl From<Parameter> for Value {
    fn from(p: Parameter) -> Self {
        Self::Parameter(p)
    }
}

impl From<Temporary> for Value {
    fn from(t: Temporary) -> Self {
        Self::Temporary(t)
    }
}

impl Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Parameter(Parameter(i)) => write!(f, "Parameter({})", i),
            Value::Temporary(Temporary(i)) => write!(f, "Temporary({})", i),
        }
    }
}

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
        self.build_operations()?;
        self.build_outputs()?;
        Ok(IR {
            operations: self.operations,
            inputs: self.inputs,
            input_names: self.input_map.keys().map(|k| k.to_string()).collect(),
            outputs: self.outputs,
            output_names: self.output_map.keys().map(|k| k.to_string()).collect(),
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

            let dims = tensor_ty
                .shape
                .as_ref()
                .ok_or(IRBuildError::ShapelessTensor(i))
                .and_then(|shape_proto| parse_shape(shape_proto))?;

            self.input_map.insert(name.as_str(), i);
            self.inputs.push(Input {
                ty: elem_ty,
                shape: dims,
            });
        }

        Ok(())
    }

    fn build_operations(&mut self) -> Result<()> {
        for (i, node) in self.model.graph.node.iter().enumerate() {
            let op_type = node
                .op_type
                .as_ref()
                .ok_or(IRBuildError::MissingOpName(i))?;
            let tmp = self.make_temporary();
            let (op, output_name) = match op_type.as_str() {
                "Add" => {
                    if node.input.len() != 2 {
                        return Err(IRBuildError::WrongInputCount {
                            index: i,
                            expected: 2,
                        });
                    }
                    if node.output.len() != 1 {
                        return Err(IRBuildError::WrongOutputCount {
                            index: i,
                            expected: 1,
                        });
                    }
                    let lhs_name = node.input[0].as_str();
                    let rhs_name = node.input[1].as_str();
                    let lhs = self.find_operand(lhs_name).ok_or_else(|| {
                        IRBuildError::UnknownInput { index: i, nth: 0 }
                    })?;
                    let rhs = self.find_operand(rhs_name).ok_or_else(|| {
                        IRBuildError::UnknownInput { index: i, nth: 1 }
                    })?;
                    let output_name = node.output[0].as_str();
                    (
                        Operation::BinOp(BinOp {
                            kind: BinOpKind::Add,
                            lhs,
                            rhs,
                        }),
                        output_name,
                    )
                }
                _ => todo!(),
            };
            self.operations.push(op);
            self.output_map.insert(output_name, tmp);
        }

        Ok(())
    }

    fn build_outputs(&mut self) -> Result<()> {
        for (i, output) in self.model.graph.output.iter().enumerate() {
            let name =
                output.name.as_ref().ok_or(IRBuildError::MissingName(i))?;
            let value = self
                .find_operand(name.as_str())
                .ok_or_else(|| IRBuildError::UnknownOutput(i))?;
            self.outputs.push(Output { value });
        }
        Ok(())
    }

    fn make_temporary(&mut self) -> Temporary {
        let idx = self.tmp_count;
        self.tmp_count += 1;
        Temporary(idx)
    }

    fn find_operand(&self, name: &str) -> Option<Value> {
        self.input_map
            .get(name)
            .copied()
            .map(|idx| Value::Parameter(idx.into()))
            .or_else(|| {
                self.output_map
                    .get(name)
                    .copied()
                    .map(|idx| Value::Temporary(idx.into()))
            })
    }
}

impl TranslateFrom<onnx::ModelProto> for IR {
    type Error = IRBuildError;
    fn translate_from(model: onnx::ModelProto) -> Result<IR> {
        let builder = IRBuilder::new(&model);
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

fn parse_shape(shape: &TensorShapeProto) -> Result<Vec<Option<usize>>> {
    let TensorShapeProto { dim: raw_dims, .. } = shape;
    let mut dims = Vec::with_capacity(raw_dims.len());
    for raw_dim in raw_dims.iter() {
        let Dimension { value, .. } = raw_dim;
        let value = value
            .as_ref()
            .and_then(|v| match v {
                &dimension::Value::DimValue(v) => Some(v),
                dimension::Value::DimParam(_) => {
                    todo!("Named parameter dimensions")
                }
            })
            .map(|v| v as usize);
        dims.push(value);
    }

    Ok(dims)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    use protobuf::Message;

    use crate::protos::onnx::ModelProto;

    fn project_path() -> Box<Path> {
        Path::new(file!())
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .into()
    }

    #[test]
    fn test_translate_simple_add() {
        let models_dir = project_path().join("test_models");

        let graph_path = models_dir.join("simple_add.onnx");
        let bytes = std::fs::read(graph_path).unwrap();
        let model = ModelProto::parse_from_bytes(&bytes[..]).unwrap();
        let ir = IR::translate_from(model).unwrap();
        assert_eq!(ir.inputs.len(), 2);
        assert_eq!(ir.input_names, vec!["X", "Y"]);
        assert_eq!(
            ir.inputs[0],
            Input {
                ty: Ty::F32,
                shape: [3, 2].into_iter().map(Some).collect(),
            }
        );
        assert_eq!(
            ir.inputs[1],
            Input {
                ty: Ty::F32,
                shape: [3, 2].into_iter().map(Some).collect(),
            }
        );

        assert_eq!(ir.outputs.len(), 1);
        assert_eq!(ir.output_names, vec!["Z"]);
        assert_eq!(
            ir.outputs[0],
            Output {
                value: Value::Temporary(Temporary(0)),
            }
        );

        assert_eq!(ir.operations.len(), 1);
        assert_eq!(
            ir.operations[0],
            Operation::BinOp(BinOp {
                kind: BinOpKind::Add,
                lhs: Value::Parameter(Parameter(0)),
                rhs: Value::Parameter(Parameter(1)),
            })
        );

        eprintln!("{:#?}", ir);
    }
}