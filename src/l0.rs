use std::collections::HashMap;
use std::fmt::Debug;

use protobuf::Enum;

use crate::l_base::{ScalarTy, Translate};
use crate::protos::onnx::tensor_proto::DataType;
use crate::protos::onnx::tensor_shape_proto::{Dimension, dimension};
use crate::protos::onnx::{self, TensorShapeProto, ValueInfoProto, type_proto};

#[derive(Clone, Debug, Default)]
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
    pub ty: ScalarTy,
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

#[derive(Default)]
pub struct IRBuilder {
    operations: Vec<Operation>,
    inputs: Vec<Input>,
    outputs: Vec<Output>,
    tmp_count: usize,
}

struct IOMap<'a> {
    input_map: HashMap<&'a str, Parameter>,
    output_map: HashMap<&'a str, Temporary>,
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

impl IRBuilder {
    fn build_inputs<'a>(
        &mut self,
        input_map: &mut HashMap<&'a str, Parameter>,
        graph: &'a onnx::GraphProto,
    ) -> Result<()> {
        for (i, input) in graph.input.iter().enumerate() {
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

            self.inputs.push(Input {
                ty: elem_ty,
                shape: dims,
            });
            input_map.insert(name.as_str(), Parameter(i));
        }

        Ok(())
    }

    fn build_operations<'a>(
        &mut self,
        io_map: &mut IOMap<'a>,
        graph: &'a onnx::GraphProto,
    ) -> Result<()> {
        for (i, node) in graph.node.iter().enumerate() {
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
                    let lhs =
                        io_map.find_operand(lhs_name).ok_or_else(|| {
                            IRBuildError::UnknownInput { index: i, nth: 0 }
                        })?;
                    let rhs =
                        io_map.find_operand(rhs_name).ok_or_else(|| {
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
            io_map.output_map.insert(output_name, tmp);
        }

        Ok(())
    }

    fn build_outputs(
        &mut self,
        io_map: &IOMap,
        graph: &onnx::GraphProto,
    ) -> Result<()> {
        for (i, output) in graph.output.iter().enumerate() {
            let name =
                output.name.as_ref().ok_or(IRBuildError::MissingName(i))?;
            let value = io_map
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
}

impl<'a> IOMap<'a> {
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

impl Translate<onnx::ModelProto, IR> for IRBuilder {
    type Config = ();
    type Error = IRBuildError;
    fn translate(
        mut self,
        raw: onnx::ModelProto,
        _config: &Self::Config,
    ) -> Result<IR> {
        let graph = raw.graph.unwrap();
        let mut io_map = IOMap {
            input_map: HashMap::with_capacity(graph.input.len()),
            output_map: HashMap::with_capacity(graph.node.len()),
        };
        {
            self.build_inputs(&mut io_map.input_map, &graph)?;
            self.build_operations(&mut io_map, &graph)?;
            self.build_outputs(&io_map, &graph)?;
        }
        let IRBuilder {
            operations,
            inputs,
            outputs,
            ..
        } = self;
        let onnx::GraphProto {
            input: raw_inputs,
            output: raw_outputs,
            ..
        } = graph;
        let input_names = raw_inputs
            .into_iter()
            .map(|onnx::ValueInfoProto { name, .. }| name.unwrap())
            .collect();
        let output_names = raw_outputs
            .into_iter()
            .map(|onnx::ValueInfoProto { name, .. }| name.unwrap())
            .collect();
        Ok(IR {
            operations,
            inputs,
            input_names,
            outputs,
            output_names,
        })
    }
}

fn parse_ty(ty: DataType) -> ScalarTy {
    match ty {
        DataType::FLOAT => ScalarTy::F32,
        DataType::INT32 => ScalarTy::S32,
        DataType::UINT32 => ScalarTy::U32,
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
pub mod test_utils {
    use std::path::Path;

    pub fn project_path() -> Box<Path> {
        Path::new(file!())
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::test_utils::project_path;

    use protobuf::Message;

    use crate::protos::onnx::ModelProto;

    #[test]
    fn test_translate_simple_add() {
        let models_dir = project_path().join("test_models");

        let graph_path = models_dir.join("simple_add.onnx");
        let bytes = std::fs::read(graph_path).unwrap();
        let model = ModelProto::parse_from_bytes(&bytes[..]).unwrap();
        let ir = IRBuilder::default().translate(model, &()).unwrap();
        assert_eq!(ir.inputs.len(), 2);
        assert!(
            ir.input_names
                .iter()
                .map(String::as_str)
                .all(|name| ["X", "Y"].contains(&name))
        );
        assert_eq!(
            ir.inputs[0],
            Input {
                ty: ScalarTy::F32,
                shape: [3, 2].into_iter().map(Some).collect(),
            }
        );
        assert_eq!(
            ir.inputs[1],
            Input {
                ty: ScalarTy::F32,
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

        // eprintln!("{:#?}", ir);
    }
}