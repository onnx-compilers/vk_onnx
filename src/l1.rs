use core::result::Result as CoreResult;

use crate::l_base::{ScalarTy, Translate};
use crate::l0;

pub use crate::l0::{Output, Parameter, Temporary, Value};

#[derive(Default, Debug, Clone)]
pub struct IR {
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Output>,
    pub instructions: Vec<Instruction>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Instruction {
    pub op: Operation,
    pub result: Argument,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Argument {
    pub ty: ScalarTy,
    pub shape: Box<[usize]>,
    // pub batch_dim: Option<BatchDimension>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Operation {
    Add {
        lhs: Value,
        rhs: Value,
    },
    Scaler {
        input: Value,
        offset: f32,
        scale: f32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchDimension {
    First,
    Last,
}

#[derive(Debug, Default, Clone)]
pub struct IRBuilder {
    inputs: Vec<Argument>,
    instructions: Vec<Instruction>,
}

#[derive(Debug, Clone)]
pub enum Error {
    MissingInputShape(usize),
    MissingInputDimension(usize, usize),
    InvalidValue(usize, Value),
    OperandTyMismatch(usize),
    ShapeMismatch(usize),
}

type Result<V> = CoreResult<V, Error>;

impl IRBuilder {
    fn build_inputs(&mut self, l0: &l0::IR) -> Result<()> {
        for (i, input) in l0.inputs.iter().enumerate() {
            let (shape, _batch_dim) = parse_shape(&input.shape)
                // NOTE: Maybe put this into a function
                .map_err(|e| match e {
                    ShapeParseError::Shapeless => Error::MissingInputShape(i),
                    ShapeParseError::MissingDimension(d) => {
                        Error::MissingInputDimension(i, d)
                    }
                })?;
            self.inputs.push(Argument {
                ty: input.ty,
                shape: shape.into_boxed_slice(),
                // batch_dim,
            });
        }
        Ok(())
    }

    fn build_instructions(&mut self, l0: &l0::IR) -> Result<()> {
        let layers = &l0.layers;
        for (i, layer) in layers.iter().enumerate() {
            let &l0::Layer {
                kind,
                inputs_range,
                attributes_range,
            } = layer;
            let inputs = l0.get_layer_inputs(inputs_range);
            match kind {
                l0::LayerKind::Add => {
                    let &[lhs, rhs] = inputs else { unreachable!() };
                    let lhs_data = self
                        .get_argument(lhs)
                        .ok_or(Error::InvalidValue(i, lhs))?;
                    let rhs_data = self
                        .get_argument(rhs)
                        .ok_or(Error::InvalidValue(i, rhs))?;
                    let ty = matching_ty(lhs_data.ty, rhs_data.ty)
                        .ok_or(Error::OperandTyMismatch(i))?;
                    let shape /* , batch_dim */ =
                        match_shape(&lhs_data, &rhs_data)
                            .map_err(|()| Error::ShapeMismatch(i))?;
                    let result = Argument {
                        ty,
                        shape,
                        // batch_dim,
                    };
                    let op = Operation::Add { lhs, rhs };
                    self.instructions.push(Instruction { op, result });
                }

                l0::LayerKind::Scaler => {
                    let &[input] = inputs else { unreachable!() };
                    let input_data = self
                        .get_argument(input)
                        .ok_or(Error::InvalidValue(i, input))?;
                    // TODO: Validate input type
                    let attributes =
                        l0.get_layer_attributes(attributes_range.unwrap());
                    let l0::Attribute {
                        data: l0::AttributeData::Floats(offsets),
                        ..
                    } = attributes
                        .iter()
                        .find(|attr| attr.name == "offset")
                        .unwrap()/* else { panic!("Wrong attribute kind") }*/;
                    let l0::Attribute {
                        data: l0::AttributeData::Floats(scales),
                        ..
                    } = attributes
                        .iter()
                        .find(|attr| attr.name == "scale")
                        .unwrap();
                    assert_eq!(offsets.len(), 1);
                    assert_eq!(scales.len(), 1);
                    let offset = offsets[0];
                    let scale = scales[0];
                    let op = Operation::Scaler {
                        input,
                        offset,
                        scale,
                    };
                    let result = Argument {
                        ty: input_data.ty,
                        shape: input_data.shape.clone(),
                    };
                    self.instructions.push(Instruction { op, result });
                }
            }
        }
        Ok(())
    }

    fn get_argument(&self, value: Value) -> Option<&Argument> {
        match value {
            Value::Parameter(Parameter(i)) => self.inputs.get(i),
            Value::Temporary(Temporary(i)) => {
                self.instructions.get(i).map(|instr| &instr.result)
            }
        }
    }
}

fn matching_ty(lhs: ScalarTy, rhs: ScalarTy) -> Option<ScalarTy> {
    if lhs == rhs { Some(lhs) } else { None }
}

fn match_shape(
    lhs: &Argument,
    rhs: &Argument,
) -> CoreResult<Box<[usize]> /* , Option<BatchDimension> */, ()> {
    if lhs.shape == rhs.shape
    /* && lhs.batch_dim == rhs.batch_dim */
    {
        Ok(lhs.shape.clone() /* , lhs.batch_dim */)
    } else {
        Err(())
    }
}

enum ShapeParseError {
    Shapeless,
    MissingDimension(usize),
}

fn parse_shape(
    raw_shape: &[Option<usize>],
) -> CoreResult<(Vec<usize>, Option<BatchDimension>), ShapeParseError> {
    use ShapeParseError::*;

    if raw_shape.is_empty() {
        return Err(Shapeless);
    }

    let (raw_shape, batch_dim) = if let [_] = raw_shape {
        (raw_shape, None)
    } else if let [None, rest @ ..] = raw_shape {
        (rest, Some(BatchDimension::First))
    } else if let [rest @ .., None] = raw_shape {
        (rest, Some(BatchDimension::Last))
    } else {
        (raw_shape, None)
    };

    let shape = raw_shape
        .iter()
        .enumerate()
        .map(|(i, dim)| dim.ok_or(MissingDimension(i)))
        .collect::<CoreResult<Vec<_>, _>>()?;

    Ok((shape, batch_dim))
}

impl Translate<l0::IR, IR> for IRBuilder {
    type Config = ();
    type Error = Error;

    fn translate(mut self, l0: l0::IR, _config: &Self::Config) -> Result<IR> {
        self.inputs.reserve(l0.inputs.len());
        self.instructions.reserve(l0.layers.len());
        self.build_inputs(&l0)?;
        self.build_instructions(&l0)?;

        Ok(IR {
            inputs: self.inputs,
            instructions: self.instructions,
            outputs: l0.outputs,
        })
    }
}

impl IR {
    pub fn make_input(&mut self, argument: Argument) -> Parameter {
        let i = self.inputs.len();
        self.inputs.push(argument);
        Parameter(i)
    }

    pub fn add_output(&mut self, value: Value) {
        self.outputs.push(Output { value });
    }

    pub fn append_op(&mut self, op: Operation, result: Argument) -> Temporary {
        let i = self.instructions.len();
        self.instructions.push(Instruction { op, result });
        Temporary(i)
    }

    pub fn output_data(&self, idx: usize) -> Option<&Argument> {
        let Output { value } = self.outputs.get(idx)?;
        match value {
            &Value::Parameter(Parameter(i)) => self.inputs.get(i),
            &Value::Temporary(Temporary(i)) => {
                self.instructions.get(i).map(|instr| &instr.result)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_add_parse() {
        let shape = Box::new([None, Some(2)]);
        let ir = l0::IR {
            input_names: ["X", "Y"].iter().map(<&str>::to_string).collect(),
            output_names: ["Z"].iter().map(<&str>::to_string).collect(),
            inputs: vec![
                l0::Input {
                    shape: shape.clone(),
                    ty: ScalarTy::F32,
                },
                l0::Input {
                    shape: shape,
                    ty: ScalarTy::F32,
                },
            ],
            outputs: vec![l0::Output {
                value: Value::Parameter(0.into()),
            }],
            layers: vec![l0::Layer {
                kind: l0::LayerKind::Add,
                inputs_range: (0, 2),
                attributes_range: None,
            }],
            layer_inputs: vec![
                Value::Parameter(0.into()),
                Value::Parameter(1.into()),
            ],
            layer_attributes: vec![],
        };

        let ir = IRBuilder::default().translate(ir, &()).unwrap();
        assert_eq!(ir.inputs.len(), 2);
        assert_eq!(ir.outputs.len(), 1);
        assert_eq!(ir.instructions.len(), 1);
    }
}
