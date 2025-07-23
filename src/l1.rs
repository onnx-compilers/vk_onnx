use core::result::Result as CoreResult;

use crate::l_base::TranslateFrom;
use crate::l0;

pub use crate::l0::{Output, Parameter, Temporary, Ty, Value};

#[derive(Debug, Clone)]
pub struct IR {
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Output>,
    pub instructions: Vec<Instruction>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Instruction {
    pub op: Operation,
    pub result: Argument,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Argument {
    pub ty: Ty,
    pub shape: Vec<usize>,
    pub batch_dim: Option<BatchDimension>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operation {
    BinOp(BinOp<Value>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinOp<T: PartialEq + Eq + Clone> {
    pub kind: BinOpKind,
    pub lhs: T,
    pub rhs: T,
    // pub broadcast: BroadcastConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOpKind {
    Add,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchDimension {
    First,
    Last,
}

#[derive(Debug, Clone)]
struct IRBuilder<'a> {
    l0: &'a l0::IR,
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

enum InstructionError {
    TyMismatch,
    InvalidValue(Value),
    ShapeMismatch,
}

type Result<V> = CoreResult<V, Error>;

impl<'a> IRBuilder<'a> {
    fn new(l0: &'a l0::IR) -> Self {
        let inputs = Vec::with_capacity(l0.inputs.len());
        Self {
            l0,
            inputs,
            instructions: Vec::with_capacity(l0.operations.len()),
        }
    }

    fn build(mut self) -> Result<IR> {
        self.build_inputs()?;
        self.build_instructions()?;

        Ok(IR {
            inputs: self.inputs,
            instructions: self.instructions,
            outputs: self.l0.outputs.clone(),
        })
    }

    fn build_inputs(&mut self) -> Result<()> {
        for (i, input) in self.l0.inputs.iter().enumerate() {
            let (shape, batch_dim) = parse_shape(input.shape.as_slice())
                // NOTE: Maybe put this into a function
                .map_err(|e| match e {
                    ShapeParseError::Shapeless => Error::MissingInputShape(i),
                    ShapeParseError::MissingDimension(d) => {
                        Error::MissingInputDimension(i, d)
                    }
                })?;
            self.inputs.push(Argument {
                ty: input.ty,
                shape,
                batch_dim,
            });
        }
        Ok(())
    }

    fn build_instructions(&mut self) -> Result<()> {
        let operations = &self.l0.operations;
        for (i, op) in operations.iter().enumerate() {
            match op {
                &l0::Operation::BinOp(l0::BinOp { kind, lhs, rhs }) => {
                    let instr = self.make_binop(kind, lhs, rhs).map_err(
                        |e| match e {
                            InstructionError::TyMismatch => {
                                Error::OperandTyMismatch(i)
                            }
                            InstructionError::InvalidValue(v) => {
                                Error::InvalidValue(i, v)
                            }
                            InstructionError::ShapeMismatch => {
                                Error::ShapeMismatch(i)
                            }
                        },
                    )?;
                    self.instructions.push(instr);
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

    fn make_binop(
        &self,
        kind: l0::BinOpKind,
        lhs: Value,
        rhs: Value,
    ) -> CoreResult<Instruction, InstructionError> {
        use InstructionError::*;
        let lhs_data = self.get_argument(lhs).ok_or(InvalidValue(lhs))?;
        let rhs_data = self.get_argument(rhs).ok_or(InvalidValue(rhs))?;
        let (op, result) = match kind {
            l0::BinOpKind::Add => {
                let result = self.make_add(lhs_data, rhs_data)?;
                let op = Operation::BinOp(BinOp {
                    kind: BinOpKind::Add,
                    lhs,
                    rhs,
                });
                (op, result)
            }
        };

        Ok(Instruction { op, result })
    }

    fn make_add(
        &self,
        lhs: &Argument,
        rhs: &Argument,
    ) -> CoreResult<Argument, InstructionError> {
        let ty = match_ty(lhs.ty, rhs.ty)?;
        let (shape, batch_dim) = match_shape(&lhs, &rhs)?;

        Ok(Argument {
            ty,
            shape,
            batch_dim,
        })
    }
}

fn match_ty(lhs: Ty, rhs: Ty) -> CoreResult<Ty, InstructionError> {
    if lhs == rhs {
        Ok(lhs)
    } else {
        Err(InstructionError::TyMismatch)
    }
}

fn match_shape(
    lhs: &Argument,
    rhs: &Argument,
) -> CoreResult<(Vec<usize>, Option<BatchDimension>), InstructionError> {
    if lhs.shape == rhs.shape && lhs.batch_dim == rhs.batch_dim {
        Ok((lhs.shape.clone(), lhs.batch_dim))
    } else {
        Err(InstructionError::ShapeMismatch)
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

impl TranslateFrom<l0::IR> for IR {
    type Error = Error;
    fn translate_from(ir: l0::IR) -> Result<Self> {
        IRBuilder::new(&ir).build()
    }
}

impl IR {
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
        let ir = l0::IR {
            input_names: ["X", "Y"].iter().map(<&str>::to_string).collect(),
            output_names: ["Z"].iter().map(<&str>::to_string).collect(),
            inputs: vec![
                l0::Input {
                    shape: vec![None, Some(2)],
                    ty: l0::Ty::F32,
                },
                l0::Input {
                    shape: vec![None, Some(2)],
                    ty: l0::Ty::F32,
                },
            ],
            outputs: vec![l0::Output {
                value: Value::Parameter(0.into()),
            }],
            operations: vec![l0::Operation::BinOp(l0::BinOp {
                kind: l0::BinOpKind::Add,
                lhs: Value::Parameter(0.into()),
                rhs: Value::Parameter(1.into()),
            })],
        };

        let ir = IR::translate_from(ir).unwrap();
        assert_eq!(ir.inputs.len(), 2);
        assert_eq!(ir.outputs.len(), 1);
        assert_eq!(ir.instructions.len(), 1);
        eprintln!("{:#?}", ir);
        ir.outputs
            .iter()
            .enumerate()
            .map(|(i, o)| { (o, ir.output_data(i).unwrap()) })
            .for_each(|(o, d)| eprintln!("{:?} -> {:?}", o, d) );
    }
}