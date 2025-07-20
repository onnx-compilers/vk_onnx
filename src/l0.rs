use std::collections::HashMap;

use crate::l_base::TranslateFrom;
use crate::protos::onnx;

pub struct IR {
    pub instructions: Vec<Instruction>,
    pub inputs: Vec<Input>,
    pub outputs: Vec<Output>,
}

pub enum Instruction {
    BinOp(Ty, BinOp<Value>),
}

pub enum Value {
    Parameter(Parameter),
    Temporary(Temporary),
}

pub struct Parameter(pub usize);
pub struct Temporary(pub usize);

pub struct Input {
    pub name: String,
    pub ty: Ty,
    pub shape: Vec<usize>,
    pub batch: Option<Batch>,
}

pub struct Output {
    pub name: String,
    pub value: Value,
}

pub enum Batch {
    First,
    Last,
}

pub enum BinOpKind {
    Add,
}

pub struct BinOp<R> {
    pub kind: BinOpKind,
    pub lhs: R,
    pub rhs: R,
    pub shapes: BinOpShapes,
}

pub enum Ty {
    F32,
    S32,
    U32,
}

pub enum BinOpShapes {
    SS,                                  // scalar-scalar
    TS(Vec<usize>),                      // tensor-scalar
    TTExact(Vec<usize>),                 // tensor-tensor with same shape
    TTBroadcast(Vec<usize>, Vec<usize>), // tensor-tensor with broadcast
}

impl TranslateFrom<onnx::ModelProto> for IR {
    type Error = ();
    fn translate_from(model: &onnx::ModelProto) -> Result<IR, Self::Error> {
        unimplemented!()
    }
}