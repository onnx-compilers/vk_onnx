use crate::l_base::{ScalarTy, Translate};
use crate::l1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Buffer {
    pub ty: ScalarTy,
    pub dims_range: (usize, usize),
}

#[derive(Debug, Default, Clone)]
pub struct IR {
    pub inputs: Vec<OperandBufferId>,
    pub outputs: Vec<OperandBufferId>,
    pub constant_buffers: Vec<Buffer>,
    pub operand_buffers: Vec<Buffer>,
    pub dims_pool: Vec<usize>,
    pub instructions: Vec<Instr>,
    pub operations: Vec<Op>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferId {
    Constant(ConstantBufferId),
    Operand(OperandBufferId),
}

#[derive(Debug, Clone, Copy)]
pub enum Instr {
    Op(OpId),
}

#[derive(Debug, Clone, Copy)]
pub enum Op {
    Unary(UnOp),
    Binary(BinOp),
}

#[derive(Debug, Clone, Copy)]
pub struct UnOp {
    pub kind: UnOpKind,
    pub operand: UnOperand,
}

#[derive(Debug, Clone, Copy)]
pub enum UnOpKind {
    Scaler { offset: f32, scale: f32 },
}

#[derive(Debug, Clone, Copy)]
pub enum UnOperand {
    Inplace {
        operand: OperandBufferId,
    },
    NotInplace {
        result: OperandBufferId,
        operand: BufferId,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct BinOp {
    pub kind: BinOpKind,
    pub operands: BinOperands,
}

#[derive(Debug, Clone, Copy)]
pub enum BinOpKind {
    AddElementwise,
}

#[derive(Debug, Clone, Copy)]
pub enum BinOperands {
    Inplace {
        a: OperandBufferId,
        b: BufferId,
        to_lhs: bool,
    },
    NotInplace {
        result: OperandBufferId,
        lhs: BufferId,
        rhs: BufferId,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceBufferId(pub usize);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstantBufferId(pub usize);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OperandBufferId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpId(pub usize);

impl IR {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn make_input(
        &mut self,
        buffer: Buffer,
    ) -> (SourceBufferId, OperandBufferId) {
        let i = self.inputs.len();
        let id = self.make_operand_buffer(buffer);
        self.inputs.push(id);
        (SourceBufferId(i), id)
    }

    pub fn make_constant_buffer(&mut self, buffer: Buffer) -> ConstantBufferId {
        let id = self.constant_buffers.len();
        self.constant_buffers.push(buffer);
        ConstantBufferId(id)
    }

    pub fn make_operand_buffer(&mut self, buffer: Buffer) -> OperandBufferId {
        let id = self.operand_buffers.len();
        self.operand_buffers.push(buffer);
        OperandBufferId(id)
    }

    pub fn make_op(&mut self, op: Op) -> OpId {
        let id = self.operations.len();
        self.operations.push(op);
        OpId(id)
    }

    fn add_dimensions(&mut self, dims: &[usize]) -> (usize, usize) {
        let start = self.dims_pool.len();
        self.dims_pool.extend_from_slice(dims);
        let end = self.dims_pool.len();
        (start, end)
    }

    pub fn get_operand(&self, OperandBufferId(i): OperandBufferId) -> &Buffer {
        &self.operand_buffers[i]
    }

    pub fn get_shape(&self, dims_range: (usize, usize)) -> &[usize] {
        &self.dims_pool[dims_range.0..dims_range.1]
    }
}

impl From<ConstantBufferId> for BufferId {
    fn from(id: ConstantBufferId) -> Self {
        BufferId::Constant(id)
    }
}

impl From<OperandBufferId> for BufferId {
    fn from(id: OperandBufferId) -> Self {
        BufferId::Operand(id)
    }
}

#[derive(Default, Clone)]
pub struct IRBuilder {
    inputs: Vec<(SourceBufferId, OperandBufferId)>,
    intermediates: Vec<OperandBufferId>,
}

impl IRBuilder {
    fn get_value(&self, v: l1::Value) -> Option<BufferId> {
        match v {
            l1::Value::Parameter(l1::Parameter(i)) => {
                self.inputs.get(i).map(|&(_, id)| id.into())
            }
            l1::Value::Temporary(l1::Temporary(i)) => {
                self.intermediates.get(i).map(|&id| id.into())
            }
        }
    }
}

impl Translate<l1::IR, IR> for IRBuilder {
    type Config = ();
    type Error = ();
    fn translate(
        mut self,
        l1: l1::IR,
        _config: &Self::Config,
    ) -> Result<IR, Self::Error> {
        let mut ir = IR::new();

        self.inputs.reserve(l1.inputs.len());
        for &l1::Argument { ty, ref shape } in l1.inputs.iter() {
            let buffer = Buffer {
                ty,
                dims_range: ir.add_dimensions(shape),
            };
            let (src_id, buf_id) = ir.make_input(buffer);
            self.inputs.push((src_id, buf_id));
        }

        for &l1::Instruction {
            op: ref l1_op,
            result: l1::Argument { ty, ref shape },
        } in l1.instructions.iter()
        {
            let dims_range = ir.add_dimensions(shape);
            let buf = ir.make_operand_buffer(Buffer { ty, dims_range });
            self.intermediates.push(buf);
            let op = match l1_op {
                &l1::Operation::Add { lhs, rhs } => {
                    let lhs = self.get_value(lhs).unwrap();
                    let rhs = self.get_value(rhs).unwrap();
                    Op::Binary(BinOp {
                        kind: BinOpKind::AddElementwise,
                        operands: BinOperands::NotInplace {
                            lhs,
                            rhs,
                            result: buf,
                        },
                    })
                }

                &l1::Operation::Scaler {
                    input,
                    offset,
                    scale,
                } => {
                    let input = self.get_value(input).unwrap();
                    Op::Unary(UnOp {
                        kind: UnOpKind::Scaler { offset, scale },
                        operand: UnOperand::NotInplace {
                            result: buf,
                            operand: input,
                        },
                    })
                }
            };
            let id = ir.make_op(op);
            // NOTE: Maybe put this in13 a function
            ir.instructions.push(Instr::Op(id));
        }

        ir.outputs.reserve(l1.outputs.len());
        for &l1::Output { value } in l1.outputs.iter() {
            let BufferId::Operand(buf) = self.get_value(value).unwrap() else {
                panic!("yo wtf man");
            };
            ir.outputs.push(buf);
        }

        Ok(ir)
    }
}

#[cfg(test)]
mod tests {
    use crate::l1::{Argument, Operation};

    use super::*;

    #[test]
    fn test() {
        let mut l1 = l1::IR::default();
        let a = l1.make_input(Argument {
            ty: ScalarTy::F32,
            shape: Box::new([2, 2]),
        });
        let b = l1.make_input(Argument {
            ty: ScalarTy::F32,
            shape: Box::new([2, 2]),
        });
        let c = l1.append_op(
            Operation::Add {
                lhs: a.into(),
                rhs: b.into(),
            },
            Argument {
                ty: ScalarTy::F32,
                shape: Box::new([2, 2]),
            },
        );
        l1.add_output(c.into());
        let _ir = IRBuilder::default().translate(l1, &()).unwrap();
    }
}
