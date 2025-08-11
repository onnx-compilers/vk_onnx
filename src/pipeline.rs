use crate::l_base::{ScalarTy, Translate};
use crate::l1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Buffer {
    pub size: usize, // in bytes
}

#[derive(Debug, Default, Clone)]
pub struct IR {
    pub source_buffers: Vec<Buffer>,
    pub constant_buffers: Vec<Buffer>,
    pub operand_buffers: Vec<Buffer>,
    pub inputs: Vec<OperandBufferId>,
    pub outputs: Vec<OperandBufferId>,
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
    // Unary(UnOp),
    Binary(BinOp),
}

// pub enum UnOp {
//
// }

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
        res: OperandBufferId,
        lhs: BufferId,
        rhs: BufferId,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceBufferId(pub usize);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstantBufferId(pub usize);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperandBufferId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpId(pub usize);

impl IR {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn make_source_buffer(&mut self, buffer: Buffer) -> SourceBufferId {
        let id = self.source_buffers.len();
        self.source_buffers.push(buffer);
        SourceBufferId(id)
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
            let element_count: usize = shape.iter().product();
            let buffer = Buffer {
                size: element_count * size_of(ty),
            };
            let source_id = ir.make_source_buffer(buffer.clone());
            let id = ir.make_operand_buffer(buffer);
            self.inputs.push((source_id, id));
            ir.inputs.push(id);
        }
        for &l1::Instruction {
            op: ref l1_op,
            result: l1::Argument { ty, ref shape },
        } in l1.instructions.iter()
        {
            let element_count: usize = shape.iter().product();
            let buf = ir.make_operand_buffer(Buffer {
                size: element_count * size_of(ty),
            });
            self.intermediates.push(buf);
            let op = match l1_op {
                &l1::Operation::BinOp(l1::BinOp {
                    kind: l1::BinOpKind::Add,
                    lhs,
                    rhs,
                }) => {
                    let lhs = self.get_value(lhs).unwrap();
                    let rhs = self.get_value(rhs).unwrap();
                    Op::Binary(BinOp {
                        kind: BinOpKind::AddElementwise,
                        operands: BinOperands::NotInplace {
                            lhs,
                            rhs,
                            res: buf,
                        },
                    })
                }
            };
            let id = ir.make_op(op);
            // NOTE: Maybe put this in13 a function
            ir.instructions.push(Instr::Op(id));
        }
        Ok(ir)
    }
}

fn size_of(_ty: ScalarTy) -> usize {
    // XXX: Hack, must calculate/lookup sizes for respective element types
    4
}

#[cfg(test)]
mod tests {
    use crate::l1::{Argument, BinOp, BinOpKind, Operation};

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
        let _c = l1.append_op(
            Operation::BinOp(BinOp {
                kind: BinOpKind::Add,
                lhs: a.into(),
                rhs: b.into(),
            }),
            Argument {
                ty: ScalarTy::F32,
                shape: Box::new([2, 2]),
            },
        );
        let ir = IRBuilder::default().translate(l1, &()).unwrap();
        println!("{:?}", ir);
    }
}