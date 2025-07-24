use crate::l_base::{ScalarTy, TranslateFrom};

use rspirv::dr::{Builder as SPVBuilder, Module as SPVModule};
use rspirv::spirv::{self, Word};

#[derive(Debug, Default, Clone)]
pub struct IR {
    pub functions: Vec<Function>,
    pub buffers: Vec<Buffer>,
    pub blobs_data: Vec<u8>,
    pub blobs: Vec<Blob>,
    pub types: Vec<CompositeTy>,
    pub nodes: Vec<Node>,
    pub value_nodes: Vec<ValueNode>,
    pub config: IRConfig,
}

#[derive(Default, Debug, Clone)]
pub struct IRConfig {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub ret: Ty,
    pub args: Vec<Ty>,
    pub body: Vec<NodeId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Node {
    Value(Ty, ValueNodeId),
    ReturnValue(Value),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueNode {
    Load(Value),
    IAdd(Value, Value),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Buffer {
    pub blob: Option<BlobId>,
    pub ty: TyId,
    pub layout: BufferLayout,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferLayout {
    pub binding: Word,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Blob {
    pub start_idx: usize,
    pub end_idx: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ty {
    Void,
    Scalar(ScalarTy),
    Vec2(ScalarTy),
    Vec3(ScalarTy),
    Vec4(ScalarTy),
    Mat2(ScalarTy),
    Mat3(ScalarTy),
    Mat4(ScalarTy),
    Composite(TyId),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompositeTy {
    Array1(Ty, Option<usize>),
    Array2(Ty, Option<usize>, usize),
    Array3(Ty, Option<usize>, usize, usize),
    ArrayN(Ty, Option<usize>, Vec<usize>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FunctionId(pub usize);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferId(pub usize);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlobId(pub usize);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TyId(pub usize);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeId(pub usize);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValueNodeId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Parameter(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Temporary(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Value {
    Parameter(Parameter),
    Temporary(Temporary),
}

struct AssemblerTypes {
    void: Word,
    f32: Word,
    u32: Word,
    s32: Word,
}

impl TranslateFrom<IR> for SPVModule {
    type Error = rspirv::dr::Error;
    fn translate_from(ir: IR) -> Result<Self, Self::Error> {
        let mut b = SPVBuilder::new();
        b.capability(spirv::Capability::Shader);
        b.ext_inst_import("GLSL.std.450");
        b.source(
            spirv::SourceLanguage::Unknown,
            0,
            None,
            Option::<&str>::None,
        );
        b.memory_model(
            spirv::AddressingModel::Logical,
            spirv::MemoryModel::GLSL450,
        );
        let types = AssemblerTypes {
            void: b.type_void(),
            f32: b.type_float(32),
            u32: b.type_int(32, 0),
            s32: b.type_int(32, 1),
        };
        let mut spirv_functions = Vec::with_capacity(ir.functions.len());

        for func in ir.functions.iter() {
            // TODO: Move this into a function
            let Function { ret, .. } = func;
            let ret = ret.to_assembler(&ir.types, &types);
            let args = func
                .args
                .iter()
                .map(|ty| ty.to_assembler(&ir.types, &types))
                .collect::<Box<[_]>>();
            let func_ty_id = b.type_function(ret, args.clone());
            let func_id = b.begin_function(
                ret,
                None,
                spirv::FunctionControl::DONT_INLINE,
                func_ty_id,
            )?;
            let parameter_ids = args
                .into_iter()
                .map(|id| b.function_parameter(id))
                .collect::<Result<Box<[_]>, Self::Error>>()?;

            let _block_id = b.begin_block_no_label(None)?;

            let mut value_ids = Vec::new();

            // TODO: Move this into a proper function
            let get_value = |value_ids: &[Word], v| match v {
                Value::Parameter(Parameter(i)) => parameter_ids[i],
                Value::Temporary(Temporary(i)) => value_ids[i],
            };

            for &NodeId(node_id) in func.body.iter() {
                match &ir.nodes[node_id] {
                    Node::Value(ty, ValueNodeId(value_node_idx)) => {
                        assert!(*value_node_idx == value_ids.len());
                        match &ir.value_nodes[*value_node_idx] {
                            ValueNode::IAdd(lhs, rhs) => {
                                let lhs = get_value(&value_ids, *lhs);
                                let rhs = get_value(&value_ids, *rhs);
                                let op_id = b.i_add(
                                    ty.to_assembler(&ir.types, &types),
                                    None,
                                    lhs,
                                    rhs,
                                )?;
                                value_ids.push(op_id);
                            }
                            _ => todo!(),
                        }
                    }
                    &Node::ReturnValue(value) => {
                        let value = get_value(&value_ids, value);
                        b.ret_value(value)?;
                    }
                }
            }
            b.end_function()?;
            spirv_functions.push(func_id);
        }

        Ok(b.module())
    }
}

impl IR {
    pub fn new(config: IRConfig) -> Self {
        Self {
            config,
            ..Default::default()
        }
    }
    pub fn new_type(&mut self, ty: CompositeTy) -> TyId {
        let id = self.types.len();
        self.types.push(ty);
        TyId(id)
    }
    pub fn new_function(&mut self, func: Function) -> FunctionId {
        let id = self.functions.len();
        self.functions.push(func);
        FunctionId(id)
    }
    pub fn new_node(&mut self, node: Node) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(node);
        NodeId(id)
    }
    pub fn new_value_node(&mut self, value: ValueNode) -> (ValueNodeId, Value) {
        let id = self.value_nodes.len();
        self.value_nodes.push(value);
        (ValueNodeId(id), Value::temporary(id))
    }
}

impl Ty {
    fn to_assembler(
        &self,
        _composites: &[CompositeTy],
        asm_types: &AssemblerTypes,
    ) -> Word {
        match self {
            Ty::Void => asm_types.void,
            Ty::Scalar(ScalarTy::F32) => asm_types.f32,
            Ty::Scalar(ScalarTy::U32) => asm_types.u32,
            Ty::Scalar(ScalarTy::S32) => asm_types.s32,
            _ => todo!(),
        }
    }

    pub fn same_as(&self, other: &Ty, _composites: &[CompositeTy]) -> bool {
        self == other // TODO: composite comparison
    }
}

impl Value {
    pub fn temporary(i: usize) -> Self {
        Value::Temporary(Temporary(i))
    }
    pub fn parameter(i: usize) -> Self {
        Value::Parameter(Parameter(i))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rspirv::binary::{Assemble, Disassemble};

    #[test]
    fn test() {
        let mut ir = IR::new(IRConfig {});
        let (add_node, add_result) = ir.new_value_node(ValueNode::IAdd(
            Value::parameter(0),
            Value::parameter(1),
        ));
        let val = ir.new_node(Node::Value(Ty::Scalar(ScalarTy::U32), add_node));
        let ret = ir.new_node(Node::ReturnValue(add_result));
        ir.new_function(Function {
            ret: Ty::Scalar(ScalarTy::U32),
            args: [ScalarTy::U32, ScalarTy::U32]
                .into_iter()
                .map(Ty::Scalar)
                .collect(),
            body: vec![val, ret],
        });
        let module = SPVModule::translate_from(ir).unwrap();
        eprintln!("{}", module.disassemble());
        assert_ne!(module.assemble().len(), 0);
    }
}
