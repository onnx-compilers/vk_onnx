use crate::l_base::ScalarTy;

use rspirv::binary::{Assemble, Disassemble};
use rspirv::dr::Builder as SPVBuilder;
use rspirv::spirv::{self, Word};

#[derive(Debug, Clone)]
pub struct IR {
    pub functions: Vec<Function>,
    pub buffers: Vec<Buffer>,
    pub blobs_data: Vec<u8>,
    pub blobs: Vec<Blob>,
    pub types: Vec<CompositeTy>,
    pub nodes: Vec<Node>,
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
    Store(TyId, NodeId),
    Load(TyId),
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

struct AssemblerTypes {
    void: Word,
    f32: Word,
    u32: Word,
    s32: Word,
}

impl Assemble for IR {
    fn assemble_into(&self, result: &mut Vec<u32>) {
        let mut b = SPVBuilder::new();
        b.ext_inst_import("GLSL.std.450");
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
        let mut spirv_functions = Vec::with_capacity(self.functions.len());

        for func in self.functions.iter() {
            let Function { ret, .. } = func;
            let ret = ret.to_assembler(&self.types, &types);
            let args = func
                .args
                .iter()
                .map(|ty| ty.to_assembler(&self.types, &types))
                .collect::<Box<[_]>>();
            let func_ty_id = b.type_function(ret, args);
            let func_id = b.begin_function(
                ret,
                None,
                spirv::FunctionControl::DONT_INLINE,
                func_ty_id,
            );
            b.end_function().unwrap();
            spirv_functions.push(func_id);
        }

        let module = b.module();
        eprintln!("{}", module.disassemble());
        module.assemble_into(result);
    }
}

impl IR {
    pub fn new(config: IRConfig) -> Self {
        Self {
            functions: Vec::new(),
            buffers: Vec::new(),
            blobs_data: Vec::new(),
            blobs: Vec::new(),
            types: Vec::new(),
            nodes: Vec::new(),
            config,
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let mut ir = IR::new(IRConfig {});
        ir.new_function(Function {
            ret: Ty::Void,
            args: [ScalarTy::F32, ScalarTy::F32]
                .into_iter()
                .map(Ty::Scalar)
                .collect(),
            body: vec![],
        });
        let spirv = ir.assemble();
        assert_ne!(spirv.len(), 0);
    }
}


