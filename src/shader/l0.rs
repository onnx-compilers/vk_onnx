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

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub ret: Ty,
    pub args: Vec<Ty>,
    pub body: Vec<NodeId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Node {
    Value(ValueNodeId),
    ReturnValue(Value),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueNodeOp {
    Load(Value),
    IAdd(Value, Value),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueNode {
    pub ty: Ty,
    pub op: ValueNodeOp,
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

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub enum Ty {
    #[default]
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

#[derive(Debug)]
pub enum Error {
    AssemblerError(rspirv::dr::Error),
    ReturnTypeMismatch(Ty),
}

impl From<rspirv::dr::Error> for Error {
    fn from(e: rspirv::dr::Error) -> Self {
        Error::AssemblerError(e)
    }
}

struct AssemblerTypes {
    void: Word,
    f32: Word,
    u32: Word,
    s32: Word,
}

struct FunctionCompiler<'a> {
    types: &'a AssemblerTypes,
    function: &'a Function,
    ir: &'a IR,
    parameter_ids: Box<[u32]>,
    value_ids: Vec<u32>,
    func_id: u32,
}

impl TranslateFrom<IR> for SPVModule {
    type Error = Error;
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

        for i in 0..ir.functions.len() {
            let func_id =
                FunctionCompiler::new(&mut b, &types, &ir.functions[i], &ir)?
                    .compile(&mut b)?;
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
    pub fn make_type(&mut self, ty: CompositeTy) -> TyId {
        let id = self.types.len();
        self.types.push(ty);
        TyId(id)
    }
    pub fn make_function(&mut self, func: Function) -> FunctionId {
        let id = self.functions.len();
        self.functions.push(func);
        FunctionId(id)
    }
    pub fn make_node(&mut self, node: Node) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(node);
        NodeId(id)
    }
    pub fn make_value_node(
        &mut self,
        value: ValueNode,
    ) -> (ValueNodeId, Value) {
        let id = self.value_nodes.len();
        self.value_nodes.push(value);
        (ValueNodeId(id), Value::temporary(id))
    }
}

impl Function {
    pub fn new(ret: Ty) -> Self {
        Self { ret, ..Default::default() }
    }
    pub fn make_argument(&mut self, ty: Ty) -> Value {
        let id = self.args.len();
        self.args.push(ty);
        Value::Parameter(Parameter(id))
    }

    pub fn append_node(&mut self, node: NodeId) {
        self.body.push(node);
    }

    pub fn return_value(&mut self, ir: &mut IR, value: Value) {
        self.body.push(ir.make_node(Node::ReturnValue(value)));
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
        dbg!(dbg!(self) == dbg!(other)) // TODO: composite comparison
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

impl<'a> FunctionCompiler<'a> {
    pub fn new(
        b: &mut SPVBuilder,
        types: &'a AssemblerTypes,
        func: &'a Function,
        ir: &'a IR,
    ) -> Result<Self, rspirv::dr::Error> {
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
            .collect::<Result<Box<[_]>, rspirv::dr::Error>>()?;
        Ok(FunctionCompiler {
            types,
            function: func,
            ir,
            func_id,
            value_ids: Vec::new(),
            parameter_ids,
        })
    }

    fn get_value(&self, v: Value) -> Word {
        match v {
            Value::Parameter(Parameter(i)) => self.parameter_ids[i],
            Value::Temporary(Temporary(i)) => self.value_ids[i],
        }
    }

    pub fn compile(
        mut self,
        b: &mut SPVBuilder,
    ) -> Result<u32, Error> {
        let _block_id = b.begin_block(None)?;

        for &NodeId(node_id) in self.function.body.iter() {
            let node = &self.ir.nodes[node_id];
            let () = self.compile_node(b, node)?;
        }
        b.end_function()?;
        Ok(self.func_id)
    }

    fn compile_node(
        &mut self,
        b: &mut SPVBuilder,
        node: &Node,
    ) -> Result<(), Error> {
        match node {
            Node::Value(ValueNodeId(value_node_idx)) => {
                let value_node = &self.ir.value_nodes[*value_node_idx];
                let () = self.compile_value_node(b, value_node)?;
            }
            &Node::ReturnValue(value) => {
                let ty = self.get_value_type(value);
                let value = self.get_value(value);
                if !ty.same_as(&self.function.ret, &self.ir.types) {
                    return Err(Error::ReturnTypeMismatch(ty.clone()));
                }
                b.ret_value(value)?;
            }
        }
        Ok(())
    }

    fn get_value_type(&self, v: Value) -> &Ty {
        match v {
            Value::Parameter(Parameter(i)) => &self.function.args[i],
            Value::Temporary(Temporary(i)) => &self.ir.value_nodes[i].ty,
        }
    }

    fn compile_value_node(
        &mut self,
        b: &mut SPVBuilder,
        node: &ValueNode,
    ) -> Result<(), rspirv::dr::Error> {
        match node {
            ValueNode {
                ty,
                op: ValueNodeOp::IAdd(lhs, rhs),
            } => {
                let lhs = self.get_value(*lhs);
                let rhs = self.get_value(*rhs);
                let op_id = b.i_add(
                    ty.to_assembler(&self.ir.types, self.types),
                    None,
                    lhs,
                    rhs,
                )?;
                self.value_ids.push(op_id);
            }
            _ => todo!(),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rspirv::binary::{Assemble, Disassemble};

    #[test]
    fn test() {
        let mut ir = IR::new(IRConfig {});
        let mut func = Function::new(Ty::Scalar(ScalarTy::U32));
        let a = func.make_argument(Ty::Scalar(ScalarTy::U32));
        let b = func.make_argument(Ty::Scalar(ScalarTy::U32));
        let (add_node, add_result) = ir.make_value_node(ValueNode {
            ty: Ty::Scalar(ScalarTy::U32),
            op: ValueNodeOp::IAdd(a, b),
        });
        let val =
            ir.make_node(Node::Value(add_node));
        func.append_node(val);
        func.return_value(&mut ir, add_result);
        ir.make_function(func);
        let module = SPVModule::translate_from(ir).unwrap();
        eprintln!("{}", module.disassemble());
        let raw = module.assemble();
        assert_ne!(raw.len(), 0);
        let mut loader = rspirv::dr::Loader::new();
        // to make sure it's actually valid
        let () =
            rspirv::binary::parse_words(raw.as_slice(), &mut loader).unwrap();
        eprintln!("{}", loader.module().assemble() == raw);
    }
}
