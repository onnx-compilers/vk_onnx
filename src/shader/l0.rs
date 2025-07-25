use crate::l_base::{ScalarTy, TranslateFrom};

use rspirv::dr::{Builder as SPVBuilder, Module as SPVModule};
use rspirv::spirv::{self, StorageClass, Word};

#[derive(Debug, Default, Clone)]
pub struct IR {
    pub functions: Vec<Function>,
    pub buffers: Vec<Buffer>,
    pub blobs_data: Vec<u8>,
    pub blobs: Vec<Blob>,
    pub types: Vec<CompositeTy>,
    pub value_nodes: Vec<ValueNode>,
    pub config: IRConfig,
    pub entry_point: Option<FunctionId>,
}

#[derive(Default, Debug, Clone)]
pub struct IRConfig {}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub name: Option<String>,
    pub ret: Option<Ty>,
    pub args: Vec<Ty>,
    pub body: Vec<Node>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Node {
    Value(ValueNodeId),
    ReturnValue(Value),
    Return,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueNodeOp {
    Load(Value),
    Store(Value, Value),
    Alloca,
    Add(Value, Value),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ty {
    Scalar(ScalarTy),
    Vec(ScalarTy, u32),
    Mat(ScalarTy, u32),
    Composite(TyId),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompositeTy {
    Pointer(Ty, StorageClass),
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
    MissingEntrypoinName,
    ReturnInVoid,
    ReturnTypeMismatch(Ty),
}

impl From<rspirv::dr::Error> for Error {
    fn from(e: rspirv::dr::Error) -> Self {
        Error::AssemblerError(e)
    }
}

#[derive(Default)]
struct LazyTypeMapper {
    void: Option<Word>,
    composites: Box<[Option<Word>]>,
    scalars: LazyScalarTypeMapper,
    vectors: [LazyScalarTypeMapper; 3],
    matrices: [LazyScalarTypeMapper; 3],
}

#[derive(Default)]
struct LazyScalarTypeMapper {
    f32: Option<Word>,
    u32: Option<Word>,
    s32: Option<Word>,
}

struct FunctionCompiler<'a> {
    type_mapper: &'a mut LazyTypeMapper,
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
        let mut type_mapper = LazyTypeMapper::new(ir.types.len());
        let mut spirv_functions = Vec::with_capacity(ir.functions.len());

        for i in 0..ir.functions.len() {
            let func_id = FunctionCompiler::new(
                &mut b,
                &mut type_mapper,
                &ir.functions[i],
                &ir,
            )?
            .compile(&mut b)?;
            spirv_functions.push(func_id);
        }

        if let Some(FunctionId(i)) = ir.entry_point {
            b.entry_point(
                spirv::ExecutionModel::GLCompute,
                spirv_functions[i],
                ir.functions
                    .get(i)
                    .and_then(|f| f.name.clone())
                    .ok_or(Error::MissingEntrypoinName)?,
                &[],
            );
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
        let (id, _) = self.make_function_mut(func);
        id
    }
    // for when you have recursive functions and whatnot
    pub fn make_function_mut(
        &mut self,
        func: Function,
    ) -> (FunctionId, &mut Function) {
        let id = self.functions.len();
        self.functions.push(func);
        let func = &mut self.functions[id];
        (FunctionId(id), func)
    }
    pub fn make_value_node(
        &mut self,
        value: ValueNode,
    ) -> (ValueNodeId, Value) {
        let id = self.value_nodes.len();
        self.value_nodes.push(value);
        (ValueNodeId(id), Value::temporary(id))
    }

    pub fn entry_point(&mut self, func_id: FunctionId) {
        self.entry_point = Some(func_id);
    }
}

impl Function {
    pub fn new(name: Option<String>, ret: Option<Ty>) -> Self {
        Self {
            ret,
            name,
            ..Default::default()
        }
    }
    pub fn make_argument(&mut self, ty: Ty) -> Value {
        let id = self.args.len();
        self.args.push(ty);
        Value::Parameter(Parameter(id))
    }

    pub fn append_node(&mut self, node: Node) {
        self.body.push(node);
    }

    pub fn return_value(&mut self, value: Value) {
        self.body.push(Node::ReturnValue(value));
    }
}

impl Ty {
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

impl<'a> FunctionCompiler<'a> {
    pub fn new(
        b: &mut SPVBuilder,
        type_mapper: &'a mut LazyTypeMapper,
        func: &'a Function,
        ir: &'a IR,
    ) -> Result<Self, rspirv::dr::Error> {
        let Function { ret: maybe_ret, .. } = func;
        let ret = maybe_ret
            .map(|ret| type_mapper.get(b, &ir.types, &ret))
            .unwrap_or_else(|| type_mapper.get_void(b));
        let args = func
            .args
            .iter()
            .map(|ty| type_mapper.get(b, &ir.types, &ty))
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
            type_mapper,
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

    pub fn compile(mut self, b: &mut SPVBuilder) -> Result<u32, Error> {
        let _block_id = b.begin_block(None)?;

        for node in self.function.body.iter() {
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
                if let None = self.function.ret {
                    return Err(Error::ReturnInVoid);
                } else if let Some(ret) = self.function.ret {
                    if !ty.same_as(&ret, &self.ir.types) {
                        return Err(Error::ReturnTypeMismatch(ty.clone()));
                    }
                }
                b.ret_value(value)?;
            }
            &Node::Return => b.ret()?,
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
                op: ValueNodeOp::Add(lhs, rhs),
            } => {
                let lhs = self.get_value(*lhs);
                let rhs = self.get_value(*rhs);
                // TODO: Check type compatibility for operation
                let ty_id = self.type_mapper.get(b, &self.ir.types, ty);
                let inner_ty = match ty {
                    &Ty::Scalar(ty) => ty,
                    &Ty::Vec(ty, _n) => ty,
                    _ => todo!(),
                };
                let op_id = match inner_ty {
                    ScalarTy::U32 | ScalarTy::S32 => {
                        b.i_add(ty_id, None, lhs, rhs)?
                    }
                    ScalarTy::F32 => b.f_add(ty_id, None, lhs, rhs)?,
                };
                self.value_ids.push(op_id);
            }
            _ => todo!(),
        }
        Ok(())
    }
}

impl LazyTypeMapper {
    fn new(n_composites: usize) -> Self {
        Self {
            composites: vec![None; n_composites].into_boxed_slice(),
            ..Default::default()
        }
    }

    fn get_void(&mut self, b: &mut SPVBuilder) -> Word {
        *self.void.get_or_insert_with(|| b.type_void())
    }

    fn get(
        &mut self,
        b: &mut SPVBuilder,
        composites: &[CompositeTy],
        ty: &Ty,
    ) -> Word {
        match ty {
            &Ty::Scalar(ty) => self.scalars.get(b, ty),
            &Ty::Vec(ty, n) => self.scalars.get_transfer_map(
                &mut self.vectors[n as usize - 2],
                b,
                ty,
                |b, id| b.type_vector(id, n),
            ),
            &Ty::Mat(ty, n) => self.scalars.get_transfer_map(
                &mut self.matrices[n as usize - 2],
                b,
                ty,
                |b, id| b.type_matrix(id, n),
            ),
            &Ty::Composite(ty) => self.get_composite(b, composites, ty),
        }
    }

    fn get_composite(
        &mut self,
        b: &mut SPVBuilder,
        composites: &[CompositeTy],
        ty: TyId,
    ) -> Word {
        let TyId(i) = ty;
        let ty = &composites[i];
        let (is_found, id) = match self.composites[i] {
            None => match ty {
                CompositeTy::Pointer(ty, storage_class) => {
                    let base = self.get(b, composites, &ty);
                    let id = b.type_pointer(None, *storage_class, base);
                    (false, id)
                }
                _ => todo!(),
            },
            Some(id) => (true, id),
        };
        if !is_found {
            self.composites[i] = Some(id);
        }
        id
    }
}

impl LazyScalarTypeMapper {
    fn get(&mut self, b: &mut SPVBuilder, ty: ScalarTy) -> Word {
        *match ty {
            ScalarTy::F32 => self.f32.get_or_insert_with(|| b.type_float(32)),
            ScalarTy::U32 => self.u32.get_or_insert_with(|| b.type_int(32, 0)),
            ScalarTy::S32 => self.s32.get_or_insert_with(|| b.type_int(32, 1)),
        }
    }

    // TODO: Make the function return a closure which modifies the member instead
    fn get_transfer_map<F>(
        &mut self,
        other: &mut Self,
        b: &mut SPVBuilder,
        ty: ScalarTy,
        f: F,
    ) -> Word
    where
        F: FnOnce(&mut SPVBuilder, Word) -> Word,
    {
        *match ty {
            ScalarTy::F32 => other.f32.get_or_insert_with(|| {
                let id = *self.f32.get_or_insert_with(|| b.type_float(32));
                f(b, id)
            }),
            ScalarTy::U32 => other.u32.get_or_insert_with(|| {
                let id = *self.u32.get_or_insert_with(|| b.type_int(32, 0));
                f(b, id)
            }),
            ScalarTy::S32 => other.s32.get_or_insert_with(|| {
                let id = *self.s32.get_or_insert_with(|| b.type_int(32, 1));
                f(b, id)
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io::Write,
        process::{Command, Stdio},
    };

    use super::*;

    use rspirv::binary::{Assemble, Disassemble};

    #[test]
    fn test() {
        let mut ir = IR::new(IRConfig {});
        let t = Ty::Vec(ScalarTy::F32, 3);
        let mut func = Function::new(None, Some(t));
        let a = func.make_argument(t);
        let b = func.make_argument(t);
        let (add_node, add_result) = ir.make_value_node(ValueNode {
            ty: t,
            op: ValueNodeOp::Add(a, b),
        });
        func.append_node(Node::Value(add_node));
        func.return_value(add_result);
        ir.make_function(func);
        let main_func = ir.make_function(Function {
            ret: None,
            name: Some("main".into()),
            body: vec![Node::Return],
            ..Default::default()
        });
        ir.entry_point(main_func);
        let module = SPVModule::translate_from(ir).unwrap();
        eprintln!("{}", module.disassemble());
        let raw = module.assemble();
        assert_ne!(raw.len(), 0);
        let mut loader = rspirv::dr::Loader::new();
        // to make sure it's actually valid
        let () =
            rspirv::binary::parse_words(raw.as_slice(), &mut loader).unwrap();
        assert_eq!(loader.module().assemble(), raw);
        let mut child = Command::new("spirv-val")
            .arg("-")
            .stdin(Stdio::piped())
            .spawn()
            .unwrap();
        let mut stdin = child.stdin.take().unwrap();
        stdin
            .write_all(unsafe { raw.as_slice().align_to::<u8>().1 })
            .unwrap();
        stdin.flush().unwrap();
        drop(stdin);
        let output = child.wait_with_output().unwrap();
        assert!(output.status.success());
    }
}