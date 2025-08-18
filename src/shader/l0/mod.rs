pub mod constant;
pub mod ty;

use crate::l_base::{ScalarTy, Translate};

use rspirv::dr::{Builder as SPVBuilder, Module as SPVModule};
use rspirv::spirv::{self, StorageClass, Word};

pub use constant::{Constant, ConstantId, ScalarConstant};
pub use ty::{CompositeTy, CompositeTyId, PtrTyId, StructMember, Ty};

#[derive(Debug, Default, Clone)]
pub struct IR {
    functions: Vec<Function>,
    constants: Vec<Constant>,
    storage_buffers: Vec<StorageBuffer>,
    uniforms: Vec<Uniform>,
    push_constants: Vec<PushConstant>,
    builtins: Vec<(PtrTyId, spirv::BuiltIn)>,
    composite_types: Vec<CompositeTy>,
    ptr_types: Vec<Ty>,
    entry_point: Option<(FunctionId, Box<[Variable]>)>,
}

#[derive(Default, Debug, Clone, Copy)]
struct Types<'a> {
    composite: &'a [CompositeTy],
    ptr: &'a [Ty],
}

#[derive(Default, Debug, Clone)]
pub struct Config {
    pub local_size: [u32; 3],
    // TODO: Fill this in Default::default() instead of using an option
    pub version: Option<(u8, u8)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageBuffer {
    pub ty: PtrTyId,
    pub descriptor_set: u32,
    pub binding: u32,
    pub writable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Uniform {
    pub ty: PtrTyId,
    pub descriptor_set: u32,
    pub binding: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PushConstant {
    pub ty: PtrTyId,
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct Function {
    name: Option<String>,
    ret: Option<Ty>,
    args: Vec<Ty>,
    vars: Vec<(PtrTyId, Option<Value>)>,
    body: Vec<Node>,
    value_nodes: Vec<ValueNode>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Node {
    Value(ValueNodeId),
    Store(Value, Value),
    ReturnValue(Value),
    Return,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueNodeOp {
    Load(Value),
    IAdd(Value, Value),
    FAdd(Value, Value),
    // TODO: Check for access chain validity
    Access(Value, Box<[Value]>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueNode {
    pub ty: Ty,
    pub op: ValueNodeOp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FunctionId(pub usize);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValueNodeId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Parameter(pub usize);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Temporary(pub usize);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Private(pub usize);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StorageBufferId(pub usize);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UniformId(pub usize);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PushConstantId(pub usize);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuiltinId(pub usize); // NOTE: Maybe use an enum for this
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Value {
    Parameter(Parameter),
    Temporary(Temporary),
    Constant(ConstantId),
    Variable(Variable),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variable {
    Private(Private),
    Builtin(BuiltinId),
    StorageBuffer(StorageBufferId),
    Uniform(UniformId),
    PushConstant(PushConstantId),
}

impl From<Parameter> for Value {
    fn from(p: Parameter) -> Self {
        Value::Parameter(p)
    }
}

impl From<Temporary> for Value {
    fn from(t: Temporary) -> Self {
        Value::Temporary(t)
    }
}

impl From<Variable> for Value {
    fn from(v: Variable) -> Self {
        Value::Variable(v)
    }
}

impl From<Private> for Value {
    fn from(v: Private) -> Self {
        Value::Variable(Variable::Private(v))
    }
}

impl From<StorageBufferId> for Value {
    fn from(v: StorageBufferId) -> Self {
        Value::Variable(Variable::StorageBuffer(v))
    }
}

impl From<UniformId> for Value {
    fn from(v: UniformId) -> Self {
        Value::Variable(Variable::Uniform(v))
    }
}

impl From<PushConstantId> for Value {
    fn from(v: PushConstantId) -> Self {
        Value::Variable(Variable::PushConstant(v))
    }
}

impl From<BuiltinId> for Value {
    fn from(v: BuiltinId) -> Self {
        Value::Variable(Variable::Builtin(v))
    }
}

impl From<ConstantId> for Value {
    fn from(v: ConstantId) -> Self {
        Value::Constant(v)
    }
}

impl From<StorageBufferId> for Variable {
    fn from(v: StorageBufferId) -> Self {
        Variable::StorageBuffer(v)
    }
}

impl From<UniformId> for Variable {
    fn from(v: UniformId) -> Self {
        Variable::Uniform(v)
    }
}

impl From<PushConstantId> for Variable {
    fn from(v: PushConstantId) -> Self {
        Variable::PushConstant(v)
    }
}

impl From<BuiltinId> for Variable {
    fn from(v: BuiltinId) -> Self {
        Variable::Builtin(v)
    }
}

#[derive(Debug)]
pub enum Error {
    AssemblerError(rspirv::dr::Error),
    MissingEntrypoinName,
    ReturnInVoid,
    ReturnTypeMismatch(Ty),
    WrongNumberType(ScalarTy),
}

impl From<rspirv::dr::Error> for Error {
    fn from(error: rspirv::dr::Error) -> Self {
        Error::AssemblerError(error)
    }
}

#[derive(Default)]
struct LazyTypeMapper {
    void: Option<Word>,
    composites: Box<[Option<Word>]>,
    ptrs: Box<[Option<Word>]>,
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

#[derive(Debug, Clone, Copy)]
struct ShaderScope<'a> {
    constants: &'a [Word],
    builtins: &'a [Word],
    storage_buffers: &'a [Word],
    uniforms: &'a [Word],
    push_constants: &'a [Word],
}

struct FunctionCompiler<'a> {
    type_mapper: &'a mut LazyTypeMapper,
    constant_ids: &'a [Word],
    function: &'a Function,
    ir_types: Types<'a>,
    parameter_ids: Box<[u32]>,
    variable_ids: Box<[u32]>,
    value_ids: Vec<u32>,
    func_id: u32,
    scope: ShaderScope<'a>,
}

#[derive(Default)]
pub struct SPVModuleBuilder;

impl Translate<IR, SPVModule> for SPVModuleBuilder {
    type Error = Error;
    type Config = Config;
    fn translate(
        self,
        ir: IR,
        config: &Self::Config,
    ) -> Result<SPVModule, Self::Error> {
        let mut b = SPVBuilder::new();
        let (major, minor) = config
            .version
            .unwrap_or((spirv::MAJOR_VERSION, spirv::MINOR_VERSION));
        b.set_version(major, minor);
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

        let mut type_mapper = LazyTypeMapper::new(ir.types());

        let mut constant_ids = Vec::with_capacity(ir.constants.len());
        for c in ir.constants.iter() {
            let id = match c {
                &Constant::Scalar(ScalarConstant::U32(v)) => {
                    let t = type_mapper.scalars.get_u32(&mut b);
                    b.constant_bit32(t, v)
                }
                _ => todo!(),
            };
            constant_ids.push(id);
        }
        let constant_ids = constant_ids.into_boxed_slice();

        let storage_buffer_ids = ir
            .storage_buffers
            .iter()
            // TODO: Maybe put this in a declared function
            .map(
                |&StorageBuffer {
                     descriptor_set,
                     binding,
                     ty,
                     .. // writable,
                 }| {
                    let ty_id = type_mapper.get_ptr(
                        &mut b,
                        ir.types(),
                        &constant_ids,
                        ty,
                        StorageClass::StorageBuffer,
                    );
                    let id = b.variable(
                        ty_id,
                        None,
                        StorageClass::StorageBuffer,
                        None,
                    );
                    b.decorate(
                        id,
                        spirv::Decoration::DescriptorSet,
                        [descriptor_set.into()],
                    );
                    b.decorate(
                        id,
                        spirv::Decoration::Binding,
                        [binding.into()],
                    );
                    // if !writable {
                    //     b.decorate(id, spirv::Decoration::NonWritable, []);
                    // }
                    id
                },
            )
            .collect::<Box<[_]>>();

        let uniform_ids = ir
            .uniforms
            .iter()
            // TODO: Maybe put this in a declared function
            .map(
                |&Uniform {
                     descriptor_set,
                     binding,
                     ty,
                 }| {
                    let ty_id = type_mapper.get_ptr(
                        &mut b,
                        ir.types(),
                        &constant_ids,
                        ty,
                        StorageClass::Uniform,
                    );
                    let id =
                        b.variable(ty_id, None, StorageClass::Uniform, None);
                    b.decorate(
                        id,
                        spirv::Decoration::DescriptorSet,
                        [descriptor_set.into()],
                    );
                    b.decorate(
                        id,
                        spirv::Decoration::Binding,
                        [binding.into()],
                    );
                    // if !writable {
                    //     b.decorate(id, spirv::Decoration::NonWritable, []);
                    // }
                    id
                },
            )
            .collect::<Box<[_]>>();

        let push_constant_ids = ir
            .push_constants
            .iter()
            .map(|&PushConstant { ty }| {
                let ty_id = type_mapper.get_ptr(
                    &mut b,
                    ir.types(),
                    &constant_ids,
                    ty,
                    StorageClass::PushConstant,
                );
                let id =
                    b.variable(ty_id, None, StorageClass::PushConstant, None);
                id
            })
            .collect::<Box<[_]>>();

        let builtin_ids = ir
            .builtins
            .iter()
            .map(|&(ty, which)| {
                let ty_id = type_mapper.get_ptr(
                    &mut b,
                    ir.types(),
                    &constant_ids,
                    ty,
                    StorageClass::Input,
                );
                let id = b.variable(ty_id, None, StorageClass::Input, None);
                b.decorate(
                    id,
                    spirv::Decoration::BuiltIn,
                    [rspirv::dr::Operand::BuiltIn(which)],
                );
                id
            })
            .collect::<Box<[_]>>();

        let mut spirv_functions = Vec::with_capacity(ir.functions.len());

        for i in 0..ir.functions.len() {
            let func_id = FunctionCompiler::new(
                &mut b,
                &mut type_mapper,
                &constant_ids,
                &ir.functions[i],
                &ir,
                ShaderScope {
                    constants: &constant_ids,
                    storage_buffers: &storage_buffer_ids,
                    uniforms: &uniform_ids,
                    builtins: &builtin_ids,
                    push_constants: &push_constant_ids,
                },
            )?
            .compile(&mut b)?;
            spirv_functions.push(func_id);
        }

        if let Some((FunctionId(i), variables)) = ir.entry_point {
            b.entry_point(
                spirv::ExecutionModel::GLCompute,
                spirv_functions[i],
                ir.functions
                    .get(i)
                    .and_then(|f| f.name.clone())
                    .ok_or(Error::MissingEntrypoinName)?,
                variables
                    .into_iter()
                    .map(|v| match v {
                        Variable::StorageBuffer(StorageBufferId(i)) => {
                            storage_buffer_ids[i]
                        }
                        Variable::Builtin(BuiltinId(i)) => builtin_ids[i],
                        Variable::PushConstant(PushConstantId(i)) => {
                            push_constant_ids[i]
                        }
                        _ => todo!(),
                    })
                    .collect::<Box<[_]>>(),
            );

            b.execution_mode(
                spirv_functions[i],
                spirv::ExecutionMode::LocalSize,
                &config.local_size,
            );
        }

        Ok(b.module())
    }
}

impl IR {
    pub fn new() -> Self {
        Default::default()
    }

    fn types(&self) -> Types {
        Types {
            composite: &self.composite_types,
            ptr: &self.ptr_types,
        }
    }

    pub fn make_ptr_type(&mut self, ty: Ty) -> PtrTyId {
        let id = self.ptr_types.len();
        self.ptr_types.push(ty);
        PtrTyId(id)
    }

    pub fn make_composite_type(&mut self, ty: CompositeTy) -> CompositeTyId {
        let id = self.composite_types.len();
        self.composite_types.push(ty);
        CompositeTyId(id)
    }

    pub fn make_storage_buffer(
        &mut self,
        storage_buffer: StorageBuffer,
    ) -> StorageBufferId {
        let id = self.storage_buffers.len();
        self.storage_buffers.push(storage_buffer);
        StorageBufferId(id)
    }

    pub fn make_uniform(&mut self, uniform: Uniform) -> UniformId {
        let id = self.storage_buffers.len();
        self.uniforms.push(uniform);
        UniformId(id)
    }

    pub fn make_push_constant(
        &mut self,
        push_constant: PushConstant,
    ) -> PushConstantId {
        let id = self.push_constants.len();
        self.push_constants.push(push_constant);
        PushConstantId(id)
    }

    pub fn make_builtin(
        &mut self,
        ty: PtrTyId,
        builtin: spirv::BuiltIn,
    ) -> BuiltinId {
        let id = self.builtins.len();
        self.builtins.push((ty, builtin));
        BuiltinId(id)
    }

    pub fn make_constant(&mut self, constant: Constant) -> ConstantId {
        let id = self.constants.len();
        self.constants.push(constant);
        ConstantId(id)
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

    pub fn entry_point(
        &mut self,
        func_id: FunctionId,
        variables: impl Into<Box<[Variable]>>,
    ) {
        self.entry_point = Some((func_id, variables.into()));
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

    pub fn make_argument(&mut self, ty: Ty) -> Parameter {
        let id = self.args.len();
        self.args.push(ty);
        Parameter(id)
    }

    pub fn make_variable(&mut self, ty: PtrTyId) -> Private {
        let id = self.vars.len();
        self.vars.push((ty, None));
        Private(id)
    }

    pub fn append_node(&mut self, node: Node) {
        self.body.push(node);
    }

    pub fn append_value_node(&mut self, node: ValueNode) -> Temporary {
        let (id, value) = self.make_value_node(node);
        self.body.push(Node::Value(id));
        value
    }

    pub fn return_value(&mut self, value: Value) {
        self.body.push(Node::ReturnValue(value));
    }

    pub fn make_value_node(
        &mut self,
        value: ValueNode,
    ) -> (ValueNodeId, Temporary) {
        let id = self.value_nodes.len();
        self.value_nodes.push(value);
        (ValueNodeId(id), Temporary(id))
    }
}

impl Ty {
    fn same_as(&self, other: &Ty, _types: Types<'_>) -> bool {
        self == other // TODO: composite comparison
    }
}

impl ValueNode {
    pub fn new(ty: Ty, op: ValueNodeOp) -> Self {
        Self { ty, op }
    }

    pub fn iadd(ty: Ty, a: Value, b: Value) -> Self {
        Self::new(ty, ValueNodeOp::IAdd(a, b))
    }

    pub fn fadd(ty: Ty, a: Value, b: Value) -> Self {
        Self::new(ty, ValueNodeOp::FAdd(a, b))
    }

    pub fn load(ty: Ty, a: Value) -> Self {
        Self::new(ty, ValueNodeOp::Load(a))
    }

    pub fn access(
        ty: PtrTyId,
        storage_class: StorageClass,
        base: Value,
        chain: impl Into<Box<[Value]>>,
    ) -> Self {
        Self::new(
            Ty::Ptr(ty, storage_class),
            ValueNodeOp::Access(base, chain.into()),
        )
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
        constant_ids: &'a [Word],
        func: &'a Function,
        ir: &'a IR,
        scope: ShaderScope<'a>,
    ) -> Result<Self, rspirv::dr::Error> {
        let Function { ret: maybe_ret, .. } = func;
        let ret = maybe_ret
            .map(|ret| type_mapper.get(b, ir.types(), constant_ids, &ret))
            .unwrap_or_else(|| type_mapper.get_void(b));
        let args = func
            .args
            .iter()
            .map(|ty| type_mapper.get(b, ir.types(), constant_ids, &ty))
            .collect::<Box<[_]>>();
        let func_ty_id = b.type_function(ret, args.clone());
        let func_id = b.begin_function(
            ret,
            None,
            spirv::FunctionControl::DONT_INLINE,
            func_ty_id,
        )?;
        let _block_id = b.begin_block(None)?;
        let parameter_ids = args
            .into_iter()
            .map(|id| b.function_parameter(id))
            .collect::<Result<Box<[_]>, rspirv::dr::Error>>()?;
        // TODO: handle initializers
        let variable_ids = func
            .vars
            .iter()
            .map(|&(ty, _init)| {
                let ty_id = type_mapper.get_ptr(
                    b,
                    ir.types(),
                    constant_ids,
                    ty,
                    StorageClass::Function,
                );
                // let init = init.map(|init| b.variable(ty_id, None, StorageClass::Function, init));
                b.variable(ty_id, None, StorageClass::Function, None)
            })
            .collect();
        Ok(FunctionCompiler {
            type_mapper,
            constant_ids,
            function: func,
            ir_types: ir.types(),
            func_id,
            value_ids: Vec::new(),
            parameter_ids,
            variable_ids,
            scope,
        })
    }

    fn get_value(&self, v: Value) -> Word {
        match v {
            Value::Parameter(Parameter(i)) => self.parameter_ids[i],
            Value::Temporary(Temporary(i)) => self.value_ids[i],
            Value::Variable(Variable::Private(Private(i))) => {
                self.variable_ids[i]
            }
            Value::Variable(Variable::StorageBuffer(StorageBufferId(i))) => {
                self.scope.storage_buffers[i]
            }
            Value::Variable(Variable::Uniform(UniformId(i))) => {
                self.scope.uniforms[i]
            }
            Value::Variable(Variable::Builtin(BuiltinId(i))) => {
                self.scope.builtins[i]
            }
            Value::Constant(ConstantId(i)) => self.scope.constants[i],
            Value::Variable(Variable::PushConstant(PushConstantId(i))) => {
                self.scope.push_constants[i]
            }
        }
    }

    pub fn compile(mut self, b: &mut SPVBuilder) -> Result<u32, Error> {
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
                let value_node = &self.function.value_nodes[*value_node_idx];
                let () = self.compile_value_node(b, value_node)?;
            }
            &Node::ReturnValue(value) => {
                let ty = self.get_value_type(value);
                let value = self.get_value(value);
                if let None = self.function.ret {
                    return Err(Error::ReturnInVoid);
                } else if let Some(ret) = self.function.ret {
                    if !ty.same_as(&ret, self.ir_types) {
                        return Err(Error::ReturnTypeMismatch(ty.clone()));
                    }
                }
                b.ret_value(value)?;
            }
            &Node::Return => b.ret()?,
            // TODO: Type checking perhaps
            Node::Store(ptr, value) => {
                let ptr = self.get_value(*ptr);
                let value = self.get_value(*value);
                b.store(ptr, value, None, [])?;
            }
        }
        Ok(())
    }

    fn get_value_type(&self, v: Value) -> Ty {
        match v {
            Value::Parameter(Parameter(i)) => self.function.args[i],
            Value::Temporary(Temporary(i)) => self.function.value_nodes[i].ty,
            Value::Variable(Variable::Private(Private(i))) => {
                Ty::Ptr(self.function.vars[i].0, StorageClass::Function)
            }
            _ => todo!(),
        }
    }

    fn compile_value_node(
        &mut self,
        b: &mut SPVBuilder,
        node: &ValueNode,
    ) -> Result<(), Error> {
        type Op = ValueNodeOp;
        match node {
            ValueNode {
                ty,
                op: Op::IAdd(lhs, rhs),
            } => {
                let lhs = self.get_value(*lhs);
                let rhs = self.get_value(*rhs);
                // TODO: Check type compatibility for operation
                let ty_id = self.type_mapper.get(
                    b,
                    self.ir_types,
                    self.constant_ids,
                    ty,
                );
                let inner_ty = match ty {
                    &Ty::Scalar(ty) => ty,
                    &Ty::Vec(ty, _n) => ty,
                    _ => todo!(),
                };
                if !matches!(inner_ty, ScalarTy::U32 | ScalarTy::S32) {
                    return Err(Error::WrongNumberType(inner_ty));
                }
                let op_id = b.i_add(ty_id, None, lhs, rhs)?;
                self.value_ids.push(op_id);
            }

            ValueNode {
                ty,
                op: Op::FAdd(lhs, rhs),
            } => {
                let lhs = self.get_value(*lhs);
                let rhs = self.get_value(*rhs);
                // TODO: Check type compatibility for operation
                let ty_id = self.type_mapper.get(
                    b,
                    self.ir_types,
                    self.constant_ids,
                    ty,
                );
                let inner_ty = match ty {
                    &Ty::Scalar(ty) => ty,
                    &Ty::Vec(ty, _n) => ty,
                    _ => todo!(),
                };
                if !matches!(inner_ty, ScalarTy::F32) {
                    return Err(Error::WrongNumberType(inner_ty));
                }
                let op_id = b.f_add(ty_id, None, lhs, rhs)?;
                self.value_ids.push(op_id);
            }

            ValueNode {
                ty,
                op: Op::Load(ptr),
            } => {
                // TODO: Type checking perhaps
                let ty_id = self.type_mapper.get(
                    b,
                    self.ir_types,
                    self.constant_ids,
                    ty,
                );
                let ptr_id = self.get_value(*ptr);
                let id = b.load(ty_id, None, ptr_id, None, [])?;
                self.value_ids.push(id);
            }

            ValueNode {
                ty,
                op: Op::Access(base, chain),
            } => {
                let base = self.get_value(*base);
                let chain = chain
                    .iter()
                    .map(|v| self.get_value(*v))
                    .collect::<Box<[_]>>();
                let ty_id = self.type_mapper.get(
                    b,
                    self.ir_types,
                    self.constant_ids,
                    ty,
                );
                let id = b.access_chain(ty_id, None, base, chain)?;
                self.value_ids.push(id);
            }
        }
        Ok(())
    }
}

impl LazyTypeMapper {
    fn new(types: Types<'_>) -> Self {
        Self {
            composites: vec![None; types.composite.len()].into_boxed_slice(),
            ptrs: vec![None; types.ptr.len()].into_boxed_slice(),
            ..Default::default()
        }
    }

    fn get(
        &mut self,
        b: &mut SPVBuilder,
        ir_types: Types<'_>,
        constants: &[Word],
        ty: &Ty,
    ) -> Word {
        match ty {
            &Ty::Scalar(ty) => self.scalars.get(b, ty),
            &Ty::Vec(ty, n) => *self.vectors[n as usize - 2]
                .access_mut(ty)
                .get_or_insert_with(|| {
                    let base = self.scalars.get(b, ty);
                    b.type_vector(base, n)
                }),
            &Ty::Mat(ty, n) => *self.matrices[n as usize - 2]
                .access_mut(ty)
                .get_or_insert_with(|| {
                    let base = self.scalars.get(b, ty);
                    b.type_matrix(base, n)
                }),
            &Ty::Ptr(ty, storage_class) => {
                self.get_ptr(b, ir_types, constants, ty, storage_class)
            }
            &Ty::Composite(ty) => {
                self.get_composite(b, ir_types, constants, ty)
            }
        }
    }

    fn get_ptr(
        &mut self,
        b: &mut SPVBuilder,
        ir_types: Types<'_>,
        constants: &[Word],
        PtrTyId(i): PtrTyId,
        storage_class: StorageClass,
    ) -> Word {
        let Some(ty) = ir_types.ptr.get(i) else {
            unreachable!()
        };
        let base = self.get(b, ir_types, constants, ty);
        *self.ptrs[i].get_or_insert_with(|| {
            let id = b.type_pointer(None, storage_class, base);
            id
        })
    }

    fn get_void(&mut self, b: &mut SPVBuilder) -> Word {
        *self.void.get_or_insert_with(|| b.type_void())
    }

    fn get_composite(
        &mut self,
        b: &mut SPVBuilder,
        ir_types: Types<'_>,
        constants: &[Word],
        ty: CompositeTyId,
    ) -> Word {
        let CompositeTyId(i) = ty;
        let (is_found, id) = match &self.composites[i] {
            &Some(id) => (true, id),
            None => (
                false,
                self.make_composite(
                    b,
                    ir_types,
                    constants,
                    &ir_types.composite[i],
                ),
            ),
        };
        if !is_found {
            self.composites[i] = Some(id);
        }
        id
    }

    fn make_composite(
        &mut self,
        b: &mut SPVBuilder,
        ir_types: Types<'_>,
        constants: &[Word],
        ty: &CompositeTy,
    ) -> Word {
        match ty {
            CompositeTy::Array1(ty, None) => {
                self.make_runtime_array1(b, ir_types, constants, ty)
            }
            &CompositeTy::Array1(ref ty, Some(len)) => {
                self.make_array1(b, ir_types, constants, ty, len)
            }
            CompositeTy::Struct { fields, is_block } => {
                self.make_struct(b, ir_types, constants, fields, *is_block)
            }
        }
    }

    fn make_array1(
        &mut self,
        b: &mut SPVBuilder,
        ir_types: Types<'_>,
        constants: &[Word],
        elem_ty: &Ty,
        ConstantId(len_idx): ConstantId,
    ) -> Word {
        let base = self.get(b, ir_types, constants, elem_ty);
        let len = constants[len_idx];
        let id = b.type_array(base, len);
        // TODO: Array stride
        // b.decorate(id, spirv::Decoration::ArrayStride, );
        id
    }

    fn make_runtime_array1(
        &mut self,
        b: &mut SPVBuilder,
        ir_types: Types<'_>,
        constants: &[Word],
        elem_ty: &Ty,
    ) -> Word {
        let base = self.get(b, ir_types, constants, elem_ty);
        let id = b.type_runtime_array(base);
        // HACK: calculate type sizes for strides or annotate array types
        let stride = 4;
        b.decorate(
            id,
            spirv::Decoration::ArrayStride,
            [rspirv::dr::Operand::LiteralBit32(stride)],
        );
        id
    }

    fn make_struct(
        &mut self,
        b: &mut SPVBuilder,
        ir_types: Types<'_>,
        constants: &[Word],
        fields: &[StructMember],
        is_block: bool,
    ) -> Word {
        let field_ids = fields
            .iter()
            .map(|StructMember { ty, .. }| self.get(b, ir_types, constants, ty))
            .collect::<Box<[_]>>();
        let id = b.id();
        let id = b.type_struct_id(Some(id), field_ids);
        if is_block {
            b.decorate(id, spirv::Decoration::Block, []);
        }
        // TODO: Maybe use this as the size
        let _ = fields.iter().enumerate().fold(
            0,
            |offset, (i, &StructMember { writable, .. })| {
                let i = i as u32;
                if !writable {
                    b.member_decorate(
                        id,
                        i,
                        spirv::Decoration::NonWritable,
                        [],
                    );
                }
                // HACK: calculate type sizes
                let member_size = 4;
                b.member_decorate(
                    id,
                    i,
                    spirv::Decoration::Offset,
                    [rspirv::dr::Operand::LiteralBit32(offset)],
                );
                offset + member_size
            },
        );
        id
    }
}

impl LazyScalarTypeMapper {
    fn access_mut(&mut self, ty: ScalarTy) -> &mut Option<Word> {
        match ty {
            ScalarTy::F32 => &mut self.f32,
            ScalarTy::U32 => &mut self.u32,
            ScalarTy::S32 => &mut self.s32,
        }
    }

    fn get(&mut self, b: &mut SPVBuilder, ty: ScalarTy) -> Word {
        match ty {
            ScalarTy::F32 => self.get_f32(b),
            ScalarTy::U32 => self.get_u32(b),
            ScalarTy::S32 => self.get_s32(b),
        }
    }

    fn get_f32(&mut self, b: &mut SPVBuilder) -> Word {
        *self.f32.get_or_insert_with(|| b.type_float(32))
    }

    fn get_u32(&mut self, b: &mut SPVBuilder) -> Word {
        *self.u32.get_or_insert_with(|| b.type_int(32, 0))
    }

    fn get_s32(&mut self, b: &mut SPVBuilder) -> Word {
        *self.s32.get_or_insert_with(|| b.type_int(32, 1))
    }
}

#[cfg(test)]
pub mod test_utils {
    use super::*;

    pub fn make_module(config: Config) -> SPVModule {
        let mut ir = IR::new();
        let t_f32 = Ty::Scalar(ScalarTy::F32);
        let t_f32_ptr = ir.make_ptr_type(t_f32);
        let t_f32_rt_array = ir
            .make_composite_type(CompositeTy::Array1(t_f32, None))
            .into();
        let t2_input = ir.make_composite_type(CompositeTy::r#struct(
            [StructMember {
                ty: t_f32_rt_array,
                writable: false,
            }],
            true,
        ));
        let t2_output = ir.make_composite_type(CompositeTy::r#struct(
            [StructMember {
                ty: t_f32_rt_array,
                writable: true,
            }],
            true,
        ));
        let t3_input = ir.make_ptr_type(t2_input.into());
        let t3_output = ir.make_ptr_type(t2_output.into());

        let t_u32_vec3 = Ty::Vec(ScalarTy::U32, 3);
        let t_u32_vec3_ptr = ir.make_ptr_type(t_u32_vec3);
        let t_u32 = Ty::Scalar(ScalarTy::U32);
        let t_u32_ptr = ir.make_ptr_type(t_u32);
        let global_invocation_id =
            ir.make_builtin(t_u32_vec3_ptr, spirv::BuiltIn::GlobalInvocationId);

        let u32_0 = ir
            .make_constant(Constant::Scalar(ScalarConstant::U32(0)))
            .into();

        let inputs = [0, 1].map(|binding| {
            ir.make_storage_buffer(StorageBuffer {
                ty: t3_input,
                descriptor_set: 0,
                binding,
                writable: false,
            })
        });
        let output = ir.make_storage_buffer(StorageBuffer {
            ty: t3_output,
            descriptor_set: 0,
            binding: 2,
            writable: true,
        });

        let mut main = Function::new(Some("main".into()), None);
        let global_invocation_id_x_ptr =
            main.append_value_node(ValueNode::access(
                t_u32_ptr,
                StorageClass::Input,
                global_invocation_id.into(),
                [u32_0],
            ));
        let x = main.append_value_node(ValueNode::load(
            Ty::Scalar(ScalarTy::U32),
            global_invocation_id_x_ptr.into(),
        ));

        let [a, b] = inputs.map(|input| {
            let ptr_id = main.append_value_node(ValueNode::access(
                t_f32_ptr,
                StorageClass::StorageBuffer,
                input.into(),
                [u32_0, x.into()],
            ));
            main.append_value_node(ValueNode::load(t_f32, ptr_id.into()))
        });

        let output_value =
            main.append_value_node(ValueNode::fadd(t_f32, a.into(), b.into()));
        let output_access = main.append_value_node(ValueNode::access(
            t_f32_ptr,
            StorageClass::StorageBuffer,
            output.into(),
            [u32_0, x.into()],
        ));
        main.append_node(Node::Store(
            output_access.into(),
            output_value.into(),
        ));
        main.append_node(Node::Return);

        let main_func = ir.make_function(main);
        ir.entry_point(
            main_func,
            [
                global_invocation_id.into(),
                inputs[0].into(),
                inputs[1].into(),
                output.into(),
            ],
        );

        SPVModuleBuilder::default().translate(ir, &config).unwrap()
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
        let module = test_utils::make_module(Config {
            local_size: [1, 1, 1],
            ..Default::default()
        });
        eprintln!("{}", module.disassemble());
        let raw = module.assemble();
        assert_ne!(raw.len(), 0);

        // to make sure it's actually valid
        let mut loader = rspirv::dr::Loader::new();
        let () =
            rspirv::binary::parse_words(raw.as_slice(), &mut loader).unwrap();
        assert_eq!(loader.module().assemble(), raw);

        let mut child = Command::new("spirv-val")
            .arg("-")
            .stdin(Stdio::piped())
            .spawn()
            .unwrap();
        {
            let mut stdin = child.stdin.take().unwrap();
            stdin
                .write_all(unsafe { raw.as_slice().align_to::<u8>().1 })
                .unwrap();
            stdin.flush().unwrap();
        }
        let output = child.wait_with_output().unwrap();
        assert!(output.status.success());
    }
}
