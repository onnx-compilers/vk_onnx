pub mod constant;
pub mod ty;

use crate::l_base::{ScalarTy, TranslateFrom};

use rspirv::dr::{Builder as SPVBuilder, Module as SPVModule};
use rspirv::spirv::{self, StorageClass, Word};

use constant::{Constant, ConstantId, ScalarConstant};
use ty::{CompositeTy, CompositeTyId, PtrTyId, Ty};

#[derive(Debug, Default, Clone)]
pub struct IR {
    pub functions: Vec<Function>,
    pub constants: Vec<Constant>,
    pub uniforms: Vec<Uniform>,
    pub composite_types: Vec<CompositeTy>,
    pub ptr_types: Vec<Ty>,
    pub config: IRConfig,
    pub entry_point: Option<FunctionId>,
}

#[derive(Default, Debug, Clone, Copy)]
pub struct Types<'a> {
    pub composite: &'a [CompositeTy],
    pub ptr: &'a [Ty],
}

#[derive(Default, Debug, Clone)]
pub struct IRConfig {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Uniform {
    pub ty: PtrTyId,
    pub descriptor_set: u32,
    pub binding: u32,
    pub writable: bool,
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub name: Option<String>,
    pub ret: Option<Ty>,
    pub args: Vec<Ty>,
    pub vars: Vec<(PtrTyId, Option<Value>)>,
    pub body: Vec<Node>,
    pub value_nodes: Vec<ValueNode>,
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
pub struct UniformId(pub usize);
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
    Uniform(UniformId),
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

impl From<UniformId> for Value {
    fn from(v: UniformId) -> Self {
        Value::Variable(Variable::Uniform(v))
    }
}

impl From<ConstantId> for Value {
    fn from(v: ConstantId) -> Self {
        Value::Constant(v)
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
    fn from(e: rspirv::dr::Error) -> Self {
        Error::AssemblerError(e)
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
    uniforms: &'a [Word],
}

struct FunctionCompiler<'a> {
    type_mapper: &'a mut LazyTypeMapper,
    function: &'a Function,
    types: Types<'a>,
    parameter_ids: Box<[u32]>,
    variable_ids: Box<[u32]>,
    value_ids: Vec<u32>,
    func_id: u32,
    scope: ShaderScope<'a>,
}

impl TranslateFrom<IR> for SPVModule {
    type Error = Error;
    fn translate_from(ir: IR) -> Result<Self, Self::Error> {
        let mut b = SPVBuilder::new();
        b.set_version(1, 0);
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

        let uniform_ids = ir
            .uniforms
            .iter()
            // TODO: Maybe put this in a declared function
            .map(
                |&Uniform {
                     descriptor_set,
                     binding,
                     ty,
                     writable,
                 }| {
                    let ty_id = type_mapper.get_ptr(
                        &mut b,
                        ir.types(),
                        ty,
                        StorageClass::Uniform,
                    );
                    // TODO: Decorate the variable, not the type.
                    b.decorate(ty_id, spirv::Decoration::BufferBlock, []);
                    b.decorate(
                        ty_id,
                        spirv::Decoration::DescriptorSet,
                        [descriptor_set.into()],
                    );
                    b.decorate(
                        ty_id,
                        spirv::Decoration::Binding,
                        [binding.into()],
                    );
                    if !writable {
                        b.decorate(ty_id, spirv::Decoration::NonWritable, []);
                    }
                    b.variable(ty_id, None, StorageClass::Uniform, None)
                },
            )
            .collect::<Box<[_]>>();

        let constant_ids = ir
            .constants
            .iter()
            .map(|c| match c {
                &Constant::Scalar(ScalarConstant::U32(v)) => {
                    let t =
                        type_mapper.scalars.get(&mut b, ScalarTy::U32).into();
                    b.constant_bit32(t, v)
                }
                _ => todo!(),
            })
            .collect::<Box<[_]>>();

        let mut spirv_functions = Vec::with_capacity(ir.functions.len());

        for i in 0..ir.functions.len() {
            let func_id = FunctionCompiler::new(
                &mut b,
                &mut type_mapper,
                &ir.functions[i],
                &ir,
                ShaderScope {
                    uniforms: &uniform_ids,
                    constants: &constant_ids,
                },
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

    pub fn types(&self) -> Types {
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

    pub fn make_uniform(&mut self, uniform: Uniform) -> UniformId {
        let id = self.uniforms.len();
        self.uniforms.push(uniform);
        UniformId(id)
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
    pub fn same_as(&self, other: &Ty, _types: Types<'_>) -> bool {
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
        scope: ShaderScope<'a>,
    ) -> Result<Self, rspirv::dr::Error> {
        let Function { ret: maybe_ret, .. } = func;
        let ret = maybe_ret
            .map(|ret| type_mapper.get(b, ir.types(), &ret))
            .unwrap_or_else(|| type_mapper.get_void(b));
        let args = func
            .args
            .iter()
            .map(|ty| type_mapper.get(b, ir.types(), &ty))
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
                    ty,
                    StorageClass::Function,
                );
                // let init = init.map(|init| b.variable(ty_id, None, StorageClass::Function, init));
                b.variable(ty_id, None, StorageClass::Function, None)
            })
            .collect();
        Ok(FunctionCompiler {
            type_mapper,
            function: func,
            types: ir.types(),
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
            Value::Variable(Variable::Uniform(UniformId(i))) => {
                self.scope.uniforms[i]
            }
            Value::Constant(ConstantId(i)) => self.scope.constants[i],
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
                    if !ty.same_as(&ret, self.types) {
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
                let ty_id = self.type_mapper.get(b, self.types, ty);
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
                let ty_id = self.type_mapper.get(b, self.types, ty);
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
                let ty_id = self.type_mapper.get(b, self.types, ty);
                let ptr_id = self.get_value(*ptr);
                let id = b.load(ty_id, None, ptr_id, None, [])?;
                self.value_ids.push(id);
            }

            ValueNode {
                ty,
                op: Op::Access(base, chain),
            } => {
                let base = self.get_value(dbg!(*base));
                let chain = chain
                    .iter()
                    .map(|v| self.get_value(*v))
                    .collect::<Box<[_]>>();
                let ty_id = self.type_mapper.get(b, self.types, ty);
                dbg!((ty_id, base));
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

    fn get(&mut self, b: &mut SPVBuilder, types: Types<'_>, ty: &Ty) -> Word {
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
                self.get_ptr(b, types, ty, storage_class)
            }
            &Ty::Composite(ty) => self.get_composite(b, types, ty),
        }
    }

    fn get_ptr(
        &mut self,
        b: &mut SPVBuilder,
        types: Types<'_>,
        ptr_ty: PtrTyId,
        storage_class: StorageClass,
    ) -> Word {
        let Some(ty) = types.ptr.get(ptr_ty.0) else {
            unreachable!()
        };
        let base = self.get(b, types, ty);
        *self.ptrs[ptr_ty.0]
            .get_or_insert_with(|| b.type_pointer(None, storage_class, base))
    }

    fn get_void(&mut self, b: &mut SPVBuilder) -> Word {
        *self.void.get_or_insert_with(|| b.type_void())
    }

    fn get_composite(
        &mut self,
        b: &mut SPVBuilder,
        types: Types<'_>,
        ty: CompositeTyId,
    ) -> Word {
        let CompositeTyId(i) = ty;
        let (is_found, id) = match &self.composites[i] {
            &Some(id) => (true, id),
            None => (
                false,
                match &types.composite[i] {
                    CompositeTy::Array1(ty, None) => {
                        let base = self.get(b, types, ty);
                        b.type_runtime_array(base)
                    }
                    CompositeTy::Array1(ty, Some(len)) => {
                        let base = self.get(b, types, ty);
                        let u32_id = self.scalars.get(b, ScalarTy::U32);
                        let len = b.constant_bit32(u32_id, *len as u32);
                        b.type_array(base, len)
                    }
                    // TODO: Decorate members
                    CompositeTy::Struct(fields) => {
                        let fields_ids = fields
                            .iter()
                            .map(|ty| self.get(b, types, ty))
                            .collect::<Box<[_]>>();
                        b.type_struct(fields_ids)
                    }
                },
            ),
        };
        if !is_found {
            self.composites[i] = Some(id);
        }
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
        *match ty {
            ScalarTy::F32 => self.f32.get_or_insert_with(|| b.type_float(32)),
            ScalarTy::U32 => self.u32.get_or_insert_with(|| b.type_int(32, 0)),
            ScalarTy::S32 => self.s32.get_or_insert_with(|| b.type_int(32, 1)),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io::Write,
        process::{Command, Stdio},
    };

    use crate::shader::l0::constant::ScalarConstant;

    use super::*;

    use rspirv::binary::{Assemble, Disassemble};

    #[test]
    fn test() {
        let mut ir = IR::new(IRConfig {
            ..Default::default()
        });
        let t0 = Ty::Scalar(ScalarTy::F32);
        let t1 = ir.make_composite_type(CompositeTy::Array1(t0, None)).into();
        let t2 = ir.make_composite_type(CompositeTy::Struct(Box::new([t1])));
        let t3 = ir.make_ptr_type(t2.into());
        let t4 = ir.make_ptr_type(t0);

        let u32_0 = ir
            .make_constant(Constant::Scalar(ScalarConstant::U32(0)))
            .into();

        let inputs = [0, 1]
            .into_iter()
            .map(|binding| {
                ir.make_uniform(Uniform {
                    ty: t3,
                    descriptor_set: 0,
                    binding,
                    writable: false,
                })
            })
            .collect::<Box<[_]>>();
        let output = ir.make_uniform(Uniform {
            ty: t3,
            descriptor_set: 0,
            binding: 2,
            writable: true,
        });

        let mut main = Function::new(Some("main".into()), None);
        let [a, b] = &inputs
            .iter()
            .map(|&input| {
                let ptr_id = main.append_value_node(ValueNode {
                    ty: Ty::Ptr(t4, StorageClass::Uniform),
                    op: ValueNodeOp::Access(
                        input.into(),
                        Box::new([u32_0, u32_0]),
                    ),
                });
                main.append_value_node(ValueNode::load(t0, ptr_id.into()))
                    .into()
            })
            .collect::<Box<[_]>>()[..]
        else {
            panic!("goofy ahh")
        };
        let output_value = main.append_value_node(ValueNode::fadd(t0, *a, *b));
        let output_access = main.append_value_node(ValueNode {
            ty: Ty::Ptr(t4, StorageClass::Uniform),
            op: ValueNodeOp::Access(output.into(), Box::new([u32_0, u32_0])),
        });
        main.append_node(Node::Store(
            output_access.into(),
            output_value.into(),
        ));
        main.append_node(Node::Return);

        let main_func = ir.make_function(main);
        ir.entry_point(main_func);

        let module = SPVModule::translate_from(ir).unwrap();
        eprintln!("{}", module.disassemble());
        let raw = module.assemble();
        assert_ne!(raw.len(), 0);
        //
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