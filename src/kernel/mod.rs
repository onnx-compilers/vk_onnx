// pub mod l0;
// pub mod l1;

use crate::shader::l0::IR;

pub mod componentwise {
    //! ```scheme
    //! (begin-for-template
    //!  (define-struct Bindings ([inputs (Array 2 U32)]
    //!                           [output U32]
    //!                           [config U32]))
    //!
    //!  (define-enum Op (Add))
    //!
    //!  (define-struct TemplateConfig ([local-size (Array 3 U32)]
    //!                                 [item-type ScalarTy]
    //!                                 [descriptor-set U32]
    //!                                 [bindings Bindings]
    //!                                 [op Op])
    //!    #:symbol "Config")
    //!
    //!  (configuration TemplateConfig))
    //!
    //! (define-type U32 (#%type 'u32))
    //! (define-type U32-Vec3 (Vec U32 3))
    //! (define-type T-RTArray (RTArray (#%get item-type)))
    //!
    //! (define-struct Input ([values T-RTArray]) #:writable #f)
    //!
    //! (define-struct Output ([values T-RTArray]) #:writable #t)
    //!
    //! (define-struct Config ([offset U32]) #:writable #f #:export "KernelConfig")
    //!
    //! (version 460)
    //! (local-size (#%get local-size))
    //!
    //! (define global-invocation-id (#%builtin U32-Vec3 'GlobalInvocationId))
    //!
    //! (define input-a
    //!   (#%storage-buffer Input
    //!                     #:layout (#%layout #:set (#%get descriptor-set)
    //!                                        #:binding (#%get bindings inputs 0))
    //!                     #:writable #f))
    //!
    //! (define input-b
    //!   (#%storage-buffer Input
    //!                     #:layout (#%layout #:set (#%get descriptor-set)
    //!                                        #:binding (#%get bindings inputs 1))
    //!                     #:writable #f))
    //!
    //! (define output
    //!   (#%storage-buffer Output
    //!                     #:layout (#%layout #:set (#%get descriptor-set)
    //!                                        #:binding (#%get bindings output))
    //!                     #:writable #t))
    //!
    //!
    //! (define config
    //!   (#%push-constant Config
    //!                    #:layout (#%layout #:set (#%get descriptor-set)
    //!                                       #:binding (#%get bindings config))
    //!                    #:writable #f))
    //!
    //! (interface
    //!  global-invocation-id input-a input-b output config)
    //!
    //! (define (main)
    //!   (define idx (+i (load@ global-invocation-id x)
    //!                  (load@ config offset)))
    //!   (define c (#%match (#%tuple (#%get op))
    //!              [(:: Op Add)
    //!               (+ (%get item-type)
    //!                  (load@ input-a values idx)
    //!                  (load@ input-b values idx)))])
    //!   (store (@ output values idx) c))
    //! ```
    use crate::{
        l_base::{ScalarTy, Translate},
        shader::l0::{
            BuiltinId, CompositeTy, /* , CompositeTyId */
            Constant, ConstantId, Function, Node, PtrTyId, PushConstant,
            PushConstantId, ScalarConstant, StorageBuffer, StructMember,
            Temporary, Ty, ValueNode, Variable,
        },
    };
    use rspirv::spirv::{BuiltIn, StorageClass};
    use vulkano::buffer::BufferContents;

    use super::IR;

    // TODO: Maybe use vectors and matrices for paralleller computation
    // pub enum ItemTy {
    //     Scalar(ScalarTy),
    //     Vec(ScalarTy, u32),
    //     Mat(ScalarTy, u32),
    // }

    #[derive(Debug, Clone)]
    pub enum Op<B> {
        // Maybe we need the unop, binop, multiop distinction
        Add {
            inputs: [B; 2],
            result: B,
        },
        Scaler {
            input: B,
            result: B,
            offset: f32,
            scale: f32,
        },
        // Sub,
        // Mul
    }

    #[derive(Debug, Clone, BufferContents)]
    #[repr(C)]
    pub struct KernelConfig {
        pub offset: u32,
    }

    #[derive(Debug, Clone)]
    pub struct Config {
        pub item_type: ScalarTy,
        pub descriptor_set: u32,
    }

    impl Default for Config {
        fn default() -> Self {
            Self {
                item_type: ScalarTy::F32,
                descriptor_set: 0,
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct Builder {
        ir: IR,
        // TODO: Maybe persist this sort of state in such a way that multiple
        //       kernel builders can use it so that they can be merged into one
        //       shader.
        fn_main: Function,
        // t_u32: Ty,
        // t_u32_ptr: PtrTyId,
        // t_u32_vec3: Ty,
        // t_u32_vec3_ptr: PtrTyId,
        // t_config_struct: CompositeTyId,
        // t_config_ptr: PtrTyId,
        const_u32_0: ConstantId,
        bi_global_invocation_id: BuiltinId,
        pc_config: PushConstantId,
        // tmp_gii_x_ptr: Temporary,
        // tmp_gii_x: Temporary,
        // tmp_offset_ptr: Temporary,
        // tmp_offset: Temporary,
        tmp_idx: Temporary,
    }

    struct IOState {
        t_item_scalar: ScalarTy,
        t_item: Ty,
        t_item_ptr: PtrTyId,
        // t_item_rt_array: CompositeTyId,
        // t_input_struct: CompositeTyId,
        // t_output_struct: CompositeTyId,
        t_input_ptr: PtrTyId,
        t_output_ptr: PtrTyId,
    }

    impl Builder {
        pub fn new() -> Self {
            let mut ir = IR::new();

            let t_u32 = ScalarTy::U32.into();
            let t_u32_ptr = ir.make_ptr_type(t_u32);
            let t_u32_vec3 = Ty::Vec(ScalarTy::U32, 3);
            let t_u32_vec3_ptr = ir.make_ptr_type(t_u32_vec3);
            let t_config_struct =
                ir.make_composite_type(CompositeTy::r#struct(
                    [StructMember {
                        ty: t_u32,
                        writable: false,
                    }],
                    true,
                ));
            let t_config_ptr = ir.make_ptr_type(t_config_struct.into());

            let const_u32_0 =
                ir.make_constant(Constant::Scalar(ScalarConstant::U32(0)));

            let global_invocation_id =
                ir.make_builtin(t_u32_vec3_ptr, BuiltIn::GlobalInvocationId);
            let config = ir.make_push_constant(PushConstant {
                ty: t_config_ptr.into(),
            });

            let mut main = Function::new(Some("main".into()), None);

            let gii_x_ptr = main.append_value_node(ValueNode::access(
                t_u32_ptr,
                StorageClass::Input,
                global_invocation_id.into(),
                [const_u32_0.into()],
            ));
            let gii_x = main
                .append_value_node(ValueNode::load(t_u32, gii_x_ptr.into()));

            let offset_ptr = main.append_value_node(ValueNode::access(
                t_u32_ptr,
                StorageClass::Uniform,
                config.into(),
                [const_u32_0.into()],
            ));
            let offset = main
                .append_value_node(ValueNode::load(t_u32, offset_ptr.into()));
            let idx = main.append_value_node(ValueNode::iadd(
                t_u32,
                gii_x.into(),
                offset.into(),
            ));

            Self {
                ir,
                fn_main: main,
                // t_u32,
                // t_u32_ptr,
                // t_u32_vec3,
                // t_u32_vec3_ptr,
                // t_config_struct,
                // t_config_ptr,
                const_u32_0,
                bi_global_invocation_id: global_invocation_id,
                pc_config: config,
                // tmp_gii_x: gii_x,
                // tmp_gii_x_ptr: gii_x_ptr,
                // tmp_offset_ptr: offset_ptr,
                // tmp_offset: offset,
                tmp_idx: idx,
            }
        }

        fn build_add(
            &mut self,
            io_state: &IOState,
            interface: &mut Vec<Variable>,
            descriptor_set: u32,
            input_bindings: [u32; 2],
            result_binding: u32,
        ) {
            let inputs = input_bindings.map(|binding| -> Variable {
                self.ir
                    .make_storage_buffer(StorageBuffer {
                        ty: io_state.t_input_ptr,
                        descriptor_set,
                        binding,
                        writable: false,
                    })
                    .into()
            });
            let output: Variable = self
                .ir
                .make_storage_buffer(StorageBuffer {
                    ty: io_state.t_output_ptr,
                    descriptor_set,
                    binding: result_binding,
                    writable: true,
                })
                .into();

            let [a, b] = inputs.map(|input| {
                let ptr_id = self
                    .fn_main
                    .append_value_node(ValueNode::access(
                        io_state.t_item_ptr,
                        StorageClass::StorageBuffer,
                        input.into(),
                        [self.const_u32_0.into(), self.tmp_idx.into()],
                    ))
                    .into();
                self.fn_main
                    .append_value_node(ValueNode::load(
                        io_state.t_item.into(),
                        ptr_id,
                    ))
                    .into()
            });

            let c = match io_state.t_item_scalar {
                ScalarTy::F32 => self
                    .fn_main
                    .append_value_node(ValueNode::fadd(io_state.t_item, a, b)),
                ScalarTy::U32 | ScalarTy::S32 => self
                    .fn_main
                    .append_value_node(ValueNode::iadd(io_state.t_item, a, b)),
            };
            let c_ptr = self.fn_main.append_value_node(ValueNode::access(
                io_state.t_item_ptr,
                StorageClass::StorageBuffer,
                output.into(),
                [self.const_u32_0.into(), self.tmp_idx.into()],
            ));

            self.fn_main
                .append_node(Node::Store(c_ptr.into(), c.into()));
            self.fn_main.append_node(Node::Return);

            interface.extend_from_slice(&inputs[..]);
            interface.push(output);
        }

        fn build_scaler(
            &mut self,
            io_state: &IOState,
            interface: &mut Vec<Variable>,
            descriptor_set: u32,
            input_binding: u32,
            result_binding: u32,
            offset: f32,
            scale: f32,
        ) {
            let const_offset = self
                .ir
                .make_constant(Constant::Scalar(ScalarConstant::F32(offset)))
                .into();
            let const_scale = self
                .ir
                .make_constant(Constant::Scalar(ScalarConstant::F32(scale)))
                .into();

            let sb_input: Variable = self
                .ir
                .make_storage_buffer(StorageBuffer {
                    ty: io_state.t_input_ptr,
                    descriptor_set,
                    binding: input_binding,
                    writable: false,
                })
                .into();

            let sb_output: Variable = self
                .ir
                .make_storage_buffer(StorageBuffer {
                    ty: io_state.t_output_ptr,
                    descriptor_set,
                    binding: result_binding,
                    writable: true,
                })
                .into();

            let tmp_x_ptr = self
                .fn_main
                .append_value_node(ValueNode::access(
                    io_state.t_item_ptr,
                    StorageClass::StorageBuffer,
                    sb_input.into(),
                    [self.const_u32_0.into(), self.tmp_idx.into()],
                ))
                .into();
            let tmp_x = self
                .fn_main
                .append_value_node(ValueNode::load(
                    io_state.t_item.into(),
                    tmp_x_ptr,
                ))
                .into();

            if !matches!(io_state.t_item_scalar, ScalarTy::F32) {
                todo!("type for Scaler layer: {:?}", io_state.t_item_scalar);
            }

            let tmp_y1 = self
                .fn_main
                .append_value_node(ValueNode::fadd(
                    io_state.t_item,
                    tmp_x,
                    const_offset,
                ))
                .into();
            let tmp_y = self
                .fn_main
                .append_value_node(ValueNode::fmul(
                    io_state.t_item,
                    tmp_y1,
                    const_scale,
                ))
                .into();

            let tmp_y_ptr = self
                .fn_main
                .append_value_node(ValueNode::access(
                    io_state.t_item_ptr,
                    StorageClass::StorageBuffer,
                    sb_output.into(),
                    [self.const_u32_0.into(), self.tmp_idx.into()],
                ))
                .into();
            self.fn_main.append_node(Node::Store(tmp_y_ptr, tmp_y));
            self.fn_main.append_node(Node::Return);

            interface.extend_from_slice(&[sb_input, sb_output]);
        }
    }

    impl Translate<Op<u32>, IR> for Builder {
        type Error = ();
        type Config = Config;

        fn translate(
            mut self,
            op: Op<u32>,
            config: &Self::Config,
        ) -> Result<IR, Self::Error> {
            let &Config {
                item_type: t_item_scalar,
                descriptor_set,
            } = config;
            let t_item = t_item_scalar.into();
            let t_item_ptr = self.ir.make_ptr_type(t_item);
            let t_item_rt_array = self
                .ir
                .make_composite_type(CompositeTy::Array1(t_item, None));
            let t_input_struct =
                self.ir.make_composite_type(CompositeTy::r#struct(
                    [StructMember {
                        ty: t_item_rt_array.into(),
                        writable: false,
                    }],
                    true,
                ));
            let t_output_struct =
                self.ir.make_composite_type(CompositeTy::r#struct(
                    [StructMember {
                        ty: t_item_rt_array.into(),
                        writable: true,
                    }],
                    true,
                ));
            let t_input_ptr = self.ir.make_ptr_type(t_input_struct.into());
            let t_output_ptr = self.ir.make_ptr_type(t_output_struct.into());
            let io_state = IOState {
                t_item_scalar,
                t_item,
                t_item_ptr,
                // t_item_rt_array,
                // t_input_struct,
                // t_output_struct,
                t_input_ptr,
                t_output_ptr,
            };

            let mut interface = vec![
                self.bi_global_invocation_id.into(),
                self.pc_config.into(),
            ];

            match op {
                Op::Add { inputs, result } => self.build_add(
                    &io_state,
                    &mut interface,
                    descriptor_set,
                    inputs,
                    result,
                ),
                Op::Scaler {
                    input,
                    result,
                    offset,
                    scale,
                } => self.build_scaler(
                    &io_state,
                    &mut interface,
                    descriptor_set,
                    input,
                    result,
                    offset,
                    scale,
                ),
            }

            let main_id = self.ir.make_function(self.fn_main);
            self.ir.entry_point(main_id, interface.into_boxed_slice());

            Ok(self.ir)
        }
    }
}