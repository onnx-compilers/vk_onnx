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
        l_base::ScalarTy,
        shader::l0::{
            CompositeTy, Constant, Function, Node, PushConstant,
            ScalarConstant, StorageBuffer, StructMember, Ty, ValueNode,
            Variable,
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
    pub enum Op {
        Add,
        // Sub,
        // Mul
    }

    #[derive(Debug, Clone)]
    pub struct Bindings {
        pub inputs: [u32; 2],
        pub output: u32,
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
        pub bindings: Bindings,
        pub op: Op,
    }

    impl Default for Config {
        fn default() -> Self {
            Self {
                item_type: ScalarTy::F32,
                descriptor_set: 0,
                bindings: Bindings {
                    inputs: [0, 1],
                    output: 2,
                },
                op: Op::Add,
            }
        }
    }

    pub fn new(config: Config) -> IR {
        let Config {
            item_type: scalar_t_item,
            descriptor_set,
            bindings:
                Bindings {
                    inputs: input_bindings,
                    output: output_binding,
                },
            op,
        } = config;
        let mut ir = IR::new();

        let t_item = scalar_t_item.into();
        let t_u32 = ScalarTy::U32.into();
        let t_u32_ptr = ir.make_ptr_type(t_u32);
        let t_u32_vec3 = Ty::Vec(ScalarTy::U32, 3);
        let t_u32_vec3_ptr = ir.make_ptr_type(t_u32_vec3);
        let t_item_ptr = ir.make_ptr_type(t_item);

        let t_item_rt_array = ir
            .make_composite_type(CompositeTy::Array1(t_item, None))
            .into();
        let t_input_struct = ir
            .make_composite_type(CompositeTy::r#struct(
                [StructMember {
                    ty: t_item_rt_array,
                    writable: false,
                }],
                true,
            ))
            .into();
        let t_output_struct = ir
            .make_composite_type(CompositeTy::r#struct(
                [StructMember {
                    ty: t_item_rt_array,
                    writable: true,
                }],
                true,
            ))
            .into();
        let t_config_struct = ir
            .make_composite_type(CompositeTy::r#struct(
                [StructMember {
                    ty: t_u32,
                    writable: false,
                }],
                true,
            ))
            .into();
        let t_input_ptr = ir.make_ptr_type(t_input_struct);
        let t_output_ptr = ir.make_ptr_type(t_output_struct);
        let t_config_ptr = ir.make_ptr_type(t_config_struct);

        let const_u32_0 = ir
            .make_constant(Constant::Scalar(ScalarConstant::U32(0)))
            .into();

        let global_invocation_id =
            ir.make_builtin(t_u32_vec3_ptr, BuiltIn::GlobalInvocationId);

        let inputs = input_bindings.map(|binding| -> Variable {
            ir.make_storage_buffer(StorageBuffer {
                ty: t_input_ptr,
                descriptor_set,
                binding,
                writable: false,
            })
            .into()
        });
        let output: Variable = ir
            .make_storage_buffer(StorageBuffer {
                ty: t_output_ptr,
                descriptor_set,
                binding: output_binding,
                writable: true,
            })
            .into();
        let config: Variable = ir
            .make_push_constant(PushConstant { ty: t_config_ptr })
            .into();

        let mut main = Function::new(Some("main".into()), None);

        let gii_x_ptr = main
            .append_value_node(ValueNode::access(
                t_u32_ptr,
                StorageClass::Input,
                global_invocation_id.into(),
                [const_u32_0],
            ))
            .into();
        let gii_x = main
            .append_value_node(ValueNode::load(t_u32, gii_x_ptr))
            .into();

        let offset_ptr = main
            .append_value_node(ValueNode::access(
                t_u32_ptr,
                StorageClass::Uniform,
                config.into(),
                [const_u32_0],
            ))
            .into();
        let offset = main
            .append_value_node(ValueNode::load(t_u32, offset_ptr))
            .into();
        let idx = main
            .append_value_node(ValueNode::iadd(t_u32, gii_x, offset))
            .into();

        let [a, b] = inputs.map(|input| {
            let ptr_id = main
                .append_value_node(ValueNode::access(
                    t_item_ptr,
                    StorageClass::StorageBuffer,
                    input.into(),
                    [const_u32_0, idx],
                ))
                .into();
            main.append_value_node(ValueNode::load(t_item, ptr_id))
                .into()
        });

        let c = match (scalar_t_item, op) {
            (ScalarTy::F32, Op::Add) => {
                main.append_value_node(ValueNode::fadd(t_item, a, b))
            }
            (ScalarTy::U32 | ScalarTy::S32, Op::Add) => {
                main.append_value_node(ValueNode::iadd(t_item, a, b))
            }
        };
        let c_ptr = main.append_value_node(ValueNode::access(
            t_item_ptr,
            StorageClass::StorageBuffer,
            output.into(),
            [const_u32_0, idx],
        ));

        main.append_node(Node::Store(c_ptr.into(), c.into()));
        main.append_node(Node::Return);
        let main_id = ir.make_function(main);
        ir.entry_point(
            main_id,
            [
                global_invocation_id.into(),
                inputs[0],
                inputs[1],
                output,
                config,
            ],
        );

        ir
    }
}
