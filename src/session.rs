use std::alloc::{Layout, LayoutError};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use replace_with::replace_with_or_abort;
use vulkano::buffer::{Buffer, BufferContents, BufferUsage};
use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use vulkano::device::DeviceOwned;
use vulkano::pipeline::PipelineShaderStageCreateInfo;
use vulkano::shader::EntryPoint;
use vulkano::sync::GpuFuture;
use vulkano::{Validated, VulkanError};

use crate::context::{Context, MakeBufferError};
use crate::l_base::{ScalarTy, Translate};
use crate::pipeline;

#[derive(Debug)]
pub struct Session {
    ctx: Arc<Context>,
    pub input_buffers: Vec<BufferId>,
    pub output_buffers: Vec<BufferId>,
    // NOTE: Maybe keep just the input and output buffers
    buffers: Vec<Arc<Buffer>>,
    instructions: Vec<Instruction>,
    stages: Vec<GPUStage>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferId(pub usize);

#[derive(Debug)]
struct GPUStage {
    // pipeline: Arc<ComputePipeline>,
    command_buffer: Arc<PrimaryAutoCommandBuffer>,
}

#[derive(Debug)]
pub enum Instruction {
    Invoke(StageId),
}

pub struct Run<'a> {
    session: &'a mut Session,
    future: Box<dyn GpuFuture>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageId(pub usize);

#[derive(Debug, Clone)]
pub enum MakeInputBufferError {
    MakeBuffer(MakeBufferError),
    Validation(Box<vulkano::ValidationError>),
    Vulkan(VulkanError),
}

impl From<MakeBufferError> for MakeInputBufferError {
    fn from(err: MakeBufferError) -> Self {
        MakeInputBufferError::MakeBuffer(err)
    }
}

impl From<VulkanError> for MakeInputBufferError {
    fn from(err: VulkanError) -> Self {
        MakeInputBufferError::Vulkan(err)
    }
}

impl From<Box<vulkano::ValidationError>> for MakeInputBufferError {
    fn from(err: Box<vulkano::ValidationError>) -> Self {
        MakeInputBufferError::Validation(err)
    }
}

impl<T: Into<MakeInputBufferError>> From<vulkano::Validated<T>>
    for MakeInputBufferError
{
    fn from(err: vulkano::Validated<T>) -> Self {
        match err {
            vulkano::Validated::Error(err) => err.into(),
            vulkano::Validated::ValidationError(err) => err.into(),
        }
    }
}

impl Session {
    pub fn new(ctx: Arc<Context>) -> Self {
        Self {
            ctx,
            input_buffers: Vec::new(),
            output_buffers: Vec::new(),
            buffers: Vec::new(),
            instructions: Vec::new(),
            stages: Vec::new(),
        }
    }

    pub fn get_buffer(&self, BufferId(i): BufferId) -> &Arc<Buffer> {
        &self.buffers[i]
    }

    pub fn make_run(&mut self) -> Run {
        Run::new(self, vulkano::sync::now(self.ctx.device.clone()).boxed())
    }

    pub fn make_storage_buffer(
        &mut self,
        layout: Layout,
        transfer: Option<bool>,
    ) -> Result<(Arc<Buffer>, BufferId), MakeBufferError> {
        let transfer_usage = match transfer {
            Some(true) => BufferUsage::TRANSFER_SRC,
            Some(false) => BufferUsage::TRANSFER_DST,
            None => BufferUsage::empty(),
        };
        let buf = self.ctx.make_buffer(
            BufferUsage::STORAGE_BUFFER | transfer_usage,
            true,
            layout,
        )?;
        let i = self.buffers.len();
        self.buffers.push(buf.clone());
        Ok((buf, BufferId(i)))
    }

    pub fn make_stage(
        &mut self,
        entry_point: EntryPoint,
        inputs: &[BufferId],
        outputs: &[BufferId],
        push_constants: impl BufferContents,
        // push_constants_offset: u32,
    ) -> StageId {
        let stage = PipelineShaderStageCreateInfo::new(entry_point);
        let layout = self.ctx.make_pipeline_layout([&stage]);
        let pipeline = self
            .ctx
            .make_compute_pipeline(stage, layout.clone())
            .unwrap();
        let descriptor_set = self
            .ctx
            .make_descriptor_set(
                layout.set_layouts()[0].clone(),
                inputs
                    .iter()
                    .map(|&BufferId(i)| self.buffers[i].clone())
                    .chain(
                        outputs
                            .iter()
                            .map(|&BufferId(i)| self.buffers[i].clone()),
                    ),
            )
            .unwrap();
        let command_buffer = self
            .ctx
            .make_compute_pipeline_command(
                pipeline.clone(),
                [descriptor_set].into_iter(),
                0,
                push_constants,
                0,
                [1, 1, 1],
            )
            .unwrap();
        let i = self.stages.len();
        self.stages.push(GPUStage {
            // pipeline,
            command_buffer,
        });
        StageId(i)
    }
}

struct X<F: GpuFuture> {
    next: F,
}

unsafe impl<F: GpuFuture> DeviceOwned for X<F> {
    fn device(&self) -> &Arc<vulkano::device::Device> {
        self.next.device()
    }
}

unsafe impl<F: GpuFuture> GpuFuture for X<F> {
    fn cleanup_finished(&mut self) {
        eprintln!("X cleanup_finished");
        self.next.cleanup_finished();
    }

    unsafe fn build_submission(
        &self,
    ) -> Result<vulkano::sync::future::SubmitAnyBuilder, Validated<VulkanError>>
    {
        eprintln!("X build_submission");
        unsafe { self.next.build_submission() }
    }

    fn flush(&self) -> Result<(), Validated<VulkanError>> {
        eprintln!("X flush");
        self.next.flush()
    }

    unsafe fn signal_finished(&self) {
        eprintln!("X signal_finished");
        unsafe {
            self.next.signal_finished();
        }
    }

    fn queue(&self) -> Option<Arc<vulkano::device::Queue>> {
        eprintln!("X queue");
        self.next.queue()
    }

    fn queue_change_allowed(&self) -> bool {
        eprintln!("X queue_change_allowed");
        self.next.queue_change_allowed()
    }

    fn check_buffer_access(
        &self,
        buffer: &Buffer,
        range: std::ops::Range<vulkano::DeviceSize>,
        exclusive: bool,
        queue: &vulkano::device::Queue,
    ) -> Result<(), vulkano::sync::future::AccessCheckError> {
        eprintln!("X check_buffer_access");
        self.next
            .check_buffer_access(buffer, range, exclusive, queue)
    }

    fn check_image_access(
        &self,
        image: &vulkano::image::Image,
        range: std::ops::Range<vulkano::DeviceSize>,
        exclusive: bool,
        expected_layout: vulkano::image::ImageLayout,
        queue: &vulkano::device::Queue,
    ) -> Result<(), vulkano::sync::future::AccessCheckError> {
        eprintln!("X check_image_access");
        self.next.check_image_access(
            image,
            range,
            exclusive,
            expected_layout,
            queue,
        )
    }

    fn check_swapchain_image_acquired(
        &self,
        swapchain: &vulkano::swapchain::Swapchain,
        image_index: u32,
        before: bool,
    ) -> Result<(), vulkano::sync::future::AccessCheckError> {
        eprintln!("X check_swapchain_image_acquired");
        self.next
            .check_swapchain_image_acquired(swapchain, image_index, before)
    }
}

impl<'a> Run<'a> {
    pub fn new(
        session: &'a mut Session,
        future: impl Into<Box<dyn GpuFuture>>,
    ) -> Self {
        Self {
            session,
            future: Box::new(X {
                next: future.into(),
            }),
        }
    }

    pub fn run(&mut self) {
        for instr in self.session.instructions.iter() {
            match instr {
                &Instruction::Invoke(StageId(i)) => {
                    eprintln!("INVOKING STAGE {}", i);
                    replace_with_or_abort(&mut self.future, |future| {
                        Box::new(
                            future
                                .then_execute(
                                    self.session.ctx.queue.clone(),
                                    self.session.stages[i]
                                        .command_buffer
                                        .clone(),
                                )
                                .unwrap(),
                        )
                    });
                }
            }
        }
        replace_with_or_abort(&mut self.future, |future| {
            Box::new(X { next: future })
        });
    }

    pub fn wait(
        self,
        timeout: Option<Duration>,
    ) -> Result<(), vulkano::Validated<VulkanError>> {
        self.future.then_signal_fence_and_flush()?.wait(timeout)
    }
}

// fn linearize_index(shape: &[u32], index: &[usize]) -> usize {
//     let (linear_index, _) = shape
//         .iter()
//         .chain(std::iter::once(&1))
//         .rev()
//         .skip(1)
//         .zip(index.iter().rev())
//         .fold((0, 1), |(acc, x), (&dim, &i)| {
//             let acc = acc + i * x;
//             let x = x * dim as usize;
//             (acc, x)
//         });
//     linear_index
// }

#[derive(Debug, Clone)]
pub enum TranslationError {
    Layout(LayoutError),
    MakeBuffer(MakeBufferError),
    MakeInputBuffer(MakeInputBufferError),
    ShaderFromSpirV(Validated<vulkano::VulkanError>),
}

impl From<LayoutError> for TranslationError {
    fn from(err: LayoutError) -> Self {
        TranslationError::Layout(err)
    }
}

impl From<MakeBufferError> for TranslationError {
    fn from(err: MakeBufferError) -> Self {
        TranslationError::MakeBuffer(err)
    }
}

impl From<MakeInputBufferError> for TranslationError {
    fn from(err: MakeInputBufferError) -> Self {
        TranslationError::MakeInputBuffer(err)
    }
}

#[derive(Debug)]
pub struct SessionBuilder {
    session: Session,
    outputs: HashSet<pipeline::OperandBufferId>,
}

impl SessionBuilder {
    pub fn new(ctx: Arc<Context>) -> Self {
        Self {
            session: Session::new(ctx),
            outputs: HashSet::new(),
        }
    }

    fn op_add_elementwise_notinplace(
        &mut self,
        ir: &pipeline::IR,
        lhs: BufferId,
        rhs: BufferId,
        result: BufferId,
        ty: ScalarTy,
        dims_range: (usize, usize),
    ) -> Result<BufferId, <Self as Translate<pipeline::IR, Session>>::Error>
    {
        use crate::kernel::componentwise::{
            self, Bindings, Config, KernelConfig, Op,
        };
        use crate::shader::l0::{Config as ShaderConfig, SPVModuleBuilder};
        use rspirv::binary::{Assemble, Disassemble};

        let shader_ir = componentwise::new(Config {
            item_type: ty,
            descriptor_set: 0,
            bindings: Bindings {
                inputs: [0, 1],
                output: 2,
            },
            op: Op::Add,
        });
        let rspirv_ir = SPVModuleBuilder::default()
            .translate(
                shader_ir,
                &ShaderConfig {
                    local_size: [
                        ir.get_shape(dims_range).iter().product::<usize>()
                            as u32,
                        1,
                        1,
                    ],
                    version: Some((1, 3)),
                },
            )
            .unwrap();
        eprintln!("{}", rspirv_ir.disassemble());
        let spirv = rspirv_ir.assemble();
        let shader =
            unsafe { self.session.ctx.shader_from_spirv(spirv.as_slice()) }
                .map_err(TranslationError::ShaderFromSpirV)?;
        let entry_point = shader.entry_point("main").unwrap();
        let stage = self.session.make_stage(
            entry_point,
            &[lhs, rhs],
            &[result],
            KernelConfig { offset: 0 },
        );
        self.session.instructions.push(Instruction::Invoke(stage));
        Ok(result)
    }
}

impl Translate<pipeline::IR, Session> for SessionBuilder {
    type Error = TranslationError;
    type Config = ();

    fn translate(
        mut self,
        ir: pipeline::IR,
        _config: &Self::Config,
    ) -> Result<Session, Self::Error> {
        use pipeline::Instr;

        self.session.input_buffers.reserve(ir.inputs.len());
        self.session.output_buffers.reserve(ir.outputs.len());
        self.session
            .buffers
            .reserve(ir.inputs.len() + ir.outputs.len());

        self.outputs.reserve(ir.outputs.len());
        for &output in ir.outputs.iter() {
            self.outputs.insert(output);
        }

        for &pipeline::Buffer { ty, dims_range } in ir.operand_buffers.iter() {
            let element_count: usize =
                ir.dims_pool[dims_range.0..dims_range.1].iter().product();
            let size = element_count * size_of(ty);
            let _buffer = self
                .session
                .make_storage_buffer(Layout::array::<u8>(size)?, None)?;
        }

        for &pipeline::OperandBufferId(i) in ir.inputs.iter() {
            self.session.input_buffers.push(BufferId(i));
        }

        for instr in ir.instructions.iter() {
            match instr {
                &Instr::Op(pipeline::OpId(i)) => {
                    // TODO: Put this in a function or smth
                    let outputs = match &ir.operations[i] {
                        &pipeline::Op::Binary(pipeline::BinOp {
                            kind: pipeline::BinOpKind::AddElementwise,
                            operands:
                                pipeline::BinOperands::NotInplace {
                                    lhs: pipeline::BufferId::Operand(lhs),
                                    rhs: pipeline::BufferId::Operand(rhs),
                                    result,
                                },
                        }) => {
                            let pipeline::OperandBufferId(lhs_i) = lhs;
                            let pipeline::OperandBufferId(rhs_i) = rhs;
                            let pipeline::OperandBufferId(result_i) = result;
                            let lhs_id = BufferId(lhs_i);
                            let rhs_id = BufferId(rhs_i);
                            let result_id = BufferId(result_i);
                            let &pipeline::Buffer { ty, dims_range } =
                                ir.get_operand(lhs);
                            let output = self.op_add_elementwise_notinplace(
                                &ir, lhs_id, rhs_id, result_id, ty, dims_range,
                            )?;
                            [(result, output)].into_iter()
                        }
                        op => todo!("unimplemented operation: {:#?}", op),
                    };
                    for (output, buffer_id) in outputs {
                        if self.outputs.contains(&output) {
                            self.session.output_buffers.push(buffer_id);
                        }
                    }
                } // _ => todo!("unimplemented instruction: {:#?}", instr)
            }
        }

        Ok(self.session)
    }
}

fn size_of(_ty: ScalarTy) -> usize {
    // XXX: Hack, must calculate/lookup sizes for respective element types
    4
}

#[cfg(test)]
mod tests {
    use protobuf::Message;
    use vulkano::buffer::Subbuffer;

    use crate::context::Config;
    use crate::protos::onnx::ModelProto;
    use crate::{l0, l1};

    use super::*;

    #[test]
    fn test() {
        let ctx = Arc::new(
            Context::new(Config {
                name: Some("test".into()),
                ..Default::default()
            })
            .unwrap(),
        );

        let model_path = l0::test_utils::project_path()
            .join("test_models")
            .join("simple_add.onnx");
        let bytes = std::fs::read(model_path).unwrap();
        let model = ModelProto::parse_from_bytes(bytes.as_slice()).unwrap();

        let mut session = SessionBuilder::new(ctx)
            .translate(
                pipeline::IRBuilder::default()
                    .translate(
                        l1::IRBuilder::default().translate(
                            l0::IRBuilder::default()
                                .translate(model, &())
                                .unwrap(),
                            &(),
                        ).unwrap(),
                        &(),
                    )
                    .unwrap(),
                &(),
            )
            .unwrap();

        eprintln!("INPUTS: {:?}", session.input_buffers);
        eprintln!("OUTPUTS: {:?}", session.output_buffers);

        let a = Into::<Subbuffer<_>>::into(
            session.get_buffer(session.input_buffers[0]).clone(),
        )
        .reinterpret::<[f32]>();
        let b = Into::<Subbuffer<_>>::into(
            session.get_buffer(session.input_buffers[1]).clone(),
        )
        .reinterpret::<[f32]>();

        {
            let mut a_access = a.write().unwrap();
            let mut b_access = b.write().unwrap();
            for i in 0..a_access.len() {
                a_access[i] = (i + 1) as f32;
                b_access[i] = (a_access.len() - i) as f32;
            }
            eprintln!("{:?} {:?}", &a_access[..], &b_access[..]);
        }

        let mut run = session.make_run();
        run.run();
        let () = run.wait(None).unwrap();

        eprintln!("INPUTS: {:?}", session.input_buffers);
        eprintln!("OUTPUTS: {:?}", session.output_buffers);

        for (i, buffer) in session.buffers.iter().enumerate() {
            let subbuffer = Into::<Subbuffer<_>>::into(buffer.clone())
                .reinterpret::<[f32]>();
            let access = subbuffer.read().unwrap();
            eprintln!("{:?}: {:?}", BufferId(i), &access[..]);
        }
    }

    // #[test]
    // fn test_linearize_index() {
    //     assert_eq!(linearize_index(&[2, 2, 3], &[0, 0, 0]), 0);
    //     assert_eq!(linearize_index(&[2, 2, 3], &[0, 1, 0]), 3);
    //     assert_eq!(linearize_index(&[2, 2, 3], &[0, 1, 1]), 4);
    //     assert_eq!(linearize_index(&[4, 2, 3], &[3, 1, 1]), 22);
    // }
}
