use std::alloc::{Layout, LayoutError};
// use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use vulkano::buffer::{Buffer, BufferUsage, Subbuffer};
use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use vulkano::pipeline::{ComputePipeline, PipelineShaderStageCreateInfo};
use vulkano::shader::EntryPoint;
// use vulkano::pipeline::ComputePipeline;
use vulkano::VulkanError;
use vulkano::sync::GpuFuture;

use crate::context::{Context, MakeBufferError};
use crate::l_base::{ScalarTy, Translate};
use crate::pipeline;

#[derive(Debug)]
pub struct Session {
    ctx: Arc<Context>,
    pub input_buffers: Vec<InputBuffer>,
    pub output_buffers: Vec<OutputBuffer>,
    buffers: Vec<Arc<Buffer>>,
    stages: Vec<GPUStage>,
    stage_inputs_pool: Vec<BufferId>,
    stage_outputs_pool: Vec<BufferId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferId(pub usize);

#[derive(Debug)]
pub struct InputBuffer {
    pub src: Arc<Buffer>,
    pub dst: BufferId,
    pub copy_command: Arc<PrimaryAutoCommandBuffer>,
}

#[derive(Debug)]
pub struct OutputBuffer {
    pub src: BufferId,
    pub dst: Arc<Buffer>,
    pub copy_command: Arc<PrimaryAutoCommandBuffer>,
}

#[derive(Debug)]
struct GPUStage {
    pipeline: Arc<ComputePipeline>,
    inputs_range: (usize, usize),
    outputs_range: (usize, usize),
}

pub struct Run<'a> {
    session: &'a mut Session,
    future: Box<dyn GpuFuture>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipelineId(pub usize);

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
            stages: Vec::new(),
            stage_inputs_pool: Vec::new(),
            stage_outputs_pool: Vec::new(),
        }
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

    pub fn make_input_buffer(
        &mut self,
        layout: Layout,
    ) -> Result<(Arc<Buffer>, BufferId), MakeInputBufferError> {
        let (src, src_id) = self.make_storage_buffer(layout, Some(true))?;
        let (dst, dst_id) = self.make_storage_buffer(layout, Some(false))?;
        let cmd = self.ctx.make_buffer_copy_command(
            Into::<Subbuffer<[u8]>>::into(src.clone()),
            Into::<Subbuffer<[u8]>>::into(dst.clone()),
        )?;
        self.buffers.push(dst.clone());
        self.input_buffers.push(InputBuffer {
            src: src.clone(),
            dst: dst_id,
            copy_command: cmd,
        });
        Ok((src, src_id))
    }

    pub fn make_stage(
        &mut self,
        entry_point: EntryPoint,
        inputs: &[BufferId],
        outputs: &[BufferId],
    ) {
        let stage = PipelineShaderStageCreateInfo::new(entry_point);
        let layout = self.ctx.make_pipeline_layout([&stage]);
        let pipeline = self
            .ctx
            .make_compute_pipeline(stage, layout.clone())
            .unwrap();
        let inputs_range = self.make_inputs_range(inputs);
        let outputs_range = self.make_outputs_range(outputs);
        self.stages.push(GPUStage {
            pipeline,
            inputs_range,
            outputs_range,
        })
    }

    fn make_inputs_range(&mut self, inputs: &[BufferId]) -> (usize, usize) {
        let start = self.stage_inputs_pool.len();
        let end = start + inputs.len();
        self.stage_inputs_pool.extend_from_slice(inputs);
        (start, end)
    }

    fn make_outputs_range(&mut self, outputs: &[BufferId]) -> (usize, usize) {
        let start = self.stage_outputs_pool.len();
        let end = start + outputs.len();
        self.stage_outputs_pool.extend_from_slice(outputs);
        (start, end)
    }
}

impl<'a> Run<'a> {
    pub fn new(
        session: &'a mut Session,
        future: impl Into<Box<dyn GpuFuture>>,
    ) -> Self {
        Self {
            session,
            future: future.into(),
        }
    }

    pub fn transfer_inputs(&mut self, _diff: impl Iterator<Item = bool>) {
        todo!(
            "dispatch commands for copying to the buffer according to the diff"
        )
    }

    pub fn run(&mut self) {
        todo!("dispatch commands to run the vulkan pipelines")
    }

    pub fn transfer_outputs(&mut self, _diff: impl Iterator<Item = bool>) {
        todo!(
            "dispatch commands to transfer the data from the device buffers to the host"
        )
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
}

impl SessionBuilder {
    pub fn new(ctx: Arc<Context>) -> Self {
        Self {
            session: Session::new(ctx),
        }
    }

    fn make_buffer(
        &mut self,
        ir: &pipeline::IR,
        buffer: &pipeline::Buffer,
    ) -> Result<
        (Arc<Buffer>, BufferId),
        <Self as Translate<pipeline::IR, Session>>::Error,
    > {
        Ok(self.session.make_storage_buffer(
            Layout::array::<u8>(
                ir.get_shape(buffer.dims_range).iter().product::<usize>()
                    * size_of(buffer.ty),
            )?,
            None,
        )?)
    }

    fn make_input_buffer(
        &mut self,
        ir: &pipeline::IR,
        buffer: &pipeline::Buffer,
    ) -> Result<
        (Arc<Buffer>, BufferId),
        <Self as Translate<pipeline::IR, Session>>::Error,
    > {
        Ok(self.session.make_input_buffer(Layout::array::<u8>(
            ir.get_shape(buffer.dims_range).iter().product::<usize>()
                * size_of(buffer.ty),
        )?)?)
    }

    fn op_add_elementwise_notinplace(
        &mut self,
        ir: &pipeline::IR,
        lhs: &pipeline::Buffer,
        rhs: &pipeline::Buffer,
        result: &pipeline::Buffer,
    ) -> Result<(), <Self as Translate<pipeline::IR, Session>>::Error> {
        let (lhs_buf, lhs_id) = self.make_input_buffer(ir, lhs)?;
        let (rhs_buf, rhs_id) = self.make_input_buffer(ir, rhs)?;
        let (result_buf, result_id) = self.make_buffer(ir, result)?;
        // TODO: Continue here
        Ok(())
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

        // let mut input_buffer_set = HashSet::new();
        for &input in ir.inputs.iter() {
            let &pipeline::Buffer { ty, dims_range } = ir.get_operand(input);
            let element_count: usize =
                ir.dims_pool[dims_range.0..dims_range.1].iter().product();
            let size = element_count * size_of(ty);
            let _buffer =
                self.session.make_input_buffer(Layout::array::<u8>(size)?)?;
            // input_buffer_set.insert(input);
        }

        for instr in ir.instructions.iter() {
            match instr {
                &Instr::Op(pipeline::OpId(i)) => {
                    // TODO: Put this in a function or smth
                    match &ir.operations[i] {
                        &pipeline::Op::Binary(pipeline::BinOp {
                            kind: pipeline::BinOpKind::AddElementwise,
                            operands:
                                pipeline::BinOperands::NotInplace {
                                    lhs: pipeline::BufferId::Operand(lhs),
                                    rhs: pipeline::BufferId::Operand(rhs),
                                    result,
                                },
                        }) => {
                            let lhs = ir.get_operand(lhs);
                            let rhs = ir.get_operand(rhs);
                            let result = ir.get_operand(result);
                            self.op_add_elementwise_notinplace(
                                &ir, lhs, rhs, result,
                            )?;
                        }
                        op => todo!("unimplemented operation: {:#?}", op),
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
    use crate::context::Config;

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
        let session = Session::new(ctx);
    }

    // #[test]
    // fn test_linearize_index() {
    //     assert_eq!(linearize_index(&[2, 2, 3], &[0, 0, 0]), 0);
    //     assert_eq!(linearize_index(&[2, 2, 3], &[0, 1, 0]), 3);
    //     assert_eq!(linearize_index(&[2, 2, 3], &[0, 1, 1]), 4);
    //     assert_eq!(linearize_index(&[4, 2, 3], &[3, 1, 1]), 22);
    // }
}