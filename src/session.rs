use std::alloc::{Layout, LayoutError};
use std::ops::{Index, IndexMut};
use std::sync::Arc;
use std::time::Duration;

use vulkano::buffer::{
    Buffer, BufferContents, BufferReadGuard, BufferUsage, BufferWriteGuard,
    Subbuffer,
};
use vulkano::command_buffer::PrimaryAutoCommandBuffer;
// use vulkano::pipeline::ComputePipeline;
use vulkano::VulkanError;
use vulkano::sync::{GpuFuture, HostAccessError};

use crate::context::{Context, MakeBufferError};
use crate::l_base::ScalarTy;

#[derive(Debug)]
pub struct Session {
    ctx: Arc<Context>,
    pub input_buffers: Vec<(Arc<Buffer>, usize)>,
    buffers: Vec<Arc<Buffer>>,
    input_copy_commands: Vec<Arc<PrimaryAutoCommandBuffer>>,
    // pipelines: Vec<Arc<ComputePipeline>>,
}

pub struct Run<'a> {
    session: &'a mut Session,
    future: Box<dyn GpuFuture>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipelineId(pub usize);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDescriptor {
    pub shape: Box<[u32]>,
    pub ty: ScalarTy,
    pub batch_dim: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenericTensor {
    pub shape: Box<[u32]>,
    pub ty: ScalarTy,
    pub batch_dim: Option<u32>,
    buffer: Arc<vulkano::buffer::Buffer>,
}

#[derive(Debug)]
pub struct TensorView<'a, T: BufferContents> {
    pub shape: &'a [u32],
    buffer: Subbuffer<[T]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorId(pub usize);

#[derive(Debug)]
pub enum MakeTensorError {
    LayoutError(LayoutError),
    MakeBufferError(MakeBufferError),
}

#[derive(Debug)]
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

pub struct TensorWriteGuard<'a, T> {
    buffer: BufferWriteGuard<'a, [T]>,
    shape: &'a [u32],
}

pub struct TensorReadGuard<'a, T> {
    buffer: BufferReadGuard<'a, [T]>,
    shape: &'a [u32],
}

impl Session {
    pub fn new(ctx: Arc<Context>) -> Self {
        Self {
            ctx,
            input_buffers: Vec::new(),
            buffers: Vec::new(),
            input_copy_commands: Vec::new(),
            // pipelines: Vec::new(),
        }
    }

    pub fn make_run(&mut self) -> Run {
        Run::new(self, vulkano::sync::now(self.ctx.device.clone()).boxed())
    }

    pub fn make_input_buffer(
        &mut self,
        layout: Layout,
    ) -> Result<Arc<Buffer>, MakeInputBufferError> {
        let src = self.ctx.make_buffer(
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
            false,
            layout,
        )?;
        let dst = self.ctx.make_buffer(
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
            true,
            layout,
        )?;
        let i = self.buffers.len();
        self.buffers.push(dst.clone());
        self.input_buffers.push((src.clone(), i));
        let cmd = self.ctx.make_buffer_copy_command(
            Into::<Subbuffer<[u8]>>::into(src.clone()),
            Into::<Subbuffer<[u8]>>::into(dst.clone()),
        )?;
        self.input_copy_commands.push(cmd);
        Ok(src)
    }

    pub fn make_tensor(
        &self,
        TensorDescriptor {
            shape,
            ty,
            batch_dim,
        }: TensorDescriptor,
    ) -> Result<GenericTensor, MakeTensorError> {
        // NOTE: align dimensions, batch_dim is the maximum
        let total_count: u32 = shape.iter().product();
        let total_count = batch_dim
            .map(|dim| dim * total_count)
            .unwrap_or(total_count);
        // XXX: This is a hack, need type sizes
        let size = 4 * total_count;
        let layout = Layout::array::<u8>(size as usize)?;
        Ok(GenericTensor {
            shape,
            ty,
            batch_dim,
            buffer: self.ctx.make_buffer(
                BufferUsage::STORAGE_BUFFER,
                true,
                layout,
            )?,
        })
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

    pub fn transfer_outputs(&mut self) {
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

impl<'a> TryFrom<&'a GenericTensor> for TensorView<'a, f32> {
    type Error = ();
    fn try_from(value: &'a GenericTensor) -> Result<Self, Self::Error> {
        Ok(TensorView {
            shape: &value.shape,
            buffer: Into::<Subbuffer<[u8]>>::into(value.buffer.clone())
                .reinterpret::<[f32]>(),
        })
    }
}

impl<'a, T: BufferContents> TensorView<'a, T> {
    // TODO: Make custom type that wraps the read/write guards and does indexing according to the shape. also can do slicing
    pub fn write(&'a self) -> Result<TensorWriteGuard<'a, T>, HostAccessError> {
        Ok(TensorWriteGuard {
            buffer: self.buffer.write()?,
            shape: &self.shape,
        })
    }

    pub fn read(&'a self) -> Result<TensorReadGuard<'a, T>, HostAccessError> {
        Ok(TensorReadGuard {
            buffer: self.buffer.read()?,
            shape: &self.shape,
        })
    }
}

impl<'a, T: BufferContents + Copy> TensorWriteGuard<'a, T> {
    pub fn copy_from_slice(&mut self, slice: &[T]) {
        self.buffer.copy_from_slice(slice);
    }
}

impl<'a, T: BufferContents> Index<&[usize]> for TensorWriteGuard<'a, T> {
    type Output = T;
    fn index(&self, index: &[usize]) -> &Self::Output {
        &self.buffer[linearize_index(self.shape, index)]
    }
}

impl<'a, T: BufferContents> IndexMut<&[usize]> for TensorWriteGuard<'a, T> {
    fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
        &mut self.buffer[linearize_index(self.shape, index)]
    }
}

impl<'a, T: BufferContents> Index<&[usize]> for TensorReadGuard<'a, T> {
    type Output = T;
    fn index(&self, index: &[usize]) -> &Self::Output {
        &self.buffer[linearize_index(self.shape, index)]
    }
}

fn linearize_index(shape: &[u32], index: &[usize]) -> usize {
    let (linear_index, _) = shape
        .iter()
        .chain(std::iter::once(&1))
        .rev()
        .skip(1)
        .zip(index.iter().rev())
        .fold((0, 1), |(acc, x), (&dim, &i)| {
            let acc = acc + i * x;
            let x = x * dim as usize;
            (acc, x)
        });
    linear_index
}

impl From<LayoutError> for MakeTensorError {
    fn from(err: LayoutError) -> Self {
        MakeTensorError::LayoutError(err)
    }
}

impl From<MakeBufferError> for MakeTensorError {
    fn from(err: MakeBufferError) -> Self {
        MakeTensorError::MakeBufferError(err)
    }
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
        let generic_tensor = session
            .make_tensor(TensorDescriptor {
                shape: Box::new([1, 2, 3]),
                ty: ScalarTy::F32,
                batch_dim: None,
            })
            .unwrap();
        let tensor = TensorView::<f32>::try_from(&generic_tensor).unwrap();
        eprintln!("{:?}", tensor);
        {
            let mut x = tensor.write().unwrap();
            x[&[0, 0, 0]] = 1.35;
        }
        {
            let x = tensor.read().unwrap();
            assert_eq!(x[&[0, 0, 0]], 1.35);
        }
    }

    #[test]
    fn test_linearize_index() {
        assert_eq!(linearize_index(&[2, 2, 3], &[0, 0, 0]), 0);
        assert_eq!(linearize_index(&[2, 2, 3], &[0, 1, 0]), 3);
        assert_eq!(linearize_index(&[2, 2, 3], &[0, 1, 1]), 4);
        assert_eq!(linearize_index(&[4, 2, 3], &[3, 1, 1]), 22);
    }
}
