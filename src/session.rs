use std::{
    alloc::{Layout, LayoutError},
    sync::Arc,
};

use vulkano::{
    buffer::{
        BufferContents, BufferReadGuard, BufferUsage, BufferWriteGuard,
        Subbuffer,
    },
    // pipeline::ComputePipeline,
    sync::HostAccessError,
};

use crate::{
    context::{Context, MakeBufferError},
    l_base::ScalarTy,
};

#[derive(Debug)]
pub struct Session {
    ctx: Arc<Context>,
    // pipelines: Vec<Arc<ComputePipeline>>,
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

impl Session {
    pub fn new(ctx: Arc<Context>) -> Self {
        Self {
            ctx,
            // pipelines: Vec::new(),
        }
    }

    pub fn make_tensor(
        &self,
        TensorDescriptor {
            shape,
            ty,
            batch_dim,
        }: TensorDescriptor,
    ) -> Result<GenericTensor, MakeTensorError> {
        let total_count: u32 = shape.iter().product();
        let total_count = batch_dim
            .map(|dim| dim * total_count)
            .unwrap_or(total_count);
        // XXX: This is a hack
        let size = 4 * total_count;
        let layout = Layout::array::<u8>(size as usize)?;
        Ok(GenericTensor {
            shape,
            ty,
            buffer: self.ctx.make_buffer(
                BufferUsage::STORAGE_BUFFER,
                true,
                layout,
            )?,
        })
    }
}

impl GenericTensor {
    pub fn cast<'a, T: BufferContents>(&'a self) -> Option<TensorView<'a, T>> {
        Some(TensorView {
            shape: &self.shape,
            buffer: Into::<Subbuffer<[u8]>>::into(self.buffer.clone())
                .reinterpret::<[T]>(),
        })
    }
}

impl<'a, T: BufferContents> TensorView<'a, T> {
    // TODO: Make custom type that wraps the read/write guards and does indexing according to the shape. also can do slicing
    pub fn write(
        &'a self,
    ) -> Result<BufferWriteGuard<'a, [T]>, HostAccessError> {
        self.buffer.write()
    }

    pub fn read(&'a self) -> Result<BufferReadGuard<'a, [T]>, HostAccessError> {
        self.buffer.read()
    }
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
        let tensor = generic_tensor.cast::<f32>().unwrap();
        eprintln!("{:?}", tensor);
        {
            let mut x = tensor.write().unwrap();
            x[0] = 1.35;
        }
        {
            let x = tensor.read().unwrap();
            assert_eq!(x[0], 1.35);
        }
    }
}
