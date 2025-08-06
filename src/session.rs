use std::alloc::{Layout, LayoutError};
use std::ops::{Index, IndexMut};
use std::sync::Arc;

use vulkano::buffer::{
    BufferContents, BufferReadGuard, BufferUsage, BufferWriteGuard, Subbuffer,
};
// use vulkano::pipeline::ComputePipeline;
use vulkano::sync::HostAccessError;

use crate::context::{Context, MakeBufferError};
use crate::l_base::ScalarTy;

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
        let tensor  = TensorView::<f32>::try_from(&generic_tensor).unwrap();
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