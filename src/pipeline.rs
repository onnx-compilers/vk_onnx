// use std::alloc::Layout;
use std::sync::Arc;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
// use vulkano::device::physical::PhysicalDevice;
use vulkano::device::{
    Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{
    AllocationCreateInfo, DeviceLayout, MemoryTypeFilter,
    StandardMemoryAllocator,
};
use vulkano::pipeline::PipelineBindPoint;
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::{
    IntoPipelineLayoutCreateInfoError, PipelineDescriptorSetLayoutCreateInfo,
};
use vulkano::pipeline::{
    ComputePipeline, Pipeline as VKPipeline, PipelineLayout,
    PipelineShaderStageCreateInfo,
};
use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo};
use vulkano::sync::GpuFuture;
use vulkano::{VulkanLibrary, sync};

#[derive(Debug)]
pub struct Pipeline {
    device: Arc<Device>,
    queue: Arc<Queue>,
    // allocator: Arc<StandardMemoryAllocator>,
    // pipeline: Arc<ComputePipeline>,
    pub buffers: Box<[Arc<Buffer>]>,
    command_buffer: Arc<PrimaryAutoCommandBuffer>,
}

#[derive(Debug)]
pub enum CreateError {
    EntryPointMainMissing,
    NoPhysicalDevice,
    DeviceQueueLacksCompute,
    LoadingError(vulkano::LoadingError),
    VulkanError(vulkano::VulkanError),
    VulkanValidationError(Box<vulkano::ValidationError>),
    PipelineLayoutError(IntoPipelineLayoutCreateInfoError),
    VulkanAllocateBuffer(vulkano::buffer::AllocateBufferError),
}

impl From<vulkano::LoadingError> for CreateError {
    fn from(err: vulkano::LoadingError) -> Self {
        CreateError::LoadingError(err)
    }
}

impl From<vulkano::VulkanError> for CreateError {
    fn from(err: vulkano::VulkanError) -> Self {
        CreateError::VulkanError(err)
    }
}

impl From<Box<vulkano::ValidationError>> for CreateError {
    fn from(err: Box<vulkano::ValidationError>) -> Self {
        CreateError::VulkanValidationError(err)
    }
}

impl From<vulkano::buffer::AllocateBufferError> for CreateError {
    fn from(err: vulkano::buffer::AllocateBufferError) -> Self {
        CreateError::VulkanAllocateBuffer(err)
    }
}

impl<T: Into<CreateError>> From<vulkano::Validated<T>> for CreateError {
    fn from(err: vulkano::Validated<T>) -> Self {
        match err {
            vulkano::Validated::Error(err) => err.into(),
            vulkano::Validated::ValidationError(err) => err.into(),
        }
    }
}

impl From<IntoPipelineLayoutCreateInfoError> for CreateError {
    fn from(err: IntoPipelineLayoutCreateInfoError) -> Self {
        CreateError::PipelineLayoutError(err)
    }
}

impl Pipeline {
    pub fn new(spirv: &[u32], len: usize) -> Result<Self, CreateError> {
        let library = VulkanLibrary::new()?;
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                ..Default::default()
            },
        )?;
        let physical_device = instance
            .enumerate_physical_devices()?
            .next()
            .ok_or_else(|| CreateError::NoPhysicalDevice)?;

        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(_queue_family_index, queue_family_properties)| {
                queue_family_properties
                    .queue_flags
                    .contains(QueueFlags::COMPUTE)
            })
            .ok_or_else(|| CreateError::DeviceQueueLacksCompute)?
            as u32;

        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )?;
        // NOTE: Maybe handle failure
        let queue = queues.next().unwrap();

        let allocator =
            Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let buffers = (0..3)
            .map(|_| {
                Buffer::new(
                    allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::STORAGE_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                        ..Default::default()
                    },
                    // TODO: don't unwrap
                    DeviceLayout::new_unsized::<[f32]>(len as u64).unwrap()
                )
            })
            .collect::<Result<Box<[_]>, _>>()?;

        let shader_module = unsafe {
            ShaderModule::new(
                device.clone(),
                ShaderModuleCreateInfo::new(spirv),
            )?
        };
        let entry_point = shader_module
            .entry_point("main")
            .ok_or_else(|| CreateError::EntryPointMainMissing)?;
        let stage = PipelineShaderStageCreateInfo::new(entry_point);
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone())?,
        )?;
        let pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )?;
        let descriptor_set_allocator =
            Arc::new(StandardDescriptorSetAllocator::new(
                device.clone(),
                Default::default(),
            ));
        let pipeline_layout = pipeline.layout();
        let descriptor_set_layouts = pipeline_layout.set_layouts();

        let descriptor_set_layout_index = 0;
        // TODO: Don't just unwrap
        let descriptor_set_layout = descriptor_set_layouts
            .get(descriptor_set_layout_index)
            .unwrap();
        let descriptor_set = DescriptorSet::new(
            descriptor_set_allocator.clone(),
            descriptor_set_layout.clone(),
            buffers.iter().enumerate().map(|(index, buffer)| {
                WriteDescriptorSet::buffer(index as u32, buffer.clone().into())
            }),
            [],
        )?;

        let command_buffer_allocator =
            Arc::new(StandardCommandBufferAllocator::new(
                device.clone(),
                Default::default(),
            ));
        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
        )?;
        let work_group_counts = [1, 1, 1];

        let _ = command_buffer_builder
            .bind_pipeline_compute(pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                descriptor_set_layout_index as u32,
                descriptor_set,
            )?;
        let _ = unsafe { command_buffer_builder.dispatch(work_group_counts)? };

        let command_buffer = command_buffer_builder.build()?;

        Ok(Pipeline {
            device,
            queue,
            // allocator,
            // pipeline,
            buffers,
            command_buffer,
        })
    }

    pub fn run(&self) {
        sync::now(self.device.clone())
            .then_execute(self.queue.clone(), self.command_buffer.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    }
}

#[cfg(test)]
mod tests {
    use rspirv::binary::Assemble;
    use vulkano::buffer::Subbuffer;

    use crate::shader::l0::Config;

    use super::*;

    #[test]
    fn test() {
        let module = crate::shader::l0::test_utils::make_module(Config {
            local_size: [2, 1, 1],
            version: Some((1, 3)),
            ..Default::default()
        });
        let spirv = module.assemble();
        let pipeline = Pipeline::new(spirv.as_slice(), 2).unwrap();
        {
            let [buf_a, buf_b] = [0, 1].map(|i| {
                Into::<Subbuffer<[u8]>>::into(pipeline.buffers[i].clone())
                    .reinterpret::<[f32]>()
            });
            let mut a = buf_a.write().unwrap();
            let mut b = buf_b.write().unwrap();
            a[0] = 12.5;
            a[1] = 0.0;
            b[0] = 5.4;
            b[1] = 1.0;
        }
        pipeline.run();
        {
            let buf_c =
                Into::<Subbuffer<[u8]>>::into(pipeline.buffers[2].clone())
                    .reinterpret::<[f32]>();
            let c = buf_c.read().unwrap();
            assert_eq!(dbg!(&c[..]), [17.9, 1.0]);
        }
    }
}