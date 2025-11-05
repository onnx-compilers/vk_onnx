use std::alloc::Layout;
use std::sync::Arc;

use vulkano::buffer::{
    Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer,
};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo,
    PrimaryAutoCommandBuffer,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::layout::DescriptorSetLayout;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::PhysicalDevice;
// use vulkano::device::physical::PhysicalDevice;
use vulkano::device::{
    Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{
    AllocationCreateInfo, DeviceLayout, MemoryTypeFilter,
    StandardMemoryAllocator,
};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout:: PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};
use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo};
use vulkano::{Validated, Version, VulkanError, VulkanLibrary};

#[derive(Debug)]
pub struct Context {
    pub library: Arc<VulkanLibrary>,
    pub instance: Arc<Instance>,
    pub physical_device: Arc<PhysicalDevice>,
    pub device: Arc<Device>,
    // TODO: Maybe use multiple queues and have parallel pipeline execution
    // could consider the queues like registers in SSA compilers.
    pub queue: Arc<Queue>,
    pub alloc: Arc<StandardMemoryAllocator>,
    pub descriptor_set_alloc: Arc<StandardDescriptorSetAllocator>,
    pub command_buffer_alloc: Arc<StandardCommandBufferAllocator>,
}

#[derive(Default)]
pub struct Config {
    pub name: Option<String>,
    pub version: Option<Version>,
}

#[derive(Debug)]
pub enum CreateError {
    EntryPointMainMissing,
    NoPhysicalDevice,
    DeviceQueueLacksCompute,
    LibraryLoading(vulkano::LoadingError),
    InstanceCreation(vulkano::Validated<vulkano::VulkanError>),
    EnumeratePhysicalDevices(vulkano::VulkanError),
    CreateDevice(vulkano::Validated<vulkano::VulkanError>),
}

#[derive(Debug, Clone)]
pub enum MakeBufferError {
    InvalidLayout,
    Allocate(vulkano::Validated<vulkano::buffer::AllocateBufferError>),
}

#[derive(Debug)]
pub enum MakePipelineError {
    PipelineCreation(vulkano::Validated<VulkanError>),
}

impl Context {
    pub fn new(config: Config) -> Result<Self, CreateError> {
        let library = VulkanLibrary::new().map_err(CreateError::LibraryLoading)?;
        let instance = Instance::new(
            library.clone(),
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                application_name: config.name,
                application_version: config
                    .version
                    .unwrap_or(Version::major_minor(0, 0)),
                ..Default::default()
            },
        ).map_err(CreateError::InstanceCreation)?;

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .map_err(CreateError::EnumeratePhysicalDevices)?
            .filter_map(|d| {
                rate_physical_device(&d).map(|(score, queue_family_index)| {
                    (d, score, queue_family_index)
                })
            })
            .max_by_key(|(_d, score, _)| *score)
            .map(|(d, _score, qfi)| (d, qfi))
            .ok_or_else(|| CreateError::NoPhysicalDevice)?;

        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        ).map_err(CreateError::CreateDevice)?;

        let queue = queues
            .next()
            .ok_or_else(|| CreateError::DeviceQueueLacksCompute)?;

        let alloc =
            Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let descriptor_set_alloc =
            Arc::new(StandardDescriptorSetAllocator::new(
                device.clone(),
                Default::default(),
            ));
        let command_buffer_alloc =
            Arc::new(StandardCommandBufferAllocator::new(
                device.clone(),
                Default::default(),
            ));

        Ok(Context {
            library: library.clone(),
            instance: instance.clone(),
            physical_device: physical_device.clone(),
            device: device.clone(),
            queue: queue.clone(),
            alloc: alloc.clone(),
            descriptor_set_alloc: descriptor_set_alloc.clone(),
            command_buffer_alloc: command_buffer_alloc.clone(),
        })
    }

    pub fn make_buffer(
        &self,
        usage: BufferUsage,
        prefer_device: bool,
        layout: Layout,
    ) -> Result<Arc<Buffer>, MakeBufferError> {
        Ok(Buffer::new(
            self.alloc.clone(),
            BufferCreateInfo {
                usage,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: (if prefer_device {
                    MemoryTypeFilter::PREFER_DEVICE
                } else {
                    MemoryTypeFilter::PREFER_HOST
                }) | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            DeviceLayout::from_layout(layout)
                .ok_or_else(|| MakeBufferError::InvalidLayout)?,
        ).map_err(MakeBufferError::Allocate)?)
    }

    pub fn make_pipeline_layout<'a>(
        &self,
        stages: impl IntoIterator<Item = &'a PipelineShaderStageCreateInfo>,
    ) -> Arc<PipelineLayout> {
        PipelineLayout::new(
            self.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(stages)
                .into_pipeline_layout_create_info(self.device.clone())
                .unwrap(),
        )
        .unwrap()
    }

    pub fn make_compute_pipeline(
        &self,
        stage: PipelineShaderStageCreateInfo,
        layout: Arc<PipelineLayout>,
    ) -> Result<Arc<ComputePipeline>, MakePipelineError> {
        Ok(ComputePipeline::new(
            self.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        ).map_err(MakePipelineError::PipelineCreation)?)
    }

    pub fn make_descriptor_set(
        &self,
        layout: Arc<DescriptorSetLayout>,
        buffers: impl Iterator<Item = Arc<Buffer>>,
    ) -> Result<Arc<DescriptorSet>, Validated<VulkanError>> {
        DescriptorSet::new(
            self.descriptor_set_alloc.clone(),
            layout,
            buffers.enumerate().map(|(i, buffer)| {
                WriteDescriptorSet::buffer(
                    i as u32,
                    Into::<Subbuffer<[u8]>>::into(buffer.clone()),
                )
            }),
            [],
        )
    }

    pub unsafe fn shader_from_spirv(
        &self,
        spirv: &[u32],
    ) -> Result<Arc<ShaderModule>, Validated<vulkano::VulkanError>> {
        unsafe {
            ShaderModule::new(
                self.device.clone(),
                ShaderModuleCreateInfo::new(spirv),
            )
        }
    }

    pub fn make_buffer_copy_command(
        &self,
        src: Subbuffer<[u8]>,
        dst: Subbuffer<[u8]>,
    ) -> Result<Arc<PrimaryAutoCommandBuffer>, Validated<VulkanError>> {
        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_alloc.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;
        let _ = builder.copy_buffer(CopyBufferInfo::buffers(src, dst));
        builder.build()
    }

    pub fn make_compute_pipeline_command(
        &self,
        pipeline: Arc<ComputePipeline>,
        descriptor_sets: impl Iterator<Item = Arc<DescriptorSet>>,
        first_set: u32,
        push_constants: impl BufferContents,
        push_constants_offset: u32,
        group_counts: [u32; 3],
    ) -> Result<Arc<PrimaryAutoCommandBuffer>, Validated<VulkanError>> {
        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_alloc.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
        )?;
        let layout = pipeline.layout().clone();
        let _ = builder.bind_pipeline_compute(pipeline.clone())?;
        for descriptor_set in descriptor_sets {
            let _ = builder.bind_descriptor_sets(
                PipelineBindPoint::Compute,
                layout.clone(),
                first_set,
                descriptor_set,
            )?;
        }
        let _ = builder.push_constants(
            pipeline.layout().clone(),
            push_constants_offset,
            push_constants,
        )?;
        let _ = unsafe { builder.dispatch(group_counts) }?;
        builder.build()
    }
}

fn rate_physical_device(device: &PhysicalDevice) -> Option<(usize, u32)> {
    let mut score = 0;
    let Some(queue_family_index) = device
        .queue_family_properties()
        .iter()
        .position(|properties| {
            properties.queue_flags.contains(QueueFlags::COMPUTE)
        })
    else {
        return None;
    };

    let properties = device.properties();
    score += rate_device_type(properties.device_type);

    Some((score, queue_family_index as u32))
}

fn rate_device_type(
    device_type: vulkano::device::physical::PhysicalDeviceType,
) -> usize {
    use vulkano::device::physical::PhysicalDeviceType::*;
    match device_type {
        DiscreteGpu => 600,
        IntegratedGpu => 300,
        VirtualGpu => 200,
        Cpu => 100,
        Other => 0,
        _ => unreachable!(),
    }
}