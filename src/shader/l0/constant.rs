use super::ty::CompositeTyId;

#[derive(Debug, Clone, PartialEq)]
pub enum Constant {
    Scalar(ScalarConstant),
    Composite(CompositeConstant),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ScalarConstant {
    U32(u32),
    S32(i32),
    F32(f32),
}

#[derive(Debug, Clone, PartialEq)]
pub enum CompositeConstant {
    Aggregate(AggregateConstant),
    Vec(Box<[ScalarConstant]>),
    Mat(Box<[ScalarConstant]>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum AggregateConstant {
    Array(CompositeTyId, Box<[Constant]>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstantId(pub usize);

impl From<u32> for Constant {
    fn from(v: u32) -> Self {
        Constant::Scalar(ScalarConstant::U32(v))
    }
}

impl From<i32> for Constant {
    fn from(v: i32) -> Self {
        Constant::Scalar(ScalarConstant::S32(v))
    }
}

impl From<f32> for Constant {
    fn from(v: f32) -> Self {
        Constant::Scalar(ScalarConstant::F32(v))
    }
}

impl From<CompositeConstant> for Constant {
    fn from(v: CompositeConstant) -> Self {
        Constant::Composite(v)
    }
}

impl From<AggregateConstant> for Constant {
    fn from(v: AggregateConstant) -> Self {
        Constant::Composite(CompositeConstant::Aggregate(v))
    }
}