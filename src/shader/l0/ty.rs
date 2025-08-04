use crate::l_base::ScalarTy;

use rspirv::spirv::StorageClass;

use super::constant::ConstantId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ty {
    Scalar(ScalarTy),
    Vec(ScalarTy, u32),
    Mat(ScalarTy, u32),
    Ptr(PtrTyId, StorageClass),
    Composite(CompositeTyId),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompositeTy {
    Array1(Ty, Option<ConstantId>),
    // Array2(Ty, Option<usize>, usize),
    // Array3(Ty, Option<usize>, usize, usize),
    // ArrayN(Ty, Option<usize>, Box<[usize]>),
    Struct {
        fields: Box<[StructMember]>,
        is_block: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructMember {
    pub ty: Ty,
    pub writable: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PtrTyId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompositeTyId(pub usize);

impl From<CompositeTyId> for Ty {
    fn from(id: CompositeTyId) -> Self {
        Ty::Composite(id)
    }
}

impl CompositeTy {
    pub fn r#struct(
        fields: impl Into<Box<[StructMember]>>,
        is_block: bool,
    ) -> Self {
        Self::Struct {
            fields: fields.into(),
            is_block,
        }
    }
}
