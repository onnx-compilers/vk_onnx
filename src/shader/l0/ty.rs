use crate::l_base::ScalarTy;

use rspirv::spirv::StorageClass;

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
    Array1(Ty, Option<usize>),
    // Array2(Ty, Option<usize>, usize),
    // Array3(Ty, Option<usize>, usize, usize),
    // ArrayN(Ty, Option<usize>, Box<[usize]>),
    Struct(Box<[Ty]>),
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