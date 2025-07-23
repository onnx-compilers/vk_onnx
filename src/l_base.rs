pub trait TranslateFrom<T> where Self: Sized {
    type Error;
    fn translate_from(source: T) -> Result<Self, Self::Error>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarTy {
    F32,
    S32,
    U32,
}