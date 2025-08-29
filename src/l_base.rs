pub trait Translate<FromL, ToL> {
    type Config: Sized;
    type Error: Sized;
    fn translate(
        self,
        from: FromL,
        config: &Self::Config,
    ) -> Result<ToL, Self::Error>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarTy {
    F32,
    S32,
    U32,
}

#[cfg(test)]
pub mod test_utils {
    use std::path::Path;

    pub fn project_path() -> Box<Path> {
        Path::new(file!())
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .into()
    }
}