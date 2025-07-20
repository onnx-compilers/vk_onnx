pub trait TranslateFrom<T> where Self: Sized {
    type Error;
    fn translate_from(source: &T) -> Result<Self, Self::Error>;
}