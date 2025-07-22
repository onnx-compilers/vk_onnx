use crate::l0;
use crate::l_base::TranslateFrom;

pub use crate::l0::{Input, Output, Value, Parameter, Temporary};

pub struct IR {
    pub inputs: Vec<Input>,
    pub outputs: Vec<Output>
}

pub struct IRBuilder<'a> {
    inputs: &'a [Input],
    outputs: &'a [Output],
}

impl TranslateFrom<l0::IR> for IR {
    type Error = ();
    fn translate_from(ir: l0::IR) -> Result<Self, Self::Error> {
        Ok(IR {
            inputs: ir.inputs,
            outputs: ir.outputs
        })
    }
}