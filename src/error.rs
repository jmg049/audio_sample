use thiserror::Error;
pub type AudioSampleResult<T> = Result<T, AudioSampleError>;

#[derive(Error, Debug)]
pub enum AudioSampleError {
    #[error("Channel mismatch")]
    ChannelMismatch,

    #[cfg(feature = "ndarray")]
    #[error("Shape error: {0}")]
    ShapeMismatch(#[from] ndarray::ShapeError),

    #[error("Data length not divisible by channel count: total samples = {0}, channels = {1}")]
    InvalidChannelDivision(usize, usize),
}
