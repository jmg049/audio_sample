use thiserror::Error;
pub type AudioSampleResult<T> = Result<T, AudioSampleError>;

#[derive(Error, Debug)]
pub enum AudioSampleError {
    #[error("Channel mismatch")]
    ChannelMismatch,
    #[error("Shape error: {0}")]
    ShapeMismatch(#[from] ndarray::ShapeError),
}
