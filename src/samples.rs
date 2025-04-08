use bytemuck::cast_slice;
use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
    sync::Arc,
};

use crate::{AudioSample, ConvertTo};

/// Defines the storage backend for audio samples.
///
/// This enum allows `Samples` to store data in different ways,
/// enabling zero-copy operations when possible.
enum SamplesStorage<T: AudioSample> {
    /// Owned buffer - samples are owned by this struct
    Owned(Box<[T]>),

    /// Borrowed buffer - samples are borrowed and have a lifetime
    Borrowed(&'static [T]),

    /// Memory-mapped buffer - samples are viewed from a memory-mapped file
    MemoryMapped {
        mmap: Arc<memmap2::Mmap>,
        offset: usize,
        len: usize,
    },

    /// Shared buffer - samples are in an Arc for cheap cloning
    Shared(Arc<[T]>),
}

/// A collection of audio samples with support for zero-copy operations.
///
/// `Samples<T>` can store audio data in various ways, optimizing for
/// different use cases:
/// - Owned data for full control
/// - Borrowed data for zero-copy reading
/// - Memory-mapped data for efficient file access
/// - Shared data for multi-threaded processing
///
/// This struct implements `Deref<Target=[T]>`, allowing it to be used
/// like a slice in most cases.
pub struct Samples<T: AudioSample> {
    storage: SamplesStorage<T>,
}

impl<T: AudioSample> Samples<T> {
    /// Creates a new `Samples` with owned data.
    ///
    /// # Arguments
    ///
    /// * `samples` - A boxed slice containing the audio samples
    ///
    /// # Returns
    ///
    /// A new `Samples<T>` that owns the data.
    pub fn new(samples: Box<[T]>) -> Self {
        Self {
            storage: SamplesStorage::Owned(samples),
        }
    }

    /// Creates a new `Samples` that borrows data.
    ///
    /// This method allows for zero-copy operations when the data
    /// already exists elsewhere.
    ///
    /// # Arguments
    ///
    /// * `samples` - A slice of audio samples to borrow
    ///
    /// # Returns
    ///
    /// A new `Samples<T>` that borrows the data.
    pub fn from_slice(samples: &'static [T]) -> Self {
        Self {
            storage: SamplesStorage::Borrowed(samples),
        }
    }

    /// Creates a new `Samples` from a memory-mapped file.
    ///
    /// This method enables zero-copy reading directly from a file
    /// when the sample type matches the file's native format.
    ///
    /// # Arguments
    ///
    /// * `mmap` - A shared reference to a memory-mapped file
    /// * `offset` - Offset in bytes to the start of the sample data
    /// * `len` - Number of samples in the data
    ///
    /// # Returns
    ///
    /// A new `Samples<T>` that views the memory-mapped file.
    pub fn from_mmap(mmap: Arc<memmap2::Mmap>, offset: usize, len: usize) -> Self {
        Self {
            storage: SamplesStorage::MemoryMapped { mmap, offset, len },
        }
    }

    /// Creates a new `Samples` with shared data.
    ///
    /// This method is useful for multi-threaded processing where
    /// multiple threads need access to the same data.
    ///
    /// # Arguments
    ///
    /// * `samples` - A shared slice of audio samples
    ///
    /// # Returns
    ///
    /// A new `Samples<T>` that shares ownership of the data.
    pub fn from_shared(samples: Arc<[T]>) -> Self {
        Self {
            storage: SamplesStorage::Shared(samples),
        }
    }

    /// Gets the length of the samples.
    ///
    /// # Returns
    ///
    /// The number of samples.
    pub fn len(&self) -> usize {
        match &self.storage {
            SamplesStorage::Owned(samples) => samples.len(),
            SamplesStorage::Borrowed(samples) => samples.len(),
            SamplesStorage::MemoryMapped { len, .. } => *len,
            SamplesStorage::Shared(samples) => samples.len(),
        }
    }

    /// Checks if the samples are empty.
    ///
    /// # Returns
    ///
    /// `true` if there are no samples, `false` otherwise.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Gets a reference to the samples as a slice.
    ///
    /// # Returns
    ///
    /// A slice containing the audio samples.
    pub fn as_slice(&self) -> &[T] {
        match &self.storage {
            SamplesStorage::Owned(samples) => samples,
            SamplesStorage::Borrowed(samples) => samples,
            SamplesStorage::MemoryMapped { mmap, offset, len } => {
                let byte_offset = *offset;
                unsafe {
                    let ptr = mmap.as_ptr().add(byte_offset) as *const T;
                    std::slice::from_raw_parts(ptr, *len)
                }
            }
            SamplesStorage::Shared(samples) => samples,
        }
    }

    /// Gets a mutable reference to the samples as a slice.
    ///
    /// This will convert borrowed or memory-mapped samples to owned samples
    /// if necessary, since those storage modes don't allow mutation.
    ///
    /// # Returns
    ///
    /// A mutable slice containing the audio samples.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // First collect what we need to determine if conversion is necessary
        let need_conversion = match &self.storage {
            SamplesStorage::Owned(_) => false,
            SamplesStorage::Shared(arc) => Arc::strong_count(arc) > 1,
            _ => true, // Borrowed or MemoryMapped always need conversion
        };

        // If we need to convert to owned, do it now
        if need_conversion {
            let owned = self.as_slice().to_vec().into_boxed_slice();
            self.storage = SamplesStorage::Owned(owned);
        }

        // Now return a mutable reference
        match &mut self.storage {
            SamplesStorage::Owned(samples) => samples,
            SamplesStorage::Shared(arc) => {
                // We know this is safe because we checked strong_count above
                Arc::get_mut(arc).unwrap()
            }
            // These cases cannot happen after the conversion above
            _ => unreachable!(),
        }
    }

    /// Converts the samples to a new sample type.
    ///
    /// # Type Parameters
    ///
    /// * `U` - The target audio sample type
    ///
    /// # Returns
    ///
    /// A new `Samples<U>` with the converted data.
    pub fn convert<U: AudioSample>(&self) -> Samples<U>
    where
        T: ConvertTo<U>,
    {
        // Check if types are the same (potential zero-copy conversion)
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<U>() {
            // If types are the same, we can potentially reuse the storage
            match &self.storage {
                SamplesStorage::MemoryMapped { mmap, offset, len } => {
                    // Memory-mapped data can be reused directly
                    return Samples::<U>::from_mmap(Arc::clone(mmap), *offset, *len);
                }
                SamplesStorage::Shared(samples) => {
                    // If types are the same, we can reinterpret the shared data
                    // This is a safe transmute because T and U are the same type (checked above)
                    let samples_ptr = samples.as_ptr() as *const u8;
                    let samples_len = samples.len();

                    unsafe {
                        let u_slice =
                            std::slice::from_raw_parts(samples_ptr as *const U, samples_len);
                        // Create the arc directly from a Vec to avoid Box->Arc conversion issues
                        let u_vec = u_slice.to_vec();
                        return Samples::<U>::from_shared(u_vec.into());
                    }
                }
                _ => {}
            }
        }

        // If zero-copy isn't possible, convert the data
        let mut converted = Vec::with_capacity(self.len());
        for &sample in self.as_slice() {
            converted.push(sample.convert_to());
        }

        Samples::new(converted.into_boxed_slice())
    }

    /// Applies a function to each sample and returns a new `Samples`.
    ///
    /// # Arguments
    ///
    /// * `f` - A function that takes a sample and returns a new sample
    ///
    /// # Returns
    ///
    /// A new `Samples<T>` with the mapped data.
    pub fn map<F>(&self, mut f: F) -> Samples<T>
    where
        F: FnMut(T) -> T,
    {
        let mut result = Vec::with_capacity(self.len());
        for &sample in self.as_slice() {
            result.push(f(sample));
        }

        Samples::new(result.into_boxed_slice())
    }

    /// Extracts a channel from interleaved audio data.
    ///
    /// # Arguments
    ///
    /// * `channel` - The channel index to extract (0-based)
    /// * `num_channels` - The total number of channels in the data
    ///
    /// # Returns
    ///
    /// A new `Samples<T>` containing only the specified channel.
    pub fn extract_channel(&self, channel: usize, num_channels: usize) -> Samples<T> {
        if channel >= num_channels {
            return Samples::new(Box::new([]));
        }

        let mut result = Vec::with_capacity(self.len() / num_channels);
        for i in (channel..self.len()).step_by(num_channels) {
            result.push(self.as_slice()[i]);
        }

        Samples::new(result.into_boxed_slice())
    }

    /// Applies a window function to the samples.
    ///
    /// # Arguments
    ///
    /// * `window_type` - The type of window function to apply
    ///
    /// # Returns
    ///
    /// A new `Samples<T>` with the windowed data.
    pub fn window(&self, window_type: WindowType) -> Samples<T>
    where
        T: AudioSample + std::ops::Mul<f32, Output = T>,
    {
        let len = self.len();
        let mut result = Vec::with_capacity(len);

        match window_type {
            WindowType::Rectangular => {
                // Rectangular window is just a copy
                return Samples::new(self.as_slice().to_vec().into_boxed_slice());
            }
            WindowType::Hann => {
                for i in 0..len {
                    let window_value = 0.5
                        * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (len - 1) as f32).cos());
                    result.push(self.as_slice()[i] * window_value);
                }
            }
            WindowType::Hamming => {
                for i in 0..len {
                    let window_value = 0.54
                        - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (len - 1) as f32).cos();
                    result.push(self.as_slice()[i] * window_value);
                }
            }
            WindowType::Blackman => {
                for i in 0..len {
                    let x = 2.0 * std::f32::consts::PI * i as f32 / (len - 1) as f32;
                    let window_value = 0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos();
                    result.push(self.as_slice()[i] * window_value);
                }
            }
        }

        Samples::new(result.into_boxed_slice())
    }

    /// Gets the bytes representation of the samples.
    ///
    /// # Returns
    ///
    /// A slice of bytes representing the sample data.
    pub fn as_bytes(&self) -> &[u8] {
        match &self.storage {
            SamplesStorage::MemoryMapped { mmap, offset, len } => {
                let byte_len = *len * std::mem::size_of::<T>();
                &mmap[*offset..*offset + byte_len]
            }
            _ => cast_slice(self.as_slice()),
        }
    }

    /// Ensures the samples are owned, converting if necessary.
    ///
    /// This is useful when you need to guarantee that the sample data
    /// won't be affected by external changes.
    ///
    /// # Returns
    ///
    /// A new `Samples<T>` that owns its data.
    pub fn to_owned(&self) -> Samples<T> {
        match &self.storage {
            SamplesStorage::Owned(_) => {
                // Already owned, just clone
                Samples::new(self.as_slice().to_vec().into_boxed_slice())
            }
            _ => {
                // Convert to owned
                Samples::new(self.as_slice().to_vec().into_boxed_slice())
            }
        }
    }
}

impl<T: AudioSample> Deref for Samples<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: AudioSample> DerefMut for Samples<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T: AudioSample> AsRef<[T]> for Samples<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: AudioSample> AsMut<[T]> for Samples<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: AudioSample> From<Vec<T>> for Samples<T> {
    fn from(vec: Vec<T>) -> Self {
        Samples::new(vec.into_boxed_slice())
    }
}

impl<T: AudioSample> From<Box<[T]>> for Samples<T> {
    fn from(boxed: Box<[T]>) -> Self {
        Samples::new(boxed)
    }
}

impl<T: AudioSample> From<&'static [T]> for Samples<T> {
    fn from(slice: &'static [T]) -> Self {
        Samples::from_slice(slice)
    }
}

impl<T: AudioSample + Debug> Display for Samples<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Samples<{}>({} samples)",
            std::any::type_name::<T>(),
            self.len()
        )
    }
}

/// Types of window functions for audio processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowType {
    /// Rectangular window (no windowing)
    Rectangular,
    /// Hann window
    Hann,
    /// Hamming window
    Hamming,
    /// Blackman window
    Blackman,
}
