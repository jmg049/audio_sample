//! # Samples
//!
//! This module provides a flexible and efficient trait-based framework for working with audio sample sequences in Rust.
//! It defines a `Samples` trait for enhancing iterators over [AudioSample] items, allowing transformations, conversions,
//! serialization, and integration with numerical computing libraries like `ndarray`.
//!
//! ## Core Traits
//!
//! - [`Samples`]: Extends iterators of [AudioSample] to support mapping, conversion, and serialization to byte buffers.
//! - [`ConvertCollection`]: Enables conversion of entire collections (e.g., `Vec`, `Box`, `Arc`) to a different [AudioSample] type.
//! - [`AsSamples`]: Provides an ergonomic way to turn references like `&[T]`, `Vec<T>` into sample iterators.
//! - [`StructuredSamples`]: Adds structural metadata (e.g., channel layout, count) to sample iterators.
//! - [`ToNdarray`]: Converts structured samples into [ndarray::Array2] for numerical processing.
//!
//! ## Format Conversion Functions
//!
//! - [`planar_to_interleaved`]: Rearranges audio data from planar (channel-major) to interleaved (frame-major) layout.
//! - [`interleaved_to_planar`]: Reorders interleaved audio data into planar layout.
//!
//!
//! This trait-based design allows ergonomic extensions to common audio containers without requiring wrapper types,
//! and is designed for real-time and offline DSP applications, audio pipelines, or I/O contexts.

use num_traits::ToBytes;
use std::{alloc::Layout, sync::Arc};

use crate::{I24, ChannelLayout, AudioSample, ConvertTo};

use crate::{AudioSampleResult, AudioSampleError};

#[cfg(feature = "ndarray")]
use ndarray::Array2;

/// Helper function to allocate a fixed sized, heap allocated buffer of bytes.
pub(crate) fn alloc_box_buffer(len: usize) -> Box<[u8]> {
    if len == 0 {
        return <Box<[u8]>>::default();
    }
    let layout = match Layout::array::<u8>(len) {
        Ok(layout) => layout,
        Err(_) => panic!("Failed to allocate buffer of size {}", len),
    };

    let ptr = unsafe { std::alloc::alloc(layout) };
    let slice_ptr = core::ptr::slice_from_raw_parts_mut(ptr, len);
    unsafe { Box::from_raw(slice_ptr) }
}

/// Trait for iterators over audio sample types providing convenient conversions and utilities.
pub trait Samples: Iterator
where
    Self::Item: AudioSample,
{

    /// Consumes the iterator and maps each element to another sample type using `ConvertTo`.
    fn map_converted<T: AudioSample>(
        self,
    ) -> std::iter::Map<Self, fn(Self::Item) -> T>
    where
        Self: Sized,
        Self::Item: ConvertTo<T>,
    {
        fn convert<F: AudioSample, T: AudioSample>(item: F) -> T
        where
            F: ConvertTo<T>,
        {
            item.convert_to()
        }

        self.map(convert::<Self::Item, T>)
    }

    /// Clones the iterator and maps each element to another sample type using `ConvertTo`.
    fn clone_and_convert<T: AudioSample>(
        &self,
    ) -> std::iter::Map<Self, fn(Self::Item) -> T>
    where
        Self: Sized + Clone,
        Self::Item: ConvertTo<T>,
    {
        self.clone().map_converted()
    }

    /// Clones the iterator and collects it into a byte buffer using `to_ne_bytes()`.
    fn as_bytes(&self) -> Box<[u8]>
    where
        Self: Sized + Clone + ExactSizeIterator,
    {
        self.clone().into_bytes()
    }

    /// Consumes the iterator and collects it into a byte buffer using `to_ne_bytes()`.
    fn into_bytes(self) -> Box<[u8]>
    where
        Self: Sized + ExactSizeIterator,
    {
        let f_type = std::any::TypeId::of::<Self::Item>();
        let size = if f_type == std::any::TypeId::of::<I24>() {
            3
        } else {
            std::mem::size_of::<Self::Item>()
        };

        let mut bytes = alloc_box_buffer(self.len() * size);

        let mut i = 0;
        for sample in self {
            let s_buf = sample.to_ne_bytes();
            let s_bytes: &[u8] = s_buf.as_ref();
            bytes[i..i + size].copy_from_slice(s_bytes);
            i += size;
        }

        bytes
    }
}

/// Trait for defining a source of samples from a reference (e.g., &[T], Vec<T>).
pub trait AsSamples<'a> {
    type Item: AudioSample;
    type Iter: Samples<Item = Self::Item> + 'a;

    fn as_samples(&'a self) -> Self::Iter;
}

impl<'a, T: AudioSample> AsSamples<'a> for Vec<T> {
    type Item = T;
    type Iter = std::iter::Cloned<std::slice::Iter<'a, T>>;

    fn as_samples(&'a self) -> Self::Iter {
        self.iter().cloned()
    }
}

impl<'a, T: AudioSample> AsSamples<'a> for Box<[T]> {
    type Item = T;
    type Iter = std::iter::Cloned<std::slice::Iter<'a, T>>;

    fn as_samples(&'a self) -> Self::Iter {
        self.iter().cloned()
    }
}

impl<'a, T: AudioSample> AsSamples<'a> for &'a [T] {
    type Item = T;
    type Iter = std::iter::Cloned<std::slice::Iter<'a, T>>;

    fn as_samples(&'a self) -> Self::Iter {
        self.iter().cloned()
    }
}

/// Allows conversion of owned or borrowed collections of audio samples to another sample type.
pub trait ConvertCollection<T: AudioSample> {
    type Output;
    fn convert(&self) -> Self::Output;
    fn convert_into(self) -> Self::Output;
}


impl<F, T> ConvertCollection<T> for Vec<F>
where
    F: AudioSample + ConvertTo<T>,
    T: AudioSample,
{
    type Output = Vec<T>;

    fn convert(&self) -> Self::Output {
        self.iter().cloned().map(|s| s.convert_to()).collect()
    }

    fn convert_into(self) -> Self::Output {
        self.into_iter().map(|s| s.convert_to()).collect()
    }
}


impl<F, T> ConvertCollection<T> for Box<[F]>
where
    F: AudioSample + ConvertTo<T>,
    T: AudioSample,
{
    type Output = Box<[T]>;

    fn convert(&self) -> Self::Output {
        self.iter()
            .cloned()
            .map(|s| s.convert_to())
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    fn convert_into(self) -> Self::Output {
        self.into_vec()
            .into_iter()
            .map(|s| s.convert_to())
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }
}

impl<F, T> ConvertCollection<T> for Arc<[F]>
where
    F: AudioSample + ConvertTo<T>,
    T: AudioSample,
{
    type Output = Arc<[T]>;

    fn convert(&self) -> Self::Output {
        let converted: Vec<T> = self.iter().cloned().map(|s| s.convert_to()).collect();
        Arc::from(converted)
    }

    fn convert_into(self) -> Self::Output {
        let converted: Vec<T> = self.iter().cloned().map(|s| s.convert_to()).collect();
        Arc::from(converted)
    }
}

/// Trait for sample iterators that carry structural audio metadata (channel layout, etc.).
pub trait StructuredSamples: Samples + ExactSizeIterator + Clone 
where
Self::Item: AudioSample,
{
    fn layout(&self) -> ChannelLayout;
    fn channels(&self) -> usize;

    fn frames(&self) -> usize {
        self.len() / self.channels()
    }
}

#[cfg(feature = "ndarray")]
/// Trait allowing structured iterators to convert into a `ndarray::Array2` matrix of shape (channels, frames).
pub trait ToNdarray<T: AudioSample>: StructuredSamples<Item = T> {
    fn to_ndarray(&self) -> AudioSampleResult<Array2<T>>;
}

#[cfg(feature = "ndarray")]
/// Interleaved-to-column-major conversion into `ndarray::Array2`.
impl<T, I> ToNdarray<T> for I
where
    T: AudioSample,
    I: StructuredSamples<Item = T>,
{
    fn to_ndarray(&self) -> AudioSampleResult<Array2<T>> {
        if self.layout() != ChannelLayout::Interleaved {
            return Err(AudioSampleError::ChannelMismatch);
        }

        let channels = self.channels();
        let frames = self.frames();
        let data: Vec<T> = self.clone().collect();

        let mut deinterleaved = vec![T::default(); data.len()];
        for (i, sample) in data.iter().enumerate() {
            let ch = i % channels;
            let frame = i / channels;
            deinterleaved[ch * frames + frame] = *sample;
        }

        Array2::from_shape_vec((channels, frames), deinterleaved)
            .map_err(AudioSampleError::ShapeMismatch)
    }
}

// Blanket impl for any iterator that satisfies trait bounds.
impl<T, I> Samples for I
where
    T: AudioSample,
    I: Iterator<Item = T> + ExactSizeIterator<Item = T>,
{
}

/// Converts planar audio data (grouped by channel) into interleaved layout.
pub fn planar_to_interleaved<T: AudioSample + Copy, A: AsMut<[T]>>(
    mut planar_data: A,
    channels: usize,
) -> AudioSampleResult<()> {
    let data = planar_data.as_mut();
    let total_samples = data.len();
    if total_samples % channels != 0 {
        return Err(AudioSampleError::InvalidChannelDivision(total_samples, channels));
    }
    let samples_per_channel = total_samples / channels;

    let temp = data.to_vec();
    let chunks = temp.chunks_exact(samples_per_channel);
    let remainder = chunks.remainder();

    if !remainder.is_empty() {
        return Err(AudioSampleError::InvalidChannelDivision(
            total_samples,
            channels,
        ));
    }

    for (i, chunk) in chunks.enumerate() {
        for (j, &sample) in chunk.iter().enumerate() {
            data[j * channels + i] = sample;
        }
    }


    Ok(())
}

/// Converts interleaved audio data into planar format (grouped by channel).
pub fn interleaved_to_planar<T: AudioSample + Copy, A: AsMut<[T]>>(
    mut interleaved_data: A,
    channels: usize,
) -> AudioSampleResult<()> {
    let data = interleaved_data.as_mut();
    let total_samples = data.len();
    if total_samples % channels != 0 {
        return Err(AudioSampleError::InvalidChannelDivision(total_samples, channels));
    }
    let samples_per_channel = total_samples / channels;

    let temp = data.to_vec();

    for i in 0..samples_per_channel {
        for c in 0..channels {
            data[c * samples_per_channel + i] = temp[i * channels + c];
        }
    }

    Ok(())
}


#[cfg(feature = "ndarray")]
#[cfg(test)]
mod ndarray_tests {
    use std::sync::Arc;

    use crate::{samples::{interleaved_to_planar, planar_to_interleaved}, *};
    use ndarray::Array2;

    #[test]
fn test_to_ndarray_interleaved() {
    let interleaved: Vec<f32> = vec![
        1.0, 10.0, // Frame 0: ch0, ch1
        2.0, 20.0, // Frame 1
        3.0, 30.0, // Frame 2
    ];

    let expected = Array2::from_shape_vec((2, 3), vec![
        1.0, 2.0, 3.0, // ch0
        10.0, 20.0, 30.0, // ch1
    ])
    .unwrap();

    #[derive(Clone)]
    struct InterleavedSamples {
        data: Vec<f32>,
        channels: usize,
    }

    impl Iterator for InterleavedSamples {
        type Item = f32;

        fn next(&mut self) -> Option<Self::Item> {
            if self.data.is_empty() {
                None
            } else {
                Some(self.data.remove(0)) 
            }
        }
        
    }

    impl ExactSizeIterator for InterleavedSamples {
        fn len(&self) -> usize {
            self.data.len()
        }
    }


    impl StructuredSamples for InterleavedSamples {
        fn layout(&self) -> ChannelLayout {
            ChannelLayout::Interleaved
        }

        fn channels(&self) -> usize {
            self.channels
        }
    }


    let data = InterleavedSamples {
        data: interleaved.clone(),
        channels: 2,
    };

    let result = data.to_ndarray().unwrap();
    assert_eq!(result, expected);
}

#[test]
fn test_planar_interleaved_roundtrip() {
    let mut planar: Vec<i16> = vec![1, 2, 3, 10, 20, 30]; // Two channels, 3 frames
    let expected = planar.clone();

    planar_to_interleaved(&mut planar, 2).unwrap();
    interleaved_to_planar(&mut planar, 2).unwrap();

    assert_eq!(planar, expected);
}

#[test]
fn test_convert_collection_vec() {
    let input: Vec<i16> = vec![-32768, 0, 32767];
    let output: Vec<f32> = input.convert();

    assert_eq!(output.len(), 3);
    assert!(output[0] < 0.0);
    assert_eq!(output[1], 0.0);
    assert!(output[2] > 0.0);
}

#[test]
fn test_convert_collection_boxed() {
    let input: Box<[i16]> = vec![-32768, 0, 32767].into_boxed_slice();
    let output: Box<[f32]> = input.convert();

    assert_eq!(output.len(), 3);
    assert!(output[0] < 0.0);
    assert_eq!(output[1], 0.0);
    assert!(output[2] > 0.0);
}

#[test]
fn test_convert_collection_arc() {
    let input: Arc<[i16]> = Arc::from(vec![-32768, 0, 32767]);
    let output: Arc<[f32]> = input.convert();

    assert_eq!(output.len(), 3);
    assert!(output[0] < 0.0);
    assert_eq!(output[1], 0.0);
    assert!(output[2] > 0.0);
}

}