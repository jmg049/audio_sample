<div align="center">

# Audio Sample Processing & Conversion Crate

<img src="logo.png" alt="audio_sample Logo" width="200"/>

[![Crates.io](https://img.shields.io/crates/v/audio_sample.svg)](https://crates.io/crates/audio_sample)[![Docs.rs](https://docs.rs/audio_sample/badge.svg)](https://docs.rs/audio_sample)![MSRV: 1.70+](https://img.shields.io/badge/MSRV-1.70+-blue)
</div>

A trait-oriented audio processing crate focused on efficient, safe, and composable handling of audio sample sequences in Rust.

This library defines the `AudioSample` trait and builds robust abstractions around common audio workflows - including type conversion, layout reorganization, and data transformation into numerical arrays.

---

## Features

- **Zero-cost abstractions**: All traits are composable and minimize allocations.
- **Loss-aware conversions**: Converts between all supported sample formats with scaling and clamping.
- **Collection-friendly**: Convert full `Vec`, `Box`, `Arc` collections via [`ConvertCollection`].
- **Channel layout transforms**: Interleave and deinterleave with [`planar_to_interleaved`] and [`interleaved_to_planar`].
- **Ndarray integration**: Convert audio data into [`ndarray::Array2`] for further processing (via `ndarray` feature).
- **Typed iterators**: Enhance any `Iterator<Item = T>` where `T: AudioSample` via [`Samples`].

---

## Supported Sample Types

- `i16`: 16-bit signed integer (common PCM)
- `i24`: 24-bit signed integer ([i24 crate](https://crates.io/crates/i24))
- `i32`: 32-bit signed integer
- `f32`: 32-bit floating-point
- `f64`: 64-bit floating-point

---

## Feature Flags

This crate supports the following optional features:

- `ndarray`: Enables `ToNdarray` trait for converting structured audio to `ndarray::Array2`.

Enable it like so:

```toml
audio_sample = { version = "X.Y", features = ["ndarray"] }
```

## API Overview

| Trait / Type             | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| [`AudioSample`]          | Marker trait for types that can be used as audio samples (e.g., `i16`, `f32`). Enables safe, performant conversions and serialization. |
| [`ConvertTo`]            | Trait for generic sample type conversions (e.g., `i16` to `f32`, `f64` to `i24`). |
| [`ConvertCollection`]    | Provides ergonomic conversion of entire sample collections (e.g., `Vec<i16>` to `Vec<f32>`). |
| [`Samples`]              | Extends iterators of audio samples with zero-cost conversion, byte export, and cloning operations. |
| [`StructuredSamples`]    | Builds on `Samples` to add layout and channel metadata, enabling structured views and integration with libraries like `ndarray`. |
| [`ToNdarray`]            | Converts interleaved structured sample iterators into 2D `ndarray::Array2` representations. |
| [`AsSamples`]            | Enables ergonomic, non-consuming iteration over common containers like `Vec<T>` and `Box<[T]>` as sample sequences. |

For full API documentation, see [docs.rs/audio_sample](https://docs.rs/audio_sample)

## Usage Examples

### Basic Conversion

```rust
use audio_sample::ConvertTo;

let sample: i16 = i16::MAX;
let converted: f32 = sample.convert_to();
assert!((0.99..=1.0).contains(&converted));
```

### Convert an Entire Collection

```rust
use audio_sample::ConvertCollection;

let raw: Vec<i16> = vec![0, 16384, -16384, 32767];
let as_f32: Vec<f32> = raw.convert();       // Non-consuming
let as_f32: Vec<f32> = raw.convert_into();  // Consuming

```

## Conversion Semantics

### Float -> Integer

- Scaled to the full integer range
- Rounded to the nearest integer
- Clamped to avoid overflow

### Integer - Float

- Normalised into [-1.0, 1.0]
- Maintains proportional amplitude

### Bit-Depth Changes

- Widening: Scaled up (left-shifted)
- Narrowing: Scaled down (right-shifted)

## Use With Iterators

```rust
use audio_sample::Samples;

let data = vec![1i16, 2, 3];
let sum: f32 = data.iter().cloned().map(|s| s.convert_to::<f32>()).sum();
```

### Convert to ndarray Matrix

Requires ndarray feature:
```rust
use audio_sample::{StructuredSamples, ToNdarray, ChannelLayout};
use ndarray::Array2;

#[derive(Clone)]
struct Interleaved<T> {
    data: Vec<T>,
    channels: usize,
}

impl<T: Copy> Iterator for Interleaved<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.data.is_empty() {
            None
        } else {
            Some(self.data.remove(0)) // For performance, prefer `VecDeque` or `into_iter`
        }
    }
}

impl<T: audio_sample::AudioSample> StructuredSamples for Interleaved<T> {
    fn layout(&self) -> ChannelLayout { ChannelLayout::Interleaved }
    fn channels(&self) -> usize { self.channels }
}

let interleaved = Interleaved { data: vec![1.0, 10.0, 2.0, 20.0], channels: 2 };
let matrix: Array2<f32> = interleaved.to_ndarray().unwrap();
```

## Error Handling

Fallible methods (e.g., to_ndarray, interleaved_to_planar) return: ``Result<T, AudioSampleError>``

### Errors

- ``AudioSampleError::ChannelMismatch``: Channel layout inconsistency.
- ``AudioSampleError::ShapeMismatch``: Matrix shape cannot be derived.
- ``AudioSampleError::InvalidChannelDivision``: Sample count not divisible by channels.

## License

Licensed under [MIT](./LICENSE).

## Feature Requests / Contributing

Feature and pull requests are welcome!

## Benchmarks

To keep the README focused, benchmark results have been moved to [BENCHMARKS.md](./BENCHMARKS.md).

They include conversion speeds across formats (e.g., f32 -> i32, i16 -> f64, i24 -> f32) at different sample rates and durations.
