// This module provides terminal/console visualization capabilities for audio samples.
// It can be enabled with the "visualization" feature flag.

use std::any::type_name;
use std::fmt::{self, Display, Formatter};
use std::iter;
use colored;
use crate::{AudioSample, ConvertTo, Samples};

// Re-export these for external use
pub use self::statistics::SampleStatistics;
pub use self::waveform::{VisualizationStyle, WaveformOptions};

/// A structure holding statistics about audio samples
pub mod statistics {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct SampleStatistics<T>
    where
        T: AudioSample,
        f64: ConvertTo<T>,
    {
        pub min: T,
        pub max: T,
        pub avg: T,
        pub rms: T,
        pub peak_to_peak: T,
        pub len: usize,
        pub duration_secs: Option<f64>,
    }

    impl<T> Display for SampleStatistics<T>
    where
        T: AudioSample + Display,
        f64: ConvertTo<T>,
    {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            writeln!(f, "Sample Statistics:")?;
            writeln!(f, "  Count:        {}", self.len)?;

            if let Some(duration) = self.duration_secs {
                writeln!(f, "  Duration:     {:.6} seconds", duration)?;
            }

            writeln!(f, "  Min:          {}", self.min)?;
            writeln!(f, "  Max:          {}", self.max)?;
            writeln!(f, "  Peak-to-peak: {}", self.peak_to_peak)?;
            writeln!(f, "  Avg:          {}", self.avg)?;
            write!(f, "  RMS:          {}", self.rms)
        }
    }
}

/// Waveform visualization utilities
pub mod waveform {
    use super::*;
    use super::terminal::{color, TerminalOptions};

    // Define constants for the bar characters used in compact mode
    const BAR_CHARS: [char; 9] = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

    /// Available styles for visualizing audio waveforms
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum VisualizationStyle {
        /// Detailed view with vertical bars and center line
        Detailed,
        /// Compact view using Unicode block characters
        Compact,
        /// Dots style using only dots to mark sample positions
        Dots,
        /// Simple view using ASCII-only characters
        Ascii,
    }

    /// Options for waveform visualization
    #[derive(Debug, Clone)]
    pub struct WaveformOptions {
        /// Width of the visualization in characters
        pub width: usize,
        /// Height of the visualization in characters (only for detailed view)
        pub height: usize,
        /// Style of visualization
        pub style: VisualizationStyle,
        /// Whether to show the value axis
        pub show_axis: bool,
        /// Whether to show markers for amplitude (e.g., 0dB, -6dB)
        pub show_amplitude_markers: bool,
        /// Whether to show time markers
        pub show_time_markers: bool,
        /// Sample rate for time calculations (if known)
        pub sample_rate: Option<u32>,
    }

    impl Default for WaveformOptions {
        fn default() -> Self {
            Self {
                width: 80,
                height: 11,
                style: VisualizationStyle::Detailed,
                show_axis: true,
                show_amplitude_markers: false,
                show_time_markers: false,
                sample_rate: None,
            }
        }
    }

    impl WaveformOptions {
        /// Create a new set of options with default values
        pub fn new() -> Self {
            Self::default()
        }

        /// Set the width of the visualization
        pub fn width(mut self, width: usize) -> Self {
            self.width = width;
            self
        }

        /// Set the height of the visualization
        pub fn height(mut self, height: usize) -> Self {
            self.height = height;
            self
        }

        /// Set the style of the visualization
        pub fn style(mut self, style: VisualizationStyle) -> Self {
            self.style = style;
            self
        }

        /// Set whether to show the axis
        pub fn show_axis(mut self, show: bool) -> Self {
            self.show_axis = show;
            self
        }

        /// Set whether to show amplitude markers
        pub fn show_amplitude_markers(mut self, show: bool) -> Self {
            self.show_amplitude_markers = show;
            self
        }

        /// Set whether to show time markers
        pub fn show_time_markers(mut self, show: bool) -> Self {
            self.show_time_markers = show;
            self
        }

        /// Set the sample rate for time calculations
        pub fn sample_rate(mut self, rate: u32) -> Self {
            self.sample_rate = Some(rate);
            self
        }
    }

    /// Format samples as dots on a line ('.', '^', 'v')
    pub fn format_dots<T>(samples: &[T], width: usize) -> String
    where
        T: AudioSample,
    {
        if samples.is_empty() {
            return String::from("[empty samples]");
        }

        // Convert all samples to f64 for normalization
        let f64_samples: Vec<f64> = samples.iter().map(|s| s.convert_to()).collect();

        // Find min and max for scaling
        let min = f64_samples.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = f64_samples
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // Use the greater of |min| and |max| for symmetric scaling
        let abs_max = f64::max(min.abs(), max.abs());

        // For very small signals, avoid division by zero
        let scale = if abs_max < 1e-10 { 1.0 } else { abs_max };

        // Sample at regular intervals
        let step = f64::max(1.0, (f64_samples.len() as f64) / (width as f64));

        // Create the center line with dots for samples
        let mut center_line = iter::repeat('-').take(width).collect::<String>();

        for i in 0..width {
            let index = (i as f64 * step).floor() as usize;
            if index < f64_samples.len() {
                let sample = f64_samples[index] / scale;
                let char = if sample > 0.01 {
                    '^'
                } else if sample < -0.01 {
                    'v'
                } else {
                    '.'
                };

                if i < center_line.len() {
                    center_line.replace_range(i..i + 1, &char.to_string());
                }
            }
        }

        center_line
    }


    /// Format samples as a compact representation using block characters
    pub fn format_compact<T>(samples: &[T], width: usize) -> String
    where
        T: AudioSample,
    {
        if samples.is_empty() {
            return String::from("[empty samples]");
        }

        // Convert all samples to f64 for normalization
        let f64_samples: Vec<f64> = samples.iter().map(|s| s.convert_to()).collect();

        // Find min and max for scaling
        let min = f64_samples.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = f64_samples
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // Use the greater of |min| and |max| for symmetric scaling
        let abs_max = f64::max(min.abs(), max.abs());

        // For very small signals, avoid division by zero
        let scale = if abs_max < 1e-10 { 1.0 } else { abs_max };

        // Sample at regular intervals
        let step = f64::max(1.0, (f64_samples.len() as f64) / (width as f64));
        let mut result = String::with_capacity(width);

        for i in 0..width {
            let index = (i as f64 * step).floor() as usize;
            if index < f64_samples.len() {
                let normalized = f64_samples[index] / scale;
                let char_index =
                    ((normalized + 1.0) / 2.0 * (BAR_CHARS.len() - 1) as f64).round() as usize;
                result.push(BAR_CHARS[char_index.min(BAR_CHARS.len() - 1)]);
            }
        }

        result
    }

    /// Format samples as a detailed waveform with a center line
    pub fn format_detailed<T>(samples: &[T], width: usize, height: usize) -> String
    where
        T: AudioSample + ConvertTo<f64>,
    {
        if samples.is_empty() {
            return String::from("[empty samples]");
        }

        // Convert all samples to f64 for normalization
        let f64_samples: Vec<f64> = samples.iter().map(|s| s.convert_to()).collect();

        // Find min and max for scaling
        let min = f64_samples.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = f64_samples
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // Use the greater of |min| and |max| for symmetric scaling
        let abs_max = f64::max(min.abs(), max.abs());

        // For very small signals, avoid division by zero
        let scale = if abs_max < 1e-10 { 1.0 } else { abs_max };

        // Sample at regular intervals
        let step = f64::max(1.0, (f64_samples.len() as f64) / (width as f64));
        let mut sampled_indices = Vec::with_capacity(width);

        for i in 0..width {
            let index = (i as f64 * step).floor() as usize;
            if index < f64_samples.len() {
                sampled_indices.push(index);
            }
        }

        // Create the waveform using a 2D vec of chars instead of strings
        let center_line = (height - 1) / 2;
        let mut waveform = vec![vec![' '; width]; height];

        // Draw the center line (zero)
        for x in 0..width {
            waveform[center_line][x] = '─';
        }

        // Draw each sample point
        for (x, &idx) in sampled_indices.iter().enumerate() {
            if x >= width {
                break;
            }

            let normalized = f64_samples[idx] / scale;
            let bar_height = ((normalized * (center_line as f64)).round() as isize).abs();
            let bar_height = bar_height.min(center_line as isize) as usize;

            if normalized >= 0.0 {
                // Draw positive values (going up)
                for y in (center_line - bar_height)..=center_line {
                    if y < height {
                        waveform[y][x] = '│';
                    }
                }
            } else {
                // Draw negative values (going down)
                for y in center_line..(center_line + bar_height + 1) {
                    if y < height {
                        waveform[y][x] = '│';
                    }
                }
            }
        }

        // Combine all lines into final output
        waveform
            .iter()
            .map(|row| row.iter().collect::<String>())
            .collect::<Vec<String>>()
            .join("\n")
    }

    // Similarly, update the format_ascii function:
    pub fn format_ascii<T>(samples: &[T], width: usize, height: usize) -> String
    where
        T: AudioSample + ConvertTo<f64>,
    {
        if samples.is_empty() {
            return String::from("[empty samples]");
        }

        // Convert all samples to f64 for normalization
        let f64_samples: Vec<f64> = samples.iter().map(|s| s.convert_to()).collect();

        // Find min and max for scaling
        let min = f64_samples.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = f64_samples
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // Use the greater of |min| and |max| for symmetric scaling
        let abs_max = f64::max(min.abs(), max.abs());

        // For very small signals, avoid division by zero
        let scale = if abs_max < 1e-10 { 1.0 } else { abs_max };

        // Sample at regular intervals
        let step = f64::max(1.0, (f64_samples.len() as f64) / (width as f64));
        let mut sampled_indices = Vec::with_capacity(width);

        for i in 0..width {
            let index = (i as f64 * step).floor() as usize;
            if index < f64_samples.len() {
                sampled_indices.push(index);
            }
        }

        // Create the waveform using a 2D vec of chars
        let center_line = (height - 1) / 2;
        let mut waveform = vec![vec![' '; width]; height];

        // Draw the center line (zero)
        for x in 0..width {
            waveform[center_line][x] = '-';
        }

        // Draw each sample point
        for (x, &idx) in sampled_indices.iter().enumerate() {
            if x >= width {
                break;
            }

            let normalized = f64_samples[idx] / scale;
            let bar_height = ((normalized * (center_line as f64)).round() as isize).abs();
            let bar_height = bar_height.min(center_line as isize) as usize;

            if normalized >= 0.0 {
                // Draw positive values (going up)
                for y in (center_line - bar_height)..=center_line {
                    if y < height {
                        waveform[y][x] = '|';
                    }
                }
            } else {
                // Draw negative values (going down)
                for y in center_line..(center_line + bar_height + 1) {
                    if y < height {
                        waveform[y][x] = '|';
                    }
                }
            }
        }

        // Combine all lines into final output
        waveform
            .iter()
            .map(|row| row.iter().collect::<String>())
            .collect::<Vec<String>>()
            .join("\n")
    }

    // If you need to update the colored waveform function as well:
    pub fn format_colored_waveform<T>(samples: &[T], options: &TerminalOptions) -> String
    where
        T: AudioSample + ConvertTo<f64>,
    {
        if samples.is_empty() {
            return String::from("[empty samples]");
        }

        // Fall back to ASCII if Unicode is not supported
        let waveform_style = if options.unicode_support {
            waveform::VisualizationStyle::Detailed
        } else {
            waveform::VisualizationStyle::Ascii
        };

        let waveform_options = waveform::WaveformOptions {
            width: options.width,
            height: options.height.min(20),
            style: waveform_style,
            ..Default::default()
        };

        let waveform = waveform::format_with_options(samples, &waveform_options);

        // Add colors if requested
        if options.use_colors {
            if options.unicode_support {
                // Color the vertical and horizontal lines
                waveform
                    .replace("│", &format!("{}│{}", color::CYAN, color::RESET))
                    .replace("─", &format!("{}─{}", color::YELLOW, color::RESET))
            } else {
                // Color the ASCII lines
                waveform
                    .replace("|", &format!("{}|{}", color::CYAN, color::RESET))
                    .replace("-", &format!("{}-{}", color::YELLOW, color::RESET))
            }
        } else {
            waveform
        }
    }

    pub fn format_with_options<T>(samples: &[T], options: &WaveformOptions) -> String
    where
        T: AudioSample,
    {
        match options.style {
            VisualizationStyle::Detailed => format_detailed(samples, options.width, options.height),
            VisualizationStyle::Compact => format_compact(samples, options.width),
            VisualizationStyle::Dots => format_dots(samples, options.width),
            VisualizationStyle::Ascii => format_ascii(samples, options.width, options.height),
        }
    }
}

/// Multi-channel visualization utilities
pub mod multichannel {
    use super::*;

    /// Options for multi-channel visualization
    #[derive(Debug, Clone)]
    pub struct MultichannelOptions {
        /// Whether to include statistics for each channel
        pub include_stats: bool,
        /// Whether to show channel labels
        pub show_labels: bool,
        /// Custom channel labels (if None, defaults to "Channel 1", etc.)
        pub channel_labels: Option<Vec<String>>,
        /// Width of each channel visualization
        pub width: usize,
        /// Height of each channel visualization
        pub height: usize,
        /// Style to use for visualization
        pub style: waveform::VisualizationStyle,
    }

    impl Default for MultichannelOptions {
        fn default() -> Self {
            Self {
                include_stats: true,
                show_labels: true,
                channel_labels: None,
                width: 80,
                height: 7,
                style: waveform::VisualizationStyle::Detailed,
            }
        }
    }

    impl MultichannelOptions {
        /// Create a new set of options with default values
        pub fn new() -> Self {
            Self::default()
        }

        /// Set whether to include statistics
        pub fn include_stats(mut self, include: bool) -> Self {
            self.include_stats = include;
            self
        }

        /// Set whether to show channel labels
        pub fn show_labels(mut self, show: bool) -> Self {
            self.show_labels = show;
            self
        }

        /// Set custom channel labels
        pub fn channel_labels(mut self, labels: Vec<String>) -> Self {
            self.channel_labels = Some(labels);
            self
        }

        /// Set the width of each channel visualization
        pub fn width(mut self, width: usize) -> Self {
            self.width = width;
            self
        }

        /// Set the height of each channel visualization
        pub fn height(mut self, height: usize) -> Self {
            self.height = height;
            self
        }

        /// Set the style of visualization
        pub fn style(mut self, style: waveform::VisualizationStyle) -> Self {
            self.style = style;
            self
        }
    }

    /// Format interleaved multi-channel audio
    pub fn format_interleaved<T>(
        samples: &[T],
        channels: usize,
        options: &MultichannelOptions,
    ) -> String
    where
        T: AudioSample + Display,
        f64: ConvertTo<T>,
    {
        if samples.is_empty() || channels == 0 {
            return String::from("[empty samples]");
        }

        // Check if we can divide samples into channels
        let samples_per_channel = samples.len() / channels;
        if samples.len() % channels != 0 {
            return format!(
                "[warning: sample count {} is not divisible by channel count {}]",
                samples.len(),
                channels
            );
        }

        let mut result = String::new();
        result.push_str(&format!(
            "Multi-channel Audio: {} channels, {} frames\n\n",
            channels, samples_per_channel
        ));

        // Process each channel
        for ch in 0..channels {
            // Get channel label
            let channel_label = if let Some(labels) = &options.channel_labels {
                if ch < labels.len() {
                    labels[ch].clone()
                } else {
                    format!("Channel {}", ch + 1)
                }
            } else {
                format!("Channel {}", ch + 1)
            };

            // Extract samples for this channel
            let channel_samples: Vec<T> = (0..samples_per_channel)
                .map(|i| samples[ch + i * channels])
                .collect();

            // Add channel header if needed
            if options.show_labels {
                result.push_str(&format!("{}:\n", channel_label));
            }

            // Add waveform
            let waveform_options = waveform::WaveformOptions {
                width: options.width,
                height: options.height,
                style: options.style,
                ..Default::default()
            };

            result.push_str(&waveform::format_with_options(
                &channel_samples,
                &waveform_options,
            ));

            // Add channel statistics if needed
            if options.include_stats {
                let stats = calculate_statistics(&channel_samples, None);
                result.push_str(&format!(
                    "\nMin: {:.6}, Max: {:.6}, RMS: {:.6}\n",
                    stats.min, stats.max, stats.rms
                ));
            }

            result.push_str("\n\n");
        }

        result
    }

    /// Extract a single channel from interleaved multi-channel audio
    pub fn extract_channel<T>(samples: &[T], channel: usize, total_channels: usize) -> Vec<T>
    where
        T: AudioSample + Copy,
    {
        if samples.is_empty() || total_channels == 0 {
            return Vec::new();
        }

        if channel >= total_channels {
            return Vec::new();
        }

        let samples_per_channel = samples.len() / total_channels;
        let mut channel_samples = Vec::with_capacity(samples_per_channel);

        for i in 0..samples_per_channel {
            channel_samples.push(samples[channel + i * total_channels]);
        }

        channel_samples
    }

    #[cfg(feature = "ndarray")]
    /// Format audio from ndarray representation
    pub fn format_from_ndarray<T>(
        array: &ndarray::Array2<T>,
        options: &MultichannelOptions,
    ) -> String
    where
        T: AudioSample + Display,
        i16: ConvertTo<T>,
        crate::I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>
    {
        let channels = array.nrows();
        let frames = array.ncols();

        let mut result = String::new();
        result.push_str(&format!(
            "Audio Data: {} channels, {} frames\n\n",
            channels, frames
        ));

        // Process each channel
        for ch in 0..channels {
            // Get channel label
            let channel_label = if let Some(labels) = &options.channel_labels {
                if ch < labels.len() {
                    labels[ch].clone()
                } else {
                    format!("Channel {}", ch + 1)
                }
            } else {
                format!("Channel {}", ch + 1)
            };

            // Extract channel data
            let channel_data: Vec<T> = array.row(ch).to_vec();

            // Add channel header if needed
            if options.show_labels {
                result.push_str(&format!("{}:\n", channel_label));
            }

            // Add waveform
            let waveform_options = waveform::WaveformOptions {
                width: options.width,
                height: options.height,
                style: options.style,
                ..Default::default()
            };

            result.push_str(&waveform::format_with_options(
                &channel_data,
                &waveform_options,
            ));


            if options.include_stats {
                let stats = calculate_statistics(&channel_data, None);
                let stats_min: f64 = stats.min.convert_to();
                let stats_max: f64 = stats.max.convert_to();
                let stats_rms: f64 = stats.rms.convert_to();
                result.push_str(&format!(
                    "\nMin: {:.6}, Max: {:.6}, RMS: {:.6}\n",
                    stats_min, stats_max, stats_rms
                ));
            }

            result.push_str("\n\n");
        }

        result
    }
}

/// Spectral analysis utilities
pub mod spectral {
    use super::*;

    /// Options for spectral visualization
    #[derive(Debug, Clone)]
    pub struct SpectralOptions {
        /// Width of the visualization
        pub width: usize,
        /// Height of the visualization
        pub height: usize,
        /// Sample rate for frequency calculations
        pub sample_rate: u32,
        /// Whether to use dB scale
        pub use_db_scale: bool,
        /// Minimum dB value to display (if using dB scale)
        pub min_db: f64,
        /// FFT size to use
        pub fft_size: usize,
    }

    impl Default for SpectralOptions {
        fn default() -> Self {
            Self {
                width: 80,
                height: 20,
                sample_rate: 44100,
                use_db_scale: true,
                min_db: -60.0,
                fft_size: 1024,
            }
        }
    }

    impl SpectralOptions {
        /// Create a new set of options with default values
        pub fn new() -> Self {
            Self::default()
        }

        /// Set the width of the visualization
        pub fn width(mut self, width: usize) -> Self {
            self.width = width;
            self
        }

        /// Set the height of the visualization
        pub fn height(mut self, height: usize) -> Self {
            self.height = height;
            self
        }

        /// Set the sample rate for frequency calculations
        pub fn sample_rate(mut self, rate: u32) -> Self {
            self.sample_rate = rate;
            self
        }

        /// Set whether to use dB scale
        pub fn use_db_scale(mut self, use_db: bool) -> Self {
            self.use_db_scale = use_db;
            self
        }

        /// Set the minimum dB value to display
        pub fn min_db(mut self, db: f64) -> Self {
            self.min_db = db;
            self
        }

        /// Set the FFT size to use
        pub fn fft_size(mut self, size: usize) -> Self {
            self.fft_size = size;
            self
        }
    }

    /// Format a simple spectral visualization using pseudo-FFT
    ///
    /// Note: This is a very basic implementation for visualization purposes only.
    /// For real spectral analysis, use a dedicated DSP library.
    pub fn format_spectrum<T>(samples: &[T], options: &SpectralOptions) -> String
    where
        T: AudioSample,
    {
        if samples.is_empty() {
            return String::from("[empty samples]");
        }

        // This is a placeholder for a real FFT implementation
        // In a real implementation, you would use a proper FFT library

        let mut result = String::new();
        result.push_str("Frequency Spectrum (Placeholder):\n\n");

        // For now, just create a fake spectrum visualization
        // In a real implementation, you would process the FFT results

        let height = options.height;
        let width = options.width;

        // Create a fake spectrum with some peaks
        let mut spectrum = vec![vec![' '; width]; height];

        // Draw frequency axis
        for x in 0..width {
            spectrum[height - 1][x] = '-';
        }

        // Label frequencies
        let freq_labels = [
            (0.05, "50Hz"),
            (0.1, "100Hz"),
            (0.2, "1kHz"),
            (0.5, "5kHz"),
            (0.9, "15kHz"),
        ];

        for (pos, label) in freq_labels.iter() {
            let x = (pos * width as f64) as usize;
            if x < width {
                spectrum[height - 1][x] = '|';

                if x + label.len() < width {
                    for (i, c) in label.chars().enumerate() {
                        if x + i < width {
                            spectrum[height - 2][x + i] = c;
                        }
                    }
                }
            }
        }

        // Create a simple "placeholder" visualization
        // This would be replaced with actual FFT results

        // Draw some fake spectrum peaks
        let peaks = [
            (0.05, 0.7, "Low"),
            (0.2, 0.9, "Mid"),
            (0.5, 0.5, "High"),
            (0.8, 0.3, "Ultrasonic"),
        ];

        for (pos, amplitude, _) in peaks.iter() {
            let x = (pos * width as f64) as usize;
            let peak_height = (amplitude * (height - 3) as f64) as usize;

            for y in ((height - 3) - peak_height)..(height - 3) {
                if x < width && y < height {
                    spectrum[y][x] = '█';
                }
            }
        }

        // Assemble the visualization
        for row in spectrum {
            result.push_str(&row.iter().collect::<String>());
            result.push('\n');
        }

        // Add a note that this is a placeholder
        result.push_str("\nNote: This is a simplified visualization. For actual spectral analysis, use a dedicated DSP library.");

        result
    }
}


pub fn calculate_statistics<T>(
    samples: &[T],
    sample_rate: Option<u32>,
) -> statistics::SampleStatistics<T>
where
    T: AudioSample + ConvertTo<f64>,
    f64: ConvertTo<T>,
{
    if samples.is_empty() {
        return statistics::SampleStatistics {
            min: T::default(),
            max: T::default(),
            avg: T::default(),
            rms: T::default(),
            peak_to_peak: T::default(),
            len: 0,
            duration_secs: None,
        };
    }

    let mut min: f64 = samples[0].convert_to();
    let mut max: f64 = samples[0].convert_to();
    let mut sum: f64 = 0.0;
    let mut sum_squared: f64 = f64::default();

    for &sample in samples.iter() {
        let sample: f64 = sample.convert_to();

        // Update min/max
        if sample < min {
            min = sample;
        }
        if sample > max {
            max = sample;
        }

        // Update sum for average
        // We need to convert back from f64 to T, so we need another bound
        sum = sum + sample;

        // Update sum squared for RMS
        sum_squared = sum_squared + sample * sample;
    }

    // Calculate average
    let avg: f64 = sum / samples.len() as f64;

    // Calculate RMS (Root Mean Square)
    let rms: f64 = sum_squared / (samples.len() as f64);
    let rms = (rms).sqrt();

    // Calculate peak-to-peak amplitude
    let peak_to_peak = (max - min).abs();

    // Calculate duration if sample rate is provided
    let duration_secs = sample_rate.map(|rate| samples.len() as f64 / rate as f64);

    // Convert results back to T

    let min = min.convert_to();
    let max = max.convert_to();
    let avg = avg.convert_to();
    let rms = rms.convert_to();
    let peak_to_peak = peak_to_peak.convert_to();

    statistics::SampleStatistics {
        min,
        max,
        avg,
        rms,
        peak_to_peak,
        len: samples.len(),
        duration_secs,
    }
}

/// Format samples as a string with the specified options
pub fn format_samples<T>(samples: &[T], options: &waveform::WaveformOptions) -> String
where
    T: AudioSample,
{
    waveform::format_with_options(samples, options)
}

// Extension methods for the Samples struct
impl<T> Samples<T>
where
    T: AudioSample + Display,
    i16: ConvertTo<T>,
    crate::I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>
{
    /// Returns statistics about the samples including min, max, and average amplitude
    pub fn statistics(&self, sample_rate: Option<u32>) -> statistics::SampleStatistics<T> {
        calculate_statistics(&self.samples, sample_rate)
    }

    /// Formats the samples as an ASCII waveform with a specified width and height
    pub fn format_waveform(&self, width: usize, height: usize) -> String {
        waveform::format_detailed(&self.samples, width, height)
    }

    /// Formats a compact visualization of the samples
    pub fn format_compact(&self, width: usize) -> String {
        waveform::format_compact(&self.samples, width)
    }

    /// Formats a visualization with the specified options
    pub fn format_with_options(&self, options: &waveform::WaveformOptions) -> String {
        waveform::format_with_options(&self.samples, options)
    }

    /// Formats a multi-channel waveform visualization for interleaved samples
    pub fn format_multichannel(
        &self,
        channels: usize,
        options: &multichannel::MultichannelOptions,
    ) -> String {
        multichannel::format_interleaved(&self.samples, channels, options)
    }

    /// Extracts a single channel from interleaved multi-channel samples
    pub fn extract_channel(&self, channel: usize, total_channels: usize) -> Samples<T>
    where
        T: Copy,
    {
        let channel_samples = multichannel::extract_channel(&self.samples, channel, total_channels);
        Samples::from(channel_samples)
    }

    /// Formats a spectral visualization of the samples
    pub fn format_spectrum(&self, options: &spectral::SpectralOptions) -> String {
        spectral::format_spectrum(&self.samples, options)
    }

    #[cfg(feature = "ndarray")]
    /// Formats a visualization using ndarray representation
    pub fn format_from_ndarray(
        &self,
        options: &multichannel::MultichannelOptions,
    ) -> crate::error::AudioSampleResult<String> {
        use crate::error::AudioSampleResult;

        let array = self.clone().into_ndarray(2)?;
        Ok(multichannel::format_from_ndarray(&array, options))
    }
    
}

// Implement custom Display for Samples when visualization is enabled
impl<T> Display for Samples<T>
where
T: AudioSample + Display,
i16: ConvertTo<T>,
crate::I24: ConvertTo<T>,
i32: ConvertTo<T>,
f32: ConvertTo<T>,
f64: ConvertTo<T>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.samples.is_empty() {
            return write!(f, "[empty samples]");
        }

        // Get statistics
        let stats = self.statistics(None);

        // Format sample type and length
        writeln!(
            f,
            "Samples<{}>: {} samples",
            type_name::<T>(),
            self.samples.len()
        )?;

        // Format statistics
        writeln!(f, "{}", stats)?;

        // Format waveform visualization (adjust width and height as needed)
        let width = 80; // Terminal width
        let height = 11; // Waveform height

        writeln!(f, "\nWaveform:")?;
        writeln!(f, "{}", self.format_waveform(width, height))?;

        // Display a preview of the beginning and end
        let preview_len = 5.min(self.samples.len());

        if preview_len > 0 {
            write!(f, "\nPreview: [")?;
            for i in 0..preview_len {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", self.samples[i])?;
            }

            if self.samples.len() > 2 * preview_len {
                write!(f, ", ... ")?;

                for i in (self.samples.len() - preview_len)..self.samples.len() {
                    write!(f, ", {}", self.samples[i])?;
                }
            }
            write!(f, "]")?;
        }

        Ok(())
    }
}

/// Advanced audio file inspection utilities
pub mod inspect {
    use super::*;

    /// Options for audio file inspection
    #[derive(Debug, Clone)]
    pub struct InspectionOptions {
        /// Whether to include statistics
        pub include_stats: bool,
        /// Whether to include waveform visualization
        pub include_waveform: bool,
        /// Whether to include spectral visualization
        pub include_spectrum: bool,
        /// Width for visualizations
        pub width: usize,
        /// Height for waveform visualization
        pub height: usize,
        /// Sample rate (if known)
        pub sample_rate: Option<u32>,
        /// Number of channels
        pub channels: Option<u16>,
    }

    impl Default for InspectionOptions {
        fn default() -> Self {
            Self {
                include_stats: true,
                include_waveform: true,
                include_spectrum: false,
                width: 80,
                height: 11,
                sample_rate: None,
                channels: None,
            }
        }
    }

    impl InspectionOptions {
        /// Create a new set of options with default values
        pub fn new() -> Self {
            Self::default()
        }

        /// Set whether to include statistics
        pub fn include_stats(mut self, include: bool) -> Self {
            self.include_stats = include;
            self
        }

        /// Set whether to include waveform visualization
        pub fn include_waveform(mut self, include: bool) -> Self {
            self.include_waveform = include;
            self
        }

        /// Set whether to include spectral visualization
        pub fn include_spectrum(mut self, include: bool) -> Self {
            self.include_spectrum = include;
            self
        }

        /// Set the width for visualizations
        pub fn width(mut self, width: usize) -> Self {
            self.width = width;
            self
        }

        /// Set the height for waveform visualization
        pub fn height(mut self, height: usize) -> Self {
            self.height = height;
            self
        }

        /// Set the sample rate
        pub fn sample_rate(mut self, rate: u32) -> Self {
            self.sample_rate = Some(rate);
            self
        }

        /// Set the number of channels
        pub fn channels(mut self, channels: u16) -> Self {
            self.channels = Some(channels);
            self
        }
    }

    /// Format a comprehensive inspection report for audio samples
    pub fn format_inspection<T>(samples: &Samples<T>, options: &InspectionOptions) -> String
    where
        T: AudioSample + Display,
        i16: ConvertTo<T>,
        crate::I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>,
    {
        let mut result = String::new();

        // Format header
        result.push_str(&format!("Audio Inspection Report\n"));
        result.push_str(&format!("=====================\n\n"));

        // Format basic information
        result.push_str(&format!("Sample type: {}\n", type_name::<T>()));
        result.push_str(&format!("Sample count: {}\n", samples.len()));

        if let Some(rate) = options.sample_rate {
            result.push_str(&format!("Sample rate: {} Hz\n", rate));

            let duration = samples.len() as f64 / rate as f64;
            let minutes = (duration / 60.0).floor();
            let seconds = duration % 60.0;
            result.push_str(&format!("Duration: {:.0}:{:06.3}\n", minutes, seconds));
        }

        if let Some(channels) = options.channels {
            result.push_str(&format!("Channels: {}\n", channels));

            if samples.len() % channels as usize != 0 {
                result.push_str(&format!(
                    "Warning: Sample count is not divisible by channel count\n"
                ));
            }
        }

        result.push_str("\n");

        // Format statistics if requested
        if options.include_stats {
            let stats = samples.statistics(options.sample_rate);
            result.push_str(&format!("{}\n\n", stats));
        }

        // Format waveform if requested
        if options.include_waveform {
            result.push_str("Waveform:\n");

            if let Some(channels) = options.channels {
                if channels > 1 {
                    // Use multi-channel visualization
                    let mc_options = multichannel::MultichannelOptions {
                        width: options.width,
                        height: options.height / channels as usize,
                        ..Default::default()
                    };

                    result.push_str(&samples.format_multichannel(channels as usize, &mc_options));
                } else {
                    // Use single-channel visualization
                    let wf_options = waveform::WaveformOptions {
                        width: options.width,
                        height: options.height,
                        ..Default::default()
                    };

                    result.push_str(&samples.format_with_options(&wf_options));
                }
            } else {
                // Use single-channel visualization by default
                let wf_options = waveform::WaveformOptions {
                    width: options.width,
                    height: options.height,
                    ..Default::default()
                };

                result.push_str(&samples.format_with_options(&wf_options));
            }

            result.push_str("\n\n");
        }

        // Format spectrum if requested
        if options.include_spectrum {
            result.push_str("Spectrum:\n");

            let spec_options = spectral::SpectralOptions {
                width: options.width,
                height: options.height,
                sample_rate: options.sample_rate.unwrap_or(44100),
                ..Default::default()
            };

            result.push_str(&samples.format_spectrum(&spec_options));
            result.push_str("\n\n");
        }

        result
    }
}

/// Command-line utilities for audio file inspection
#[cfg(feature = "cli")]
pub mod cli {
    use super::*;
    use std::path::Path;

    /// Represents a CLI command for audio visualization
    #[derive(Debug, Clone)]
    pub enum Command {
        /// Display waveform
        Waveform {
            width: usize,
            height: usize,
            style: waveform::VisualizationStyle,
        },
        /// Display detailed statistics
        Stats,
        /// Display a spectrum
        Spectrum { width: usize, height: usize },
        /// Display a comprehensive inspection report
        Inspect { options: inspect::InspectionOptions },
        /// Extract a channel from multi-channel audio
        ExtractChannel {
            channel: usize,
            total_channels: usize,
        },
    }

    impl Command {
        /// Execute the command on the provided samples
        pub fn execute<T>(&self, samples: &Samples<T>) -> String
        where
            T: AudioSample + Display,
            f64: ConvertTo<T>,
        {
            match self {
                Command::Waveform {
                    width,
                    height,
                    style,
                } => {
                    let options = waveform::WaveformOptions {
                        width: *width,
                        height: *height,
                        style: *style,
                        ..Default::default()
                    };

                    samples.format_with_options(&options)
                }
                Command::Stats => {
                    format!("{}", samples.statistics(None))
                }
                Command::Spectrum { width, height } => {
                    let options = spectral::SpectralOptions {
                        width: *width,
                        height: *height,
                        ..Default::default()
                    };

                    samples.format_spectrum(&options)
                }
                Command::Inspect { options } => inspect::format_inspection(samples, options),
                Command::ExtractChannel {
                    channel,
                    total_channels,
                } => {
                    let extracted = samples.extract_channel(*channel, *total_channels);
                    format!("{}", extracted)
                }
            }
        }
    }

    /// Parse command-line arguments into a Command
    pub fn parse_args(args: &[String]) -> Result<Command, String> {
        if args.is_empty() {
            return Err("No command specified".to_string());
        }

        match args[0].as_str() {
            "waveform" => {
                let mut width = 80;
                let mut height = 11;
                let mut style = waveform::VisualizationStyle::Detailed;

                let mut i = 1;
                while i < args.len() {
                    match args[i].as_str() {
                        "--width" | "-w" => {
                            if i + 1 < args.len() {
                                width = args[i + 1]
                                    .parse()
                                    .map_err(|_| "Invalid width value".to_string())?;
                                i += 2;
                            } else {
                                return Err("Missing value for width".to_string());
                            }
                        }
                        "--height" | "-h" => {
                            if i + 1 < args.len() {
                                height = args[i + 1]
                                    .parse()
                                    .map_err(|_| "Invalid height value".to_string())?;
                                i += 2;
                            } else {
                                return Err("Missing value for height".to_string());
                            }
                        }
                        "--style" | "-s" => {
                            if i + 1 < args.len() {
                                style = match args[i + 1].to_lowercase().as_str() {
                                    "detailed" => waveform::VisualizationStyle::Detailed,
                                    "compact" => waveform::VisualizationStyle::Compact,
                                    "dots" => waveform::VisualizationStyle::Dots,
                                    "ascii" => waveform::VisualizationStyle::Ascii,
                                    _ => return Err("Invalid style value".to_string()),
                                };
                                i += 2;
                            } else {
                                return Err("Missing value for style".to_string());
                            }
                        }
                        _ => {
                            return Err(format!("Unknown option: {}", args[i]));
                        }
                    }
                }

                Ok(Command::Waveform {
                    width,
                    height,
                    style,
                })
            }
            "stats" => Ok(Command::Stats),
            "spectrum" => {
                let mut width = 80;
                let mut height = 20;

                let mut i = 1;
                while i < args.len() {
                    match args[i].as_str() {
                        "--width" | "-w" => {
                            if i + 1 < args.len() {
                                width = args[i + 1]
                                    .parse()
                                    .map_err(|_| "Invalid width value".to_string())?;
                                i += 2;
                            } else {
                                return Err("Missing value for width".to_string());
                            }
                        }
                        "--height" | "-h" => {
                            if i + 1 < args.len() {
                                height = args[i + 1]
                                    .parse()
                                    .map_err(|_| "Invalid height value".to_string())?;
                                i += 2;
                            } else {
                                return Err("Missing value for height".to_string());
                            }
                        }
                        _ => {
                            return Err(format!("Unknown option: {}", args[i]));
                        }
                    }
                }

                Ok(Command::Spectrum { width, height })
            }
            "inspect" => {
                let mut options = inspect::InspectionOptions::default();

                let mut i = 1;
                while i < args.len() {
                    match args[i].as_str() {
                        "--width" | "-w" => {
                            if i + 1 < args.len() {
                                options.width = args[i + 1]
                                    .parse()
                                    .map_err(|_| "Invalid width value".to_string())?;
                                i += 2;
                            } else {
                                return Err("Missing value for width".to_string());
                            }
                        }
                        "--height" | "-h" => {
                            if i + 1 < args.len() {
                                options.height = args[i + 1]
                                    .parse()
                                    .map_err(|_| "Invalid height value".to_string())?;
                                i += 2;
                            } else {
                                return Err("Missing value for height".to_string());
                            }
                        }
                        "--no-stats" => {
                            options.include_stats = false;
                            i += 1;
                        }
                        "--no-waveform" => {
                            options.include_waveform = false;
                            i += 1;
                        }
                        "--spectrum" => {
                            options.include_spectrum = true;
                            i += 1;
                        }
                        "--sample-rate" | "-sr" => {
                            if i + 1 < args.len() {
                                options.sample_rate = Some(
                                    args[i + 1]
                                        .parse()
                                        .map_err(|_| "Invalid sample rate value".to_string())?,
                                );
                                i += 2;
                            } else {
                                return Err("Missing value for sample rate".to_string());
                            }
                        }
                        "--channels" | "-c" => {
                            if i + 1 < args.len() {
                                options.channels = Some(
                                    args[i + 1]
                                        .parse()
                                        .map_err(|_| "Invalid channels value".to_string())?,
                                );
                                i += 2;
                            } else {
                                return Err("Missing value for channels".to_string());
                            }
                        }
                        _ => {
                            return Err(format!("Unknown option: {}", args[i]));
                        }
                    }
                }

                Ok(Command::Inspect { options })
            }
            "extract-channel" => {
                let mut channel = 0;
                let mut total_channels = 2;

                let mut i = 1;
                while i < args.len() {
                    match args[i].as_str() {
                        "--channel" | "-c" => {
                            if i + 1 < args.len() {
                                channel = args[i + 1]
                                    .parse()
                                    .map_err(|_| "Invalid channel value".to_string())?;
                                i += 2;
                            } else {
                                return Err("Missing value for channel".to_string());
                            }
                        }
                        "--total" | "-t" => {
                            if i + 1 < args.len() {
                                total_channels = args[i + 1]
                                    .parse()
                                    .map_err(|_| "Invalid total channels value".to_string())?;
                                i += 2;
                            } else {
                                return Err("Missing value for total channels".to_string());
                            }
                        }
                        _ => {
                            return Err(format!("Unknown option: {}", args[i]));
                        }
                    }
                }

                Ok(Command::ExtractChannel {
                    channel,
                    total_channels,
                })
            }
            _ => Err(format!("Unknown command: {}", args[0])),
        }
    }

    /// Run the specified command on a file
    pub fn run_on_file<P, T>(
        path: P,
        command: &Command,
        sample_converter: fn(&[u8]) -> Samples<T>,
    ) -> Result<String, String>
    where
        P: AsRef<Path>,
        T: AudioSample + Display,
        f64: ConvertTo<T>,
    {
        // Read the file
        let bytes = std::fs::read(path).map_err(|e| format!("Failed to read file: {}", e))?;

        // Convert the bytes to samples
        let samples = sample_converter(&bytes);

        // Execute the command
        Ok(command.execute(&samples))
    }
}

/// Module for terminal coloring and styling
pub mod terminal {
    use super::*;

    /// ANSI color codes for terminal output
    pub mod color {
        pub const RESET: &str = "\x1b[0m";
        pub const BLACK: &str = "\x1b[30m";
        pub const RED: &str = "\x1b[31m";
        pub const GREEN: &str = "\x1b[32m";
        pub const YELLOW: &str = "\x1b[33m";
        pub const BLUE: &str = "\x1b[34m";
        pub const MAGENTA: &str = "\x1b[35m";
        pub const CYAN: &str = "\x1b[36m";
        pub const WHITE: &str = "\x1b[37m";

        pub const BG_BLACK: &str = "\x1b[40m";
        pub const BG_RED: &str = "\x1b[41m";
        pub const BG_GREEN: &str = "\x1b[42m";
        pub const BG_YELLOW: &str = "\x1b[43m";
        pub const BG_BLUE: &str = "\x1b[44m";
        pub const BG_MAGENTA: &str = "\x1b[45m";
        pub const BG_CYAN: &str = "\x1b[46m";
        pub const BG_WHITE: &str = "\x1b[47m";

        pub const BOLD: &str = "\x1b[1m";
        pub const UNDERLINE: &str = "\x1b[4m";
        pub const BLINK: &str = "\x1b[5m";
        pub const REVERSE: &str = "\x1b[7m";
    }

    /// Options for terminal visualization
    #[derive(Debug, Clone)]
    pub struct TerminalOptions {
        /// Whether to use colors
        pub use_colors: bool,
        /// Whether terminal supports Unicode
        pub unicode_support: bool,
        /// Terminal width
        pub width: usize,
        /// Terminal height
        pub height: usize,
    }

    impl Default for TerminalOptions {
        fn default() -> Self {
            Self {
                use_colors: true,
                unicode_support: true,
                width: 80,
                height: 24,
            }
        }
    }

    impl TerminalOptions {
        /// Create a new set of options with default values
        pub fn new() -> Self {
            Self::default()
        }

        /// Set whether to use colors
        pub fn use_colors(mut self, use_colors: bool) -> Self {
            self.use_colors = use_colors;
            self
        }

        /// Set whether terminal supports Unicode
        pub fn unicode_support(mut self, unicode_support: bool) -> Self {
            self.unicode_support = unicode_support;
            self
        }

        /// Set terminal width
        pub fn width(mut self, width: usize) -> Self {
            self.width = width;
            self
        }

        /// Set terminal height
        pub fn height(mut self, height: usize) -> Self {
            self.height = height;
            self
        }
    }

    /// Format a colored waveform for the terminal
    pub fn format_colored_waveform<T>(samples: &[T], options: &TerminalOptions) -> String
    where
        T: AudioSample,
    {
        if samples.is_empty() {
            return String::from("[empty samples]");
        }

        // Fall back to ASCII if Unicode is not supported
        let waveform_style = if options.unicode_support {
            waveform::VisualizationStyle::Detailed
        } else {
            waveform::VisualizationStyle::Ascii
        };

        let waveform_options = waveform::WaveformOptions {
            width: options.width,
            height: options.height.min(20),
            style: waveform_style,
            ..Default::default()
        };

        let mut waveform = waveform::format_with_options(samples, &waveform_options);

        // Add colors if requested
        if options.use_colors {
            // Convert all samples to f64 for analysis
            let f64_samples: Vec<f64> = samples.iter().map(|s| s.convert_to()).collect();

            // Find max amplitude for color scaling
            let max_amplitude = f64_samples
                .iter()
                .cloned()
                .map(f64::abs)
                .fold(0.0, f64::max);

            // Add color codes - replacing vertical bars with colored versions
            if options.unicode_support {
                waveform = waveform.replace("│", &format!("{}│{}", color::CYAN, color::RESET));
            } else {
                waveform = waveform.replace("|", &format!("{}|{}", color::CYAN, color::RESET));
            }

            // Color the center line
            if options.unicode_support {
                waveform = waveform.replace("─", &format!("{}─{}", color::YELLOW, color::RESET));
            } else {
                waveform = waveform.replace("-", &format!("{}-{}", color::YELLOW, color::RESET));
            }
        }

        waveform
    }
}

#[cfg(feature = "visualization")]
fn generate_sine_wave(freq: f64, sample_rate: u32, num_samples: usize) -> Samples<f32> {
    use std::f64::consts::PI;

    let mut samples = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let t = i as f64 / sample_rate as f64;
        let amplitude = (2.0 * PI * freq * t).sin() as f32;
        samples.push(amplitude);
    }

    Samples::from(samples)
}

#[cfg(feature = "visualization")]
fn generate_stereo_audio(
    freq_left: f64,
    freq_right: f64,
    sample_rate: u32,
    num_frames: usize,
) -> Samples<f32> {
    use std::f64::consts::PI;

    let mut samples = Vec::with_capacity(num_frames * 2);

    for i in 0..num_frames {
        let t = i as f64 / sample_rate as f64;

        // Left channel
        let left = (2.0 * PI * freq_left * t).sin() as f32;
        samples.push(left);

        // Right channel
        let right = (2.0 * PI * freq_right * t).sin() as f32;
        samples.push(right);
    }

    Samples::from(samples)
}

// Example of a command-line application using the visualization feature
#[cfg(feature = "visualization")]
#[cfg(feature = "cli")]
mod cli_app {
    use crate::visualization::cli;
    use crate::{AudioSample, Samples};
    use std::env;

    pub fn main() {
        let args: Vec<String> = env::args().collect();

        if args.len() < 3 {
            println!("Usage: {} <command> <file> [options]", args[0]);
            println!("\nCommands:");
            println!("  waveform      Display a waveform visualization");
            println!("  stats         Display statistics about the audio");
            println!("  spectrum      Display a spectrum visualization");
            println!("  inspect       Perform a comprehensive inspection");
            println!("  extract-channel Extract a channel from multi-channel audio");
            return;
        }

        // Parse command
        let command = match cli::parse_args(&args[1..]) {
            Ok(cmd) => cmd,
            Err(e) => {
                eprintln!("Error: {}", e);
                return;
            }
        };

        // File path is the second argument
        let file_path = &args[2];

        // Run the command on the file
        // Note: This example assumes f32 samples, in a real application
        // you would determine the sample type from the file format
        let result = match cli::run_on_file(file_path, &command, |bytes| {
            // This is just a placeholder converter function
            // In a real application, you would parse the file format properly
            Samples::<f32>::from(bytes)
        }) {
            Ok(output) => output,
            Err(e) => {
                eprintln!("Error: {}", e);
                return;
            }
        };

        // Print the result
        println!("{}", result);
    }
}
