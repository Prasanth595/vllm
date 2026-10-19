// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use image::RgbImage;

struct Kernel {
    start: usize,
    weights: Vec<i32>,
}

fn cubic(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        ((1.5 * x - 2.5) * x) * x + 1.0
    } else if x < 2.0 {
        ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
    } else {
        0.0
    }
}

fn kernels(input: usize, output: usize) -> (Vec<Kernel>, u32) {
    let scale = input as f64 / output as f64;
    let support = 2.0 * scale.max(1.0);
    let invscale = 1.0 / scale.max(1.0);
    let mut max_weight = 0.0_f64;
    let weights: Vec<_> = (0..output)
        .map(|i| {
            let center = scale * (i as f64 + 0.5);
            let start = ((center - support + 0.5) as isize).max(0) as usize;
            let end = ((center + support + 0.5) as usize).min(input);
            let mut weights: Vec<_> =
                (start..end).map(|j| cubic((j as f64 - center + 0.5) * invscale)).collect();
            let total: f64 = weights.iter().sum();
            for weight in &mut weights {
                *weight /= total;
                max_weight = max_weight.max(*weight);
            }
            (start, weights)
        })
        .collect();
    // Torch's uint8 antialiased resize quantizes weights to signed 16 bits.
    let precision = (0..22)
        .find(|p| (0.5 + max_weight * (1_u32 << (p + 1)) as f64) as i32 >= 32768)
        .unwrap_or(22);
    let kernels = weights
        .into_iter()
        .map(|(start, weights)| Kernel {
            start,
            weights: weights
                .into_iter()
                .map(|w| (w * (1_u32 << precision) as f64).round() as i32)
                .collect(),
        })
        .collect();
    (kernels, precision)
}

pub(super) fn bicubic(image: &RgbImage, width: u32, height: u32) -> RgbImage {
    let mut horizontal = image.clone();
    if width != image.width() {
        let (kernels, precision) = kernels(image.width() as usize, width as usize);
        horizontal = RgbImage::new(width, image.height());
        for y in 0..image.height() {
            for (x, kernel) in kernels.iter().enumerate() {
                for c in 0..3 {
                    let sum = kernel.weights.iter().enumerate().fold(
                        1_i32 << (precision - 1),
                        |sum, (j, weight)| {
                            sum + *weight * image.get_pixel((kernel.start + j) as u32, y)[c] as i32
                        },
                    );
                    horizontal.get_pixel_mut(x as u32, y)[c] =
                        (sum >> precision).clamp(0, 255) as u8;
                }
            }
        }
    }
    if height == image.height() {
        return horizontal;
    }
    let (kernels, precision) = kernels(image.height() as usize, height as usize);
    let mut output = RgbImage::new(width, height);
    for (y, kernel) in kernels.iter().enumerate() {
        for x in 0..width {
            for c in 0..3 {
                let sum = kernel.weights.iter().enumerate().fold(
                    1_i32 << (precision - 1),
                    |sum, (j, weight)| {
                        sum + *weight * horizontal.get_pixel(x, (kernel.start + j) as u32)[c] as i32
                    },
                );
                output.get_pixel_mut(x, y as u32)[c] = (sum >> precision).clamp(0, 255) as u8;
            }
        }
    }
    output
}
