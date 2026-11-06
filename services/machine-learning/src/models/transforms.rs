use std::io::Cursor;

use anyhow::{Context, Result};
use half::f16;
use image::{DynamicImage, ImageReader, RgbImage};
use ndarray::{Array3, ArrayD};
use opencv::core::{Mat, Size};
use opencv::imgproc;
use opencv::prelude::*;

use crate::pil::{self, Resampling};
use crate::pillow_jxl::{self, JxlImage};
use crate::sessions::ort::Tensor;

const DECODE_THREADS: isize = 0;

pub fn resize_pil(image: &RgbImage, size: u32, resample: Resampling) -> RgbImage {
    let (width, height) = image.dimensions();
    if width < height {
        pil::resize(
            image,
            size,
            ((f64::from(height) / f64::from(width)) * f64::from(size)) as u32,
            resample,
        )
    } else {
        pil::resize(
            image,
            ((f64::from(width) / f64::from(height)) * f64::from(size)) as u32,
            size,
            resample,
        )
    }
}

pub fn crop_pil(image: &RgbImage, size: u32) -> RgbImage {
    let left = (f64::from(image.width()) / 2.0 - f64::from(size) / 2.0) as i64;
    let upper = (f64::from(image.height()) / 2.0 - f64::from(size) / 2.0) as i64;
    pil::crop(
        image,
        left,
        upper,
        left + i64::from(size),
        upper + i64::from(size),
    )
}

pub fn to_numpy(image: &RgbImage) -> Array3<f32> {
    let (width, height) = image.dimensions();
    Array3::from_shape_fn((height as usize, width as usize, 3), |(y, x, c)| {
        f32::from(image.get_pixel(x as u32, y as u32)[c]) / 255.0
    })
}

pub fn normalize(image: &mut Array3<f32>, mean: &[f32; 3], std: &[f32; 3]) {
    let scale = std.map(|value| 1.0f32 / value);
    let shift = [mean[0] / std[0], mean[1] / std[1], mean[2] / std[2]];
    for ((_, _, c), value) in image.indexed_iter_mut() {
        *value *= scale[c];
        *value -= shift[c];
    }
}

pub fn decode_pil(image_bytes: &[u8]) -> Result<RgbImage> {
    if pillow_jxl::accept(image_bytes) {
        return match pillow_jxl::open(image_bytes, DECODE_THREADS)? {
            JxlImage::Jpeg(jpeg) => decode_pil(&jpeg),
            JxlImage::Raw {
                mode,
                width,
                height,
                data,
            } => pil::frombytes_rgb(mode, width, height, &data),
        };
    }
    if image_bytes.starts_with(&[0xff, 0xd8, 0xff])
        && let Ok(decoded) = turbojpeg::decompress(image_bytes, turbojpeg::PixelFormat::RGB)
        && decoded.pitch == decoded.width * 3
        && let Some(image) =
            RgbImage::from_raw(decoded.width as u32, decoded.height as u32, decoded.pixels)
    {
        return Ok(image);
    }
    let reader = ImageReader::new(Cursor::new(image_bytes))
        .with_guessed_format()
        .context("cannot identify image file")?;
    let decoder = reader
        .into_decoder()
        .context("cannot identify image file")?;
    let image = DynamicImage::from_decoder(decoder).context("cannot identify image file")?;
    Ok(to_rgb(image))
}

fn to_rgb(image: DynamicImage) -> RgbImage {
    match image {
        DynamicImage::ImageRgb8(image) => image,
        other => other.to_rgb8(),
    }
}

pub fn is_py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

pub fn clean_text(text: &str, canonicalize: bool) -> String {
    let text = text
        .split(is_py_space)
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    if canonicalize {
        text.chars()
            .filter(|c| !c.is_ascii_punctuation())
            .collect::<String>()
            .to_lowercase()
    } else {
        text
    }
}

pub fn serialize_np_array(values: &[f32]) -> String {
    serde_json::to_string(values).expect("floats serialize")
}

pub fn widen(tensor: Tensor) -> Result<ArrayD<f32>> {
    match tensor {
        Tensor::F32(array) => Ok(array),
        Tensor::F16(array) => Ok(array.mapv(f16::to_f32)),
        Tensor::U8(array) => Ok(array.mapv(f32::from)),
        Tensor::I32(array) => Ok(array.mapv(|value| value as f32)),
        Tensor::I64(array) => Ok(array.mapv(|value| value as f32)),
    }
}

pub fn rgb_mat(image: &RgbImage) -> Result<Mat> {
    let (width, height) = image.dimensions();
    let flat = Mat::new_rows_cols_with_data(height as i32, (width * 3) as i32, image.as_raw())?;
    Ok(flat.reshape(3, height as i32)?.try_clone()?)
}

pub fn mat_rgb(mat: &Mat) -> Result<RgbImage> {
    let size = mat.size()?;
    let data = mat.data_bytes()?.to_vec();
    RgbImage::from_raw(size.width as u32, size.height as u32, data)
        .context("mat is not a packed RGB image")
}

pub fn cv_resize(image: &RgbImage, width: u32, height: u32) -> Result<RgbImage> {
    let source = rgb_mat(image)?;
    let mut resized = Mat::default();
    imgproc::resize(
        &source,
        &mut resized,
        Size::new(width as i32, height as i32),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;
    mat_rgb(&resized)
}

pub fn letterbox(image: &RgbImage, size: u32) -> Result<(RgbImage, f64)> {
    let (width, height) = image.dimensions();
    let (new_height, new_width) = if height > width {
        (
            size,
            (f64::from(size) * f64::from(width) / f64::from(height)) as u32,
        )
    } else {
        (
            (f64::from(size) * f64::from(height) / f64::from(width)) as u32,
            size,
        )
    };
    let resized = cv_resize(image, new_width, new_height)?;
    let mut canvas = RgbImage::new(size, size);
    image::imageops::replace(&mut canvas, &resized, 0, 0);
    Ok((canvas, f64::from(new_height) / f64::from(height)))
}
