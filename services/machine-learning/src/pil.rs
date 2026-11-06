use anyhow::{Context, Result, bail};
use image::RgbImage;

const PRECISION_BITS: i32 = 32 - 8 - 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Resampling {
    Nearest,
    Box,
    Bilinear,
    Hamming,
    Bicubic,
    Lanczos,
}

impl Resampling {
    pub fn parse(name: &str) -> Result<Self> {
        Ok(match name.to_lowercase().as_str() {
            "nearest" => Self::Nearest,
            "box" => Self::Box,
            "bilinear" => Self::Bilinear,
            "hamming" => Self::Hamming,
            "bicubic" => Self::Bicubic,
            "lanczos" => Self::Lanczos,
            other => bail!("{other:?}"),
        })
    }

    fn support(self) -> f64 {
        match self {
            Self::Nearest | Self::Box => 0.5,
            Self::Bilinear | Self::Hamming => 1.0,
            Self::Bicubic => 2.0,
            Self::Lanczos => 3.0,
        }
    }

    fn filter(self, x: f64) -> f64 {
        match self {
            Self::Nearest | Self::Box => {
                if x > -0.5 && x <= 0.5 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Bilinear => {
                let x = x.abs();
                if x < 1.0 { 1.0 - x } else { 0.0 }
            }
            Self::Hamming => {
                let x = x.abs();
                if x == 0.0 {
                    1.0
                } else if x >= 1.0 {
                    0.0
                } else {
                    let x = x * std::f64::consts::PI;
                    x.sin() / x * (0.54f32 as f64 + 0.46f32 as f64 * x.cos())
                }
            }
            Self::Bicubic => {
                let a = -0.5;
                let x = x.abs();
                if x < 1.0 {
                    ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
                } else if x < 2.0 {
                    (((x - 5.0) * x + 8.0) * x - 4.0) * a
                } else {
                    0.0
                }
            }
            Self::Lanczos => {
                if (-3.0..3.0).contains(&x) {
                    sinc(x) * sinc(x / 3.0)
                } else {
                    0.0
                }
            }
        }
    }
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let x = x * std::f64::consts::PI;
    x.sin() / x
}

struct Coefficients {
    ksize: usize,
    bounds: Vec<(usize, usize)>,
    weights: Vec<i32>,
}

fn precompute_coeffs(
    in_size: usize,
    in0: f32,
    in1: f32,
    out_size: usize,
    filter: Resampling,
) -> Coefficients {
    let scale = f64::from(in1 - in0) / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = filter.support() * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let mut bounds = Vec::with_capacity(out_size);
    let mut prekk = vec![0.0f64; out_size * ksize];
    let inv_filterscale = 1.0 / filterscale;
    for xx in 0..out_size {
        let center = f64::from(in0) + (xx as f64 + 0.5) * scale;
        let mut ww = 0.0;
        let xmin = ((center - support + 0.5) as i64).max(0);
        let xmax = ((center + support + 0.5) as i64).min(in_size as i64) - xmin;
        let k = &mut prekk[xx * ksize..(xx + 1) * ksize];
        for (x, slot) in k.iter_mut().enumerate().take(xmax.max(0) as usize) {
            let w = filter.filter(((x as i64 + xmin) as f64 - center + 0.5) * inv_filterscale);
            *slot = w;
            ww += w;
        }
        if ww != 0.0 {
            for value in k.iter_mut().take(xmax.max(0) as usize) {
                *value /= ww;
            }
        }
        bounds.push((xmin as usize, xmax.max(0) as usize));
    }
    let weights = prekk
        .iter()
        .map(|value| {
            if *value < 0.0 {
                (-0.5 + value * f64::from(1i32 << PRECISION_BITS)) as i32
            } else {
                (0.5 + value * f64::from(1i32 << PRECISION_BITS)) as i32
            }
        })
        .collect();
    Coefficients {
        ksize,
        bounds,
        weights,
    }
}

fn clip8(value: i32) -> u8 {
    (value >> PRECISION_BITS).clamp(0, 255) as u8
}

const ROUNDING: i32 = 1 << (PRECISION_BITS - 1);

fn horizontal(
    input: &[u8],
    in_width: usize,
    out_width: usize,
    rows: usize,
    offset: usize,
    coefficients: &Coefficients,
) -> Vec<u8> {
    let mut pixels = vec![0u8; out_width * rows * 3];
    let windows: Vec<(&[i32], usize, usize)> = coefficients
        .bounds
        .iter()
        .enumerate()
        .map(|(xx, (xmin, xmax))| {
            let start = xx * coefficients.ksize;
            (
                &coefficients.weights[start..start + xmax],
                xmin * 3,
                (xmin + xmax) * 3,
            )
        })
        .collect();
    for (yy, out_row) in pixels.chunks_exact_mut(out_width * 3).enumerate() {
        let line = &input[(yy + offset) * in_width * 3..(yy + offset + 1) * in_width * 3];
        for (out, (weights, first, last)) in out_row.as_chunks_mut::<3>().0.iter_mut().zip(&windows)
        {
            let mut ss = [ROUNDING; 3];
            for (pixel, weight) in line[*first..*last].as_chunks::<3>().0.iter().zip(*weights) {
                for band in 0..3 {
                    ss[band] = ss[band].wrapping_add(i32::from(pixel[band]).wrapping_mul(*weight));
                }
            }
            for band in 0..3 {
                out[band] = clip8(ss[band]);
            }
        }
    }
    pixels
}

fn vertical(input: &[u8], width: usize, out_height: usize, coefficients: &Coefficients) -> Vec<u8> {
    let stride = width * 3;
    let mut pixels = vec![0u8; stride * out_height];
    let mut sums = vec![0i32; stride];
    for (yy, out_row) in pixels.chunks_exact_mut(stride).enumerate() {
        let (ymin, ymax) = coefficients.bounds[yy];
        let start = yy * coefficients.ksize;
        sums.fill(ROUNDING);
        for (y, weight) in coefficients.weights[start..start + ymax].iter().enumerate() {
            let line = &input[(ymin + y) * stride..(ymin + y + 1) * stride];
            for (sum, pixel) in sums.iter_mut().zip(line) {
                *sum = sum.wrapping_add(i32::from(*pixel).wrapping_mul(*weight));
            }
        }
        for (out, sum) in out_row.iter_mut().zip(&sums) {
            *out = clip8(*sum);
        }
    }
    pixels
}

pub fn resize(image: &RgbImage, width: u32, height: u32, resample: Resampling) -> RgbImage {
    if image.dimensions() == (width, height) {
        return image.clone();
    }
    if width == 0 || height == 0 {
        return RgbImage::new(width, height);
    }
    if resample == Resampling::Nearest {
        return resize_nearest(image, width, height);
    }
    let (in_width, in_height) = (image.width() as usize, image.height() as usize);
    let (xsize, ysize) = (width as usize, height as usize);
    let need_horizontal = xsize != in_width;
    let need_vertical = ysize != in_height;
    let horizontal_first = !((in_height as i64 - ysize as i64) > 0
        && (in_height as i64 - ysize as i64) > (in_width as i64 - xsize as i64) * 2);

    let mut vertical_coefficients = None;
    if (need_horizontal && horizontal_first) || need_vertical {
        vertical_coefficients = Some(precompute_coeffs(
            in_height,
            0.0,
            in_height as f32,
            ysize,
            resample,
        ));
    }
    let mut ybox = (0, in_height);
    if need_horizontal && horizontal_first {
        let coefficients = vertical_coefficients.as_mut().expect("computed above");
        let first = coefficients.bounds[0].0;
        let last = coefficients.bounds[ysize - 1].0 + coefficients.bounds[ysize - 1].1;
        if first != 0 {
            for bound in &mut coefficients.bounds {
                bound.0 -= first;
            }
        }
        ybox = (first, last);
    }
    let horizontal_coefficients =
        need_horizontal.then(|| precompute_coeffs(in_width, 0.0, in_width as f32, xsize, resample));

    let source = image.as_raw();
    let pixels = if horizontal_first {
        let narrowed = horizontal_coefficients.as_ref().map(|coefficients| {
            horizontal(
                source,
                in_width,
                xsize,
                ybox.1 - ybox.0,
                ybox.0,
                coefficients,
            )
        });
        match (&narrowed, &vertical_coefficients) {
            (Some(narrowed), Some(coefficients)) if need_vertical => {
                vertical(narrowed, xsize, ysize, coefficients)
            }
            (None, Some(coefficients)) if need_vertical => {
                vertical(source, in_width, ysize, coefficients)
            }
            _ => narrowed.expect("one pass runs when the size changes"),
        }
    } else {
        let coefficients = vertical_coefficients.as_ref().expect("computed above");
        let shortened = vertical(source, in_width, ysize, coefficients);
        match &horizontal_coefficients {
            Some(coefficients) => horizontal(&shortened, in_width, xsize, ysize, 0, coefficients),
            None => shortened,
        }
    };
    RgbImage::from_raw(width, height, pixels).expect("sized buffer")
}

fn coord(value: f64) -> i64 {
    if value < 0.0 { -1 } else { value as i64 }
}

fn resize_nearest(image: &RgbImage, width: u32, height: u32) -> RgbImage {
    let xstep = f64::from(image.width() as f32) / f64::from(width);
    let ystep = f64::from(image.height() as f32) / f64::from(height);
    let mut columns = Vec::with_capacity(width as usize);
    let mut xo = 0.0 + xstep * 0.5;
    for _ in 0..width {
        let xin = coord(xo);
        columns.push(
            (0..i64::from(image.width()))
                .contains(&xin)
                .then_some(xin as usize),
        );
        xo += xstep;
    }
    let source = image.as_raw();
    let stride = image.width() as usize * 3;
    let mut pixels = vec![0u8; width as usize * height as usize * 3];
    let mut yo = 0.0 + ystep * 0.5;
    for row in pixels.chunks_exact_mut(width as usize * 3) {
        let yin = coord(yo);
        if (0..i64::from(image.height())).contains(&yin) {
            let line = source[yin as usize * stride..(yin as usize + 1) * stride]
                .as_chunks::<3>()
                .0;
            for (pixel, column) in row.as_chunks_mut::<3>().0.iter_mut().zip(&columns) {
                if let Some(column) = column {
                    *pixel = line[*column];
                }
            }
        }
        yo += ystep;
    }
    RgbImage::from_raw(width, height, pixels).expect("sized buffer")
}

pub fn crop(image: &RgbImage, left: i64, upper: i64, right: i64, lower: i64) -> RgbImage {
    let width = (right - left).max(0) as u32;
    let height = (lower - upper).max(0) as u32;
    RgbImage::from_fn(width, height, |x, y| {
        let sx = left + i64::from(x);
        let sy = upper + i64::from(y);
        if sx >= 0 && sy >= 0 && sx < i64::from(image.width()) && sy < i64::from(image.height()) {
            *image.get_pixel(sx as u32, sy as u32)
        } else {
            image::Rgb([0, 0, 0])
        }
    })
}

fn floor(value: f64) -> i64 {
    if value < 0.0 {
        value.floor() as i64
    } else {
        value as i64
    }
}

fn bicubic(v1: f64, v2: f64, v3: f64, v4: f64, d: f64) -> f64 {
    let p1 = v2;
    let p2 = -v1 + v3;
    let p3 = 2.0 * (v1 - v2) + v3 - v4;
    let p4 = -v1 + v2 - v3 + v4;
    p1 + d * (p2 + d * (p3 + d * p4))
}

struct Source<'a> {
    pixels: &'a [u8],
    width: i64,
    height: i64,
}

impl Source<'_> {
    fn bicubic_filter(&self, xin: f64, yin: f64) -> Option<[u8; 3]> {
        if xin < 0.0 || xin >= self.width as f64 || yin < 0.0 || yin >= self.height as f64 {
            return None;
        }
        let xin = xin - 0.5;
        let yin = yin - 0.5;
        let x = floor(xin);
        let y = floor(yin);
        let dx = xin - x as f64;
        let dy = yin - y as f64;
        let columns: [usize; 4] =
            std::array::from_fn(|i| (x - 1 + i as i64).clamp(0, self.width - 1) as usize * 3);
        let rows: [usize; 4] = std::array::from_fn(|i| {
            (y - 1 + i as i64).clamp(0, self.height - 1) as usize * self.width as usize * 3
        });
        let mut out = [0u8; 3];
        for (band, slot) in out.iter_mut().enumerate() {
            let [v1, v2, v3, v4] = rows.map(|row| {
                let line = &self.pixels[row + band..];
                bicubic(
                    f64::from(line[columns[0]]),
                    f64::from(line[columns[1]]),
                    f64::from(line[columns[2]]),
                    f64::from(line[columns[3]]),
                    dx,
                )
            });
            let value = bicubic(v1, v2, v3, v4, dy);
            *slot = if value <= 0.0 {
                0
            } else if value >= 255.0 {
                255
            } else {
                value as u8
            };
        }
        Some(out)
    }
}

pub fn transform_perspective_bicubic(
    image: &RgbImage,
    width: u32,
    height: u32,
    data: &[f64; 8],
) -> RgbImage {
    if width == 0 || height == 0 {
        return RgbImage::new(width, height);
    }
    let [a0, a1, a2, a3, a4, a5, a6, a7] = *data;
    let source = Source {
        pixels: image.as_raw(),
        width: i64::from(image.width()),
        height: i64::from(image.height()),
    };
    let mut pixels = vec![0u8; width as usize * height as usize * 3];
    for (y, row) in pixels.chunks_exact_mut(width as usize * 3).enumerate() {
        let yin = y as f64 + 0.5;
        for (x, pixel) in row.as_chunks_mut::<3>().0.iter_mut().enumerate() {
            let xin = x as f64 + 0.5;
            let xx = (a0 * xin + a1 * yin + a2) / (a6 * xin + a7 * yin + 1.0);
            let yy = (a3 * xin + a4 * yin + a5) / (a6 * xin + a7 * yin + 1.0);
            if let Some(value) = source.bicubic_filter(xx, yy) {
                *pixel = value;
            }
        }
    }
    RgbImage::from_raw(width, height, pixels).expect("sized buffer")
}

type PixelToRgb = fn(&[u8]) -> [u8; 3];

fn f2l(value: f32) -> u8 {
    if value <= 0.0 {
        0
    } else if value >= 255.0 {
        255
    } else {
        value as u8
    }
}

pub fn frombytes_rgb(mode: &str, width: u32, height: u32, data: &[u8]) -> Result<RgbImage> {
    let (size, convert): (usize, PixelToRgb) = match mode {
        "RGB" | "RGBA" => (if mode == "RGB" { 3 } else { 4 }, |pixel| {
            [pixel[0], pixel[1], pixel[2]]
        }),
        "L" => (1, |pixel| [pixel[0]; 3]),
        "LA" => (2, |pixel| [pixel[0]; 3]),
        "I;16" => (2, |pixel| [if pixel[1] == 0 { pixel[0] } else { 255 }; 3]),
        "F" => (4, |pixel| {
            [f2l(f32::from_ne_bytes([pixel[0], pixel[1], pixel[2], pixel[3]])); 3]
        }),
        other => bail!("unrecognized image mode {other}"),
    };
    let needed = width as usize * height as usize * size;
    if data.len() < needed {
        bail!("not enough image data");
    }
    let pixels = data[..needed]
        .chunks_exact(size)
        .flat_map(convert)
        .collect();
    RgbImage::from_raw(width, height, pixels).context("sized buffer")
}

pub fn rotate_90(image: &RgbImage) -> RgbImage {
    let (width, height) = image.dimensions();
    RgbImage::from_fn(height, width, |x, y| *image.get_pixel(width - 1 - y, x))
}

#[cfg(test)]
mod tests {
    use image::Rgb;

    use super::*;

    const LEFT: Rgb<u8> = Rgb([10, 20, 30]);
    const RIGHT: Rgb<u8> = Rgb([40, 50, 60]);

    #[test]
    fn sixteen_bit_gray_clamps_to_white_like_pillow() {
        let data = [7, 0, 0, 1, 255, 0, 255, 255];
        assert_eq!(
            frombytes_rgb("I;16", 4, 1, &data).unwrap().into_raw(),
            [7, 7, 7, 255, 255, 255, 255, 255, 255, 255, 255, 255]
        );
    }

    #[test]
    fn float_gray_truncates_within_the_byte_range_like_pillow() {
        let data: Vec<u8> = [-3.0f32, 0.99, 127.9, 300.0, f32::NAN]
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        assert_eq!(
            frombytes_rgb("F", 5, 1, &data).unwrap().into_raw(),
            [0, 0, 0, 0, 0, 0, 127, 127, 127, 255, 255, 255, 0, 0, 0]
        );
    }

    #[test]
    fn alpha_is_dropped_and_gray_repeated() {
        let rgba = frombytes_rgb("RGBA", 1, 1, &[1, 2, 3, 4]).unwrap();
        assert_eq!(rgba.into_raw(), [1, 2, 3]);
        let la = frombytes_rgb("LA", 2, 1, &[9, 200, 8, 100]).unwrap();
        assert_eq!(la.into_raw(), [9, 9, 9, 8, 8, 8]);
    }

    #[test]
    fn short_data_is_an_error() {
        assert!(frombytes_rgb("RGB", 2, 1, &[1, 2, 3]).is_err());
    }

    #[test]
    fn nearest_steps_like_pillow_across_columns() {
        let image = RgbImage::from_fn(2, 1, |x, _| if x == 0 { LEFT } else { RIGHT });
        let resized = resize(&image, 7, 1, Resampling::Nearest);
        let row: Vec<Rgb<u8>> = resized.pixels().copied().collect();
        assert_eq!(row, [LEFT, LEFT, LEFT, LEFT, RIGHT, RIGHT, RIGHT]);
    }

    #[test]
    fn nearest_steps_like_pillow_across_rows() {
        let image = RgbImage::from_fn(1, 2, |_, y| if y == 0 { LEFT } else { RIGHT });
        let resized = resize(&image, 1, 7, Resampling::Nearest);
        let column: Vec<Rgb<u8>> = resized.pixels().copied().collect();
        assert_eq!(column, [LEFT, LEFT, LEFT, LEFT, RIGHT, RIGHT, RIGHT]);
    }
}
