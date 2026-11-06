use anyhow::Result;
use image::RgbImage;
use nalgebra::Matrix2;
use ndarray::{ArrayD, Axis, IxDyn};
use opencv::core::{Mat, Size};
use opencv::imgproc;

use crate::models::transforms::{mat_rgb, rgb_mat};
use crate::numpy::pairwise_sum;

pub const ALIGNED_SIZE: u32 = 112;
pub const DET_STRIDES: [i64; 3] = [8, 16, 32];
pub const ANCHORS_PER_CELL: usize = 2;

pub const ARCFACE_DST: [[f32; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

fn anchor_centers(size: i64, stride: i64) -> Vec<[f32; 2]> {
    let cells = size / stride;
    let mut centers = Vec::with_capacity((cells * cells) as usize * ANCHORS_PER_CELL);
    for y in 0..cells {
        for x in 0..cells {
            let center = [x as f32 * stride as f32, y as f32 * stride as f32];
            for _ in 0..ANCHORS_PER_CELL {
                centers.push(center);
            }
        }
    }
    centers
}

fn ensure_dims(array: ArrayD<f32>, ndim: usize) -> ArrayD<f32> {
    if array.ndim() >= ndim {
        return array;
    }
    let mut shape = vec![1; ndim - array.ndim()];
    shape.extend_from_slice(array.shape());
    array
        .into_shape_with_order(IxDyn(&shape))
        .expect("same number of elements")
}

fn rows(head: &ArrayD<f32>) -> Vec<f32> {
    head.index_axis(Axis(0), 0).iter().copied().collect()
}

pub struct Decoded {
    pub scores: Vec<f32>,
    pub boxes: Vec<[f32; 4]>,
    pub kps: Vec<[f32; 10]>,
}

pub fn decode_scrfd(heads: Vec<ArrayD<f32>>, size: i64) -> Decoded {
    let heads: Vec<ArrayD<f32>> = heads.into_iter().map(|head| ensure_dims(head, 4)).collect();
    let levels = DET_STRIDES.len();
    let mut decoded = Decoded {
        scores: Vec::new(),
        boxes: Vec::new(),
        kps: Vec::new(),
    };
    for (level, stride) in DET_STRIDES.iter().enumerate() {
        let centers = anchor_centers(size, *stride);
        let stride = *stride as f32;
        let scores = rows(&heads[level]);
        let distances = rows(&heads[level + levels]);
        let offsets = rows(&heads[level + 2 * levels]);
        let groups = scores.len() / centers.len().max(1);
        for group in 0..groups {
            for (index, center) in centers.iter().enumerate() {
                let anchor = group * centers.len() + index;
                decoded.scores.push(scores[anchor]);
                let d = &distances[anchor * 4..anchor * 4 + 4];
                decoded.boxes.push([
                    center[0] - d[0] * stride,
                    center[1] - d[1] * stride,
                    center[0] + d[2] * stride,
                    center[1] + d[3] * stride,
                ]);
                let o = &offsets[anchor * 10..anchor * 10 + 10];
                let mut kps = [0.0f32; 10];
                for (point, slot) in kps.iter_mut().enumerate() {
                    *slot = center[point % 2] + o[point] * stride;
                }
                decoded.kps.push(kps);
            }
        }
    }
    decoded
}

#[derive(Clone, Copy)]
struct Rect2d {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

impl Rect2d {
    fn empty(&self) -> bool {
        self.width <= 0.0 || self.height <= 0.0
    }

    fn area(&self) -> f64 {
        self.width * self.height
    }

    fn intersect(&self, other: &Self) -> Self {
        let none = Self {
            x: 0.0,
            y: 0.0,
            width: 0.0,
            height: 0.0,
        };
        if self.empty() || other.empty() {
            return none;
        }
        let (x_min, x_max) = if self.x < other.x {
            (self, other)
        } else {
            (other, self)
        };
        let (y_min, y_max) = if self.y < other.y {
            (self, other)
        } else {
            (other, self)
        };
        if (x_min.x < 0.0 && x_min.x + x_min.width < x_max.x)
            || (y_min.y < 0.0 && y_min.y + y_min.height < y_max.y)
        {
            return none;
        }
        let rect = Self {
            width: (x_min.width - (x_max.x - x_min.x)).min(x_max.width),
            height: (y_min.height - (y_max.y - y_min.y)).min(y_max.height),
            x: x_max.x,
            y: y_max.y,
        };
        if rect.empty() { none } else { rect }
    }
}

fn rect_overlap(a: &Rect2d, b: &Rect2d) -> f32 {
    let area_a = a.area();
    let area_b = b.area();
    let jaccard_distance = if area_a + area_b <= f64::EPSILON {
        0.0
    } else {
        let intersection = a.intersect(b).area();
        1.0 - intersection / (area_a + area_b - intersection)
    };
    1.0f32 - jaccard_distance as f32
}

pub fn nms(boxes: &[[f32; 4]], scores: &[f32], threshold: f32) -> Vec<usize> {
    let rects: Vec<Rect2d> = boxes
        .iter()
        .map(|b| Rect2d {
            x: f64::from(b[0]),
            y: f64::from(b[1]),
            width: f64::from(b[2] - b[0]),
            height: f64::from(b[3] - b[1]),
        })
        .collect();
    let mut order: Vec<(f32, usize)> = scores
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, score)| *score > 0.0)
        .map(|(i, s)| (s, i))
        .collect();
    order.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut keep: Vec<usize> = Vec::new();
    for (_, index) in order {
        if keep
            .iter()
            .all(|kept| rect_overlap(&rects[index], &rects[*kept]) <= threshold)
        {
            keep.push(index);
        }
    }
    keep
}

fn mean_rows(points: &[[f32; 2]; 5]) -> [f32; 2] {
    let mut total = [0.0f32; 2];
    for point in points {
        total[0] += point[0];
        total[1] += point[1];
    }
    [total[0] / 5.0, total[1] / 5.0]
}

pub fn umeyama(src: &[[f32; 2]; 5], dst: &[[f32; 2]; 5]) -> [[f32; 3]; 2] {
    let src_mean = mean_rows(src);
    let dst_mean = mean_rows(dst);
    let src_c = src.map(|p| [p[0] - src_mean[0], p[1] - src_mean[1]]);
    let dst_c = dst.map(|p| [p[0] - dst_mean[0], p[1] - dst_mean[1]]);
    let mut cov = [[0.0f32; 2]; 2];
    for (i, row) in cov.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            let mut total = 0.0f32;
            for k in 0..5 {
                total += dst_c[k][i] * src_c[k][j];
            }
            *cell = total / 5.0;
        }
    }
    let svd = Matrix2::new(cov[0][0], cov[0][1], cov[1][0], cov[1][1]).svd(true, true);
    let u = svd.u.expect("requested u");
    let vt = svd.v_t.expect("requested v_t");
    let s = svd.singular_values;
    let d = (u * vt).determinant().signum() as f64;
    let u = u.cast::<f64>();
    let vt = vt.cast::<f64>();
    let diag = Matrix2::new(1.0, 0.0, 0.0, d);
    let rotation = u * diag * vt;
    let squares: Vec<f32> = src_c
        .iter()
        .flat_map(|p| [p[0] * p[0], p[1] * p[1]])
        .collect();
    let trace = f64::from(s[0]) * 1.0 + f64::from(s[1]) * d;
    let scale = trace / f64::from(0.0 + pairwise_sum(&squares)) * 5.0;
    let scaled = rotation * scale;
    let mean = [f64::from(src_mean[0]), f64::from(src_mean[1])];
    let translation = [
        f64::from(dst_mean[0]) - (scaled[(0, 0)] * mean[0] + scaled[(0, 1)] * mean[1]),
        f64::from(dst_mean[1]) - (scaled[(1, 0)] * mean[0] + scaled[(1, 1)] * mean[1]),
    ];
    [
        [
            scaled[(0, 0)] as f32,
            scaled[(0, 1)] as f32,
            translation[0] as f32,
        ],
        [
            scaled[(1, 0)] as f32,
            scaled[(1, 1)] as f32,
            translation[1] as f32,
        ],
    ]
}

pub fn align_face(image: &Mat, kps: &[[f32; 2]; 5]) -> Result<RgbImage> {
    let matrix = umeyama(kps, &ARCFACE_DST);
    let matrix = Mat::from_slice_2d(&matrix)?;
    let mut crop = Mat::default();
    imgproc::warp_affine_def(
        image,
        &mut crop,
        &matrix,
        Size::new(ALIGNED_SIZE as i32, ALIGNED_SIZE as i32),
    )?;
    mat_rgb(&crop)
}

pub fn image_mat(image: &RgbImage) -> Result<Mat> {
    rgb_mat(image)
}
