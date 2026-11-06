pub fn pairwise_sum(values: &[f32]) -> f32 {
    let n = values.len();
    if n < 8 {
        let mut total = 0.0f32;
        for value in values {
            total += value;
        }
        total
    } else if n <= 128 {
        let mut partial = [0.0f32; 8];
        partial.copy_from_slice(&values[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for (j, slot) in partial.iter_mut().enumerate() {
                *slot += values[i + j];
            }
            i += 8;
        }
        let mut total = ((partial[0] + partial[1]) + (partial[2] + partial[3]))
            + ((partial[4] + partial[5]) + (partial[6] + partial[7]));
        while i < n {
            total += values[i];
            i += 1;
        }
        total
    } else {
        let mut half = n / 2;
        half -= half % 8;
        pairwise_sum(&values[..half]) + pairwise_sum(&values[half..])
    }
}

pub fn sum(values: &[f32]) -> f32 {
    0.0 + pairwise_sum(values)
}

pub fn argmax(values: &[f32]) -> usize {
    if let Some(index) = values.iter().position(|value| value.is_nan()) {
        return index;
    }
    let best = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    values
        .iter()
        .position(|value| *value == best)
        .unwrap_or_default()
}

pub fn argsort_stable<T: PartialOrd>(values: &[T]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_by(|a, b| {
        values[*a]
            .partial_cmp(&values[*b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices
}

#[cfg(test)]
mod tests {
    use super::argmax;

    #[test]
    fn argmax_takes_the_first_of_equal_maxima() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0, 3.0]), 1);
        assert_eq!(argmax(&[-0.0, 0.0]), 0);
        assert_eq!(argmax(&[0.0, -0.0]), 0);
    }

    #[test]
    fn argmax_returns_the_first_nan() {
        assert_eq!(argmax(&[1.0, f32::NAN, 9.0, f32::NAN]), 1);
    }

    #[test]
    fn argmax_of_nothing_above_negative_infinity_is_the_first() {
        assert_eq!(argmax(&[f32::NEG_INFINITY; 3]), 0);
        assert_eq!(argmax(&[]), 0);
    }
}
