pub fn compute_sum(a: i32, b: i32) -> i32 {
    a + b
}

fn main() {
    let result = compute_sum(10, 32);
    println!("Hello, world! Compute result: {}", result);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_sum() {
        assert_eq!(compute_sum(10, 32), 42);
    }
}

