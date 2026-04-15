pub struct LeaningRateManager<T> {
    max_rate: T,
    min_rate: T,
}

impl<T> LeaningRateManager<T> {
    pub fn new(max_rate: T, min_rate: T) -> Self {
        LeaningRateManager { max_rate, min_rate }
    }

	
}
