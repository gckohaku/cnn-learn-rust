use serde::{Deserialize, Serialize};

static DATA_VALUE: usize = 30;
static IMAGE_CHANNEL: usize = 3;
static IMAGE_HEIGHT: usize = 5;
static IMAGE_WIDTH: usize = 5;

#[derive(Debug, Serialize, Deserialize)]
pub struct SimpleDataset {
    pub data: [SmallData; DATA_VALUE],
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SmallData {
    pub image: [[[f64; IMAGE_WIDTH]; IMAGE_HEIGHT]; IMAGE_CHANNEL],
    pub label: [f64; IMAGE_CHANNEL],
}
