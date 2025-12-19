use std::{fs::File, io::Read};

use ron::ser::PrettyConfig;

use crate::simple_dataset::dataset::{SimpleDataset, SmallData};

pub fn test() {
    let mut file = File::open("src/simple_dataset/ron_data.ron").unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();

    let deserialized: SimpleDataset = ron::from_str(&contents).unwrap();

    dbg!(&deserialized);
}
