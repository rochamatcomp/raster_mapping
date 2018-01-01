#[macro_use]
extern crate ndarray;

#[macro_use]
extern crate approx;

extern crate gdal;
extern crate itertools;

use std::collections::HashMap;
use std::path::Path;
use gdal::raster::Dataset;
use ndarray::Array2;
use itertools::zip;
pub use approx::{AbsDiffEq, RelativeEq, UlpsEq};

#[derive(Debug, PartialEq)]
pub struct Raster<T>{
    pub data: Array2<T>
}

pub trait Mapping<TypeData>{
    fn get_data(filename: &String) -> TypeData;
    fn new(filename: String) -> Self;
    fn algebra(maps: HashMap<String, f32>) -> Self;
}

impl Mapping<Array2<f32>> for Raster<f32>
{
    fn get_data(filename: &String) -> Array2<f32>{
        let path = Path::new(filename);
        let dataset = Dataset::open(path).unwrap();
        let shape = dataset.size();
        let buffer = dataset.read_full_raster_as::<f32>(1).unwrap();
        let (cols, rows) = shape;
        Array2::from_shape_vec((rows, cols), buffer.data).unwrap()
    }
    
    fn new(filename: String) -> Self{
        Raster::<f32>{
            data: Self::get_data(&filename)
        }
    }
    
    fn algebra(maps: HashMap<String, f32>) -> Self{
        let mut data: Array2<f32>;
        let mut weight: f32;
        let mut result: Array2<f32> = array![[]];

        let first = maps.iter().next();  
        match first {
            // Result receives the first data
            Some(map) => {
                data = Self::get_data(map.0);
                weight = *map.1;
                result = weight * data;
            }
            // Zero items.
            None => println!("No maps.")
        }
        
        for map in maps.iter().skip(1){
            data = Self::get_data(map.0);
            weight = *map.1;
            result = result + weight * data;
        }
        
        Raster::<f32>{
            data: result
        }
    }
}

impl<T: AbsDiffEq> AbsDiffEq for Raster<T>
where
    T::Epsilon: Copy,
{
    type Epsilon = T::Epsilon;
    fn default_epsilon() -> T::Epsilon {
        T::default_epsilon()
    }
    
    fn abs_diff_eq(&self, other: &Self, epsilon: T::Epsilon) -> bool {
        for (item1, item2) in zip(&self.data, &other.data){
            if !T::abs_diff_eq(item1, item2, epsilon){
                return false;
            }
        }
        true
    }
}

impl<T: RelativeEq> RelativeEq for Raster<T>
where
    T::Epsilon: Copy,
{
    fn default_max_relative() -> T::Epsilon {
        T::default_max_relative()
    }
    
    fn relative_eq(&self, other: &Self, epsilon: T::Epsilon, max_relative: T::Epsilon) -> bool {
        
        for (item1, item2) in zip(&self.data, &other.data){
            if !T::relative_eq(item1, item2, epsilon, max_relative){
                return false;
            }
        }
        true
    }
}

impl<T: UlpsEq> UlpsEq for Raster<T>
where
    T::Epsilon: Copy,
{
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }
    
    fn ulps_eq(&self, other: &Self, epsilon: T::Epsilon, max_ulps: u32) -> bool {
        for (item1, item2) in zip(&self.data, &other.data){
            if !T::ulps_eq(item1, item2, epsilon, max_ulps){
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod test_approx {
    use super::*;

    #[test]
    fn test_abs_diff_eq(){
        let raster1 = Raster::new("data/data1.asc".to_string());
        let raster2 = Raster::<f32>{
            data: Array2::from_shape_vec((3, 4), vec![0.388889, 0.513889, 0.638889, 0.805556, 0.444447, 0.583333, 0.680556, -32768.0, 0.625, 0.625, -32768.0, -32768.0]).unwrap()
        };

        assert_abs_diff_eq!(raster1, raster2);
    }

    #[test]
    fn test_relative_eq(){
        let raster1 = Raster::new("data/data1.asc".to_string());
        let raster2 = Raster::<f32>{
            data: Array2::from_shape_vec((3, 4), vec![0.388889, 0.513889, 0.638889, 0.805556, 0.444447, 0.583333, 0.680556, -32768.0, 0.625, 0.625, -32768.0, -32768.0]).unwrap()
        };

        assert_relative_eq!(raster1, raster2);
    }

    #[test]
    fn test_ulps_eq(){
        let raster1 = Raster::new("data/data1.asc".to_string());
        let raster2 = Raster::<f32>{
            data: Array2::from_shape_vec((3, 4), vec![0.388889, 0.513889, 0.638889, 0.805556, 0.444447, 0.583333, 0.680556, -32768.0, 0.625, 0.625, -32768.0, -32768.0]).unwrap()
        };
        
        assert_ulps_eq!(raster1, raster2, max_ulps = 6);
    }
}

#[cfg(test)]
mod test_algebra {
    use super::*;

    #[test]
    fn test_new_data(){
        let weight1: f32 = 0.4;
        let weight2: f32 = 0.2;
        let weight3: f32 = 0.2;
        let weight4: f32 = 0.2;
        
        let raster1 = Raster::new("data/data1.asc".to_string());
        let raster2 = Raster::new("data/data2.asc".to_string());
        let raster3 = Raster::new("data/data3.asc".to_string());
        let raster4 = Raster::new("data/data4.asc".to_string());
        let result = Raster::new("data/result.asc".to_string());

        let combination = Raster::<f32>{
            data: weight1 * raster1.data + weight2 * raster2.data + weight3 * raster3.data + weight4 * raster4.data
        };

        assert_relative_eq!(combination, result, epsilon = 1e-5f32);

    }

    #[test]
    fn test_new_algebra(){
        let mut maps: HashMap<String, f32> = HashMap::new();

        maps.insert("data/data1.asc".to_string(), 0.4);
        maps.insert("data/data2.asc".to_string(), 0.2);
        maps.insert("data/data3.asc".to_string(), 0.2);
        maps.insert("data/data4.asc".to_string(), 0.2);
        
        let result = Raster::new("data/result.asc".to_string());
        let combination = Raster::algebra(maps);

        assert_relative_eq!(combination, result, epsilon = 1e-5f32);
    }

    #[test]
    fn test_new_algebra_zero(){
        let maps: HashMap<String, f32> = HashMap::new();
        let empty: Array2<f32> = array![[]];
        let result = Raster::<f32>{data: empty};
        let combination = Raster::algebra(maps);

        assert_eq!(combination, result);
    }
}
