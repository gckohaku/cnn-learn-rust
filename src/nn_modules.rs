pub mod linear;
pub mod builders;

pub trait NNModule {
	fn forward<T, R>(tensor: T) -> R;
}