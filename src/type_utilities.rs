// 型リストの終点
struct HNil;

// 二つの型をリストにする HCons を再帰的に利用することでリストを作成する
struct HCons<Head, Tail>(Head, Tail);

// 型リストであることを示す
trait HList {}

impl HList for HNil {}
impl<Head, Tail> HList for HCons<Head, Tail> {}

/*
    型リスト全体にトレイトを適用する
*/
// 型リストのメンバーであることを示す
trait Member<H, Index> {}

// 型リストのインデックスで利用
struct Here;
struct There<Index>(Index);

// トレイト適用部分
// まず、型リストの先頭に対して適用
impl<Head, Tail: HList> Member<HCons<Head, Tail>, Here> for Head {}
// 上の条件に合わなかったものは、この条件での適用を試みる
// この際、Tail の先頭要素を Head として上の条件に合えば適用される
// この操作を再帰的に HNil まで繰り返す
impl<T, Head, Tail, Index> Member<HCons<Head, Tail>, There<Index>> for T where T: Member<Tail, Index>
{}

// 型リスト作成マクロ
macro_rules! HList {
	() => {HNil};
	($head:ty $(,)*) => {HCons<$head, HNill>};
	($head:ty, $($tail:tt)*) => {HCons<$head, HCons!($($tail)*)>}
}
