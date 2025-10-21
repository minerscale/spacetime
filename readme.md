# spacetime

## the most nightly tensor library there is

I wanted to see if I could make a tensor library completely generic in the rust type system. Turns out it's possible, just barely using a dozen or so unstable or internal rust features. This will never, ever work on stable. Yet still, the semantics for using the library are quite nice.

You can create a new 2x3x2 tensor like this `Tensor::<{&[2, 3, 2]}>::zero()`. You can take tensor products, do contractions, the lot. If the program compiles, you've written a valid tensor transform. The type system statically guarantees that all tensors are in the right dimension.

This also means that every kind of tensor is monomorphised and optimised by the compiler specially before use. I have to imagine that this is a good idea.

Example:

```rust
use spacetime::Tensor;

let minkowski_metric = Tensor::<{&[4, 4]}, f32>::diag([-1.0, 1.0, 1.0, 1.0]);

println!("{minkowski_metric}");
println!("{}", Tensor::<{&[3]}, f32>::new([1.0, 2.0, 3.0]).product(&minkowski_metric));
```
