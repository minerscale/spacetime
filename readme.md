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

Do be warned though, this library is hilariously jank. In order to get the type system to play dice I have a function with this signature:

```rust
pub const fn concatenate(a: &'static [usize], b: &'static [usize]) -> &'static [usize]
```

That is, I'm concatenating a slice and producing a new static slice in a const function. How do you actually produce such a value? Why of course you do it differently depending on whether we are in compile time or runtime. The compile time evaluation is especially cursed, using the unstable const_allocate and const_make_global to save the new signature into the binary as a new symbol.

```rust
const fn const_concatenate(a: &'static [usize], b: &'static [usize]) -> &'static [usize] {
        unsafe {
            let buffer = core::intrinsics::const_allocate(
                (a.len() + b.len()) * size_of::<usize>(),
                align_of::<usize>(),
            )
            .cast();
            ptr::copy_nonoverlapping(a.as_ptr(), buffer, a.len());
            ptr::copy_nonoverlapping(b.as_ptr(), buffer.add(a.len()), b.len());

            intrinsics::const_make_global(buffer.cast());
            slice::from_raw_parts(buffer.cast(), a.len() + b.len())
        }
    }
    fn runtime_concatenate(a: &'static [usize], b: &'static [usize]) -> &'static [usize] {
        let mut vec = Vec::new();
        vec.extend_from_slice(a);
        vec.extend_from_slice(b);
        vec.leak()
    }
    intrinsics::const_eval_select((a, b), const_concatenate, runtime_concatenate)
```

Really, this is all to get around the inability to do stuff like this:

```rust
pub struct Tensor<const N: usize, const D: [usize; N], T>
```

It would be nice to do that one day though I'm sure there's a really good reason we can't do it.
