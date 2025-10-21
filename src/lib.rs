#![allow(internal_features)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(adt_const_params)]
#![feature(unsized_const_params)]
#![feature(iterator_try_reduce)]
#![feature(core_intrinsics)]
#![feature(const_heap)]
#![feature(const_eval_select)]
#![feature(trait_alias)]

use std::fmt::Display;
use std::intrinsics;
use std::intrinsics::const_make_global;
use std::ptr;
use std::slice;

use maths_traits::algebra::UnitalSemiring;

pub const fn product(d: &[usize]) -> usize {
    let mut acc = 1;

    let mut i = 0;

    while i < d.len() {
        acc *= d[i];

        i += 1;
    }

    acc
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Tensor<const D: &'static [usize], T: UnitalSemiring>
where
    [(); product(D)]:,
{
    pub components: [T; product(D)],
}

impl<const D: &'static [usize], T: UnitalSemiring> Tensor<D, T>
where
    [(); product(D)]:,
{
    pub fn zero() -> Self {
        Self {
            components: core::array::from_fn(|_| T::zero()),
        }
    }

    pub fn identity() -> Self {
        Self {
            components: {
                let mut ret: [T; product(D)] = core::array::from_fn(|_| T::zero());

                let mut indices = vec![0; D.len()];

                for item in &mut ret {
                    *item = match indices.iter().try_reduce(|x, y| (x == y).then_some(x)) {
                        Some(_) => T::one(),
                        None => T::zero(),
                    };

                    for (i, &d) in indices.iter_mut().zip(D) {
                        *i += 1;

                        if *i >= d {
                            *i = 0;
                        } else {
                            break;
                        }
                    }
                }

                ret
            },
        }
    }

    pub const fn new(components: [T; product(D)]) -> Self {
        Self { components }
    }
}

// ungodly evil compile time slice concatenation
pub const fn concatenate(a: &'static [usize], b: &'static [usize]) -> &'static [usize] {
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
}

pub const fn remove_axes_and_concatenate(
    a: &'static [usize],
    p: usize,
    b: &'static [usize],
    q: usize,
) -> &'static [usize] {
    assert!(p < a.len());
    assert!(q < b.len());

    assert!(
        a[p] == b[q],
        "Tensor contraction requires that the contracted axes have the same dimension. \
             For example, with `a.contract::<P, Q, _>(b)`, `a[P]` must equal `b[Q]`."
    );

    const fn const_remove_axes_and_concatenate(
        a: &'static [usize],
        p: usize,
        b: &'static [usize],
        q: usize,
    ) -> &'static [usize] {
        unsafe {
            let len = (a.len() - 1) + (b.len() - 1);

            let buffer =
                core::intrinsics::const_allocate(len * size_of::<usize>(), align_of::<usize>())
                    .cast();

            // --- copy everything from A except A[pa] ---
            let mut dst = buffer;
            let mut i = 0;
            while i < a.len() {
                if i != p {
                    ptr::write(dst, a[i]);
                    dst = dst.add(1);
                }
                i += 1;
            }

            // --- copy everything from B except B[pb] ---
            let mut j = 0;
            while j < b.len() {
                if j != q {
                    ptr::write(dst, b[j]);
                    dst = dst.add(1);
                }
                j += 1;
            }

            intrinsics::const_make_global(buffer.cast());
            slice::from_raw_parts(buffer, len)
        }
    }
    fn runtime_remove_axes_and_concatenate(
        a: &'static [usize],
        p: usize,
        b: &'static [usize],
        q: usize,
    ) -> &'static [usize] {
        let mut v = Vec::with_capacity(a.len() + b.len() - 2);
        v.extend_from_slice(&a[..p]);
        v.extend_from_slice(&a[p + 1..]);
        v.extend_from_slice(&b[..q]);
        v.extend_from_slice(&b[q + 1..]);
        v.leak()
    }
    intrinsics::const_eval_select(
        (a, p, b, q),
        const_remove_axes_and_concatenate,
        runtime_remove_axes_and_concatenate,
    )
}

pub const fn remove_axes_two(a: &'static [usize], p: usize, q: usize) -> &'static [usize] {
    assert!(p < a.len(), "First axis to remove is out of bounds");
    assert!(q < a.len(), "Second axis to remove is out of bounds");
    assert!(p != q, "Cannot remove the same axis twice");

    const fn const_remove_axes_two(a: &'static [usize], p: usize, q: usize) -> &'static [usize] {
        unsafe {
            let len = a.len() - 2;
            let buffer = core::intrinsics::const_allocate(
                len * core::mem::size_of::<usize>(),
                core::mem::align_of::<usize>(),
            )
            .cast();

            let mut dst = buffer;
            let mut i = 0;
            while i < a.len() {
                if i != p && i != q {
                    ptr::write(dst, a[i]);
                    dst = dst.add(1);
                }
                i += 1;
            }

            const_make_global(buffer.cast());
            slice::from_raw_parts(buffer, len)
        }
    }

    fn runtime_remove_axes_two(a: &'static [usize], p: usize, q: usize) -> &'static [usize] {
        let mut v = Vec::with_capacity(a.len() - 2);
        for (i, &val) in a.iter().enumerate() {
            if i != p && i != q {
                v.push(val);
            }
        }
        v.leak()
    }

    intrinsics::const_eval_select((a, p, q), const_remove_axes_two, runtime_remove_axes_two)
}

impl<const A: &'static [usize], T: UnitalSemiring> Tensor<A, T>
where
    [(); product(A)]:,
{
    pub fn product<const B: &'static [usize]>(
        &self,
        other: &Tensor<B, T>,
    ) -> Tensor<{ concatenate(A, B) }, T>
    where
        [(); product(B)]:,
        [(); product(concatenate(A, B))]:,
    {
        let mut ret: Tensor<{ concatenate(A, B) }, T> = Tensor::zero();

        let mut index_self = 0;
        let mut index_other = 0;

        for item in &mut ret.components {
            *item = self.components[index_self].clone() * other.components[index_other].clone();

            if index_self + 1 >= self.components.len() {
                index_self = 0;
                index_other += 1;
            } else {
                index_self += 1;
            };
        }

        ret
    }

    pub fn contract<const P: usize, const Q: usize, const B: &'static [usize]>(
        &self,
        other: &Tensor<B, T>,
    ) -> Tensor<{ remove_axes_and_concatenate(A, P, B, Q) }, T>
    where
        [(); product(B)]:,
        [(); product(remove_axes_and_concatenate(A, P, B, Q))]:,
    {
        // output shape and result tensor
        let out_shape = remove_axes_and_concatenate(A, P, B, Q);
        let mut result = Tensor::<{ remove_axes_and_concatenate(A, P, B, Q) }, T>::zero();

        // compute strides (row-major: last index fastest)
        let mut stride_a = vec![1usize; A.len()];
        if A.len() >= 2 {
            for i in (0..A.len() - 1).rev() {
                stride_a[i] = stride_a[i + 1] * A[i + 1];
            }
        }

        let mut stride_b = vec![1usize; B.len()];
        if B.len() >= 2 {
            for i in (0..B.len() - 1).rev() {
                stride_b[i] = stride_b[i + 1] * B[i + 1];
            }
        }

        let mut stride_out = vec![1usize; out_shape.len()];
        if out_shape.len() >= 2 {
            for i in (0..out_shape.len() - 1).rev() {
                stride_out[i] = stride_out[i + 1] * out_shape[i + 1];
            }
        }

        // temporary multi-index buffers
        let mut ia = vec![0usize; A.len()];
        let mut ib = vec![0usize; B.len()];
        let mut out_multi = vec![0usize; out_shape.len()];

        // For each output element, decode multi-index, map into ia/ib, then sum over contracted index
        for lin_out in 0..result.components.len() {
            // decode linear -> multi for out_shape
            {
                let mut rem = lin_out;
                for i in 0..out_shape.len() {
                    out_multi[i] = rem / stride_out[i];
                    rem %= stride_out[i];
                }
            }

            // map out_multi into ia and ib for indices not contracted
            // ordering of remove_axes_and_concatenate is: A without p, then B without q
            let mut k = 0;
            for i in 0..A.len() {
                if i != P {
                    ia[i] = out_multi[k];
                    k += 1;
                }
            }
            for j in 0..B.len() {
                if j != Q {
                    ib[j] = out_multi[k];
                    k += 1;
                }
            }

            // sum over contracted axis r
            let mut acc = T::zero();
            let contract_dim = A[P];
            for r in 0..contract_dim {
                ia[P] = r;
                ib[Q] = r;

                // compute linear indices for A and B via dot product of coords * strides
                let mut idx_a = 0usize;
                for (coord, &s) in ia.iter().zip(&stride_a) {
                    idx_a += coord * s;
                }
                let mut idx_b = 0usize;
                for (coord, &s) in ib.iter().zip(&stride_b) {
                    idx_b += coord * s;
                }

                acc += self.components[idx_a].clone() * other.components[idx_b].clone();
            }

            result.components[lin_out] = acc;
        }

        result
    }

    pub fn lower_index<const N: usize, const B: &'static [usize]>(
        &self,
        metric: &Tensor<B, T>,
    ) -> Tensor<{ remove_axes_and_concatenate(A, N, B, 0) }, T>
    where
        [(); product(B)]:,
        [(); product(remove_axes_and_concatenate(A, N, B, 0))]:,
    {
        self.contract::<N, 0, _>(metric)
    }

    pub fn raise_index<const N: usize, const B: &'static [usize]>(
        &self,
        metric: &Tensor<B, T>,
    ) -> Tensor<{ remove_axes_and_concatenate(A, N, B, 1) }, T>
    where
        [(); product(B)]:,
        [(); product(remove_axes_and_concatenate(A, N, B, 1))]:,
    {
        self.contract::<N, 1, _>(metric)
    }

    pub fn diag<const K: usize>(signature: [T; K]) -> Tensor<{ make_diag(K) }, T>
    where
        [(); product(make_diag(K))]:,
    {
        let mut ret = Tensor::zero();
        for (idx, val) in signature.iter().enumerate() {
            ret.components[idx + idx * A[0]] = val.clone();
        }

        ret
    }

    pub fn trace<const P: usize, const Q: usize>(&self) -> Tensor<{ remove_axes_two(A, P, Q) }, T>
    where
        [(); product(remove_axes_two(A, P, Q))]:,
        [(); product(A)]:,
    {
        let out_shape = remove_axes_two(A, P, Q);
        let mut result = Tensor::<{ remove_axes_two(A, P, Q) }, T>::zero();

        let mut ia = vec![0; A.len()];
        let mut ir = vec![0; out_shape.len()];

        let unpack = |mut idx: usize, shape: &[usize], out: &mut [usize]| {
            for i in (0..shape.len()).rev() {
                out[i] = idx % shape[i];
                idx /= shape[i];
            }
        };
        let pack = |indices: &[usize], shape: &[usize]| -> usize {
            let mut mul = 1;
            let mut acc = 0;
            for (i, &d) in shape.iter().rev().enumerate() {
                let v = indices[shape.len() - 1 - i];
                acc += v * mul;
                mul *= d;
            }
            acc
        };

        // Iterate over all result elements
        for lin_r in 0..result.components.len() {
            unpack(lin_r, out_shape, &mut ir);

            // fill remaining axes into ia
            let mut k = 0;
            for i in 0..A.len() {
                if i != P && i != Q {
                    ia[i] = ir[k];
                    k += 1;
                }
            }

            // sum along contracted axes
            let mut acc = T::zero();
            for r in 0..A[P] {
                ia[P] = r;
                ia[Q] = r;
                let idx_a = pack(&ia, A);
                acc += self.components[idx_a].clone();
            }

            result.components[lin_r] = acc;
        }

        result
    }
}

pub const fn make_diag(k: usize) -> &'static [usize] {
    const fn const_make_diag(k: usize) -> &'static [usize] {
        unsafe {
            let buffer: *mut usize =
                core::intrinsics::const_allocate(2 * size_of::<usize>(), align_of::<usize>())
                    .cast();

            buffer.write(k);
            buffer.add(1).write(k);

            const_make_global(buffer.cast());

            slice::from_raw_parts(buffer.cast(), 2)
        }
    }
    fn runtime_make_diag(k: usize) -> &'static [usize] {
        vec![k; 2].leak()
    }

    intrinsics::const_eval_select((k,), const_make_diag, runtime_make_diag)
}

impl<const D: &'static [usize], T: Display + Clone + UnitalSemiring> Display for Tensor<D, T>
where
    [(); product(D)]:,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn helper<T: Display>(
            components: &[T],
            dims: &[usize],
            indent: usize,
            f: &mut std::fmt::Formatter<'_>,
        ) -> Result<(), std::fmt::Error> {
            if dims.len() == 0 {
                assert!(components.len() == 1);

                write!(f, "{}", components[0])?;
            } else if dims.len() == 1 {
                // Last dimension: print as a row
                write!(f, "{}", " ".repeat(indent))?;
                write!(f, "[")?;
                for (i, v) in components.iter().enumerate() {
                    write!(f, "{}", v)?;
                    if i + 1 != components.len() {
                        write!(f, ", ")?;
                    }
                }
                writeln!(f, "]")?;
            } else {
                // Recursive case: split into sub-tensors
                let chunk_size = dims[1..].iter().product();
                write!(f, "{}", " ".repeat(indent))?;
                writeln!(f, "[")?;
                for chunk in components.chunks(chunk_size) {
                    helper(chunk, &dims[1..], indent + 2, f)?;
                }
                write!(f, "{}", " ".repeat(indent))?;
                writeln!(f, "]")?;
            }

            Ok(())
        }

        helper(&self.components, D, 0, f)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPACETIME: &'static [usize; 2] = &[4, 4];

    #[test]
    fn test_tensor_product() {
        let a = Tensor::<{ &[2, 3] }, f32>::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Tensor::<{ &[3, 2] }, f32>::new([7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

        let prod = a.product(&b);

        // Length of components should be 2*3*3*2 = 36
        assert_eq!(prod.components.len(), 36);

        // Spot check a few values
        assert_eq!(prod.components[0], 1.0 * 7.0);
        assert_eq!(prod.components[1], 2.0 * 7.0);
        assert_eq!(prod.components[6], 1.0 * 8.0);
    }

    #[test]
    fn test_tensor_contraction() {
        let a = Tensor::<{ &[2, 3, 2] }, f32>::new([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ]);
        let b = Tensor::<{ &[2, 2] }, f32>::identity();

        // Contract axes 0 of `a` with 0 of `b` (should sum over first dimension)
        let c = a.contract::<0, 0, _>(&b);

        // The resulting shape should be [3,2,2] with axis 0 removed
        assert_eq!(c.components.len(), 12);

        // Spot check sums: for example, c[0] = a[0,0,0] + a[1,0,0] = 1 + 7
        assert_eq!(c.components[0], 1.0);
    }

    #[test]
    fn test_trace() {
        let a = Tensor::<{ &[2, 3, 2] }, f32>::new([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ]);

        let t = a.trace::<0, 2>();

        // Shape after trace: [3]
        assert_eq!(t.components.len(), 3);

        // Expected sums: t[0] = a[0,0,0]+a[1,0,1] = 1+8 = 9
        assert_eq!(t.components[0], 9.0);
        assert_eq!(t.components[1], 13.0); // 2+11
        assert_eq!(t.components[2], 17.0); // 3+12
    }

    #[test]
    fn raise_lower_symmetric_vs_asymmetric() {
        let a = Tensor::<SPACETIME, f32>::new([
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        ]);

        let symmetric = Tensor::<SPACETIME, f32>::diag([-1.0, 1.0, 1.0, 1.0]);

        let asymmetric = Tensor::<SPACETIME, f32>::new([
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        ]);

        let raised_sym = a.raise_index::<0, _>(&symmetric);
        let lowered_sym = a.lower_index::<0, _>(&symmetric);
        assert_eq!(
            raised_sym, lowered_sym,
            "Symmetric metric should yield identical tensors"
        );

        let raised_asym = a.raise_index::<0, _>(&asymmetric);
        let lowered_asym = a.lower_index::<0, _>(&asymmetric);
        assert_ne!(
            raised_asym, lowered_asym,
            "Asymmetric metric should yield different tensors"
        );
    }

    #[test]
    fn test_identity_tensor() {
        let id = Tensor::<{ &[3, 3] }, f32>::identity();

        // Diagonal elements should be 1, off-diagonal 0
        for i in 0..3 {
            for j in 0..3 {
                let idx = i * 3 + j;
                if i == j {
                    assert_eq!(id.components[idx], 1.0);
                } else {
                    assert_eq!(id.components[idx], 0.0);
                }
            }
        }
    }
}
