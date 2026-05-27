# Views And Lifetime

`ArrayView<T>` is a non-owning view into an `Array<T>` buffer. It stores a data
pointer plus shape and stride metadata. It does not keep the source array alive.

## Safe Pattern

```cpp
auto a = litenp::ones<float>({10});
auto v = a.view();
// use v while a is still alive
```

## Dangerous Pattern

```cpp
auto v = litenp::ones<float>({10}).view();
```

The temporary `Array<float>` is destroyed at the end of the statement, so `v`
does not own valid storage after that point.

## Mutation And Metadata

`litenp` uses semantic metadata for uniform and arange arrays. Mutable views
invalidate that metadata when exposing a writable pointer or writable element
reference. This keeps operations such as `sum(a)` correct after writes through a
view or slice.

Code that writes through raw pointers obtained from `data()` is allowed, but the
library cannot track individual writes after the pointer has escaped. Metadata is
cleared when the writable pointer is handed out.

## Const Element Access

`Array<T>::operator[](std::size_t) const` returns `T` by value. This allows
virtual arrays such as `arange` and `eye` to serve element reads without
materializing storage.

