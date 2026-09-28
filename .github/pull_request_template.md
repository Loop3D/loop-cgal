<!-- The PR title becomes the changelog entry via release-please, so make it a
     conventional commit: feat:, fix:, perf:, refactor:, test:, docs:, chore:.
     Use feat!: or fix!: when the Python API or a return value changes meaning. -->

## What & why

<!-- What changes, and what problem it solves. Link the issue if there is one. -->

## How to verify

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
pytest
```

<!-- Paste the evidence: the failing case before, and the same case after. -->

## Risk

<!-- Delete the lines that do not apply. -->

- **Kernel paths** — does this behave the same with `use_exact_kernel=True` and `False`?
  They are separate code paths and a fix to one often leaves the other wrong.
- **Geometry** — is output expected to be bit-identical? If not, say what moved: vertex
  and face counts, area, bounds. Reordered vertices with identical geometry is fine, but
  say so, because the PLY/md5 will differ.
- **Degenerate input** — CGAL crashes rather than raising on coincident vertices,
  zero-area faces and planes co-aligned with an edge. If this touches clipping,
  corefinement or a kernel round trip, say which of those you tried.
- **Platform** — a crash or hang that does not reproduce on macOS/arm64 may still be
  live on Linux/x86-64 (see Loop3D/loop-cgal#14). Verify crash-class fixes on Linux.
- **API** — changed a signature, default, or the meaning of a return value? Note it here
  and mark the title breaking.

## Checklist

- [ ] A test that fails without this change and passes with it
- [ ] `pytest` passes locally
- [ ] Both kernels covered where the change touches clipping or conversion
