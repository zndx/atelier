# Labeled values must vary

Batch 13 of the restarted coverage run wrote four tables whose labeled column was a fixed string. Sibling columns called `rng`. The gate rejected each one with "generator returned one constant," because the NHSVM row uses the labeled column's values.

The skill now says those values are drawn with `rng`, so two calls can return different strings. The retry already in flight still has the previous skill text. The next pack copies this sentence.
