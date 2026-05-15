# Subject-wise train/val split

State Farm dataset has 26 unique drivers (`subject` column); each driver
appears across most/all 10 classes. A random 80/20 split places the same
driver in both train and val, so the model memorizes driver identity
(face, shirt, seat, lighting) instead of action — typical val accuracy
~99%, which collapses on out-of-distribution images.

**Decision:** partition by subject. Held-out drivers `p022, p035, p047,
p056, p075` (5/26 ≈ 19%) appear only in val. Expected honest val
accuracy is 60–80% rather than 99%, but generalizes to the OOD demo
images and matches the dataset's intended evaluation (test set in the
original Kaggle competition contains entirely new drivers).

**Trade-off accepted:** lower headline accuracy in exchange for an
evaluation number that actually reflects generalization. Random-split
training is kept as a tier-3 fallback only — to be run as a context
baseline if the subject-wise val acc collapses, never as the primary
result.
