# No horizontal flip augmentation

Horizontal flip is the default first augmentation in virtually every
image-classification pipeline, so the next person reading `augment.py`
will assume it was forgotten and add it back. It is deliberately
omitted here.

**Reason:** 4 of the 10 State Farm classes are left/right-specific —
`c1` (texting right) vs `c3` (texting left), `c2` (phone right) vs
`c4` (phone left). Horizontal flip turns a `c1` image into a visually
correct `c3` image while keeping the `c1` label, injecting label
noise across ~40% of the training set.

**Alternative considered:** label-swap HFlip (flip the image and remap
c1↔c3, c2↔c4). Rejected because the remaining 6 classes have no
mirror counterpart, so the implementation branches per class and the
gain (doubling effective data for 4 classes) does not justify the
complexity for a 5-hour training budget. Heavy non-geometric
augmentation (CutMix, RandomErasing, color jitter, grayscale) is used
instead to break driver-identity shortcuts without touching the
left/right signal.
