"""fusion_demo: cross-modality (HAADF-STEM / EDS) image fusion via PLS regression.

This package demonstrates a simplified, self-contained version of a
cross-modality tomographic image fusion pipeline: given aligned orthoslices
(or small volumes) from multiple imaging modalities/elements, it builds a
set of engineered feature images per modality and fits a Partial Least
Squares (PLS) regression model to predict a target element's (denoised)
signal from the other modalities.
"""

__version__ = "0.1.0"
