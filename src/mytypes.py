"""Define the types used over the package (for typing hint)
"""

import numpy as np
import typing
from nptyping import NDArray, Shape, Float32

ColVector = NDArray[Shape["Any, 1"], Float32]
RowVector = NDArray[Shape["1, Any"], Float32]

Array = NDArray[Shape["Any"], Float32]
Array2D = NDArray[Shape["Any, Any"], Float32]
Array3D = NDArray[Shape["Any, Any, Any"], Float32]
Array4D = NDArray[Shape["Any, Any, Any, Any"], Float32]