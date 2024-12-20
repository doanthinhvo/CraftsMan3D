# Data Format

## Directory Structure
The training data is organized into two main folders:

### 1. Images
- Contains RGB images and alpha masks
- File format: JPEG
- Paired data structure (RGB + mask)

### 2. Surfaces 
- Contains 3D point cloud surface representations
- File format: `.npz` (NumPy compressed)

## Surface Data Specification

### File Structure
- Key: `"surface"` (single key)
- Data type: `float16` (half-precision)
- Shape: `(16384, 6)`

### Point Features
Each point has 6 features:
1. Coordinates (3D)
   - `x, y, z` normalized to [-1, 1]
2. Normal vectors
   - `nx, ny, nz` unit vectors in [-1, 1]

### Data Properties
- Value range: [-1, 1] (normalized)
- Statistical measures:
  - Mean: -0.008 (approximate)
  - Standard deviation: 0.534 (approximate)