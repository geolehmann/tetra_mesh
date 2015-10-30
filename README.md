# tetimport
Tetrahedral raytracing using Tetgen
===================================

Ray-tracing and path tracing have been a constant research area of computer graphics and
other disciplines, e.g. seismic raytracing in geophysics etc.

A major drawback of this techniques is the high computational cost, in particular
due to ray-triangle intersection and finding the intersected triangle. A common solution is to use
acceleration structures like BVHs (Bounding volume hierarchies), Octrees or Grids.

In their 2008 paper "Accelerating Raytracing Using COnstrained Tetrahedralizations" Lagae & Dutre
explored the idea of using a tetrahedral mesh for fast search of the intersection point. Their idea
was to subdivide the complete scene into tetrahedra and use neighbor-relations between tetrahedra
to quickly find the triangle intersected by a ray.

This project aims to expand their idea by using Nvidia CUDA to speed up the calculations.
Tetgen developed by Hang Si(http://wias-berlin.de/software/tetgen/) is used for tetrahedralization
of the scene.

**Current status (08/10/2015):**

Import Tetgen files .node/.ele/.face/.edge/.neigh  - done!  
Ray-tetrahedra intersection routine - done!  
Ray-in-tetrahedra-testing routine - done!  

Todo:  
Function for finding tetrahedra with camera position  
Replace Vec class with CUDA kernel
Ray-triangle intersection routine for raytracing




