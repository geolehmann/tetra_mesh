Tetrahedral raytracing using Tetgen
===================================

Ray-tracing and path tracing have been a constant research area of computer graphics and
other disciplines, e.g. seismic raytracing in geophysics or heat flow modeling.

A major drawback of this techniques is the high computational cost, in particular
due to ray-mesh intersection and finding the intersected triangle. A common solution is to use
acceleration structures like BVHs (Bounding volume hierarchies), Octrees or Grids.

In their 2008 paper "Accelerating Raytracing Using Constrained Tetrahedralizations" Lagae & Dutré
explored the idea of using a tetrahedral mesh for fast search of the intersection point. Their approach
was to subdivide the complete scene into tetrahedra and use neighbor-relations between tetrahedra
to quickly find the triangle intersected by a ray.

This project aims to expand their idea by using GPGPU with Nvidia CUDA to speed up the calculations.
The Tetgen software developed by Hang Si(http://wias-berlin.de/software/tetgen/) is used for tetrahedralization
of the scene.  
  
**Current status (01/04/2016):**  

Import Tetgen files .node/.ele/.face/.edge/.neigh  - done!  
Ray-tetrahedra intersection routine - done!  
Ray-in-tetrahedra-testing routine - done!  
Find tetrahedra with camer position - done!  
CUDA kernel for ray traversal - done!  
Ray-triangle intersection routine for raytracing - done!  
visualize depth information - done! 
mesh in unified memory - done!!  
cuda raytracing kernel - done!!  
screen output - done!!  
keyboard input - done!!  
mouse input - done!!  
support for specular/refractive/diffuse materials - done!!  
path tracing - done!!!  
  
Todo:  
improve path tracer performance 
test with bigger scenes  


 




