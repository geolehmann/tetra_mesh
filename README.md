Tetrahedral pathtracing using Tetgen
===================================

Ray-tracing and path tracing have been a constant research area in computer graphics and
other disciplines, e.g. seismic raytracing in geophysics or heat flow modeling. The tetrahedral mesh structure
is suitable for modeling deformation of volumetric objects and fracture generation such as in rock mechanics.

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

<p align="center">
  <img src="https://raw.githubusercontent.com/clehmann-geo/tetra_mesh/master/Alte%20Versuche/14_05_mesh_visualization.jpg" width="350"/>
</p>
Fig. 1. Visualization of the tetrahedral mesh. 


The pathtracer code is based on the pathtracer by Samuel Lapere (https://github.com/straaljager/GPU-path-tracing-tutorial-3)
and the smallpt pathtracer (http://kevinbeason.com/smallpt/).

**References:**

Hang Si (2015). _TetGen, a Delaunay-Based Quality Tetrahedral Mesh Generator._ ACM Trans. on Mathematical Software. 41 (2), Article 11 (February 2015), 36 pages.

Lagae, A. and Dutré, P. (2008). _Accelerating Ray Tracing using Constrained Tetrahedralizations._ Computer Graphics Forum, 27: 1303–1312.

Sanzenbacher, S. (2010) Darstellung von Kaustiken und Lichtstreuung in Interaktiven Anwendungen. _Unpublished diploma thesis, Institut für Visualisierung und Interaktive Systeme, University Stuttgart._
  
**Current status (04/12/2016):**  

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
visualize mesh - done!!  
print debug info onscreen - done!!
  
Todo:   
small bugfixing
include participating media into the pathtracer  
Implement mesh deformation - Müller et al. (2015)  
properties of tetrahedra - friction etc.
