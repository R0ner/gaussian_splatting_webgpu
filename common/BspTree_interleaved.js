// 02562 Rendering Framework
// Inspired by BSP tree in GEL (https://www2.compute.dtu.dk/projects/GEL/)
// BSP tree in GEL originally written by Bent Dalgaard Larsen.
// This file written by Jeppe Revall Frisvad, 2023
// Copyright (c) DTU Compute 2023

const max_objects = 4; // maximum number of objects in a leaf
const max_level = 20;  // maximum number of levels in the tree
const f_eps = 1.0e-6;
const d_eps = 1.0e-12;
const BspNodeType = {
  bsp_x_axis: 0,
  bsp_y_axis: 1,
  bsp_z_axis: 2,
  bsp_leaf:   3,
};
var tree_objects = [];
var root = null;
var treeIds, bspTree, bspPlanes;

function AccObj(idx, v0, v1, v2)
{
  this.prim_idx = idx;
  this.bbox = new Aabb(v0, v1, v2);
  return this;
}

function BspTree(objects)
{
  this.max_level = max_level;
  this.count = objects.length;
  this.id = 0;
  this.bbox = new Aabb();
  for(var i = 0; i < objects.length; ++i)
    this.bbox.include(objects[i].bbox);
  subdivide_node(this, this.bbox, 0, objects);
  return this;
}

function subdivide_node(node, bbox, level, objects)
{
  const TESTS = 4;

  if(objects.length <= max_objects || level == max_level)
  {
    node.axis_leaf = BspNodeType.bsp_leaf;
    node.id = tree_objects.length;
    node.count = objects.length;
    node.plane = 0.0;

    for(var i = 0; i < objects.length; ++i)
      tree_objects.push(objects[i]);
  }
  else
  {
    let left_objects = [];
    let right_objects = [];
    node.left = new Object();
    node.right = new Object();

    var min_cost = 1.0e27;
    for(var i = 0; i < 3; ++i)
    {
      for(var k = 1; k < TESTS; ++k)
      {
        let left_bbox = new Aabb(bbox);
        let right_bbox = new Aabb(bbox);
        const max_corner = bbox.max[i];
        const min_corner = bbox.min[i];
        const center = (max_corner - min_corner)*k/TESTS + min_corner;
        left_bbox.max[i] = center;
        right_bbox.min[i] = center;

        // Try putting the triangles in the left and right boxes
        var left_count = 0;
        var right_count = 0;
        for(var j = 0; j < objects.length; ++j)
        {
          let obj = objects[j];
          left_count += left_bbox.intersects(obj.bbox);
          right_count += right_bbox.intersects(obj.bbox);
        }

        const cost = left_count*left_bbox.area() + right_count*right_bbox.area();
        if(cost < min_cost)
        {
          min_cost = cost;
          node.axis_leaf = i;
          node.plane = center;
          node.left.count = left_count;
          node.left.id = 0;
          node.right.count = right_count;
          node.right.id = 0;
        }
      }
    }
    
    // Now chose the right splitting plane
    const max_corner = bbox.max[node.axis_leaf];
    const min_corner = bbox.min[node.axis_leaf];
    const size = max_corner - min_corner;
    const diff = f_eps < size/8.0 ? size/8.0 : f_eps;
    let center = node.plane;

    if(node.left.count == 0)
    {
      // Find min position of all triangle vertices and place the center there
      center = max_corner;
      for(var j = 0; j < objects.length; ++j)
      {
        let obj = objects[j];
        obj_min_corner = obj.bbox.min[node.axis_leaf];
        if(obj_min_corner < center)
          center = obj_min_corner;
      }
      center -= diff;
    }
    if(node.right.count == 0)
    {
      // Find max position of all triangle vertices and place the center there
      center = min_corner;
      for(var j = 0; j < objects.length; ++j)
      {
        let obj = objects[j];
        obj_max_corner = obj.bbox.max[node.axis_leaf];
        if(obj_max_corner > center)
          center = obj_max_corner;
      }
      center += diff;
    }

    node.plane = center;
    let left_bbox = new Aabb(bbox);
    let right_bbox = new Aabb(bbox);
    left_bbox.max[node.axis_leaf] = center;
    right_bbox.min[node.axis_leaf] = center;

    // Now put the triangles in the right and left node
    for(var j = 0; j < objects.length; ++j)
    {
      let obj = objects[j];
      if(left_bbox.intersects(obj.bbox))
        left_objects.push(obj);
      if(right_bbox.intersects(obj.bbox))
        right_objects.push(obj);
    }

    objects = [];
    subdivide_node(node.left, left_bbox, level + 1, left_objects);
    subdivide_node(node.right, right_bbox, level + 1, right_objects);
  }
}

function build_bsp_tree(drawingInfo, device, buffers)
{
  var objects = [];
  for(var i = 0; i < drawingInfo.indices.length/4; ++i) {
    let face = [drawingInfo.indices[i*4]*8, drawingInfo.indices[i*4 + 1]*8, drawingInfo.indices[i*4 + 2]*8];
    let v0 = vec3(drawingInfo.attribs[face[0]], drawingInfo.attribs[face[0] + 1], drawingInfo.attribs[face[0] + 2]);
    let v1 = vec3(drawingInfo.attribs[face[1]], drawingInfo.attribs[face[1] + 1], drawingInfo.attribs[face[1] + 2]);
    let v2 = vec3(drawingInfo.attribs[face[2]], drawingInfo.attribs[face[2] + 1], drawingInfo.attribs[face[2] + 2]);
    let acc_obj = new AccObj(i, v0, v1, v2);
    objects.push(acc_obj);
  }
  root = new BspTree(objects);
  treeIds = new Uint32Array(tree_objects.length);
  for(var i = 0; i < tree_objects.length; ++i)
    treeIds[i] = tree_objects[i].prim_idx;
  const bspTreeNodes = (1<<(max_level + 1)) - 1;
  bspPlanes = new Float32Array(bspTreeNodes);
  bspTree = new Uint32Array(bspTreeNodes*4);
  
  function build_bsp_array(node, level, branch)
  {
    if(level > max_level)
      return;
    let idx = (1<<level) - 1 + branch;
    bspTree[idx*4] = node.axis_leaf + (node.count<<2);
    bspTree[idx*4 + 1] = node.id;
    bspTree[idx*4 + 2] = (1<<(level + 1)) - 1 + 2*branch;
    bspTree[idx*4 + 3] = (1<<(level + 1)) + 2*branch;
    bspPlanes[idx] = node.plane;
    if(node.axis_leaf === BspNodeType.bsp_leaf)
      return;
    build_bsp_array(node.left, level + 1, branch*2);
    build_bsp_array(node.right, level + 1, branch*2 + 1);
  }
  build_bsp_array(root, 0, 0);

  buffers.attribs = device.createBuffer({
    size: drawingInfo.attribs.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
  });
  device.queue.writeBuffer(buffers.attribs, 0, drawingInfo.attribs);

  buffers.colors = device.createBuffer({
    size: drawingInfo.colors.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
  });
  device.queue.writeBuffer(buffers.colors, 0, drawingInfo.colors);

//   buffers.indices = device.createBuffer({
//     size: drawingInfo.indices.byteLength, 
//     usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
//   });
//   device.queue.writeBuffer(buffers.indices, 0, drawingInfo.indices);
  
  buffers.treeIds = device.createBuffer({
    size: treeIds.byteLength, 
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
  });
  device.queue.writeBuffer(buffers.treeIds, 0, treeIds);

  buffers.bspTree = device.createBuffer({
    size: bspTree.byteLength, 
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
  });
  device.queue.writeBuffer(buffers.bspTree, 0, bspTree);

  buffers.bspPlanes = device.createBuffer({
    size: bspPlanes.byteLength, 
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
  });
  device.queue.writeBuffer(buffers.bspPlanes, 0, bspPlanes);

  const bbox = flatten([vec4(root.bbox.min), vec4(root.bbox.max)]);
  buffers.aabb = device.createBuffer({
    size: bbox.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffers.aabb, 0, bbox);

  return buffers;
}


function intersect_triangle(drawingInfo, r, hit, idx)
{
  let face_x = drawingInfo.indices[idx*4]*8;
  let face_y = drawingInfo.indices[idx*4 + 1]*8;
  let face_z = drawingInfo.indices[idx*4 + 2]*8;
  let v0 = vec3(drawingInfo.attribs[face_x], drawingInfo.attribs[face_x + 1], drawingInfo.attribs[face_x + 2]);
  let v1 = vec3(drawingInfo.attribs[face_y], drawingInfo.attribs[face_y + 1], drawingInfo.attribs[face_y + 2]);
  let v2 = vec3(drawingInfo.attribs[face_z], drawingInfo.attribs[face_z + 1], drawingInfo.attribs[face_z + 2]);
  let e0 = subtract(v1, v0);
  let e1 = subtract(v2, v0);
  let n = cross(e0, e1);
  let denom = dot(r.direction, n);
  if(Math.abs(denom) < 1.0e-8) { return false; }
  let a = vec3((v0[0] - r.origin[0])/denom, (v0[1] - r.origin[1])/denom, (v0[2] - r.origin[2])/denom);
  let t = dot(a, n);
  if(t < r.tmin || t > r.tmax) { return false; }
  let b = cross(a, r.direction);
  let beta = dot(b, e1);
  let gamma = -dot(b, e0);
  if(beta >= 0.0 && gamma >= 0.0 && beta + gamma <= 1.0) {
    hit.has_hit = true;
    hit.dist = t;
    hit.position = vec3(r.origin[0] + t*r.direction[0], r.origin[1] + t*r.direction[1], r.origin[2] + t*r.direction[2]);
    hit.normal = normalize(n);
    return true;
  }
  return false;
}

function intersect_min_max(r)
{
  var tmin = -1.0e32;
  var tmax = 1.0e32;
  for(var i = 0; i < 3; ++i)
    if(Math.abs(r.direction[i]) > 1.0e-8)
    {
      const p1 = (root.bbox.min[i] - r.origin[i])/r.direction[i];
      const p2 = (root.bbox.max[i] - r.origin[i])/r.direction[i];
      const pmin = Math.min(p1, p2);
      const pmax = Math.max(p1, p2);
      tmin = Math.max(tmin, pmin);
      tmax = Math.min(tmax, pmax);
    }
  if(tmin > tmax || tmin > r.tmax || tmax < r.tmin)
    return false;
  r.tmin = Math.max(tmin - 1.0e-4, r.tmin);
  r.tmax = Math.min(tmax + 1.0e-4, r.tmax);
  return true;
}

function intersect_bsp_array(ray, hit)
{
  let branch_node = new Uint32Array(max_level*2);
  let branch_ray = new Float32Array(max_level*2);
  let branch_lvl = 0;
  let near_node = 0;
  let far_node = 0;
  let t = 0.0;
  let node = 0;
  for(let i = 0; i <= max_level; ++i)
  {
    let node_axis_leaf = bspTree[node*4]&3;
    if(node_axis_leaf === BspNodeType.bsp_leaf)
    {
      const node_count = bspTree[node*4]>>2;
      let found = false;
      for(let j = 0; j < node_count; ++j)
      {
        const node_id = bspTree[node*4 + 1];
        const obj_idx = treeIds[node_id + j];
        if(intersect_triangle(ray, hit, obj_idx)) {
          ray.tmax = hit.dist;
          found = true;
        }
      }
      if(found) { return true; }
      else if(branch_lvl === 0) { return false; }
      else {
        --branch_lvl;
        i = branch_node[branch_lvl*2];
        node = branch_node[branch_lvl*2 + 1];
        ray.tmin = branch_ray[branch_lvl*2];
        ray.tmax = branch_ray[branch_lvl*2 + 1];
        continue;
      }
    }

    const axis_direction = ray.direction[node_axis_leaf];
    const axis_origin = ray.origin[node_axis_leaf];
    if(axis_direction >= 0.0) {
      near_node = bspTree[node*4 + 2]; // left
      far_node = bspTree[node*4 + 3];  // right
    }
    else {
      near_node = bspTree[node*4 + 3]; // right
      far_node = bspTree[node*4 + 2];  // left
    }
    
    const node_plane = bspPlanes[node];
    const denom = Math.abs(axis_direction) < d_eps ? d_eps : axis_direction;
    t = (node_plane - axis_origin)/denom;

    if(t > ray.tmax) { node = near_node; }
    else if(t < ray.tmin) { node = far_node; }
    else {
      branch_node[branch_lvl*2] = i;
      branch_node[branch_lvl*2 + 1] = far_node;
      branch_ray[branch_lvl*2] = t;
      branch_ray[branch_lvl*2 + 1] = ray.tmax;
      ++branch_lvl;
      ray.tmax = t;
      node = near_node;
    }
  }
  return false;
}

function intersect_trimesh(ray, hit)
{
  var subtree = [];
  var near_node = null;
  var far_node = null;
  var t = 0.0;
  var node = root;
  for(let i = 0; i <= root.max_level; ++i)
  {
    if(node.axis_leaf === BspNodeType.bsp_leaf)
    {
      var found = false;
      for(let j = 0; j < node.count; ++j)
      {
        const obj = tree_objects[node.id + j];
        if(intersect_triangle(ray, hit, obj.prim_idx))
        {
          ray.tmax = hit.dist;
          found = true;
        }
      }
      if(found)
        return true;
      else if(subtree.length === 0)
        return false;
      else
      {
        branch = subtree.pop();
        i = branch.i;
        ray.tmin = branch.tmin;
        ray.tmax = branch.tmax;
        node = branch.node;
        continue;
      }
    }

    const axis_direction = ray.direction[node.axis_leaf];
    const axis_origin = ray.origin[node.axis_leaf];
    if(axis_direction >= 0.0)
    {
      near_node = node.left;
      far_node = node.right;
    }
    else
    {
      near_node = node.right;
      far_node = node.left;
    }
    
    const denom = Math.abs(axis_direction) < d_eps ? d_eps : axis_direction;
    t = (node.plane - axis_origin)/denom;

    if(t > ray.tmax)
      node = near_node;
    else if(t < ray.tmin)
      node = far_node;
    else 
    {
      var branch = new Object();
      branch.i = i;
      branch.tmin = t;
      branch.tmax = ray.tmax;
      branch.node = far_node;
      subtree.push(branch);
      ray.tmax = t;
      node = near_node;
    }
  }
}

function intersect_node(ray, hit, node)
{
  if(node.axis_leaf === BspNodeType.bsp_leaf)
  {
    var found = false;
    for(var i = 0; i < node.count; ++i)
    {
      const obj = tree_objects[node.id + i];
      if(intersect_triangle(ray, hit, obj.prim_idx))
      {
        ray.tmax = hit.dist;
        found = true;
      }
    }
    return found;
  }
  else
  {
    var near_node = null;
    var far_node = null;
    const axis_direction = ray.direction[node.axis_leaf];
    const axis_origin = ray.origin[node.axis_leaf];
    if(axis_direction >= 0.0)
    {
      near_node = node.left;
      far_node = node.right;
    }
    else
    {
      near_node = node.right;
      far_node = node.left;
    }

    var t = 0.0;
    if(Math.abs(axis_direction) < d_eps)
      t = (node.plane - axis_origin)/d_eps;
    else
      t = (node.plane - axis_origin)/axis_direction;

    if(t > ray.tmax)
      return intersect_node(ray, hit, near_node);
    else if(t < ray.tmin)
      return intersect_node(ray, hit, far_node);
    else 
    {
      var t_max = ray.tmax;
      ray.tmax = t;
      if(intersect_node(ray, hit, near_node))
        return true;
      else
      {
        ray.tmin = t;
        ray.tmax = t_max;
        return intersect_node(ray, hit, far_node);
      }
    }
  }
}