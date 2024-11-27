// Axis-aligned bounding box (Aabb)

function Aabb(v0, v1, v2)
{
  if(v2) {
    this.min = vec3(Math.min(v0[0], Math.min(v1[0], v2[0])), Math.min(v0[1], Math.min(v1[1], v2[1])), Math.min(v0[2], Math.min(v1[2], v2[2])));
    this.max = vec3(Math.max(v0[0], Math.max(v1[0], v2[0])), Math.max(v0[1], Math.max(v1[1], v2[1])), Math.max(v0[2], Math.max(v1[2], v2[2])));
  }
  else if(v1) {
    this.min = vec3(v0[0], v0[1], v0[2]);
    this.max = vec3(v1[0], v1[1], v1[2]);
  }
  else if(v0) {
    this.min = vec3(v0.min[0], v0.min[1], v0.min[2]);
    this.max = vec3(v0.max[0], v0.max[1], v0.max[2]);
  }
  else {
    this.min = vec3(1.0e37, 1.0e37, 1.0e37);
    this.max = vec3(-1.0e37, -1.0e37, -1.0e37);
  }
  return this;
}

Aabb.prototype.include = function(x)
{
  if(x.min && x.max) {
    for(var i = 0; i < 3; ++i) {
      this.min[i] = Math.min(this.min[i], x.min[i]);
      this.max[i] = Math.max(this.max[i], x.max[i]);
    }
  }
  else {
    for(var i = 0; i < 3; ++i) {
      this.min[i] = Math.min(this.min[i], x[i]);
      this.max[i] = Math.max(this.max[i], x[i]);
    }
  }
}

Aabb.prototype.set = function(v0, v1, v2)
{
  if(v2) {
    this.min = vec3(Math.min(v0[0], Math.min(v1[0], v2[0])), Math.min(v0[1], Math.min(v1[1], v2[1])), Math.min(v0[2], Math.min(v1[2], v2[2])));
    this.max = vec3(Math.max(v0[0], Math.max(v1[0], v2[0])), Math.max(v0[1], Math.max(v1[1], v2[1])), Math.max(v0[2], Math.max(v1[2], v2[2])));
  }
  else if(v1) {
    this.min = v0;
    this.max = v1;
  }
  else {
    this.min = vec3(1.0e37, 1.0e37, 1.0e37);
    this.max = vec3(-1.0e37, -1.0e37, -1.0e37);
  }
}

Aabb.prototype.center = function(dim)
{
  if(dim)
    return (this.min[dim] + this.max[dim])*0.5;
  return vec3((this.min[0] + this.max[0])*0.5, (this.min[1] + this.max[1])*0.5, (this.min[2] + this.max[2])*0.5);
}

Aabb.prototype.extent = function(dim)
{
  if(dim)
    return this.max[dim] - this.min[dim];
  return vec3(this.max[0] - this.min[0], this.max[1] - this.min[1], this.max[2] - this.min[2]);
}

Aabb.prototype.volume = function()
{
  let d = this.extent();
  return d[0]*d[1]*d[2];
}

Aabb.prototype.area = function()
{
  return 2.0*this.halfArea();
}

Aabb.prototype.halfArea = function()
{
  let d = this.extent();
  return d[0]*d[1] + d[1]*d[2] + d[2]*d[0];
}

Aabb.prototype.longestAxis = function()
{
  let d = this.extent();
  if(d[0] > d[1])
    return d[0] > d[2] ? 0 : 2;
  return d[1] > d[2] ? 1 : 2;
}

Aabb.prototype.maxExtent = function()
{
  return this.extent(this.longestAxis());
}

Aabb.prototype.intersects = function(other)
{
  if(other.min[0] > this.max[0] || other.max[0] < this.min[0] ) return false;
  if(other.min[1] > this.max[1] || other.max[1] < this.min[1] ) return false;
  if(other.min[2] > this.max[2] || other.max[2] < this.min[2] ) return false;
  return true;
}
