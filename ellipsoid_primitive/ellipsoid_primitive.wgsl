struct VSOut {
    @builtin(position) position: vec4f,
    @location(0) coords : vec2f,
};

struct FSOut {
    @location(0) frame: vec4f,
    @location(1) accum: vec4f,
};

struct Uniforms {
    aspect: f32,
    cam_const: f32,
    canvas_width: u32,
    canvas_height: u32,
    use_texture: u32,
    frame_count: u32,
    render_object: i32,
};

struct Aabb {
    min: vec3f,
    max: vec3f,
};

struct Attributes {
    vposition: vec3f,
    normal: vec3f,
};

// Material struct
struct Material {
    color: vec4f,
    emission: vec4f,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var renderTexture: texture_2d<f32>;
@group(0) @binding(2) var<storage> attribs: array<Attributes>;
@group(0) @binding(3) var<storage> meshFaces: array<vec4u>;
@group(0) @binding(4) var<storage> materials: array<Material>;
@group(0) @binding(5) var<storage> lightIndices: array<u32>;
@group(0) @binding(6) var<storage> treeIds: array<u32>;
@group(0) @binding(7) var<storage> bspTree: array<vec4u>;
@group(0) @binding(8) var<storage> bspPlanes: array<f32>;
@group(0) @binding(9) var<uniform> aabb: Aabb;

const MAX_LEVEL = 20u;
const BSP_LEAF = 3u;
var<private> branch_node: array<vec2u, MAX_LEVEL>;
var<private> branch_ray: array<vec2f, MAX_LEVEL>;

// Define Ray struct
struct Ray {
    origin: vec3f,
    direction: vec3f,
    tmin: f32,
    tmax: f32
};

// Hit information
struct HitInfo {
    has_hit: bool,
    dist: f32,
    position: vec3f,
    normal: vec3f,
    // color: vec3f, This is replaced by diffuse and ambient compoments (below)
    diffuse: vec3f,
    ambient: vec3f, 
    emission: vec3f,
    specular: f32, // Specular reflectance (rho_s)
    shininess: f32, // Phong exponent (s)
    ior1_over_ior2: f32, // Reciprocal relative index of refraction (n1/n2)
    shader: u32, // Shader index
    texcoords: vec2f,
    emit: bool,
    rgb_factor: vec3f,
    extinction_coef: vec3f,
};

struct Onb {
    tangent: vec3f,
    binormal: vec3f,
    normal: vec3f,
};

// Light struct
struct Light {
    L_i: vec3f,
    w_i: vec3f,
    dist: f32
};

// PRNG xorshift seed generator by NVIDIA
fn tea(val0: u32, val1: u32) -> u32 {
    const N = 16u; // User specified number of iterations
    var v0 = val0; var v1 = val1; var s0 = 0u;
    for(var n = 0u; n < N; n++) {
        s0 += 0x9e3779b9;
        v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
        v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e);
    }
    return v0;
}

// Generate random unsigned int in [0, 2^31)
fn mcg31(prev: ptr<function, u32>) -> u32 {
    const LCG_A = 1977654935u; // Multiplier from Hui-Ching Tang [EJOR 2007]
    *prev = (LCG_A * (*prev)) & 0x7FFFFFFF;
    return *prev;
}

// Generate random float in [0, 1)
fn rnd(prev: ptr<function, u32>) -> f32 {
    return f32(mcg31(prev)) / f32(0x80000000);
}

fn check_ray_hit(r: Ray, hit: HitInfo, t_prime: f32) -> bool {
    if((t_prime < r.tmin) || (t_prime > r.tmax) || (hit.has_hit && (t_prime > hit.dist))) {
        return true;
    }
    return false;
}

// Given spherical coordinates, where theta is the polar angle and phi is the
// azimuthal angle, this function returns the corresponding direction vector
fn spherical_direction(sin_theta: f32, cos_theta: f32, phi: f32) -> vec3f {
    return vec3f(sin_theta*cos(phi), sin_theta*sin(phi), cos_theta);
}

// Given a direction vector v sampled around the z-axis of a local coordinate system,
// this function applies the same rotation to v as is needed to rotate the z-axis to
// the actual direction n that v should have been sampled around
// [Frisvad, Journal of Graphics Tools 16, 2012;
// Duff et al., Journal of Computer Graphics Techniques 6, 2017].
fn rotate_to_normal(n: vec3f, v: vec3f) -> vec3f {
    let s = sign(n.z + 1.0e-16f);
    let a = -1.0f/(1.0f + abs(n.z));
    let b = n.x*n.y*a;
    return vec3f(1.0f + n.x*n.x*a, b, -s*n.x)*v.x + vec3f(s*b, s*(1.0f + n.y*n.y*a), -n.y)*v.y + n*v.z;
}

// Fresnel reflectance.
fn fresnel_R(cos_theta_i: f32, cos_theta_t: f32, ni_over_nt: f32) -> f32 {
    // ni_over_nt: Recpiprocal relative index of refraction (n_i / n_t).
    // Denote arguments in order: a, b, and c.
    let ca = ni_over_nt * cos_theta_i;
    let cb = ni_over_nt * cos_theta_t;
    let r_perp = (ca - cos_theta_t) / (ca + cos_theta_t);
    let r_para = (cos_theta_i - cb) / (cos_theta_i + cb);
    return 0.5 * (r_perp * r_perp + r_para * r_para);
}

fn intersect_min_max(r: ptr<function, Ray>) -> bool {
    let p1 = (aabb.min - r.origin)/r.direction;
    let p2 = (aabb.max - r.origin)/r.direction;
    let pmin = min(p1, p2);
    let pmax = max(p1, p2);
    let tmin = max(pmin.x, max(pmin.y, pmin.z));
    let tmax = min(pmax.x, min(pmax.y, pmax.z));
    if(tmin > tmax || tmin > r.tmax || tmax < r.tmin) {
    return false;
    }
    r.tmin = max(tmin - 1.0e-3f, r.tmin);
    r.tmax = min(tmax + 1.0e-3f, r.tmax);
    return true;
}

fn intersect_trimesh(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> bool {
    var branch_lvl = 0u;
    var near_node = 0u;
    var far_node = 0u;
    var t = 0.0f;
    var node = 0u;
    for(var i = 0u; i <= MAX_LEVEL; i++) {
        let tree_node = bspTree[node];
        let node_axis_leaf = tree_node.x&3u;
        if(node_axis_leaf == BSP_LEAF) {
            // A leaf was found
            let node_count = tree_node.x>>2u;
            let node_id = tree_node.y;
            var found = false;
            for(var j = 0u; j < node_count; j++) {
                let obj_idx = treeIds[node_id + j];
                if(intersect_triangle(*r, hit, i32(obj_idx))) {
                    r.tmax = hit.dist;
                    found = true;
                }
            }
            if(found) { return true; }
            else if(branch_lvl == 0u) { return false; }
            else {
                branch_lvl--;
                i = branch_node[branch_lvl].x;
                node = branch_node[branch_lvl].y;
                r.tmin = branch_ray[branch_lvl].x;
                r.tmax = branch_ray[branch_lvl].y;
                continue;
            }
        }
        let axis_direction = r.direction[node_axis_leaf];
        let axis_origin = r.origin[node_axis_leaf];
        if(axis_direction >= 0.0f) {
        near_node = tree_node.z; // left
        far_node = tree_node.w; // right
        }
        else {
        near_node = tree_node.w; // right
        far_node = tree_node.z; // left
        }
        let node_plane = bspPlanes[node];
        let denom = select(axis_direction, 1.0e-8f, abs(axis_direction) < 1.0e-8f);
        t = (node_plane - axis_origin)/denom;
        if(t > r.tmax) { node = near_node; }
        else if(t < r.tmin) { node = far_node; }
        else {
            branch_node[branch_lvl].x = i;
            branch_node[branch_lvl].y = far_node;
            branch_ray[branch_lvl].x = t;
            branch_ray[branch_lvl].y = r.tmax;
            branch_lvl++;
            r.tmax = t;
            node = near_node;
        }
    }
    return false;
}

fn intersect_plane(r: Ray, hit: ptr<function, HitInfo>, position: vec3f, onb: Onb) -> bool {
    let denominator = dot(r.direction, onb.normal);
    
    if(abs(denominator) < 1e-8) {
        return false;
    }
    
    let t_prime = dot(position - r.origin, onb.normal) / denominator;

    if(check_ray_hit(r, *hit, t_prime)) {
        return false;
    }

    (*hit).has_hit = true;
    (*hit).dist = t_prime;
    (*hit).position = r.origin + t_prime * r.direction;
    (*hit).normal = onb.normal;
    (*hit).shader = 1;
    
    (*hit).ambient = vec3f(0.1, 0.7, 0.0);
    (*hit).diffuse = vec3f(0.1, 0.7, 0.0);
    

    return true;
}

fn intersect_triangle(r: Ray, hit: ptr<function, HitInfo>, faceidx: i32) -> bool {
    let v_indices = meshFaces[faceidx];
    let v0 = attribs[v_indices[0]].vposition;
    let e0 = attribs[v_indices[1]].vposition - v0;
    let e1 = attribs[v_indices[2]].vposition - v0;
    
    let normal = cross(e0, e1);
    
    let a = v0 - r.origin; 
    let k = dot(r.direction, normal);
    
    let t_prime = dot(a, normal) / k;
    
    if(check_ray_hit(r, *hit, t_prime)) {
        return false;
    }

    let h = cross(a, r.direction);
    
    let beta = dot(h, e1) / k;
    let gamma = -dot(h, e0) / k; 

    if((beta < 0) || (gamma < 0) || ((beta + gamma) > 1)) {
        return false;
    }
    
    // let material = materials[matIndices[faceidx]];
    // Material index is the last element of the meshface/vertex indices.
    let material = materials[v_indices[3]];

    (*hit).has_hit = true;
    (*hit).dist = t_prime;
    (*hit).position = r.origin + t_prime * r.direction;
    (*hit).normal = normalize(
        (1 - beta - gamma) * attribs[v_indices[0]].normal + 
        beta * attribs[v_indices[1]].normal + 
        gamma * attribs[v_indices[2]].normal
    );
    (*hit).emission = material.emission.rgb;
    (*hit).diffuse = material.color.rgb;
    (*hit).shader = 1;

    return true;
}

fn intersect_sphere(r: Ray, hit: ptr<function, HitInfo>, center: vec3f, radius: f32, shader: u32) -> bool {
    let omc = r.origin - center;
    let b2 = dot(omc, r.direction);
    let c = dot(omc, omc) - radius * radius;
    let b2_sq = b2 * b2;
    
    if(b2_sq - c < 0) {
        return false;
    }
    let u = sqrt(b2_sq - c);
    let t1_prime = - b2 - u;
    let t2_prime = - b2 + u;

    var t_prime = 0.0;

    if(t1_prime >= r.tmin) {
        t_prime = t1_prime;
    }
    else {
        t_prime = t2_prime;
    }

    if(check_ray_hit(r, *hit, t_prime)) {
        return false;
    }

    (*hit).has_hit = true;
    (*hit).dist = t_prime;
    (*hit).position = r.origin + t_prime * r.direction;
    (*hit).normal = normalize((*hit).position - center);
    (*hit).ambient = vec3f(0.0, 0.0, 0.0);
    (*hit).diffuse = vec3f(0.0, 0.0, 0.0);
    (*hit).ior1_over_ior2 = 1.0 / 1.5; // Hit from outside of the material.
    (*hit).shader = shader;
    
    if(dot(r.direction, (*hit).normal) > 0.0) {
        (*hit).ior1_over_ior2 = 1.0 / (*hit).ior1_over_ior2;
        (*hit).normal = -(*hit).normal;
    }
    
    return true;
}

fn intersect_ellipsoid(r: Ray, hit: ptr<function, HitInfo>, center: vec3f, radii: vec3f, rotation: mat3x3f, shader: u32) -> bool {
    // Adapted from https://iquilezles.org/articles/intersectors/ and
    // additional code by GRAPHDECO research group, https://team.inria.fr/graphdeco.
    
    let r_e_origin  = rotation * (r.origin - center);
    let r_e_direction  = rotation * r.direction;
    
    // let ocn = (r.origin - center) / radii;
    // let rdn = r.direction / radii;
    let ocn = r_e_origin / radii;
    let rdn = r_e_direction / radii;
    
    let a = dot( rdn, rdn );
    let b = dot( ocn, rdn );
    let c = dot( ocn, ocn );

    var h = b*b - a*(c-1.0);

    if(h < 0.0) {
        return false;
    }
    h = sqrt(h);
    let t1_prime = (- b - h) / a;
    let t2_prime = (- b + h) / a;

    var t_prime = 0.0;

    if(t1_prime >= r.tmin) {
        t_prime = t1_prime;
    }
    else {
        t_prime = t2_prime;
    }

    if(check_ray_hit(r, *hit, t_prime)) {
        return false;
    }

    let position_e = r_e_origin + t_prime * r_e_direction;
    let normal_e = normalize(position_e / (radii * radii));

    (*hit).has_hit = true;
    (*hit).position = r.origin + t_prime * r.direction;
    (*hit).dist = t_prime;
    (*hit).normal = normalize(normal_e * rotation);
    (*hit).ambient = vec3f(1.0, 0.0, 0.0);
    (*hit).diffuse = vec3f(1.0, 1.0, 1.0);
    (*hit).ior1_over_ior2 = 1.0 / 1.5; // Hit from outside of the material.
    (*hit).shader = shader;
    
    if(dot(r.direction, (*hit).normal) > 0.0) {
        (*hit).ior1_over_ior2 = 1.0 / (*hit).ior1_over_ior2;
        (*hit).normal = -(*hit).normal;
    }
    
    return true;
}

fn intersect_scene(r: ptr<function, Ray>, hit : ptr<function, HitInfo>) -> bool {
    // Define scene data as constants.
    // Call an intersection function for each object.
    // For each intersection found, update (*r).tmax and store additional info about the hit.

    let center_glass = vec3f(278.0, 280.0, 279.5); 
    // let center_glass = vec3f(278.0, 300.0, 250); 
    let radius_glass = 90.0;
    let radii_glass = vec3f(90.0, 60.0, 60.0) * 1.25;
    let extinction_coef_glass = vec3f(0.0, 0.0, 0.0);
    // let extinction_coef_glass = vec3f(0.0, 0.01, 0.0);
    // let rotmat_glass = mat3x3f(
    //     1.0, 0.0, 0.0,
    //     0.0, 1.0, 0.0,
    //     0.0, 0.0, 1.0,
    // );
    let ang = radians(0.0);
    let rotmat_glass = mat3x3f(
        cos(ang), sin(ang), 0.0, 
        -sin(ang), cos(ang), 0.0,
        0.0, 0.0, 1.0,
    );

    // let center_mirror = vec3f(420.0, 90.0, 370.0); 
    // let radius_mirror = 90.0;

    // let did_hit_glass_sphere = intersect_sphere(*r, hit, center_glass, radius_glass, 6);
    // if(did_hit_glass_sphere) { 
    //     (*r).tmax = (*hit).dist; 
    //     (*hit).extinction_coef = extinction_coef_glass;
    // }
    
    let did_hit_glass_ellipsoid = intersect_ellipsoid(*r, hit, center_glass, radii_glass, rotmat_glass, 1);
    if(did_hit_glass_ellipsoid) { 
        (*r).tmax = (*hit).dist; 
        (*hit).extinction_coef = extinction_coef_glass;
    }
    // let did_hit_glass_mirror = intersect_sphere(*r, hit, center_mirror, radius_mirror, 3);
    // if(did_hit_glass_mirror) { (*r).tmax = (*hit).dist; }

    if (intersect_min_max(r)) {
        let did_hit_trimesh = intersect_trimesh(r, hit);
        if(did_hit_trimesh) { (*r).tmax = (*hit).dist; }
    }

    return (*hit).has_hit;
}

fn sample_area_light_mc(pos: vec3f, seed: ptr<function, u32>) -> Light {
    // Use Monte Carlo integration to sample the area light (one sample at a time)
    var light: Light;
    
    var n_lights = i32(arrayLength(&lightIndices));

    // Get random triangle idx.
    let p = rnd(seed);
    let triangle_idx = lightIndices[u32(floor(p * f32(n_lights)))];
    
    // Get random point on the randomly drawn triangle.
    let v_indices = meshFaces[triangle_idx];
    let p1 = sqrt(rnd(seed));
    let p2 = rnd(seed);
    let weights = vec3f(1.0 - p1, (1 - p2) * p1, p2 * p1);
    let x = (
        weights[0] * attribs[v_indices[0]].vposition + 
        weights[1] * attribs[v_indices[1]].vposition + 
        weights[2] * attribs[v_indices[2]].vposition
    );
    let n = normalize(
        weights[0] * attribs[v_indices[0]].normal + 
        weights[1] * attribs[v_indices[1]].normal + 
        weights[2] * attribs[v_indices[2]].normal
    );
    let A = length(
        cross(
            attribs[v_indices[1]].vposition - attribs[v_indices[0]].vposition, 
            attribs[v_indices[2]].vposition - attribs[v_indices[0]].vposition
        )
    ) / 2.0;

    let material = materials[v_indices[3]];
    
    let direction = x - pos;

    light.dist = length(direction);
    light.w_i = direction / light.dist;

    var V = 1.0;
    // Check for occlusion.
    let occluded = occlusion(pos, light.w_i, light.dist);
    if(occluded) {
        V = 0.0;
    }
    
    light.L_i = material.emission.rgb * V * dot(n, -light.w_i) / (light.dist * light.dist) * f32(n_lights) * A;

    return light;
}

fn occlusion(origin: vec3f, direction: vec3f, dist: f32) -> bool {
    // Check whether a ray intersects a diffuse object.
    var ray: Ray;
    ray.origin = origin;
    ray.direction = direction;
    ray.tmin = 1e-2; 
    ray.tmax = dist - 1e-2;

    var hit = get_hitinfo();

    return intersect_scene(&ray, &hit);
}

fn lambertian(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, seed: ptr<function, u32>) -> vec3f {
    const pi = radians(180.0);

    let light = sample_area_light_mc((*hit).position, seed);

    var L_direct = (*hit).diffuse / pi * dot((*hit).normal, light.w_i)  * light.L_i;

    var L_emission = vec3f(0.0);
    if((*hit).emit) {
        L_emission = (*hit).emission;
    }

    let T = (*hit).rgb_factor * (*hit).diffuse;
    let P = (T[0] + T[1] + T[2]) / 3.0;
    
    L_direct *= (*hit).rgb_factor;
    L_emission *= (*hit).rgb_factor;
    if(rnd(seed) < P) {
        (*hit).rgb_factor = T / P;    
        (*hit).has_hit = false;
        (*hit).emit = false;

        (*r).origin = (*hit).position;
        // Sample new direction from cosine-weighted hemisphere.
        let theta = acos(sqrt(1.0 - rnd(seed)));
        let phi = 2.0 * pi * rnd(seed);

        (*r).direction = rotate_to_normal((*hit).normal, spherical_direction(sin(theta), cos(theta), phi));
        (*r).tmin = 1e-2;
        (*r).tmax = 1e14;
    }
    
    return L_direct + L_emission;
}

fn mirror(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
    // Overwrite the old ray with the reflected ray.
    (*r).origin = (*hit).position;
    (*r).direction = reflect((*r).direction, (*hit).normal);
    (*r).tmin = 1e-2; 
    (*r).tmax = 1e14;

    (*hit).has_hit = false;
    (*hit).emit = true;

    return vec3f(0.0);
}

// Use Fresnel reflectance.
fn transparent(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, seed: ptr<function, u32>) -> vec3f {
    // Find direction of the refracted ray.
    let cos_theta_i = dot(-(*r).direction, (*hit).normal);
    let cos_theta_t_sq = 1.0 - (*hit).ior1_over_ior2 * (*hit).ior1_over_ior2 * (1.0 - cos_theta_i * cos_theta_i); 

    // Get Fresnel reflectance 
    var R = 1.0; // Total internal reflection.
    if(cos_theta_t_sq >= 0.0) {
        // Not total internal reflection.
        R = fresnel_R(cos_theta_i, sqrt(cos_theta_t_sq), (*hit).ior1_over_ior2);
    }
    // Overwrite the old ray with the refracted ray.
    (*r).origin = (*hit).position;

    if(rnd(seed) < R) {
        // Total internal reflection
        (*r).direction = reflect((*r).direction, (*hit).normal);
    }
    else {
        (*r).direction = (*hit).ior1_over_ior2 * (cos_theta_i * (*hit).normal + (*r).direction) - (*hit).normal * sqrt(cos_theta_t_sq);
    }

    // Transparent material is hit from the inside.
    if((*hit).ior1_over_ior2 > 1.0) {
        // distance travelled in medium is hit.dist.
        let T = exp(-(*hit).extinction_coef * (*hit).dist);
        let P = (T[0] + T[1] + T[2]) / 3.0;
        if(rnd(seed) < P) {
            (*hit).rgb_factor *= T / P;
        }
        else {
            return vec3f(0.0);
        }
    }

    (*r).tmin = 1e-2; 
    (*r).tmax = 1e14;

    (*hit).dist = 0.0;
    (*hit).has_hit = false;
    (*hit).emit = true;

    return vec3f(0.0);
}

fn shade(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, seed: ptr<function, u32>) -> vec3f {
    switch (*hit).shader {
        case 1 { return lambertian(r, hit, seed); }
        // case 2 { return phong(r, hit); }
        case 3 { return mirror(r, hit); }
        // case 4 { return refractive(r, hit); }
        // case 5 { return glossy(r, hit); }
        case 6 { return transparent(r, hit, seed); }
        case default { return (*hit).ambient; }
    }
}

fn get_camera_ray(ipcoords: vec2f) -> Ray {
    // Implement ray generation (WGSL has vector operations like normalize and cross)
    const eye = vec3f(277.0, 275.0, -570.0);
    const lookat = vec3f(277.0, 275.0, 0.0);
    const up = vec3f(0.0, 1.0, 0.0);
    let d = uniforms.cam_const;

    let v = normalize(lookat-eye);
    let b1 = normalize(cross(v, up));
    let b2 = cross(b1, v);

    var ray: Ray;
    ray.origin = eye;
    ray.direction = normalize(b1 * ipcoords.x + b2 * ipcoords.y + v * d);
    ray.tmin = 0.0; 
    ray.tmax = 1e14;
    return ray;
}

fn get_hitinfo() -> HitInfo {
    var hit = HitInfo(
        false, // has_hit
        0.0, // dist
        vec3f(0.0), // position
        vec3f(0.0), // normal
        vec3f(0.0), // diffuse
        vec3f(0.0), // ambient
        vec3f(0.0), // emission
        0.0, // specular
        1.0, // shininess
        1.0, // ior1_over_ior2
        0, // shader
        vec2f(0.0), // texcoords
        true, // emit
        vec3f(1.0), // rgb_factor
        vec3f(0.0), // extinction_coef
    );
    return hit;
}

@vertex
fn main_vs(@builtin(vertex_index) VertexIndex : u32) -> VSOut {
    const pos = array<vec2f, 4>(vec2f(-1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(1.0, -1.0));
    var vsOut: VSOut;
    vsOut.position = vec4f(pos[VertexIndex], 0.0, 1.0);
    vsOut.coords = pos[VertexIndex];
    return vsOut;
}

@fragment
fn main_fs(@builtin(position) fragcoord: vec4f, @location(0) coords: vec2f) -> FSOut {
    // const bgcolor = vec4f(0.1, 0.3, 0.6, 1.0);
    const bgcolor = vec4f(0.0, 0.0, 0.0, 1.0);
    const max_depth = 100;
    const gamma = 1.0;

    // Get random jitter
    let launch_idx = u32(fragcoord.y)*uniforms.canvas_width + u32(fragcoord.x);
    var t = tea(launch_idx, uniforms.frame_count);
    let jitter = vec2f(rnd(&t), rnd(&t))/f32(uniforms.canvas_height);

    var result = vec3f(0.0);

    let uv = vec2f(coords.x*uniforms.aspect*0.5f, coords.y*0.5f) + jitter;

    var r = get_camera_ray(uv);
    var hit = get_hitinfo();

    for(var i = 0; i < max_depth; i++) {
        if(intersect_scene(&r, &hit)) { result += shade(&r, &hit, &t); }
        else { result += bgcolor.rgb; break; }
        if(hit.has_hit) { break; }
    }

    let curr_sum = textureLoad(renderTexture, vec2u(fragcoord.xy), 0).rgb*f32(uniforms.frame_count);
    let accum_color = (clamp(result, vec3f(0.0), vec3f(1e14)) + curr_sum)/f32(uniforms.frame_count + 1u);
    
    var fsOut: FSOut;
    
    fsOut.frame = vec4f(pow(accum_color, vec3f(1.0/gamma)), 1.0);
    fsOut.accum = vec4f(accum_color, 1.0);
    
    // return vec4f(pow(result, vec3f(1.0 / gamma)), bgcolor.a);
    return fsOut;
}