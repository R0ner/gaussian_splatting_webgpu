struct VSOut {
    @builtin(position) position: vec4f,
    @location(0) coords : vec2f,
};

struct Uniforms {
    aspect: f32,
    cam_const: f32,
    shader_index_splat: u32,
    shader_index_matte: u32,
    use_texture: u32,
    subdivs: i32,
    render_object: i32,
};

struct Aabb {
    min: vec3f,
    max: vec3f,
};

// struct Attributes {
//     vposition: vec3f,
//     normal: vec3f,
// };

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage> means: array<vec3f>;
@group(0) @binding(2) var<storage> scales: array<vec3f>;
@group(0) @binding(3) var<storage> rots: array<mat3x3f>;
@group(0) @binding(4) var<storage> colors: array<vec4f>;
@group(0) @binding(5) var<storage> aabbs: array<Aabb>;
@group(0) @binding(6) var<storage> children: array<u32>;

// @group(0) @binding(1) var<storage> jitter: array<vec2f>;
// @group(0) @binding(2) var<storage> attribs: array<Attributes>;
// @group(0) @binding(3) var<storage> meshFaces: array<vec4u>;
// @group(0) @binding(4) var<storage> materials: array<Material>;
// @group(0) @binding(5) var<storage> lightIndices: array<u32>;
// @group(0) @binding(6) var<storage> treeIds: array<u32>;
// @group(0) @binding(7) var<storage> bspTree: array<vec4u>;
// @group(0) @binding(8) var<storage> bspPlanes: array<f32>;
// @group(0) @binding(9) var<uniform> aabb: Aabb;

const MAX_LEVEL = 10u;
const SQRT_TAU = sqrt(6.28318531);
// const BSP_LEAF = 3u;
// var<private> branch_node: array<vec2u, MAX_LEVEL>;
// var<private> branch_ray: array<vec2f, MAX_LEVEL>;

// Define Ray struct
struct Ray {
    origin: vec3f,
    direction: vec3f,
    tmin: f32,
    tmax: f32
};

// Material struct
struct Material {
    color: vec4f,
    emission: vec4f,
}

// Hit information
struct HitInfo {
    has_hit: bool,
    dist: f32,
    position: vec3f,
    normal: vec3f,
    color: vec4f,
    diffuse: vec3f,
    ambient: vec3f,
    ior1_over_ior2: f32, // Reciprocal relative index of refraction (n1/n2)
    shader: u32, // Shader index
    factor: f32,
    transmittance: vec3f,
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

fn check_ray_hit(r: Ray, hit: HitInfo, t_prime: f32) -> bool {
    if((t_prime < r.tmin) || (t_prime > r.tmax) || (hit.has_hit && (t_prime > hit.dist))) {
        return true;
    }
    return false;
}

fn intersect_min_max(r: ptr<function, Ray>, aabb: Aabb) -> bool {
    let p1 = (aabb.min - r.origin)/r.direction;
    let p2 = (aabb.max - r.origin)/r.direction;
    let pmin = min(p1, p2);
    let pmax = max(p1, p2);
    let tmin = max(pmin.x, max(pmin.y, pmin.z));
    let tmax = min(pmax.x, min(pmax.y, pmax.z));
    if(tmin > tmax || tmin > r.tmax || tmax < r.tmin) {
        return false;
    }
    // r.tmin = max(tmin - 1.0e-3f, r.tmin);
    r.tmax = min(tmax + 1.0e-3f, r.tmax);
    return true;
}

fn intersect_box(r: ptr<function, Ray>, aabb: Aabb) -> bool {
    return false;
}

// fn intersect_trimesh(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> bool {
//     var branch_lvl = 0u;
//     var near_node = 0u;
//     var far_node = 0u;
//     var t = 0.0f;
//     var node = 0u;
//     return false;
//     for(var i = 0u; i <= MAX_LEVEL; i++) {
//         let tree_node = bspTree[node];
//         let node_axis_leaf = tree_node.x&3u;
//         if(node_axis_leaf == BSP_LEAF) {
//             // A leaf was found
//             let node_count = tree_node.x>>2u;
//             let node_id = tree_node.y;
//             var found = false;
//             for(var j = 0u; j < node_count; j++) {
//                 let obj_idx = treeIds[node_id + j];
//                 if(intersect_triangle(*r, hit, i32(obj_idx))) {
//                     r.tmax = hit.dist;
//                     found = true;
//                 }
//             }
//             if(found) { return true; }
//             else if(branch_lvl == 0u) { return false; }
//             else {
//                 branch_lvl--;
//                 i = branch_node[branch_lvl].x;
//                 node = branch_node[branch_lvl].y;
//                 r.tmin = branch_ray[branch_lvl].x;
//                 r.tmax = branch_ray[branch_lvl].y;
//                 continue;
//             }
//         }
//         let axis_direction = r.direction[node_axis_leaf];
//         let axis_origin = r.origin[node_axis_leaf];
//         if(axis_direction >= 0.0f) {
//         near_node = tree_node.z; // left
//         far_node = tree_node.w; // right
//         }
//         else {
//         near_node = tree_node.w; // right
//         far_node = tree_node.z; // left
//         }
//         let node_plane = bspPlanes[node];
//         let denom = select(axis_direction, 1.0e-8f, abs(axis_direction) < 1.0e-8f);
//         t = (node_plane - axis_origin)/denom;
//         if(t > r.tmax) { node = near_node; }
//         else if(t < r.tmin) { node = far_node; }
//         else {
//             branch_node[branch_lvl].x = i;
//             branch_node[branch_lvl].y = far_node;
//             branch_ray[branch_lvl].x = t;
//             branch_ray[branch_lvl].y = r.tmax;
//             branch_lvl++;
//             r.tmax = t;
//             node = near_node;
//         }
//     }
//     return false;
// }

// fn intersect_plane(r: Ray, hit: ptr<function, HitInfo>, position: vec3f, onb: Onb) -> bool {
//     let denominator = dot(r.direction, onb.normal);
    
//     if(abs(denominator) < 1e-8) {
//         return false;
//     }
    
//     let t_prime = dot(position - r.origin, onb.normal) / denominator;

//     if(check_ray_hit(r, *hit, t_prime)) {
//         return false;
//     }

//     (*hit).has_hit = true;
//     (*hit).dist = t_prime;
//     (*hit).position = r.origin + t_prime * r.direction;
//     (*hit).normal = onb.normal;
//     (*hit).shader = uniforms.shader_index_matte;
    
//     (*hit).ambient = vec3f(0.1, 0.7, 0.0);
//     (*hit).diffuse = vec3f(0.1, 0.7, 0.0);
    

//     return true;
// }

fn intersect_splats(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, shader: u32) -> bool {
    var idx = 0u;
    var offset = 0u;
    var n_children = 0u;
    var is_leaf = 0u;
    var did_hit_ellipsoid = false;
    var done = false;
  
    if(!intersect_min_max(r, aabbs[idx])) {
        return false;
    }
    offset = children[idx * 3 + 0];
    n_children = children[idx * 3 + 1];
    is_leaf = children[idx * 3 + 2];  
    for(var i = 0u; i <= MAX_LEVEL; i++) {
        for(var j = 0u; j < n_children; j++) {
            idx = children[j + offset];
            if(is_leaf == 1) {
                did_hit_ellipsoid = intersect_ellipsoid(*r, hit, means[idx], scales[idx], rots[idx], shader);
                if(did_hit_ellipsoid) {
                    (*r).tmax = (*hit).dist;
                    (*hit).ambient = colors[idx].rgb;
                    (*hit).diffuse = colors[idx].rgb;
                    (*hit).color = colors[idx];
                    done = true;
                }
            }
            else {
                if(intersect_min_max(r, aabbs[idx])) {
                    offset = children[idx * 3 + 0];
                    n_children = children[idx * 3 + 1];
                    is_leaf = children[idx * 3 + 2];
                }
            }
        }
        if(done) {
            return true;
        }
    }
    return false;
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
    (*hit).diffuse = vec3f(1.0, 1.0, 1.0);
    (*hit).ior1_over_ior2 = 1.0 / 1.5; // Hit from outside of the material.
    (*hit).shader = shader;
    
    if(dot(r.direction, (*hit).normal) > 0.0) {
        (*hit).ior1_over_ior2 = 1.0 / (*hit).ior1_over_ior2;
        (*hit).normal = -(*hit).normal;
    }
    
    return true;
}

fn intersect_ellipsoid(r: Ray, hit: ptr<function, HitInfo>, center: vec3f, scale: vec3f, rotation: mat3x3f, shader: u32) -> bool {
    // Adapted from https://iquilezles.org/articles/intersectors/ and
    // additional code by GRAPHDECO research group, https://team.inria.fr/graphdeco.
    let scale_clamped = clamp(scale, vec3f(1e-2), vec3f(1e14));

    let stds = scale_clamped; 
    var radii = scale_clamped;
    
    if(shader == 2) {
        radii *= 3.0;
    } 

    let r_e_origin  = rotation * (r.origin - center);
    let r_e_direction  = rotation * r.direction;

    let ocn = r_e_origin / (radii);
    let rdn = r_e_direction / (radii);
    
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
        return false;
    }

    if(check_ray_hit(r, *hit, t_prime)) {
        return false;
    }

    let position_e = r_e_origin + t_prime * r_e_direction;
    let normal_e = normalize(position_e / (radii * radii) );

    let v = normalize(r_e_direction / stds);
    let p = r_e_origin / stds;
    let vp = dot(v, p);
    let factor = SQRT_TAU * exp(( vp * vp - dot(p, p) ) / 2.0);
    // let factor = exp((pow(dot(p_t, n_t), 2.0) - dot(p_t, p_t)) / 2.0);
    // let factor = exp((-dot(p_t, p_t) + pow(dot(p_t, n_t), 2.0)) / 2.0) / radians(360.0);
    // let factor = exp((-dot(p_t, p_t) + pow(dot(p_t, n_t), 2.0)) / 2.0) / radians(360.0);

    (*hit).has_hit = true;
    (*hit).position = r.origin + t_prime * r.direction;
    (*hit).dist = t_prime;
    (*hit).normal = normalize(normal_e * rotation);
    (*hit).ambient = vec3f(1.0, 0.0, 0.0);
    (*hit).diffuse = vec3f(1.0, 1.0, 1.0);
    // (*hit).ior1_over_ior2 = 1.0 / 1.5; // Hit from outside of the material.
    (*hit).shader = shader;

    (*hit).factor = factor;
    // (*hit).factor = 1.0;
    
    // if(dot(r.direction, (*hit).normal) > 0.0) {
    //     (*hit).ior1_over_ior2 = 1.0 / (*hit).ior1_over_ior2;
    //     (*hit).normal = -(*hit).normal;
    // }
    
    return true;
}

fn intersect_scene(r: ptr<function, Ray>, hit : ptr<function, HitInfo>) -> bool {
    // Define scene data as constants.
    // Call an intersection function for each object.
    // For each intersection found, update (*r).tmax and store additional info about the hit.

    let center_glass = vec3f(130.0, 90.0, 250.0); 
    // let radius_glass = 90.0;
    // let radii_glass = vec3f(50.0 * uniforms.aspect, 90.0, 90.0);
    let rotmat_glass = mat3x3f(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    );

    let center_mirror = vec3f(420.0, 90.0, 370.0); 
    let radius_mirror = 90.0;

    // var did_hit = false;
    // var scale = vec3f(1.0);
    // for(var i = 0u; i < u32(arrayLength(&means)); i++) {
    //     did_hit = intersect_ellipsoid(
    //         *r, 
    //         hit, 
    //         means[i], 
    //         scales[i], 
    //         rots[i],
    //         uniforms.shader_index_splat,
    //     );
    //     if(did_hit) {
    //         (*r).tmax = (*hit).dist;
    //         (*hit).ambient = colors[i].rgb;
    //         (*hit).diffuse = colors[i].rgb;
    //         (*hit).color = colors[i];
    //     }
    // }

    intersect_splats(r, hit, uniforms.shader_index_splat);
    // let mean = means[0];
    // let scale = scales[0];
    // let rot = rots[0];
    // let color = colors[0];


    // did_hit = intersect_ellipsoid(*r, hit, vec3f(0.0), vec3f(1.0), rotmat_glass, 2);
    // if(did_hit) {
    //     (*r).tmax = (*hit).dist;
    //     (*hit).diffuse = vec3f(1.0);
    //     (*hit).color = vec4f(1.0);
    // }

    return (*hit).has_hit;
}

// fn sample_point_light(pos: vec3f) -> Light {
//     const light_pos = vec3f(0.0, 1.0, 0.0);
//     const pi = radians(180.0);
//     const intensity = vec3f(pi, pi, pi);
    
//     let direction = light_pos - pos;
    
//     var light: Light;
    
//     light.dist = length(direction);
//     light.L_i = intensity / (light.dist * light.dist);
//     light.w_i = direction / light.dist;
    
//     return light;
// }

fn sample_directional_light(pos: vec3f) -> Light {
    const pi = radians(180.0);
    const intensity = vec3f(pi, pi, pi);
    
    const direction = normalize(vec3f(-1.0));
    
    var light: Light;
    
    light.dist = 1e14;
    light.L_i = intensity;
    light.w_i = direction;
    
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

fn lambertian(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
    const pi = radians(180.0);
    var V = 1.0;
    // let light = sample_point_light((*hit).position);
    let light = sample_directional_light((*hit).position);
    // let light = sample_area_light((*hit).position);

    // Check for occlusion.
    let occluded = occlusion((*hit).position, light.w_i, light.dist);
    if(occluded) {
        V = 0.0;
    }
    // let L_r = (*hit).diffuse / pi * V / (light.dist * light.dist) * dot((*hit).normal, light.w_i)  * light.L_i;
    let L_r = (*hit).diffuse / pi * V * dot((*hit).normal, light.w_i)  * light.L_i;

    // return L_r;
    return L_r;
}

fn mirror(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
    // Overwrite the old ray with the reflected ray.
    (*r).origin = (*hit).position;
    (*r).direction = reflect((*r).direction, (*hit).normal);
    (*r).tmin = 1e-2; 
    // r.tmax = 1e14;

    (*hit).has_hit = false;
    // (*hit).depth += 1;
    return vec3f(0.0, 0.0, 0.0);
}

fn refractive(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
    // Find direction of the refracted ray.
    let cos_theta_i = dot(-(*r).direction, (*hit).normal);
    let cos_theta_t_sq = 1.0 - (*hit).ior1_over_ior2 * (*hit).ior1_over_ior2 * (1.0 - cos_theta_i * cos_theta_i); 

    // Overwrite the old ray with the refracted ray.
    (*r).origin = (*hit).position;
    if(cos_theta_t_sq < 0.0) {
        (*r).direction = reflect((*r).direction, (*hit).normal);
    }
    else {
        (*r).direction = (*hit).ior1_over_ior2 * (cos_theta_i * (*hit).normal + (*r).direction) - (*hit).normal * sqrt(cos_theta_t_sq);
    }
    (*r).tmin = 1e-2; 
    r.tmax = 1e14;

    (*hit).has_hit = false;

    return vec3f(0.0, 0.0, 0.0);
}

fn splat(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
    (*r).origin = (*hit).position;
    (*r).tmin = 1e-4;
    (*r).tmax = 1e14;

    let f = (*hit).color.a * (*hit).factor;
    // let color = (*hit).color.rgb * (*hit).color.a * (*hit).factor * (*hit).transmittance;
    let color = (*hit).color.rgb * f * (*hit).transmittance;
    (*hit).transmittance *= 1.0 - f;
    
    if(((*hit).transmittance[0] + (*hit).transmittance[1] + (*hit).transmittance[2]) / 3 > 1e-3) {
        (*hit).has_hit = false;
    }
    
    return color;
}

fn shade(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
    switch (*hit).shader {
        case 1 { return lambertian(r, hit); }
        case 2 { return splat(r, hit); }
        case 3 { return mirror(r, hit); }
        case 4 { return refractive(r, hit); }
        // case 5 { return glossy(r, hit); }
        case default { return (*hit).ambient; }
    }
}

fn get_camera_ray(ipcoords: vec2f) -> Ray {
    // Implement ray generation (WGSL has vector operations like normalize and cross)
    // const eye = vec3f(277.0, 275.0, -570.0);
    // const lookat = vec3f(277.0, 275.0, 0.0);
    // const up = vec3f(0.0, 1.0, 0.0);
    
    // const eye = vec3f(0.0, -120.0, -200.0);
    // const lookat = vec3f(0.0, 100.0, 0.0);
    // const up = vec3f(0.0, 1.0, 0.0);
    // let d = uniforms.cam_const;
    
    // const eye = vec3f(0.0, 0.0, -3.0);
    // const lookat = vec3f(0.0, 0.0, 0.0);
    // const up = vec3f(0.0, 1.0, 0.0);
    // let d = uniforms.cam_const;

    // Kettle    
    // const eye = vec3f(0.0, 0.0, -5.0);
    // const lookat = vec3f(-0.8, 0.2, -2.0);
    // const up = vec3f(1.0, 0.0, 0.0);

    // Nisser
    const eye = vec3f(0.0, 6.0, -5.0);
    const lookat = vec3f(0.25, 3.25, 0.0);
    const up = vec3f(0.0, -1.0, 0.0);

    let d = uniforms.cam_const * 2.0;


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
        vec4f(0.0), // color
        vec3f(0.0), // diffuse
        vec3f(0.0), // ambient
        1.0, // ior1_over_ior2
        0, // shader
        1.0, // factor
        vec3f(1.0), // transmittance
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
fn main_fs(@location(0) coords: vec2f) -> @location(0) vec4f {
    const bgcolor = vec4f(0.1, 0.3, 0.6, 1.0);
    // const bgcolor = vec4f(0.0, 0.0, 0.0, 1.0);
    // const max_depth = 10;
    let max_depth = uniforms.subdivs;
    const gamma = 1.0;

    var result = vec3f(0.0);

    // let uv = vec2f(coords.x*uniforms.aspect*0.5f, coords.y*0.5f);
    let uv = vec2f(coords.x*0.5f, coords.y*0.5f);
    // for(var k = 0; k < uniforms.subdivs * uniforms.subdivs; k++) {
    //     var r = get_camera_ray(vec2f(uv[0] + jitter[k][0], uv[1] + jitter[k][1]));

    //     var hit = get_hitinfo();
        
    //     for(var i = 0; i < max_depth; i++) {
    //         if(intersect_scene(&r, &hit)) { result += shade(&r, &hit); }
    //         else {
    //             if(!(hit.shader == 2)) {
    //                 result += bgcolor.rgb;
    //             } 
    //             else {
    //                 result += hit.transmittance * bgcolor.rgb;
    //             }
    //              break; }
    //         if(hit.has_hit) { break; }
    //     }      
    // }
    // result = result / f32(uniforms.subdivs * uniforms.subdivs);

    var r = get_camera_ray(vec2f(uv[0], uv[1]));
    var hit = get_hitinfo();
    var any_hit = false;
    var max_depth_reached = true;
    
    for(var i = 0; i < max_depth; i++) {
        if(intersect_scene(&r, &hit)) {
            any_hit = true;
            result += shade(&r, &hit); 
        }
        else {
            result += hit.transmittance * bgcolor.rgb;
            max_depth_reached = false;
            break; 
        }
        if(hit.has_hit) {
            max_depth_reached = false;
            break; 
        }
    }
    if(any_hit & max_depth_reached) {
        result += hit.transmittance * bgcolor.rgb;
    }
    
    return vec4f(pow(result, vec3f(1.0 / gamma)), bgcolor.a);
}
