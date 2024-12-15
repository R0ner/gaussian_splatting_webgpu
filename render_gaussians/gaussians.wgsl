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

const MAX_LEVEL = 1000u;
const SQRT_TAU = sqrt(6.28318531);

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
    last_hit: u32,
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
    return true;
}

fn intersect_box(r: ptr<function, Ray>, aabb: Aabb) -> f32 {
    let p1 = (aabb.min - r.origin - 1e-2)/r.direction;
    let p2 = (aabb.max - r.origin + 1e-2)/r.direction;
    let pmin = min(p1, p2);
    let pmax = max(p1, p2);
    let tmin = max(pmin.x, max(pmin.y, pmin.z));
    let tmax = min(pmax.x, min(pmax.y, pmax.z));
    if(tmin > tmax || tmin > r.tmax || tmax < r.tmin) {
        return -1.0;
    }
    return max(min(tmin, tmax), 1e-4);
}

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
    // If we are not within the aabb of the Gaussian splatting, exit.
    if(!intersect_min_max(r, aabbs[0])) {
        return false;
    }
    var idx = 0u;
    var child0 = 0u;
    var child1 = 0u;
    var child_idx = 0u;
    var closest_hit_idx = 0u;

    // var bounds0 = vec4f(0.0);
    // var bounds1 = vec4f(0.0);
    var dist0 = 0.0;
    var dist1 = 0.0;

    var offset = 0u;
    var n_children = 0u;
    var is_leaf = 0u;

    var done = false;

    var stack = array<u32, 130>();
    var stack_idx = 0u;
    // idx = 0;    
    for(var i = 0u; i <= MAX_LEVEL; i++) {
        idx = stack[stack_idx];
        if(i > 0) {
            while (idx == 0) {
                if(stack_idx == 0) {
                    return false;
                }
                stack_idx -= 1;
                idx = stack[stack_idx];
            }
        }
        stack[stack_idx] = 0;
        
        offset = children[idx * 3 + 0];
        n_children = children[idx * 3 + 1];
        is_leaf = children[idx * 3 + 2];
        
        if(is_leaf == 1) {
            // Ellipsoid intersections.
            // Indices correspond to ellipsoids.
            closest_hit_idx = 0u;
            r.tmin = 0.0;
            for(var j = 0u; j < n_children; j++) {
                child_idx = children[j + offset];
                if(child_idx == (*hit).last_hit) {
                    continue;
                }
                if(intersect_ellipsoid(*r, hit, means[child_idx], scales[child_idx], rots[child_idx], shader)) {
                    (*r).tmax = (*hit).dist;
                    (*hit).last_hit = child_idx;
                    closest_hit_idx = child_idx;
                    done = true;
                }
            }
            if(done) {
                (*hit).ambient = colors[closest_hit_idx].rgb;
                (*hit).diffuse = colors[closest_hit_idx].rgb;
                (*hit).color = colors[closest_hit_idx];
                return true; 
            }
            // If we are in a leaf node and did not intersect any ellipsoids (not done), we continue the search.
            // Importantly, we'll just end up in this spot again if don't do anything!
            // stack_idx -= 1;
            // idx = stack[stack_idx];
        }
        else {
            // Box intersections.
            // Indices correspond to aabbs.
            child0 = children[0 + offset];
            child1 = children[1 + offset];

            dist0 = intersect_box(r, aabbs[child0]);
            dist1 = intersect_box(r, aabbs[child1]);
            if(dist0 < dist1) {
                if(dist1 > 0.0) {
                    stack_idx += 1;
                    stack[stack_idx] = child1;
                }
                if(dist0 > 0.0) {
                    stack_idx += 1;
                    stack[stack_idx] = child0;
                }
            }
            else {
                if(dist0 > 0.0) {
                    stack_idx += 1;
                    stack[stack_idx] = child0;
                }
                if(dist1 > 0.0) {
                    stack_idx += 1;
                    stack[stack_idx] = child1;
                }
            }
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
    // let scale_clamped = clamp(scale, vec3f(1), vec3f(1e14));
    let scale_clamped = max(scale, vec3f(9e-1));
    // let scale_clamped = scale;

    let stds = scale_clamped; 
    var radii = scale_clamped;
    
    // if(shader == 2) {
    //     radii *= 3.0;
    // } 
    radii *= 3.0;

    let r_e_origin  = rotation * (r.origin - center);
    let r_e_direction  = rotation * r.direction;

    let ocn = r_e_origin / radii;
    let rdn = r_e_direction / radii;
    // let rdn = normalize(r_e_direction / radii);
    
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

    if(t1_prime-1e-4 > r.tmin) {
        t_prime = t1_prime;
    }
    else {
        return false;
    }

    if(check_ray_hit(r, *hit, t_prime)) {
        return false;
    }

    let position_e = r_e_origin + t_prime * r_e_direction;
    if(shader == 2) {

        let v = normalize(r_e_direction / stds);
        let p = r_e_origin / stds;
        let vp = dot(v, p);
        let factor = SQRT_TAU * exp(( vp * vp - dot(p, p) ) / 2.0);

        (*hit).factor = factor;
    }
    else {
        let normal_e = normalize(position_e / (radii * radii) );
        (*hit).normal = normalize(normal_e * rotation);
        (*hit).ambient = vec3f(1.0, 0.0, 0.0);
        (*hit).diffuse = vec3f(1.0, 1.0, 1.0);
        (*hit).ior1_over_ior2 = 1.0 / 1.5; // Hit from outside of the material.
    }
    
    (*hit).has_hit = true;
    (*hit).position = r.origin + t_prime * r.direction;
    (*hit).dist = t_prime;
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

    intersect_splats(r, hit, uniforms.shader_index_splat);

    return (*hit).has_hit;
}

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
    (*r).tmin = 0.0;
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
    const eye = vec3f(2.0, 5.0, -5.0) * 100;
    const lookat = vec3f(0.25, 3.25, 0.0) * 100;
    const up = vec3f(0.0, -1.0, 0.0);

    let d = uniforms.cam_const * 2.0;


    let v = normalize(lookat-eye);
    let b1 = normalize(cross(v, up));
    let b2 = cross(b1, v);

    var ray: Ray;
    ray.origin = eye;
    ray.direction = normalize(b1 * ipcoords.x + b2 * ipcoords.y + v * d);
    ray.tmin = 1e-4; 
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
        1000000, // 
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

    let uv = vec2f(coords.x*0.5f, coords.y*0.5f);

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
