struct VSOut {
    @builtin(position) position: vec4f,
    @location(0) coords : vec2f,
};

struct Uniforms {
    aspect: f32,
    cam_const: f32,
    shader_index_sphere: u32,
    shader_index_matte: u32,
    use_texture: u32,
    subdivs: i32,
    render_object: i32,
};

// struct Aabb {
//     min: vec3f,
//     max: vec3f,
// };

// struct Attributes {
//     vposition: vec3f,
//     normal: vec3f,
// };

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage> jitter: array<vec2f>;
@group(0) @binding(2) var<storage> means: array<vec3f>;
@group(0) @binding(3) var<storage> scales: array<vec3f>;
@group(0) @binding(4) var<storage> rots: array<mat3x3f>;
@group(0) @binding(5) var<storage> colors: array<vec4f>;

// @group(0) @binding(2) var<storage> attribs: array<Attributes>;
// @group(0) @binding(3) var<storage> meshFaces: array<vec4u>;
// @group(0) @binding(4) var<storage> materials: array<Material>;
// @group(0) @binding(5) var<storage> lightIndices: array<u32>;
// @group(0) @binding(6) var<storage> treeIds: array<u32>;
// @group(0) @binding(7) var<storage> bspTree: array<vec4u>;
// @group(0) @binding(8) var<storage> bspPlanes: array<f32>;
// @group(0) @binding(9) var<uniform> aabb: Aabb;

// const MAX_LEVEL = 20u;
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

// fn intersect_min_max(r: ptr<function, Ray>) -> bool {
//     let p1 = (aabb.min - r.origin)/r.direction;
//     let p2 = (aabb.max - r.origin)/r.direction;
//     let pmin = min(p1, p2);
//     let pmax = max(p1, p2);
//     let tmin = max(pmin.x, max(pmin.y, pmin.z));
//     let tmax = min(pmax.x, min(pmax.y, pmax.z));
//     if(tmin > tmax || tmin > r.tmax || tmax < r.tmin) {
//     return false;
//     }
//     r.tmin = max(tmin - 1.0e-3f, r.tmin);
//     r.tmax = min(tmax + 1.0e-3f, r.tmax);
//     return true;
// }

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
    (*hit).shader = uniforms.shader_index_matte;
    
    (*hit).ambient = vec3f(0.1, 0.7, 0.0);
    (*hit).diffuse = vec3f(0.1, 0.7, 0.0);
    

    return true;
}

// fn intersect_triangle(r: Ray, hit: ptr<function, HitInfo>, faceidx: i32) -> bool {
//     let v_indices = meshFaces[faceidx];
//     let v0 = attribs[v_indices[0]].vposition;
//     let e0 = attribs[v_indices[1]].vposition - v0;
//     let e1 = attribs[v_indices[2]].vposition - v0;
    
//     let normal = cross(e0, e1);
    
//     let a = v0 - r.origin; 
//     let k = dot(r.direction, normal);
    
//     let t_prime = dot(a, normal) / k;
    
//     if(check_ray_hit(r, *hit, t_prime)) {
//         return false;
//     }

//     let h = cross(a, r.direction);
    
//     let beta = dot(h, e1) / k;
//     let gamma = -dot(h, e0) / k; 

//     if((beta < 0) || (gamma < 0) || ((beta + gamma) > 1)) {
//         return false;
//     }
    
//     // let material = materials[matIndices[faceidx]];
//     // Material index is the last element of the meshface/vertex indices.
//     let material = materials[v_indices[3]];

//     (*hit).has_hit = true;
//     (*hit).dist = t_prime;
//     (*hit).position = r.origin + t_prime * r.direction;
//     (*hit).normal = normalize((1 - beta - gamma) * attribs[v_indices[0]].normal + beta * attribs[v_indices[1]].normal + gamma * attribs[v_indices[2]].normal);
//     (*hit).ambient = material.emission.rgb;
//     (*hit).diffuse = material.color.rgb;
//     (*hit).shader = 1;

//     for(var i = 0; i < i32(arrayLength(&lightIndices)); i++) {
//         if(u32(faceidx) == lightIndices[i]) {
//             (*hit).shader = 0;
//         }
//     }

//     return true;
// }

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

    let center_glass = vec3f(130.0, 90.0, 250.0); 
    // let radius_glass = 90.0;
    let radii_glass = vec3f(50.0 * uniforms.aspect, 90.0, 90.0);
    let rotmat_glass = mat3x3f(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    );

    let center_mirror = vec3f(420.0, 90.0, 370.0); 
    let radius_mirror = 90.0;

    var did_hit = false;
    for(var i = 0u; i < u32(arrayLength(&means)); i++) {
        did_hit = intersect_ellipsoid(*r, hit, means[i], scales[i], rots[i], 1);
        if(did_hit) {
            (*r).tmax = (*hit).dist;
            (*hit).diffuse = colors[i].rgb;
        }
    }

    // if (intersect_min_max(r)) {
    //     let did_hit_trimesh = intersect_trimesh(r, hit);
    //     if(did_hit_trimesh) { (*r).tmax = (*hit).dist; }
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

// fn sample_area_light(pos: vec3f) -> Light {
//     // Use approximation for a distant area light.
//     // Compute area light parameters.
//     //  x_e: Center of area light bbox
//     //  n_e_f: Light normal of face
//     //  light.w_i: Directions from pos towards light source (omega_i)
//     var light: Light;
    
//     var x_e = vec3f(0.0, 0.0, 0.0);
//     var n_lights = i32(arrayLength(&lightIndices));
//     for(var i = 0; i < n_lights; i++) {
//         var v_indices = meshFaces[lightIndices[i]];
//         x_e += attribs[v_indices[0]].vposition + attribs[v_indices[1]].vposition + attribs[v_indices[2]].vposition; 
//     }
    
//     x_e /= f32(n_lights) * 3.0;

//     let direction = x_e - pos;

//     light.dist = length(direction);
//     light.w_i = direction / light.dist;

//     var n_e_f = vec3f(0.0, 0.0, 0.0);
//     var A_e_f = 0.0;
//     var I_e = vec3f(0.0, 0.0, 0.0);
//     for(var i = 0; i < n_lights; i++) {
//         var v_indices = meshFaces[lightIndices[i]];  
//         var material = materials[v_indices[3]]; 
//         n_e_f = normalize(attribs[v_indices[0]].normal + attribs[v_indices[1]].normal + attribs[v_indices[2]].normal); 
//         A_e_f = length(
//             cross(
//                 attribs[v_indices[1]].vposition - attribs[v_indices[0]].vposition, 
//                 attribs[v_indices[2]].vposition - attribs[v_indices[0]].vposition
//             )
//         ) / 2.0;
//         I_e += dot(-light.w_i, n_e_f) * A_e_f * material.emission.rgb;  
//     }
//     light.L_i = I_e;

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

fn shade(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
    switch (*hit).shader {
        case 1 { return lambertian(r, hit); }
        // case 2 { return phong(r, hit); }
        case 3 { return mirror(r, hit); }
        case 4 { return refractive(r, hit); }
        // case 5 { return glossy(r, hit); }
        case default { return (*hit).ambient; }
    }
}

// fn texture_nearest(texture: texture_2d<f32>, texcoords: vec2f, repeat: bool) -> vec3f {
//     let res = textureDimensions(texture);
//     let st = select(clamp(texcoords, vec2f(0), vec2f(1)), texcoords - floor(texcoords), repeat);
//     let ab = st*vec2f(res);
//     let UV = vec2u(ab + 0.5) % res;
//     let texcolor = textureLoad(texture, UV, 0);
//     return texcolor.rgb;
// }

// fn texture_linear(texture: texture_2d<f32>, texcoords: vec2f, repeat: bool) -> vec3f {
//     let res = textureDimensions(texture);
//     let st = select(clamp(texcoords, vec2f(0), vec2f(1)), texcoords - floor(texcoords), repeat);
//     let ab = st*vec2f(res);
//     let UV0 = vec2u(ab);
//     let UV1 = vec2u(ceil(ab)) % res; // opposite corner
//     let c = ab - vec2f(UV0);

//     let texcolor = mix(
//         mix(textureLoad(texture, UV0, 0), textureLoad(texture, vec2u(UV1[0], UV0[1]), 0), c[0]),
//         mix(textureLoad(texture, vec2u(UV0[0], UV1[1]), 0), textureLoad(texture, UV1, 0), c[0]),
//         c[1]
//     );

//     return texcolor.rgb;
// }

fn get_camera_ray(ipcoords: vec2f) -> Ray {
    // Implement ray generation (WGSL has vector operations like normalize and cross)
    let a = jitter[0];
    // const eye = vec3f(277.0, 275.0, -570.0);
    // const lookat = vec3f(277.0, 275.0, 0.0);
    // const up = vec3f(0.0, 1.0, 0.0);
    
    const eye = vec3f(0.0, -120.0, -200.0);
    const lookat = vec3f(0.0, 100.0, 0.0);
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
        vec4f(0.0), // color
        vec3f(0.0), // diffuse
        vec3f(0.0), // ambient
        1.0, // ior1_over_ior2
        0, // shader
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
    const max_depth = 10;
    const gamma = 2.0;

    var result = vec3f(0.0);

    // let uv = vec2f(coords.x*uniforms.aspect*0.5f, coords.y*0.5f);
    let uv = vec2f(coords.x*0.5f, coords.y*0.5f);
    for(var k = 0; k < uniforms.subdivs * uniforms.subdivs; k++) {
        var r = get_camera_ray(vec2f(uv[0] + jitter[k][0], uv[1] + jitter[k][1]));

        var hit = get_hitinfo();
        
        for(var i = 0; i < max_depth; i++) {
            if(intersect_scene(&r, &hit)) { result += shade(&r, &hit); }
            else { result += bgcolor.rgb; break; }
            if(hit.has_hit) { break; }
        }      
    }
    result = result / f32(uniforms.subdivs * uniforms.subdivs);
    return vec4f(pow(result, vec3f(1.0 / gamma)), bgcolor.a);
}
