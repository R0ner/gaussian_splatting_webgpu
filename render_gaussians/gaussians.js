"use strict";
window.onload = function () { main(); }

const shader_path = "gaussians.wgsl";

async function fetchShaderCode(url) {
    const response = await fetch(url);
    return await response.text();
}

async function main() {
    // var subdivslider = document.getElementById("subdivslider");
    var fpscounter = document.getElementById("fps-counter");
    var splat_shader_menu = document.getElementById("splat_shader_menu");

    addEventListener("wheel", (event) => {
        update = true;
        cam_const *= 1.0 + 2.5e-4 * event.deltaY;
    });

    document.onkeydown = (event) => {
        update = true;
        switch (event.key) {
            case "+":
                // aspect *= 1.1;
                subdivs += 1;
                break
            case "-":
                subdivs -= 1;
                // aspect *= 0.9;
                break
        }
        subdivs = Math.max(subdivs, 1);
    }

    // subdivslider.oninput = function(event) {
    //     update = true;
    //     subdivs = subdivslider.value;
    // };

    splat_shader_menu.addEventListener(
        "click",
        async function() {     
            const expr = splat_shader_menu.selectedIndex;
            switch (expr) {
                case 1:
                    shader_index_splat = 2;
                    break;
            
                case 2:
                    shader_index_splat = 0;
                    break;
            
                default:
                    break;
            }
            update = true;
            // console.log(shader_index_splat);
        }
    );

    const gpu_options = new Object(); 
    gpu_options.powerPreference = "high-performance";
    // gpu_options.powerPreference = "low-power";
    const adapter = await navigator.gpu.requestAdapter(gpu_options);
    if (adapter) {
        console.log(adapter);
    }
    const device = await adapter.requestDevice();
    const canvas = document.getElementById("webgpu-canvas");
    const context = canvas.getContext("gpupresent") || canvas.getContext("webgpu");
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device: device,
        format: canvasFormat,
    });

    // Load the WGSL shader code from an external file
    const shaderCode = await fetchShaderCode(shader_path);

    const wgsl = device.createShaderModule({
        code: shaderCode
    });

    const pipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: {
            module: wgsl,
            entryPoint: "main_vs",
        },
        fragment: {
            module: wgsl,
            entryPoint: "main_fs",
            targets: [{ format: canvasFormat }]
        },
        primitive: {
            topology: "triangle-strip",
        },
    });

    // let jitter = new Float32Array(200); // allowing subdivs from 1 to 10
    
    var buffers = new Object()

    // Create uniform buffer
    buffers.uniforms = device.createBuffer({
        size: 28, // number of bytes 2 * 4 = 8 bytes for floats and 5 * 4 = 16 bytes for selections.
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // buffers.jitter = device.createBuffer({
    //     size: jitter.byteLength,
    //     usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
    // });
    
    // var obj_filename = "../resources/objects/CornellBox.obj";
    // var drawingInfo = await readOBJFile(obj_filename, 1, true); // file name, scale, ccw vertices
    // buffers = load_object(drawingInfo, device, buffers); 
    
    // var gaussianInfo = await read_gaussians("../resources/objects/private/gaussians.json");

    // var gaussianInfo = await read_gaussians("../resources/objects/private/kedel_1688.json");
    // var gaussianInfo = await read_gaussians("../resources/objects/private/kedel_16932.json");
    // var gaussianInfo = await read_gaussians("../resources/objects/private/kedel_34103_full.json");
    
    // var gaussianInfo = await read_gaussians("../resources/objects/private/kedel_100.json");
    // var spatialdataInfo = await read_spatial_data("../resources/objects/private/kedel_100_spatial_data.json");

    // var gaussianInfo = await read_gaussians("../resources/objects/private/nisse_full.json");
    // var spatialdataInfo = await read_spatial_data("../resources/objects/private/nisse_full_spatial.json");

    var gaussianInfo = await read_gaussians("../resources/objects/private/nisse_15k.json");
    var spatialdataInfo = await read_spatial_data("../resources/objects/private/nisse_15k_spatial.json");
   
    // console.log(gaussianInfo);
    // console.log(spatialdataInfo);

    buffers.means = device.createBuffer({
        size: gaussianInfo.means.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
    });
    device.queue.writeBuffer(buffers.means, 0, gaussianInfo.means);
    
    buffers.scales = device.createBuffer({
        size: gaussianInfo.scales.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
    });
    device.queue.writeBuffer(buffers.scales, 0, gaussianInfo.scales);

    buffers.rots = device.createBuffer({
        size: gaussianInfo.rots.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
    });
    device.queue.writeBuffer(buffers.rots, 0, gaussianInfo.rots);

    buffers.colors = device.createBuffer({
        size: gaussianInfo.colors.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
    });
    device.queue.writeBuffer(buffers.colors, 0, gaussianInfo.colors);
    
    // buffers.mins = device.createBuffer({
    //     size: spatialdataInfo.mins.byteLength,
    //     usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
    // });
    // device.queue.writeBuffer(buffers.mins, 0, spatialdataInfo.mins);
    
    // buffers.maxs = device.createBuffer({
    //     size: spatialdataInfo.maxs.byteLength,
    //     usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
    // });
    // device.queue.writeBuffer(buffers.maxs, 0, spatialdataInfo.maxs);
    
    buffers.aabbs = device.createBuffer({
        size: spatialdataInfo.aabbs.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
    });
    device.queue.writeBuffer(buffers.aabbs, 0, spatialdataInfo.aabbs);
    
    buffers.children = device.createBuffer({
        size: spatialdataInfo.children.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
    });
    device.queue.writeBuffer(buffers.children, 0, spatialdataInfo.children);


    // var bindGroup = get_bindgroup(buffers, device, pipeline);
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: buffers.uniforms } },
            { binding: 1, resource: { buffer: buffers.means } },
            { binding: 2, resource: { buffer: buffers.scales } },
            { binding: 3, resource: { buffer: buffers.rots } },
            { binding: 4, resource: { buffer: buffers.colors } },
            // { binding: 5, resource: { buffer: buffers.mins } },
            // { binding: 6, resource: { buffer: buffers.maxs } },
            { binding: 5, resource: { buffer: buffers.aabbs } },
            { binding: 6, resource: { buffer: buffers.children } },
        ],
    });

    var cam_const = 1.2;
    // var cam_const = 1.0;
    var aspect = 1.0;
    var shader_index_splat = 0;
    var shader_index_matte = 1;
    var use_texture = 1;
    var subdivs = 1;
    var render_object = 1;
    var uniforms = new Float32Array([aspect, cam_const]);
    var uniforms_selection = new Uint32Array([shader_index_splat, shader_index_matte, use_texture, subdivs, render_object]);

    var last_time = performance.now();
    var current_time = performance.now();
    var frame_count = 0;
    var fps = 0;
    var update = true;
    function render(device, context, pipeline, bindGroup) {
        current_time = performance.now();
        frame_count++;

        // Calculate FPS every second
        if (current_time - last_time >= 1000) {
            fps = frame_count;
            frame_count = 0;
            last_time = current_time;
            fpscounter.textContent = `FPS: ${fps}`;
        }

        if (update) {
            // Pass uniforms.
            uniforms[0] = aspect;
            uniforms[1] = cam_const;
            uniforms_selection[0] = shader_index_splat;
            uniforms_selection[1] = shader_index_matte;
            uniforms_selection[2] = use_texture;
            uniforms_selection[3] = subdivs;
            uniforms_selection[4] = render_object;
            // compute_jitters(jitter, 1 / canvas.height, subdivs);
            device.queue.writeBuffer(buffers.uniforms, 0, uniforms);
            device.queue.writeBuffer(buffers.uniforms, 8, uniforms_selection);
            // device.queue.writeBuffer(buffers.jitter, 0, jitter);

            // Create a render pass in a command buffer and submit it.
            let encoder = device.createCommandEncoder();
            let pass = encoder.beginRenderPass({
                colorAttachments: [{
                    view: context.getCurrentTexture().createView(),
                    loadOp: "clear",
                    storeOp: "store",
                }]
            });
            // Render pass commands
            pass.setBindGroup(0, bindGroup);
            pass.setPipeline(pipeline);
            pass.draw(4);
            pass.end();
            device.queue.submit([encoder.finish()]);

            update = false;
        }
        window.requestAnimationFrame(function () { render(device, context, pipeline, bindGroup) });
    }
    render(device, context, pipeline, bindGroup);
}

async function load_texture(device, filename) {
    const response = await fetch(filename);
    const blob = await response.blob();
    const img = await createImageBitmap(blob, { colorSpaceConversion: 'none' });
    const texture = device.createTexture({
        size: [img.width, img.height, 1],
        format: "rgba8unorm",
        usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT
    });
    device.queue.copyExternalImageToTexture(
        { source: img, flipY: true },
        { texture: texture },
        { width: img.width, height: img.height },
    );
    return texture;
}

// function compute_jitters(jitter, pixelsize, subdivs) {
//     const step = pixelsize / subdivs;
//     if (subdivs < 2) {
//         jitter[0] = 0.0;
//         jitter[1] = 0.0;
//     }
//     else {
//         for (var i = 0; i < subdivs; ++i)
//             for (var j = 0; j < subdivs; ++j) {
//                 const idx = (i * subdivs + j) * 2;
//                 jitter[idx] = (Math.random() + j) * step - pixelsize * 0.5;
//                 jitter[idx + 1] = (Math.random() + i) * step - pixelsize * 0.5;
//             }
//     }
// }

function load_object(drawingInfo, device, buffers) {
    const n_materials = drawingInfo.materials.length;
    let materialsData = new Float32Array(n_materials * 8);
    for (let i = 0; i < n_materials; i++) {
        let color = drawingInfo.materials[i].color;
        let emission = drawingInfo.materials[i].emission;
        let j = i * 8;
        materialsData[j + 0] = color.r; 
        materialsData[j + 1] = color.g; 
        materialsData[j + 2] = color.b; 
        materialsData[j + 3] = color.a;
        materialsData[j + 4] = emission.r; 
        materialsData[j + 5] = emission.g; 
        materialsData[j + 6] = emission.b; 
        materialsData[j + 7] = emission.a; 
    }
    
    // const n_indices = drawingInfo.indices.length;
    // let indices = new Uint32Array(n_indices * 4);
    // for (let i = 0; i < n_indices; i++) {
    //     let j = i * 4;
    //     indices[j + 0] = drawingInfo.indices[j + 0];
    //     indices[j + 1] = drawingInfo.indices[j + 1];
    //     indices[j + 2] = drawingInfo.indices[j + 2];
    //     indices[j + 3] = drawingInfo.mat_indices[i];
    // }

    // const indicesBuffer = device.createBuffer({
    //     size: indices.byteLength,
    //     usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
    // });
    // device.queue.writeBuffer(indicesBuffer, 0, indices);

    // const materialsBuffer = device.createBuffer({
    //     size: materialsData.byteLength,
    //     usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
    // });
    // device.queue.writeBuffer(materialsBuffer, 0, materialsData);

    // const mat_indicesBuffer = device.createBuffer({
    //     size: drawingInfo.mat_indices.byteLength,
    //     usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
    // });
    // device.queue.writeBuffer(mat_indicesBuffer, 0, drawingInfo.mat_indices);
    
    buffers.light_indices = device.createBuffer({
        size: drawingInfo.light_indices.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
    });
    device.queue.writeBuffer(buffers.light_indices, 0, drawingInfo.light_indices);
    // var buffers = new Object()
    buffers = build_bsp_tree(drawingInfo, device, buffers)

    buffers.indices = device.createBuffer({
        size: drawingInfo.indices.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
    });
    device.queue.writeBuffer(buffers.indices, 0, drawingInfo.indices);

    buffers.materials = device.createBuffer({
        size: materialsData.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
    });
    device.queue.writeBuffer(buffers.materials, 0, materialsData);

    return buffers;
}

function get_bindgroup(buffers, device, pipeline) {
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: buffers.uniforms } },
            { binding: 1, resource: { buffer: buffers.jitter } },
            { binding: 2, resource: { buffer: buffers.attribs } },
            { binding: 3, resource: { buffer: buffers.indices } },
            { binding: 4, resource: { buffer: buffers.materials } },
            { binding: 5, resource: { buffer: buffers.light_indices } },
            { binding: 6, resource: { buffer: buffers.treeIds } },
            { binding: 7, resource: { buffer: buffers.bspTree } },
            { binding: 8, resource: { buffer: buffers.bspPlanes } },
            { binding: 9, resource: { buffer: buffers.aabb } },
        ],
    });
    return bindGroup;
}

function isPromiseResolved(promise) {
    return Promise.race([promise, Promise.resolve('pending')]).then((value) => {
      return value !== 'pending';
    });
}

async function read_gaussians(path) {
    const response = await fetch(path); // Load the JSON file
    var jsonData = await response.json(); // Parse JSON
    jsonData =  dictionaryToArray(jsonData);

    const n_gaussians = jsonData.length ;
    var means = new Float32Array(n_gaussians * 4);
    var rots = new Float32Array(n_gaussians * 12);
    var scales = new Float32Array(n_gaussians * 4);
    // var opacities = new Float32Array(n_guassians);
    var colors = new Float32Array(n_gaussians * 4);
    for (let idx = 0; idx < n_gaussians; idx++) {
        const entry = jsonData[idx];
        means[idx * 4 + 0] = entry.xyz[0]; 
        means[idx * 4 + 1] = entry.xyz[1];
        means[idx * 4 + 2] = entry.xyz[2];
        
        scales[idx * 4 + 0] = entry.scale[0]; 
        scales[idx * 4 + 1] = entry.scale[1];
        scales[idx * 4 + 2] = entry.scale[2];
        
        colors[idx * 4 + 0] = entry.color_rgb[0]; 
        colors[idx * 4 + 1] = entry.color_rgb[1];
        colors[idx * 4 + 2] = entry.color_rgb[2];
        colors[idx * 4 + 3] = entry.opacity;
        
        const rot = entry.rot.flat();
        for (let i = 0; i < 3; i++) {
            rots[idx * 12 + i * 4 + 0] = rot[i * 3 + 0];    
            rots[idx * 12 + i * 4 + 1] = rot[i * 3 + 1];    
            rots[idx * 12 + i * 4 + 2] = rot[i * 3 + 2];     
        }
    }
    return new GaussianInfo(means, rots, scales, colors);
}

async function read_spatial_data(path) {
    const response = await fetch(path); // Load the JSON file
    var jsonData = await response.json(); // Parse JSON
    jsonData =  dictionaryToArray(jsonData);
    
    const n_boxes = jsonData.length;

    // Count the total number of children.
    var n_children = 0;
    for (let idx = 0; idx < n_boxes; idx++) {
        const entry = jsonData[idx];
        n_children += entry.child_indices.length; 
    }

    var aabbs = new Float32Array((2 * n_boxes) * 4);
    var children = new Uint32Array(n_children + n_boxes * 3);
    
    var offset = n_boxes * 3;
    for (let idx = 0; idx < n_boxes; idx++) {
        const entry = jsonData[idx];
        // console.log(entry);
        aabbs[idx * 8 + 0] = entry.mins[0];
        aabbs[idx * 8 + 1] = entry.mins[1];
        aabbs[idx * 8 + 2] = entry.mins[2];

        aabbs[idx * 8 + 4 + 0] = entry.maxs[0];
        aabbs[idx * 8 + 4 + 1] = entry.maxs[1];
        aabbs[idx * 8 + 4 + 2] = entry.maxs[2];
        
        const entry_children = entry.child_indices;
        const n_entry_children = entry_children.length;
        children[idx * 3 + 0] = offset; // where the indices begins in this list.
        children[idx * 3 + 1] = n_entry_children; // number of children to loop over.
        children[idx * 3 + 2] = entry.mins[3]; // is leaf: Whether to loop over boxes or ellipses.
        for (let i = 0; i < n_entry_children; i++) {
            children[offset + i] =  entry_children[i];
        }
        offset += n_entry_children;
    }
    return new SpatialDataInfo(aabbs, children);
}


const dictionaryToArray = (dictionary) => {
    return Object.keys(dictionary).map(key => (
        dictionary[key]
    ));
};

class GaussianInfo {
    constructor(means, rots, scales, colors) {
        this.means = means;
        this.rots = rots;
        this.scales = scales;
        this.colors = colors;
    }
}

class SpatialDataInfo {
    constructor(aabbs, children) {
        this.aabbs = aabbs;
        this.children = children;
    }
}