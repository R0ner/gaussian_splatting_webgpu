"use strict";
window.onload = function () { main(); }

const shader_path = "ellipsoid_primitive.wgsl";

async function fetchShaderCode(url) {
    const response = await fetch(url);
    return await response.text();
}

async function main() {
    var fpscounter = document.getElementById("fps-counter");
    var update_check = document.getElementById("update");


    update_check.oninput = function(event) {
        update = update_check.checked;
        console.log(update_check.checked);
    }

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
            targets: [
                { format: canvasFormat },
                { format: "rgba32float"},
            ]
        },
        primitive: {
            topology: "triangle-strip",
        },
    });
    
    var buffers = new Object()

    // Create uniform buffer
    buffers.uniforms = device.createBuffer({
        size: 28, // number of bytes 2 * 4 = 8 bytes for floats and 5 * 4 = 16 bytes for selections.
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    var obj_filename = "../resources/objects/CornellBox.obj";
    var drawingInfo = await readOBJFile(obj_filename, 1, true); // file name, scale, ccw vertices
    buffers = load_object(drawingInfo, device, buffers); 

    let textures = new Object();
    textures.width = canvas.width;
    textures.height = canvas.height;
    textures.renderSrc = device.createTexture({
        size: [canvas.width, canvas.height],
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        format: "rgba32float",
    });
    textures.renderDst = device.createTexture({
        size: [canvas.width, canvas.height],
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        format: "rgba32float",
    });

    var bindGroup = get_bindgroup(buffers, device, pipeline, textures);

    var cam_const = 1.0;
    var aspect = 1.0;
    var use_texture = 1;

    var frame_count = 0;
    var render_object = 1;
    var uniforms = new Float32Array([aspect, cam_const]);
    var uniforms_selection = new Uint32Array([canvas.width, canvas.height, use_texture, frame_count, render_object]);

    var last_time = performance.now();
    var current_time = performance.now();
    var last_frame_count = 0;
    var update = true;
    function render(device, context, pipeline, textures, bindGroup) {
        current_time = performance.now();

        // Calculate FPS every second
        if (current_time - last_time >= 1000) {
            fpscounter.textContent = `FPS: ${frame_count - last_frame_count}`;
            last_time = current_time;
            last_frame_count = frame_count;
        }

        if (update) {
            frame_count++;
    
            // Pass uniforms.
            uniforms[0] = aspect;
            uniforms[1] = cam_const;
            uniforms_selection[0] = canvas.width;
            uniforms_selection[1] = canvas.height;
            uniforms_selection[2] = use_texture;
            uniforms_selection[3] = frame_count;
            uniforms_selection[4] = render_object;
            device.queue.writeBuffer(buffers.uniforms, 0, uniforms);
            device.queue.writeBuffer(buffers.uniforms, 8, uniforms_selection);

            // Create a render pass in a command buffer and submit it.
            let encoder = device.createCommandEncoder();
            let pass = encoder.beginRenderPass({
                colorAttachments: [
                    {
                        view: context.getCurrentTexture().createView(), 
                        loadOp: "clear",
                        storeOp: "store",
                    },
                    { 
                        view: textures.renderSrc.createView(), 
                        loadOp: "load", 
                        storeOp: "store", 
                    },
                ]
            });
            // Render pass commands
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.draw(4);
            pass.end();

            encoder.copyTextureToTexture(
                { texture: textures.renderSrc }, 
                { texture: textures.renderDst },
                [textures.width, textures.height]
            );
            
            device.queue.submit([encoder.finish()]);

            // update = false;
        }
        window.requestAnimationFrame(function () { render(device, context, pipeline, textures, bindGroup) });
    }
    render(device, context, pipeline, textures, bindGroup);
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

    buffers.light_indices = device.createBuffer({
        size: drawingInfo.light_indices.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
    });
    device.queue.writeBuffer(buffers.light_indices, 0, drawingInfo.light_indices);

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

function get_bindgroup(buffers, device, pipeline, textures) {
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: buffers.uniforms } },
            { binding: 1, resource: textures.renderDst.createView() },
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