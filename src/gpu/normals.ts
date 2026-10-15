import faceNormalsShader from "../shaders/faceNormals.wgsl?raw";
import vertexNormalsShader from "../shaders/vertexNormals.wgsl?raw";

const WG_SIZE = 64; // matches @workgroup_size in the compute shaders

export interface NormalTargets {
  positionBuffer: GPUBuffer;
  indexBuffer: GPUBuffer;
  normalBuffer: GPUBuffer;
  csrOffsetBuffer: GPUBuffer;
  csrFaceBuffer: GPUBuffer;
  faceCount: number;
  vertexCount: number;
}

export interface NormalGenerator {
  generate(targets: NormalTargets): void;
}

//  pass 1 (per triangle): un-normalized face normals,
//  pass 2 (per vertex): gather incident faces via CSR, sum, normalize, (average)
export function createNormalGenerator(device: GPUDevice): NormalGenerator {
  // create compute pipeline for face normal pass
  const facePipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: faceNormalsShader }),
      entryPoint: "main",
    },
  });

  // create compute pipeline for vertex normal pass
  const vertexPipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: vertexNormalsShader }),
      entryPoint: "main",
    },
  });

  const generate = (t: NormalTargets) => {
    // create a buffer for face normals: face count * 3 component per vector (x, y, z) * 4bytes per component (32bit)
    const faceNormalBuffer = device.createBuffer({
      size: t.faceCount * 3 * 4,
      usage: GPUBufferUsage.STORAGE,
    });

    // bind-group for the face-normal pipeline
    const faceBind = device.createBindGroup({
      layout: facePipeline.getBindGroupLayout(0),
      entries: [
        // vertex position buffer
        { binding: 0, resource: { buffer: t.positionBuffer } },
        // vertex index buffer
        { binding: 1, resource: { buffer: t.indexBuffer } },
        // face normal buffer: this will be where we write face normal
        { binding: 2, resource: { buffer: faceNormalBuffer } },
      ],
    });

    // bind group for vertex-normals pipeline
    const vertBind = device.createBindGroup({
      layout: vertexPipeline.getBindGroupLayout(0),
      entries: [
        // face normals buffer we wrote
        { binding: 0, resource: { buffer: faceNormalBuffer } },
        // the offset buffer from CSR (holds the boundaries for each vertex)
        { binding: 1, resource: { buffer: t.csrOffsetBuffer } },
        // the face buffer holds the face indices each vertex touches
        { binding: 2, resource: { buffer: t.csrFaceBuffer } },
        // this is the vertex normal buffer we will write into
        { binding: 3, resource: { buffer: t.normalBuffer } },
      ],
    });

    // Two separate passes so pass 1 writes are visible to pass 2
    // create command encoder
    const encoder = device.createCommandEncoder();
    const p1 = encoder.beginComputePass();
    // set face pipeline with shader module
    p1.setPipeline(facePipeline);
    // set bindgroups with buffers for face normal computations
    p1.setBindGroup(0, faceBind);
    // dispatch workgroups which holds WG_SIZE threads  (one thread one face)
    p1.dispatchWorkgroups(Math.ceil(t.faceCount / WG_SIZE));
    // end first pass
    p1.end();

    // second pass
    const p2 = encoder.beginComputePass();
    // set vertex pipeline with shader module
    p2.setPipeline(vertexPipeline);
    // set bindgroups with buffers for vertex normal computations
    p2.setBindGroup(0, vertBind);
    // dispatch workgroups which holds WG_SIZE threads (one thread one vertex)
    p2.dispatchWorkgroups(Math.ceil(t.vertexCount / WG_SIZE));
    p2.end();
    // submit the command buffer to be executed in GPU
    device.queue.submit([encoder.finish()]);

    // Destroy is deferred until the GPU is done with the submitted work.
    faceNormalBuffer.destroy();
  };

  // return closure
  return { generate };
}
