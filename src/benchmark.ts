import shaders from './shaders.wgsl'

// Benchmark parameters.
// TODO: Make these configurable.
let arraySize = 8 * 1024 * 1024;
let workgroupSize = 256; // TODO: Use overridable constant to set in shader.
let iterations = 1000;

// WebGPU objects.
let device: GPUDevice = null;
let queue: GPUQueue = null;
let aBuffer: GPUBuffer = null;
let bBuffer: GPUBuffer = null;
let cBuffer: GPUBuffer = null;

const run = async () => {
  setStatus('Initializing...');

  // Initialize the WebGPU device.
  const adapter = await navigator.gpu.requestAdapter();
  device = await adapter.requestDevice();
  queue = device.queue;

  // Create the compute pipelines.
  const module = device.createShaderModule({ code: shaders });
  const pipeline = device.createComputePipeline({
    compute: {
      module: module,
      entryPoint: 'run',
    },
  });

  // Create the buffers.
  aBuffer = device.createBuffer({
    size: arraySize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  bBuffer = device.createBuffer({
    size: arraySize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  cBuffer = device.createBuffer({
    size: arraySize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  // Create the bind groups.
  // Cycle through buffers with this pattern:
  //   b = a
  //   c = b
  //   a = c
  const bindGroups = [
    device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0, resource: { buffer: aBuffer },
        },
        {
          binding: 1, resource: { buffer: bBuffer },
        },
      ],
    }),
    device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0, resource: { buffer: bBuffer },
        },
        {
          binding: 1, resource: { buffer: cBuffer },
        },
      ],
    }),
    device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0, resource: { buffer: cBuffer },
        },
        {
          binding: 1, resource: { buffer: aBuffer },
        },
      ],
    })
  ];

  // Set up the shader invocations.
  const commandEncoder = device.createCommandEncoder();
  const initEncoder = commandEncoder.beginComputePass();
  initEncoder.setPipeline(pipeline);
  for (let i = 0; i < iterations; i++) {
    initEncoder.setBindGroup(0, bindGroups[i % 3]);
    initEncoder.dispatch(arraySize / workgroupSize);
  }
  initEncoder.endPass();

  // Submit the commands.
  setStatus('Running...');
  queue.submit([commandEncoder.finish()]);
  queue.onSubmittedWorkDone().then(
    () => validate(),
    () => setStatus('Kernel execution failed.'));
}

async function validate() {
  setStatus('Validating...');

  // Map the final output onto the host.
  const stagingBuffer = device.createBuffer({
    size: arraySize * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let finalBuffer = [aBuffer, bBuffer, cBuffer][iterations % 3];
  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(finalBuffer, 0,
    stagingBuffer, 0,
    arraySize * 4);
  queue.submit([commandEncoder.finish()]);
  await stagingBuffer.mapAsync(GPUMapMode.READ, 0, arraySize * 4).then(
    null, () => setStatus('Failed to map buffer.'));

  // Check that all values are correct.
  const values = new Float32Array(
    stagingBuffer.getMappedRange(0, arraySize * 4)
  );
  if (!values.every((value: number) => value === iterations)) {
    const idx =
      values.findIndex((value: number, index: number) => value !== iterations);
    console.log(
      "Error at index " + idx + ": " + values[idx] + " != " + iterations);
    setStatus("Validation failed.");
  } else {
    setStatus('Finished.');
  }
}

function setStatus(status: string) {
  document.getElementById("status").innerHTML = status;
}

document.querySelector('#run').addEventListener('click', run);
