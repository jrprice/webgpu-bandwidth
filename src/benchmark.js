var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
// Benchmark parameters.
let arraySize;
let workgroupSize;
let iterations;
// WebGPU objects.
let device = null;
let queue = null;
let aBuffer = null;
let bBuffer = null;
let cBuffer = null;
var Type;
(function (Type) {
    Type[Type["Int8"] = 0] = "Int8";
    Type[Type["Int16"] = 1] = "Int16";
    Type[Type["Int32"] = 2] = "Int32";
})(Type || (Type = {}));
;
const run = () => __awaiter(this, void 0, void 0, function* () {
    // Initialize the WebGPU device.
    const powerPref = document.getElementById('powerpref');
    const adapter = yield navigator.gpu.requestAdapter({
        powerPreference: powerPref.selectedOptions[0].value,
    });
    device = yield adapter.requestDevice();
    queue = device.queue;
    // Get the selected number from a drop-down menu.
    const getSelectedNumber = (id) => {
        let list = document.getElementById(id);
        return Number(list.selectedOptions[0].value);
    };
    arraySize = getSelectedNumber('arraysize') * 1024 * 1024;
    workgroupSize = getSelectedNumber('wgsize');
    iterations = getSelectedNumber('iterations');
    yield run_single(Type.Int32);
    yield run_single(Type.Int16);
    yield run_single(Type.Int8);
});
const run_single = (type) => __awaiter(this, void 0, void 0, function* () {
    // Generate the type-specific parts of the shader.
    let name = '<unknown>';
    let bytesPerElement = 0;
    let storeType = '<not-set>';
    let loadStore = '';
    switch (type) {
        case Type.Int32:
            name = 'Int32';
            bytesPerElement = 4;
            storeType = 'u32';
            loadStore = `
fn load(i : u32) -> u32 {
  return in.data[i];
}
fn store(i : u32, value : u32) {
  out.data[i] = value;
}
    `;
            break;
        case Type.Int16:
            name = 'Int16';
            bytesPerElement = 2;
            storeType = 'atomic<u32>';
            loadStore = `
fn load(i : u32) -> u32 {
  let word = atomicLoad(&in.data[i / 2u]);
  return (word >> ((i%2u)*16u)) & 0xFFFFu;
}
fn store(i : u32, value : u32) {
  let prev = atomicLoad(&out.data[i / 2u]);
  let shift = (i % 2u) * 16u;
  let mask = (prev ^ (value<<shift)) & (0xFFFFu << shift);
  atomicXor(&out.data[i / 2u], mask);
}
    `;
            break;
        case Type.Int8:
            name = 'Int8';
            bytesPerElement = 1;
            storeType = 'atomic<u32>';
            loadStore = `
fn load(i : u32) -> u32 {
  let word = atomicLoad(&in.data[i / 4u]);
  return (word >> ((i%4u)*8u)) & 0xFFu;
}
fn store(i : u32, value : u32) {
  let prev = atomicLoad(&out.data[i / 4u]);
  let shift = (i % 4u) * 8u;
  let mask = (prev ^ (value<<shift)) & (0xFFu << shift);
  atomicXor(&out.data[i / 4u], mask);
}
    `;
            break;
    }
    // Construct the full shader source.
    let wgsl = `
  struct Array {
    data : array<${storeType}>;
  };
  [[group(0), binding(0)]] var<storage, read_write> in : Array;
  [[group(0), binding(1)]] var<storage, read_write> out : Array;`;
    wgsl += loadStore;
    wgsl += `
  [[stage(compute), workgroup_size(${workgroupSize})]]
  fn run([[builtin(global_invocation_id)]] gid : vec3<u32>) {
    let i = gid.x;
    store(i, load(i) + 1u);
  }
`;
    // Create the compute pipeline.
    const module = device.createShaderModule({ code: wgsl });
    const pipeline = device.createComputePipeline({
        compute: {
            module: module,
            entryPoint: 'run',
        },
    });
    // Create the buffers.
    aBuffer = device.createBuffer({
        size: arraySize * bytesPerElement,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    bBuffer = device.createBuffer({
        size: arraySize * bytesPerElement,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    cBuffer = device.createBuffer({
        size: arraySize * bytesPerElement,
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
    // Do a single warm-up run to make sure all resources are ready.
    {
        const commandEncoder = device.createCommandEncoder();
        const initEncoder = commandEncoder.beginComputePass();
        initEncoder.setPipeline(pipeline);
        initEncoder.setBindGroup(0, bindGroups[0]);
        initEncoder.dispatch(arraySize / workgroupSize);
        initEncoder.endPass();
        queue.submit([commandEncoder.finish()]);
        yield queue.onSubmittedWorkDone();
    }
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
    setStatus(name, `Running...`);
    const start = performance.now();
    queue.submit([commandEncoder.finish()]);
    yield queue.onSubmittedWorkDone();
    const end = performance.now();
    setStatus(name, `Validating...`);
    yield validate(name, type, bytesPerElement);
    // Output the runtime and achieved bandwidth.
    const ms = end - start;
    const dataMoved = arraySize * bytesPerElement * 2 * iterations;
    const bytesPerSecond = dataMoved / (ms / 1000.0);
    const gbytesPerSecond = (bytesPerSecond * 1e-9).toFixed(1);
    setStatus(name, ms.toFixed(1) + ` ms  (${gbytesPerSecond} GB/s)`);
});
function validate(name, type, bytesPerElement) {
    return __awaiter(this, void 0, void 0, function* () {
        // Map the final output onto the host.
        const stagingBuffer = device.createBuffer({
            size: arraySize * bytesPerElement,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        let finalBuffer = [aBuffer, bBuffer, cBuffer][iterations % 3];
        const commandEncoder = device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(finalBuffer, 0, stagingBuffer, 0, arraySize * bytesPerElement);
        queue.submit([commandEncoder.finish()]);
        yield stagingBuffer.mapAsync(GPUMapMode.READ).then(null, () => setStatus(name, 'Failed to map buffer.'));
        const mapped = stagingBuffer.getMappedRange();
        // Check that all values are correct.
        let values;
        let ref = iterations;
        switch (type) {
            case Type.Int8:
                values = new Uint8Array(mapped);
                ref %= 256;
                break;
            case Type.Int16:
                values = new Uint16Array(mapped);
                ref %= 65536;
                break;
            case Type.Int32:
                values = new Uint32Array(mapped);
                break;
        }
        if (!values.every((value) => value === ref)) {
            const idx = values.findIndex((value, index) => value !== ref);
            console.log("Error at index " + idx + ": " + values[idx] + " != " + ref);
            setStatus(name, "Validation failed.");
            throw 'validation failed';
        }
    });
}
function setStatus(name, status) {
    document.getElementById(name + "-status").innerHTML = status;
}
document.querySelector('#run').addEventListener('click', run);
