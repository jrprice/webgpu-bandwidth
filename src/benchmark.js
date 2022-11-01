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
let iterationsCoalesced;
let iterationsRandom;
// WebGPU objects.
let device = null;
let queue = null;
let aBuffer = null;
let bBuffer = null;
let cBuffer = null;
let indexBuffer1 = null;
let indexBuffer2 = null;
var Type;
(function (Type) {
    Type[Type["Int8"] = 0] = "Int8";
    Type[Type["Int16"] = 1] = "Int16";
    Type[Type["Int32"] = 2] = "Int32";
})(Type || (Type = {}));
;
var Pattern;
(function (Pattern) {
    Pattern[Pattern["coalesced"] = 0] = "coalesced";
    Pattern[Pattern["random"] = 1] = "random";
})(Pattern || (Pattern = {}));
;
const run = () => __awaiter(this, void 0, void 0, function* () {
    // Initialize the WebGPU device.
    const powerPref = document.getElementById('powerpref');
    if (!navigator.gpu) {
        setStatus(null, 'WebGPU not supported (or bad Origin Trial token).');
        return;
    }
    const adapter = yield navigator.gpu.requestAdapter({
        powerPreference: powerPref.selectedOptions[0].value,
    });
    if (!adapter) {
        setStatus(null, 'Failed to get a WebGPU adapter.');
        return;
    }
    device = yield adapter.requestDevice();
    if (!device) {
        setStatus(null, 'Failed to get a WebGPU device.');
        return;
    }
    queue = device.queue;
    // Get the selected number from a drop-down menu.
    const getSelectedNumber = (id) => {
        let list = document.getElementById(id);
        return Number(list.selectedOptions[0].value);
    };
    arraySize = getSelectedNumber('arraysize') * 1024;
    workgroupSize = getSelectedNumber('wgsize');
    iterationsCoalesced = getSelectedNumber('iterations-coalesced');
    iterationsRandom = getSelectedNumber('iterations-random');
    // Generate two random index buffers.
    indexBuffer1 = device.createBuffer({
        size: arraySize * 4,
        usage: GPUBufferUsage.STORAGE,
        mappedAtCreation: true,
    });
    indexBuffer2 = device.createBuffer({
        size: arraySize * 4,
        usage: GPUBufferUsage.STORAGE,
        mappedAtCreation: true,
    });
    // Initialize the indices as a sequence from 0...(N-1).
    let indices1 = new Uint32Array(indexBuffer1.getMappedRange());
    let indices2 = new Uint32Array(indexBuffer2.getMappedRange());
    for (let i = 0; i < arraySize; i++) {
        indices1[i] = i;
        indices2[i] = i;
    }
    // Perform a Fisher–Yates shuffle on each index array.
    let i = arraySize;
    while (i != 0) {
        const random1 = Math.floor(Math.random() * i);
        const random2 = Math.floor(Math.random() * i);
        i--;
        [indices1[i], indices1[random1]] = [indices1[random1], indices1[i]];
        [indices2[i], indices2[random2]] = [indices2[random2], indices2[i]];
    }
    indexBuffer1.unmap();
    indexBuffer2.unmap();
    // Run all the benchmarks.
    for (const pattern of [Pattern.coalesced, Pattern.random]) {
        yield run_single(Type.Int32, pattern);
        yield run_single(Type.Int16, pattern);
        yield run_single(Type.Int8, pattern);
    }
});
const run_single = (type, pattern) => __awaiter(this, void 0, void 0, function* () {
    // Generate the type-specific parts of the shader.
    let name = '<unknown>';
    let bytesPerElement = 0;
    let iterations = 0;
    let storeType = '<not-set>';
    let loadStore = '';
    switch (type) {
        case Type.Int32:
            name = 'Int32';
            bytesPerElement = 4;
            storeType = 'u32';
            loadStore = `
fn load(i : u32) -> u32 {
  return in[i];
}
fn store(i : u32, value : u32) {
  out[i] = value;
}
    `;
            break;
        case Type.Int16:
            name = 'Int16';
            bytesPerElement = 2;
            storeType = 'atomic<u32>';
            loadStore = `
fn load(i : u32) -> u32 {
  let word = atomicLoad(&in[i / 2u]);
  return (word >> ((i%2u)*16u)) & 0xFFFFu;
}
fn store(i : u32, value : u32) {
  let prev = atomicLoad(&out[i / 2u]);
  let shift = (i % 2u) * 16u;
  let mask = (prev ^ (value<<shift)) & (0xFFFFu << shift);
  atomicXor(&out[i / 2u], mask);
}
    `;
            break;
        case Type.Int8:
            name = 'Int8';
            bytesPerElement = 1;
            storeType = 'atomic<u32>';
            loadStore = `
fn load(i : u32) -> u32 {
  let word = atomicLoad(&in[i / 4u]);
  return (word >> ((i%4u)*8u)) & 0xFFu;
}
fn store(i : u32, value : u32) {
  let prev = atomicLoad(&out[i / 4u]);
  let shift = (i % 4u) * 8u;
  let mask = (prev ^ (value<<shift)) & (0xFFu << shift);
  atomicXor(&out[i / 4u], mask);
}
    `;
            break;
    }
    // Generate the buffer indexing expression from the access pattern.
    let index;
    switch (pattern) {
        case Pattern.coalesced:
            name += "-coalesced";
            iterations = iterationsCoalesced;
            index = `gid.x`;
            break;
        case Pattern.random:
            name += "-random";
            iterations = iterationsRandom;
            index = `indices[gid.x]`;
            break;
    }
    // Construct the full shader source.
    let wgsl = `
  @group(0) @binding(0) var<storage, read_write> in : array<${storeType}>;
  @group(0) @binding(1) var<storage, read_write> out : array<${storeType}>;
  @group(0) @binding(2) var<storage, read_write> indices : array<u32>;`;
    wgsl += loadStore;
    wgsl += `
  @compute @workgroup_size(${workgroupSize})
  fn run(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = ${index};
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
        layout: "auto",
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
    //   b = a [index1]
    //   c = b [index2]
    //   a = c [index1]
    //   b = a [index2]
    //   c = b [index1]
    //   a = c [index2]
    let bindGroups = [];
    for (let i = 0; i < 6; i++) {
        let entries = [
            {
                binding: 0, resource: { buffer: [aBuffer, bBuffer, cBuffer][i % 3] },
            },
            {
                binding: 1, resource: { buffer: [bBuffer, cBuffer, aBuffer][i % 3] },
            },
        ];
        if (pattern === Pattern.random) {
            entries.push({
                binding: 2, resource: { buffer: [indexBuffer1, indexBuffer2][i % 2] },
            });
        }
        bindGroups.push(device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0), entries,
        }));
    }
    // Do a single warm-up run to make sure all resources are ready.
    {
        const commandEncoder = device.createCommandEncoder();
        const initEncoder = commandEncoder.beginComputePass();
        initEncoder.setPipeline(pipeline);
        initEncoder.setBindGroup(0, bindGroups[0]);
        initEncoder.dispatchWorkgroups(arraySize / workgroupSize);
        initEncoder.end();
        queue.submit([commandEncoder.finish()]);
        yield queue.onSubmittedWorkDone();
    }
    // Set up the shader invocations.
    const commandEncoder = device.createCommandEncoder();
    const initEncoder = commandEncoder.beginComputePass();
    initEncoder.setPipeline(pipeline);
    for (let i = 0; i < iterations; i++) {
        initEncoder.setBindGroup(0, bindGroups[i % 3]);
        initEncoder.dispatchWorkgroups(arraySize / workgroupSize);
    }
    initEncoder.end();
    // Submit the commands.
    setStatus(name, `Running...`);
    const start = performance.now();
    queue.submit([commandEncoder.finish()]);
    yield queue.onSubmittedWorkDone();
    const end = performance.now();
    setStatus(name, `Validating...`);
    yield validate(name, type, bytesPerElement, iterations);
    // Output the runtime and achieved bandwidth.
    const ms = end - start;
    const dataMoved = arraySize * bytesPerElement * 2 * iterations;
    const bytesPerSecond = dataMoved / (ms / 1000.0);
    const msStr = ms.toFixed(1).padStart(7);
    const gbsStr = (bytesPerSecond * 1e-9).toFixed(1).padStart(6);
    let result = msStr + ' ms';
    if (pattern !== Pattern.random) {
        result += `   ${gbsStr} GB/s`;
    }
    setStatus(name, result);
});
function validate(name, type, bytesPerElement, iterations) {
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
    if (name) {
        document.getElementById(name + "-status").innerHTML = status;
    }
    else {
        document.getElementById("status").innerHTML = status;
    }
}
document.querySelector('#run').addEventListener('click', run);
