import shaders from './shaders.wgsl'

const run = async () => {
  // Initialize the WebGPU device.
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();
  const queue = device.queue;

  document.getElementById("status").innerHTML = 'Running...';

  // TODO: Implement benchmark.

  document.getElementById("status").innerHTML = 'Finished.';
}

document.querySelector('#run').addEventListener('click', run);
