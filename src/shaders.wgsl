struct Array {
  data : array<f32>;
};

[[group(0), binding(0)]] var<storage, read_write> in : Array;
[[group(0), binding(1)]] var<storage, read_write> out : Array;

[[stage(compute), workgroup_size(256)]]
fn run([[builtin(global_invocation_id)]] gid : vec3<u32>) {
  let i = gid.x;
  out.data[i] = in.data[i] + 1.0;
}
