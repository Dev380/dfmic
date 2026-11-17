//! The three DeepFilterNet3_ll graphs, driven one frame at a time on the NPU.
//!
//! Contract with `tools/dfnpy/convert.py` (which produces the IR):
//!   * every graph takes a fixed window of `frames` frames plus each GRU's
//!     output history, and returns outputs for the whole window plus the state
//!     after the newest frame;
//!   * only the **newest** frame of a windowed output is valid, because the
//!     causal convolutions need their predecessors as history;
//!   * each GRU advances exactly one frame per inference.
//!
//! So this type keeps two histories per graph — a ring of past input frames (for
//! the convolutions) and a ring of past GRU outputs (for the recurrence) — and
//! hands back one frame's worth of outputs per call.

pub mod ffi;

use std::collections::VecDeque;
use std::ffi::{c_void, CStr};
use std::path::Path;

use dfn_dsp::{EncoderOutput, ModelRuntime, NB_DF, NB_ERB, DF_ORDER};
use num_complex::Complex32;

use crate::ffi::{
    cstring, ov_compiled_model_t, ov_core_t, ov_infer_request_t, ov_property, ov_tensor_t,
};

/// Per-graph shape of the conversion, mirroring `GRAPH_FRAMES` in convert.py.
struct GraphSpec {
    name: &'static str,
    frames: usize,
    /// Number of leading inputs that carry audio features (the rest are state).
    n_inputs: usize,
    /// Time axis of each of those inputs.
    input_time_axis: &'static [usize],
    /// Number of leading outputs that carry results (the rest are state).
    n_outputs: usize,
    /// Time axis of each of those outputs.
    output_time_axis: &'static [usize],
}

const ENC: GraphSpec = GraphSpec {
    name: "enc",
    frames: 6,
    n_inputs: 2,
    input_time_axis: &[2, 2],
    n_outputs: 7,
    output_time_axis: &[2, 2, 2, 2, 1, 2, 1],
};

const ERB_DEC: GraphSpec = GraphSpec {
    name: "erb_dec",
    frames: 3,
    n_inputs: 5,
    input_time_axis: &[1, 2, 2, 2, 2],
    n_outputs: 1,
    output_time_axis: &[2],
};

const DF_DEC: GraphSpec = GraphSpec {
    name: "df_dec",
    frames: 3,
    n_inputs: 2,
    input_time_axis: &[1, 2],
    n_outputs: 2,
    output_time_axis: &[1, 1],
};

fn product(dims: &[i64]) -> usize {
    dims.iter().product::<i64>() as usize
}

/// Write one frame's block into a windowed tensor at position `t` along `axis`.
fn scatter_frame(dest: &mut [f32], dims: &[i64], axis: usize, t: usize, src: &[f32]) {
    let block = product(&dims[axis + 1..]);
    let blocks = product(&dims[..axis]);
    let stride = block * dims[axis] as usize;
    for b in 0..blocks {
        let at = b * stride + t * block;
        dest[at..at + block].copy_from_slice(&src[b * block..(b + 1) * block]);
    }
}

/// Read one frame's block out of a windowed tensor at position `t` along `axis`.
fn gather_frame(dest: &mut [f32], src: &[f32], dims: &[i64], axis: usize, t: usize) {
    let block = product(&dims[axis + 1..]);
    let blocks = product(&dims[..axis]);
    let stride = block * dims[axis] as usize;
    for b in 0..blocks {
        let at = b * stride + t * block;
        dest[b * block..(b + 1) * block].copy_from_slice(&src[at..at + block]);
    }
}

#[derive(Clone)]
struct Port {
    name: String,
    dims: Vec<i64>,
    elems: usize,
}

/// Create a tensor over a host buffer and bind it to an infer request input.
fn set_input(
    req: *mut ov_infer_request_t,
    name: &str,
    dims: &[i64],
    data: &mut [f32],
    keep: &mut Vec<*mut ov_tensor_t>,
) -> Result<(), String> {
    let mut shape = ffi::ov_shape {
        rank: dims.len() as i64,
        dims: std::ptr::null_mut(),
    };
    ffi::check(
        "ov_shape_create",
        unsafe {
            ffi::ov_shape_create(
                dims.len() as i64,
                dims.as_ptr(),
                &mut shape as *mut _ as *mut ffi::ov_shape_t,
            )
        },
    )?;
    let mut tensor: *mut ov_tensor_t = std::ptr::null_mut();
    let status = unsafe {
        ffi::ov_tensor_create_from_host_ptr(
            ffi::OV_ELEMENT_TYPE_F32,
            shape,
            data.as_mut_ptr() as *mut c_void,
            &mut tensor,
        )
    };
    unsafe { ffi::ov_shape_free(&mut shape as *mut _ as *mut ffi::ov_shape_t) };
    ffi::check("ov_tensor_create_from_host_ptr", status)?;
    keep.push(tensor);

    let cname = cstring(name);
    ffi::check(
        "ov_infer_request_set_tensor",
        unsafe { ffi::ov_infer_request_set_tensor(req, cname.as_ptr(), tensor) },
    )
    .map_err(String::from)
}

/// Read output tensor `index` into `dest`, keeping only the newest frame of a
/// windowed output (or the whole tensor when `time_axis` is `None`).
fn read_output(
    req: *mut ov_infer_request_t,
    index: usize,
    dims: &[i64],
    time_axis: Option<usize>,
    frames: usize,
    dest: &mut [f32],
) -> Result<(), String> {
    let mut tensor: *mut ov_tensor_t = std::ptr::null_mut();
    ffi::check(
        "ov_infer_request_get_output_tensor_by_index",
        unsafe { ffi::ov_infer_request_get_output_tensor_by_index(req, index, &mut tensor) },
    )?;
    let mut ptr: *mut c_void = std::ptr::null_mut();
    ffi::check(
        "ov_tensor_data",
        unsafe { ffi::ov_tensor_data(tensor, &mut ptr) },
    )?;
    let mut elems = 0usize;
    ffi::check(
        "ov_tensor_get_size",
        unsafe { ffi::ov_tensor_get_size(tensor, &mut elems) },
    )?;
    let src = unsafe { std::slice::from_raw_parts(ptr as *const f32, elems) };
    if std::env::var("DFMIC_DEBUG_RAW").is_ok() {
        eprintln!("RAW out[{index}] dims={dims:?} elems={elems} first={:?}", &src[..elems.min(8)]);
    }

    match time_axis {
        Some(axis) => gather_frame(dest, src, dims, axis, frames - 1),
        None => {
            let n = dest.len().min(elems);
            dest[..n].copy_from_slice(&src[..n]);
        }
    }
    unsafe { ffi::ov_tensor_free(tensor) };
    Ok(())
}

/// One converted graph with its state and history.
pub struct Graph {
    cm: *mut ov_compiled_model_t,
    req: *mut ov_infer_request_t,
    spec: &'static GraphSpec,
    inputs: Vec<Port>,
    /// State inputs, sized per frame (not per window).
    states: Vec<Port>,
    outputs: Vec<Port>,
    /// (output index, state index) for each state output.
    state_outs: Vec<(usize, usize)>,
    /// Reusable host buffers for building inputs.
    in_buf: Vec<Vec<f32>>,
    /// Past frames per input, oldest first, always exactly `frames` long.
    in_hist: Vec<VecDeque<Vec<f32>>>,
    /// Past GRU outputs per state, oldest first, always `frames - 1` long.
    state_hist: Vec<VecDeque<Vec<f32>>>,
    out_buf: Vec<Vec<f32>>,
    pub infer_count: u64,
}

impl Graph {
    fn compile(
        core: *mut ov_core_t,
        xml: &Path,
        spec: &'static GraphSpec,
        device: &str,
        precision: &str,
    ) -> Result<Self, String> {
        let path = cstring(&xml.to_string_lossy());
        let dev = cstring(device);
        let key = cstring("INFERENCE_PRECISION_HINT");
        let val = cstring(precision);
        let props = [ov_property {
            key: key.as_ptr(),
            value: val.as_ptr() as *const c_void,
        }];
        let mut cm: *mut ov_compiled_model_t = std::ptr::null_mut();
        let status = unsafe {
            ffi::ov_core_compile_model_from_file_props(
                core,
                path.as_ptr(),
                dev.as_ptr(),
                1,
                props.as_ptr(),
                &mut cm,
            )
        };
        ffi::check("ov_core_compile_model_from_file_props", status)
            .map_err(|e| format!("compiling {} on {}: {e}", spec.name, device))?;

        let mut req: *mut ov_infer_request_t = std::ptr::null_mut();
        ffi::check(
            "ov_compiled_model_create_infer_request",
            unsafe { ffi::ov_compiled_model_create_infer_request(cm, &mut req) },
        )?;

        let mut graph = Self {
            cm,
            req,
            spec,
            inputs: Vec::new(),
            states: Vec::new(),
            outputs: Vec::new(),
            state_outs: Vec::new(),
            in_buf: Vec::new(),
            in_hist: Vec::new(),
            state_hist: Vec::new(),
            out_buf: Vec::new(),
            infer_count: 0,
        };
        graph.discover_ports()?;
        graph.allocate();
        Ok(graph)
    }

    fn discover_ports(&mut self) -> Result<(), String> {
        let mut n_in = 0usize;
        let mut n_out = 0usize;
        ffi::check(
            "ov_compiled_model_inputs_size",
            unsafe { ffi::ov_compiled_model_inputs_size(self.cm, &mut n_in) },
        )?;
        ffi::check(
            "ov_compiled_model_outputs_size",
            unsafe { ffi::ov_compiled_model_outputs_size(self.cm, &mut n_out) },
        )?;

        for i in 0..n_in {
            self.inputs.push(self.input_port(i)?);
        }
        for i in 0..n_out {
            self.outputs.push(self.output_port(i)?);
        }

        // The converter appends state tensors after the feature tensors, so the
        // first `n_inputs` inputs are features and the rest are state.
        if self.inputs.len() < self.spec.n_inputs || self.outputs.len() < self.spec.n_outputs {
            return Err(format!(
                "{}: expected at least {} inputs and {} outputs, model has {} and {}",
                self.spec.name,
                self.spec.n_inputs,
                self.spec.n_outputs,
                self.inputs.len(),
                self.outputs.len()
            ));
        }

        let state_inputs: Vec<Port> = self.inputs.drain(self.spec.n_inputs..).collect();
        let state_outputs: Vec<Port> = self.outputs.drain(self.spec.n_outputs..).collect();
        if state_inputs.len() != state_outputs.len() {
            return Err(format!(
                "{}: {} state inputs but {} state outputs",
                self.spec.name,
                state_inputs.len(),
                state_outputs.len()
            ));
        }

        // State inputs are [frames - 1, 1, H]; the matching outputs are [1, H].
        let mut states = Vec::new();
        for (i, port) in state_inputs.iter().enumerate() {
            if port.elems % (self.spec.frames - 1) != 0 {
                return Err(format!(
                    "{}: state input {} has {} elements, not divisible by {}",
                    self.spec.name,
                    port.name,
                    port.elems,
                    self.spec.frames - 1
                ));
            }
            let per_frame = port.elems / (self.spec.frames - 1);
            if per_frame != state_outputs[i].elems {
                return Err(format!(
                    "{}: state {} has {per_frame} elements per frame but its output has {}",
                    self.spec.name, port.name, state_outputs[i].elems
                ));
            }
            states.push(Port {
                name: port.name.clone(),
                dims: port.dims.clone(),
                elems: per_frame,
            });
            self.state_outs.push((self.spec.n_outputs + i, i));
        }
        self.states = states;

        // Validate the window shape, so a converter change cannot silently corrupt
        // audio by feeding misaligned frames.
        for (i, port) in self.inputs.iter().enumerate() {
            let axis = self.spec.input_time_axis[i];
            if port.dims.get(axis).copied() != Some(self.spec.frames as i64) {
                return Err(format!(
                    "{}: input {} has shape {:?}, expected axis {axis} to be {}",
                    self.spec.name, port.name, port.dims, self.spec.frames
                ));
            }
        }
        for (i, port) in self.outputs.iter().enumerate() {
            let axis = self.spec.output_time_axis[i];
            if port.dims.get(axis).copied() != Some(self.spec.frames as i64) {
                return Err(format!(
                    "{}: output {} has shape {:?}, expected axis {axis} to be {}",
                    self.spec.name, port.name, port.dims, self.spec.frames
                ));
            }
        }
        Ok(())
    }

    fn port_at(&self, index: usize, output: bool) -> Result<Port, String> {
        let mut port_ptr = std::ptr::null_mut();
        if output {
            ffi::check(
                "ov_compiled_model_output_by_index",
                unsafe { ffi::ov_compiled_model_output_by_index(self.cm, index, &mut port_ptr) },
            )?;
        } else {
            ffi::check(
                "ov_compiled_model_input_by_index",
                unsafe { ffi::ov_compiled_model_input_by_index(self.cm, index, &mut port_ptr) },
            )?;
        }
        let port = unsafe { read_port(port_ptr) };
        unsafe { ffi::ov_output_const_port_free(port_ptr) };
        port
    }

    fn input_port(&self, index: usize) -> Result<Port, String> {
        self.port_at(index, false)
    }

    fn output_port(&self, index: usize) -> Result<Port, String> {
        self.port_at(index, true)
    }

    /// Elements in one frame of output `i`.
    fn per_frame(&self, i: usize, port: &Port) -> usize {
        let axis = self.spec.output_time_axis[i];
        let mut n = 1usize;
        for (d, size) in port.dims.iter().enumerate() {
            if d != axis {
                n *= *size as usize;
            }
        }
        n
    }

    fn allocate(&mut self) {
        self.in_buf = self.inputs.iter().map(|p| vec![0.0f32; p.elems]).collect();
        self.in_hist = self
            .inputs
            .iter()
            .map(|p| {
                VecDeque::from(vec![
                    vec![0.0f32; p.elems / self.spec.frames];
                    self.spec.frames
                ])
            })
            .collect();
        self.state_hist = self
            .states
            .iter()
            .map(|p| VecDeque::from(vec![vec![0.0f32; p.elems]; self.spec.frames - 1]))
            .collect();
        // Only the newest frame of a windowed output is valid, so the output
        // buffers hold exactly one frame: sizing them for the whole window would
        // hand the next graph a buffer with stale data past the valid block.
        self.out_buf = self
            .outputs
            .iter()
            .enumerate()
            .map(|(i, p)| vec![0.0f32; self.per_frame(i, p)])
            .collect();
    }

    /// Clear all history without recompiling (used when a stream restarts).
    pub fn reset(&mut self) {
        self.allocate();
        self.infer_count = 0;
    }

    /// Push one frame through the graph. `frame` holds one block per feature input.
    fn push(&mut self, frame: &[&[f32]]) -> Result<(), String> {
        assert_eq!(frame.len(), self.inputs.len(), "{}: frame arity", self.spec.name);

        for (i, value) in frame.iter().enumerate() {
            self.in_hist[i].pop_front();
            self.in_hist[i].push_back(value.to_vec());
        }

        // Build the windowed feature tensors from the frame histories.
        for i in 0..self.inputs.len() {
            let dims = self.inputs[i].dims.clone();
            let axis = self.spec.input_time_axis[i];
            let history: Vec<Vec<f32>> = self.in_hist[i].iter().cloned().collect();
            let mut buf = std::mem::take(&mut self.in_buf[i]);
            for (t, f) in history.iter().enumerate() {
                scatter_frame(&mut buf, &dims, axis, t, f);
            }
            self.in_buf[i] = buf;
        }

        // Stack the GRU state history into its [frames - 1, 1, H] tensor.
        let mut state_flat: Vec<Vec<f32>> = self
            .states
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let mut flat = Vec::with_capacity(p.elems * (self.spec.frames - 1));
                for h in self.state_hist[i].iter() {
                    flat.extend_from_slice(h);
                }
                flat
            })
            .collect();

        let req = self.req;
        let mut keep: Vec<*mut ov_tensor_t> = Vec::new();
        for i in 0..self.inputs.len() {
            set_input(req, &self.inputs[i].name, &self.inputs[i].dims, &mut self.in_buf[i], &mut keep)?;
        }
        for i in 0..self.states.len() {
            set_input(req, &self.states[i].name, &self.states[i].dims, &mut state_flat[i], &mut keep)?;
        }

        let status = unsafe { ffi::ov_infer_request_infer(req) };
        for t in keep {
            unsafe { ffi::ov_tensor_free(t) };
        }
        ffi::check("ov_infer_request_infer", status)?;
        self.infer_count += 1;

        // Read results: newest frame of windowed outputs, whole state tensors.
        for i in 0..self.outputs.len() {
            let dims = self.outputs[i].dims.clone();
            let axis = self.spec.output_time_axis[i];
            let mut buf = std::mem::take(&mut self.out_buf[i]);
            let res = read_output(req, i, &dims, Some(axis), self.spec.frames, &mut buf);
            self.out_buf[i] = buf;
            res?;
        }
        for (out_index, state_index) in self.state_outs.clone() {
            let dims = self.states[state_index].dims.clone();
            let mut next = vec![0.0f32; self.states[state_index].elems];
            read_output(req, out_index, &dims, None, 1, &mut next)?;
            self.state_hist[state_index].pop_front();
            self.state_hist[state_index].push_back(next);
        }
        Ok(())
    }

    /// Newest frame of output `i`.
    pub fn output(&self, i: usize) -> &[f32] {
        &self.out_buf[i]
    }
}

// OpenVINO's C API has no thread affinity: these objects must not be *used*
// concurrently, but moving ownership to another thread is fine, which is exactly
// what the daemon does when it hands the pipeline to its audio worker.
unsafe impl Send for Graph {}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe {
            if !self.req.is_null() {
                ffi::ov_infer_request_free(self.req);
            }
            if !self.cm.is_null() {
                ffi::ov_compiled_model_free(self.cm);
            }
        }
    }
}

unsafe fn read_port(port: *mut ffi::ov_output_const_port_t) -> Result<Port, String> {
    let mut shape = ffi::ov_shape {
        rank: 0,
        dims: std::ptr::null_mut(),
    };
    ffi::check(
        "ov_const_port_get_shape",
        unsafe {
            ffi::ov_const_port_get_shape(port, &mut shape as *mut _ as *mut ffi::ov_shape_t)
        },
    )?;
    let dims: Vec<i64> = unsafe { std::slice::from_raw_parts(shape.dims, shape.rank as usize) }.to_vec();
    unsafe { ffi::ov_shape_free(&mut shape as *mut _ as *mut ffi::ov_shape_t) };

    let mut name_ptr: *mut std::ffi::c_char = std::ptr::null_mut();
    let name = if unsafe { ffi::ov_port_get_any_name(port, &mut name_ptr) } == ffi::OK
        && !name_ptr.is_null()
    {
        let s = unsafe { CStr::from_ptr(name_ptr) }.to_string_lossy().into_owned();
        unsafe { ffi::ov_free(name_ptr) };
        s
    } else {
        String::from("<unnamed>")
    };

    Ok(Port {
        name,
        elems: product(&dims),
        dims,
    })
}

/// All three graphs plus the owning OpenVINO core.
pub struct NpuRuntime {
    core: *mut ov_core_t,
    enc: Graph,
    erb_dec: Graph,
    df_dec: Graph,
    pub device: String,
}

impl NpuRuntime {
    /// Load the converted IR from `ir_dir` and compile it for `device`.
    ///
    /// `cache_dir`, when given, lets OpenVINO cache compiled blobs so a service
    /// restart skips the multi-second NPU compile.
    pub fn load(ir_dir: &Path, device: &str, cache_dir: Option<&Path>) -> Result<Self, String> {
        let mut core: *mut ov_core_t = std::ptr::null_mut();
        ffi::check("ov_core_create", unsafe { ffi::ov_core_create(&mut core) })?;

        if let Some(dir) = cache_dir {
            let dir_s = cstring(&dir.to_string_lossy());
            let dev = cstring(device);
            let k_cache = cstring("CACHE_DIR");
            let v_mode = cstring("OPTIMIZE_SPEED");
            let k_mode = cstring("CACHE_MODE");
            let props = [
                ov_property {
                    key: k_cache.as_ptr(),
                    value: dir_s.as_ptr() as *const c_void,
                },
                ov_property {
                    key: k_mode.as_ptr(),
                    value: v_mode.as_ptr() as *const c_void,
                },
            ];
            let status =
                unsafe { ffi::ov_core_set_properties(core, dev.as_ptr(), 2, props.as_ptr()) };
            if let Err(e) = ffi::check("ov_core_set_properties", status) {
                // Caching is an optimisation; never fail the daemon over it.
                eprintln!("dfmic: warning: could not enable model caching: {e}");
            }
        }

        let enc = Graph::compile(core, &ir_dir.join("enc.xml"), &ENC, device, "f16")?;
        let erb_dec = Graph::compile(core, &ir_dir.join("erb_dec.xml"), &ERB_DEC, device, "f16")?;
        let df_dec = Graph::compile(core, &ir_dir.join("df_dec.xml"), &DF_DEC, device, "f16")?;
        Ok(Self {
            core,
            enc,
            erb_dec,
            df_dec,
            device: device.to_string(),
        })
    }

    pub fn reset(&mut self) {
        self.enc.reset();
        self.erb_dec.reset();
        self.df_dec.reset();
    }

    /// Total inferences performed, for status reporting.
    pub fn infer_count(&self) -> u64 {
        self.enc.infer_count
    }
}

unsafe impl Send for NpuRuntime {}

impl Drop for NpuRuntime {
    fn drop(&mut self) {
        unsafe {
            if !self.core.is_null() {
                ffi::ov_core_free(self.core);
            }
        }
    }
}

/// Lists devices OpenVINO can see, for `dfmic doctor`.
pub fn available_devices() -> Result<Vec<String>, String> {
    let mut core: *mut ov_core_t = std::ptr::null_mut();
    ffi::check("ov_core_create", unsafe { ffi::ov_core_create(&mut core) })?;
    let mut devices = ffi::ov_available_devices {
        devices: std::ptr::null_mut(),
        size: 0,
    };
    let status = unsafe { ffi::ov_core_get_available_devices(core, &mut devices) };
    let result = ffi::check("ov_core_get_available_devices", status);
    let out = if result.is_ok() {
        let names = unsafe { std::slice::from_raw_parts(devices.devices, devices.size) };
        names
            .iter()
            .map(|p| unsafe { CStr::from_ptr(*p) }.to_string_lossy().into_owned())
            .collect()
    } else {
        Vec::new()
    };
    unsafe {
        ffi::ov_available_devices_free(&mut devices);
        ffi::ov_core_free(core);
    }
    result.map_err(|e| e.to_string())?;
    Ok(out)
}

impl ModelRuntime for NpuRuntime {
    fn encode(&mut self, feat_erb: &[f32], feat_spec: &[f32]) -> EncoderOutput {
        self.enc
            .push(&[feat_erb, feat_spec])
            .expect("encoder inference failed");
        EncoderOutput {
            e0: self.enc.output(0).to_vec(),
            e1: self.enc.output(1).to_vec(),
            e2: self.enc.output(2).to_vec(),
            e3: self.enc.output(3).to_vec(),
            emb: self.enc.output(4).to_vec(),
            c0: self.enc.output(5).to_vec(),
            lsnr: self.enc.output(6)[0],
        }
    }

    fn erb_mask(&mut self, enc: &EncoderOutput) -> [f32; NB_ERB] {
        self.erb_dec
            .push(&[&enc.emb, &enc.e3, &enc.e2, &enc.e1, &enc.e0])
            .expect("erb decoder inference failed");
        let mut gains = [0.0f32; NB_ERB];
        gains.copy_from_slice(&self.erb_dec.output(0)[..NB_ERB]);
        gains
    }

    fn df_coefs(&mut self, enc: &EncoderOutput) -> Vec<Complex32> {
        self.df_dec
            .push(&[&enc.emb, &enc.c0])
            .expect("df decoder inference failed");
        self.df_dec
            .output(0)
            .chunks_exact(2)
            .map(|c| Complex32::new(c[0], c[1]))
            .take(NB_DF * DF_ORDER)
            .collect()
    }
}
